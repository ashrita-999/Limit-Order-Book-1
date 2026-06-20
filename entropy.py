import math
import random
import heapq
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

try:
    from sortedcontainers import SortedList
except ImportError as exc:
    raise ImportError(
        "This fast LOB version requires sortedcontainers. Install it with: pip install sortedcontainers"
    ) from exc

EPS = 1e-9
CURRENT_TIME = 0

# =========================================================
# CONFIG: SHIP / ENTROPY GRID
# =========================================================

GLOBAL_SEED = 123

# Mainland (LOB economy A)
MAIN_N = 500
MAIN_BASE_MONEY = 100.0
MAIN_BASE_GOODS = 100.0

# Grid over mainland macrostate
#M_FACTORS = np.geomspace(0.2, 10, 8)
#G_FACTORS = np.geomspace(10, 10, 1)

M_FACTORS = np.linspace(1.0, 2.0, 10)   # M from 50k to 100k
G_FACTORS = np.linspace(1.0, 2.0, 10)   # G from 50k to 100k

# Ship meter
SHIP_N = 50
SHIP_BASE_MONEY = 50.0
SHIP_BASE_GOODS = 50.0
ALPHA_SHIP = 5.0
NU_SHIP = 5.0

# Timing
EQ_STEPS_UNCOUPLED = 6_000_000
STEPS_COUPLED = 5_000_000
P_CONTACT = 0.2

# Meter averaging window
METER_AVG_WINDOW = 800_000

# Number of seeds per grid point
N_SEEDS_PER_GRIDPOINT = 1

# Plotting
SHOW_3D = True

# =========================================================
# CONFIG: LOG-UTILITY LOB MODEL
# =========================================================

# Tick grid
TICK_SIZE = 0.01
P_MIN = TICK_SIZE

# Fill window
P_FILL_MIN = 1e-9
K_MAX = int(math.ceil(-math.log(P_FILL_MIN)))

# Bootstrap / seeding phase
SEED_STEPS = 100

# Log-utility action weighting
BETA_ACTION = 1.0
DO_NOTHING_FLOOR_FRAC = 1e-12
LAMBDA_FILL = 0.1

ORDER_EXP_LAM = 0.001
Q_TRADE = 1.0

# Weak contact magnitudes
EPS_M = 1.0
EPS_G = 1.0

# Meter-only measurement
METER_ONLY = True


# =========================================================
# HELPERS: TICK GRID
# =========================================================

def round_to_tick(p: float) -> float:
    if not np.isfinite(p):
        return P_MIN
    return max(P_MIN, round(p / TICK_SIZE) * TICK_SIZE)


def floor_to_tick(p: float) -> float:
    if not np.isfinite(p):
        return P_MIN
    return max(P_MIN, math.floor(p / TICK_SIZE) * TICK_SIZE)


def price_to_tick(p: float) -> int:
    return int(round(float(p) / TICK_SIZE))


def tick_to_price(tk: int) -> float:
    return float(tk) * TICK_SIZE


# =========================================================
# CORE: AGENT / ORDER
# =========================================================

class Agent:
    """
    Cobb-Douglas utility:
        u = m^(alpha-1) * g^(nu-1)

    Mainland agents are heterogeneous.
    Ship agents are homogeneous with ALPHA_SHIP, NU_SHIP.
    """
    def __init__(self, agent_id, money, goods, alpha=None, nu=None):
        self.id = int(agent_id)
        self.M = float(money)
        self.G = float(goods)

        if alpha is None:
            self.alpha = random.uniform(1.0 + 1e-6, 100.0)
        else:
            self.alpha = float(alpha)

        if nu is None:
            self.nu = random.uniform(1.0 + 1e-6, 100.0)
        else:
            self.nu = float(nu)

        self.a = (self.alpha - 1.0) / (self.alpha + self.nu - 2.0)

        # Fast per-agent resting order indices.
        # These lists are cleaned lazily inside invalidate_agent_orders.
        self.resting_bids = []
        self.resting_asks = []

    def utility_at(self, M: float, G: float) -> float:
        M = max(float(M), EPS)
        G = max(float(G), EPS)
        return (M ** (self.alpha - 1.0)) * (G ** (self.nu - 1.0))

    def utility(self) -> float:
        return self.utility_at(self.M, self.G)

    def log_utility_at(self, M: float, G: float) -> float:
        M = max(float(M), EPS)
        G = max(float(G), EPS)
        return (self.alpha - 1.0) * math.log(M) + (self.nu - 1.0) * math.log(G)

    def log_utility(self) -> float:
        return self.log_utility_at(self.M, self.G)

    def MUM(self) -> float:
        M = max(self.M, EPS)
        G = max(self.G, EPS)
        return (self.alpha - 1.0) * (M ** (self.alpha - 2.0)) * (G ** (self.nu - 1.0))

    def MUG(self) -> float:
        M = max(self.M, EPS)
        G = max(self.G, EPS)
        return (self.nu - 1.0) * (M ** (self.alpha - 1.0)) * (G ** (self.nu - 2.0))

    def implied_price(self) -> float:
        return self.MUG() / max(self.MUM(), EPS)


class Order:
    _next_uid = 0

    def __init__(self, agent, price, qty, arrival_time, lam=ORDER_EXP_LAM):
        self.uid = Order._next_uid
        Order._next_uid += 1

        self.agent = agent
        self.price = float(price)
        self.qty = float(qty)
        self.arrival_time = int(arrival_time)
        self.active = True
        lifetime = np.random.exponential(scale=1 / lam) if lam > 0 else float("inf")
        self.expiration_time = self.arrival_time + lifetime


# =========================================================
# LOG-UTILITY LOB MODEL
# =========================================================

class LOB:
    def __init__(self, global_money_cap: float):
        # Sorted ascending by price.
        # Best bid is self.bids[-1], best ask is self.asks[0].
        self.bids = SortedList(key=lambda o: o.price)
        self.asks = SortedList(key=lambda o: o.price)

        # Expiry heaps for cheap cancellation.
        # Entries are (expiration_time, uid, order). Old entries for already-filled
        # or already-removed orders are harmless because removals are lazy.
        self.bid_expiry_heap = []
        self.ask_expiry_heap = []

        self.trade_prices = []
        self.trade_qtys = []
        self.trade_times = []

        self.P_MAX = floor_to_tick(global_money_cap)
        self.MIN_TICK = price_to_tick(P_MIN)
        self.MAX_TICK = price_to_tick(self.P_MAX)

        # MH diagnostics
        self.mh_attempts = 0
        self.mh_accepts = 0
        self.mh_rejects = 0
        self.mh_logR = []

    def _is_live_bid(self, order) -> bool:
        global CURRENT_TIME
        return (
            order.active
            and order.qty > EPS
            and CURRENT_TIME < order.expiration_time
            and order.agent.M + 1e-12 >= order.price * order.qty
        )

    def _is_live_ask(self, order) -> bool:
        global CURRENT_TIME
        return (
            order.active
            and order.qty > EPS
            and CURRENT_TIME < order.expiration_time
            and order.agent.G + 1e-12 >= order.qty
        )

    def _remove_bid_order(self, order):
        order.active = False
        self.bids.discard(order)

    def _remove_ask_order(self, order):
        order.active = False
        self.asks.discard(order)

    def _clean_best_bid(self):
        while self.bids and not self._is_live_bid(self.bids[-1]):
            order = self.bids.pop(-1)
            order.active = False

    def _clean_best_ask(self):
        while self.asks and not self._is_live_ask(self.asks[0]):
            order = self.asks.pop(0)
            order.active = False

    def invalidate_agent_orders(self, agent: Agent):
        """
        Preserve the old model's affordability/inventory cancellation rule after an
        agent's holdings change, without scanning the whole book.

        Any resting bid by this agent that is no longer affordable is cancelled forever.
        Any resting ask by this agent that is no longer backed by inventory is cancelled forever.
        """
        live_bids = []
        for order in agent.resting_bids:
            if not order.active:
                continue
            if order.agent.M + 1e-12 < order.price * order.qty:
                self._remove_bid_order(order)
            else:
                live_bids.append(order)
        agent.resting_bids = live_bids

        live_asks = []
        for order in agent.resting_asks:
            if not order.active:
                continue
            if order.agent.G + 1e-12 < order.qty:
                self._remove_ask_order(order)
            else:
                live_asks.append(order)
        agent.resting_asks = live_asks

    def best_bid(self):
        self._clean_best_bid()
        return self.bids[-1] if self.bids else None

    def best_ask(self):
        self._clean_best_ask()
        return self.asks[0] if self.asks else None

    def add_bid(self, order):
        self.bids.add(order)
        order.agent.resting_bids.append(order)
        heapq.heappush(self.bid_expiry_heap, (order.expiration_time, order.uid, order))

    def add_ask(self, order):
        self.asks.add(order)
        order.agent.resting_asks.append(order)
        heapq.heappush(self.ask_expiry_heap, (order.expiration_time, order.uid, order))

    def cancel_stale_orders(self):
        """
        Cheap expiry cancellation.

        Old version scanned the whole book every step. This version only pops orders
        whose expiration time has passed. Invalid orders caused by balance/inventory
        changes are cleaned lazily when they reach the touch via best_bid/best_ask.
        """
        global CURRENT_TIME

        while self.bid_expiry_heap and self.bid_expiry_heap[0][0] <= CURRENT_TIME:
            _, _, order = heapq.heappop(self.bid_expiry_heap)
            order.active = False
            self.bids.discard(order)

        while self.ask_expiry_heap and self.ask_expiry_heap[0][0] <= CURRENT_TIME:
            _, _, order = heapq.heappop(self.ask_expiry_heap)
            order.active = False
            self.asks.discard(order)

        self._clean_best_bid()
        self._clean_best_ask()

    @staticmethod
    def _p_fill_from_k(k: int) -> float:
        return float(math.exp(-k * LAMBDA_FILL))

    def _touch_or_synthetic(self, agent: Agent) -> tuple[float, float]:
        bb = self.best_bid()
        ba = self.best_ask()

        if bb is not None and ba is not None:
            return bb.price, ba.price

        p_star = round_to_tick(max(agent.implied_price(), P_MIN))

        if bb is None and ba is None:
            b_ref = max(P_MIN, p_star - TICK_SIZE)
            a_ref = max(P_MIN, p_star)
            return b_ref, a_ref

        if bb is None and ba is not None:
            a_ref = ba.price
            b_ref = max(P_MIN, a_ref - TICK_SIZE)
            return b_ref, a_ref

        b_ref = bb.price
        a_ref = b_ref + TICK_SIZE
        return b_ref, a_ref

    def agent_generates_order(self, agent: Agent, arrival_time: int, t: int):
        """
        Fast vectorised version of the log-utility action-selection rule.

        Main speed choices:
        - Cache alpha-1, nu-1, M0, G0 and logU0 as local variables.
        - Evaluate passive bid/ask windows with NumPy arrays.
        - Store candidate actions in preallocated NumPy buffers rather than a
          Python list of tuples.
        - Sample the final action using np.cumsum + np.searchsorted.

        Intended action probabilities are unchanged. Individual trajectories are
        not bit-for-bit identical because floating-point accumulation and sampling
        order are slightly different.
        """
        if t < SEED_STEPS:
            u = random.random()
            side = "ask" if u < agent.a else "bid"

            p = round_to_tick(max(agent.implied_price(), P_MIN))
            p = min(max(p, P_MIN), self.P_MAX)

            if side == "ask" and agent.G < Q_TRADE:
                return None, None
            if side == "bid" and agent.M < p * Q_TRADE:
                return None, None

            return side, Order(agent, p, Q_TRADE, arrival_time)

        # Cache agent state and exponents locally.
        am1 = agent.alpha - 1.0
        nm1 = agent.nu - 1.0
        M0 = agent.M
        G0 = agent.G

        logM0 = math.log(max(M0, EPS))
        logG0 = math.log(max(G0, EPS))
        logU0 = am1 * logM0 + nm1 * logG0

        # Preallocate enough room for 2 market actions plus both passive windows.
        # Actual windows may be clipped by MIN_TICK/MAX_TICK and affordability.
        n_max = 2 + 2 * K_MAX
        prices_buf = np.empty(n_max, dtype=float)
        logw_buf = np.empty(n_max, dtype=float)
        sides_buf = np.empty(n_max, dtype=np.int8)  # 1 = bid, 0 = ask
        n_actions = 0

        bb = self.best_bid()
        ba = self.best_ask()

        # -------------------------
        # Market buy
        # -------------------------
        if ba is not None:
            p_exec = ba.price
            if M0 >= p_exec * Q_TRADE:
                M_after = M0 - p_exec * Q_TRADE
                G_after = G0 + Q_TRADE
                logU_after = am1 * math.log(max(M_after, EPS)) + nm1 * math.log(max(G_after, EPS))
                logw = BETA_ACTION * (logU_after - logU0)
                if math.isfinite(logw):
                    prices_buf[n_actions] = p_exec
                    logw_buf[n_actions] = logw
                    sides_buf[n_actions] = 1
                    n_actions += 1

        # -------------------------
        # Market sell
        # -------------------------
        if bb is not None:
            p_exec = bb.price
            if G0 >= Q_TRADE:
                M_after = M0 + p_exec * Q_TRADE
                G_after = G0 - Q_TRADE
                logU_after = am1 * math.log(max(M_after, EPS)) + nm1 * math.log(max(G_after, EPS))
                logw = BETA_ACTION * (logU_after - logU0)
                if math.isfinite(logw):
                    prices_buf[n_actions] = p_exec
                    logw_buf[n_actions] = logw
                    sides_buf[n_actions] = 0
                    n_actions += 1

        # -------------------------
        # Passive bids / asks
        # -------------------------
        b_ref, a_ref = self._touch_or_synthetic(agent)
        a_tk = price_to_tick(a_ref)
        b_tk = price_to_tick(b_ref)

        # Passive bids
        if M0 > EPS:
            lo = max(self.MIN_TICK, a_tk - K_MAX)
            hi = min(self.MAX_TICK, a_tk - 1)

            if hi >= lo:
                ticks = np.arange(lo, hi + 1, dtype=np.int64)
                p_arr = ticks.astype(float) * TICK_SIZE
                k_arr = a_tk - ticks

                M_after = M0 - p_arr * Q_TRADE
                affordable = M_after >= EPS

                if np.any(affordable):
                    logG_after = math.log(max(G0 + Q_TRADE, EPS))
                    logU_after = am1 * np.log(np.maximum(M_after, EPS)) + nm1 * logG_after
                    logw_arr = -k_arr * LAMBDA_FILL + BETA_ACTION * (logU_after - logU0)
                    mask = affordable & np.isfinite(logw_arr)

                    n_valid = int(np.sum(mask))
                    if n_valid > 0:
                        end = n_actions + n_valid
                        prices_buf[n_actions:end] = p_arr[mask]
                        logw_buf[n_actions:end] = logw_arr[mask]
                        sides_buf[n_actions:end] = 1
                        n_actions = end

        # Passive asks
        if G0 >= Q_TRADE:
            lo = max(self.MIN_TICK, b_tk + 1)
            hi = min(self.MAX_TICK, b_tk + K_MAX)

            if hi >= lo:
                ticks = np.arange(lo, hi + 1, dtype=np.int64)
                p_arr = ticks.astype(float) * TICK_SIZE
                k_arr = ticks - b_tk

                G_after = G0 - Q_TRADE

                if G_after >= EPS:
                    logG_after = math.log(max(G_after, EPS))
                    M_after = M0 + p_arr * Q_TRADE
                    logU_after = am1 * np.log(np.maximum(M_after, EPS)) + nm1 * logG_after
                    logw_arr = -k_arr * LAMBDA_FILL + BETA_ACTION * (logU_after - logU0)
                    mask = np.isfinite(logw_arr)

                    n_valid = int(np.sum(mask))
                    if n_valid > 0:
                        end = n_actions + n_valid
                        prices_buf[n_actions:end] = p_arr[mask]
                        logw_buf[n_actions:end] = logw_arr[mask]
                        sides_buf[n_actions:end] = 0
                        n_actions = end

        if n_actions == 0:
            return None, None

        prices_arr = prices_buf[:n_actions]
        logw_arr = logw_buf[:n_actions]
        sides_arr = sides_buf[:n_actions]

        max_logw = float(np.max(logw_arr))
        weights = np.exp(logw_arr - max_logw)

        total_trade_w = float(np.sum(weights))
        if total_trade_w <= 0.0 or not np.isfinite(total_trade_w):
            return None, None

        # Sample directly from trade actions (no do-nothing option)
        r = random.random() * total_trade_w

        cdf = np.cumsum(weights)
        chosen_idx = int(np.searchsorted(cdf, r, side="left"))
        if chosen_idx >= n_actions:
            chosen_idx = n_actions - 1

        chosen_side = "bid" if sides_arr[chosen_idx] == 1 else "ask"
        chosen_price = float(prices_arr[chosen_idx])

        return chosen_side, Order(agent, chosen_price, Q_TRADE, arrival_time)

    def execute_trade_mh(self, bid_order, ask_order, aggressor_side, t_now: int):
        price = ask_order.price if aggressor_side == "bid" else bid_order.price

        buyer = bid_order.agent
        seller = ask_order.agent

        if buyer is seller:
            return False

        max_by_orders = min(bid_order.qty, ask_order.qty)
        max_by_seller = seller.G
        max_by_buyer = buyer.M / max(price, EPS)

        qty = min(max_by_orders, max_by_seller, max_by_buyer)
        if qty <= EPS:
            return False

        cost = price * qty

        Mb_new = buyer.M - cost
        Gb_new = buyer.G + qty
        Ms_new = seller.M + cost
        Gs_new = seller.G - qty

        if Mb_new < -1e-12 or Gs_new < -1e-12:
            return False

        self.mh_attempts += 1

        log_u_before = buyer.log_utility() + seller.log_utility()
        log_u_after = buyer.log_utility_at(Mb_new, Gb_new) + seller.log_utility_at(Ms_new, Gs_new)
        logR = log_u_after - log_u_before
        self.mh_logR.append(float(logR))

        if logR >= 0.0:
            accept = True
        else:
            #accept = (random.random() < math.exp(logR))
            accept = True

        if not accept:
            self.mh_rejects += 1
            return False

        self.mh_accepts += 1

        buyer.M = Mb_new
        buyer.G = Gb_new
        seller.M = Ms_new
        seller.G = Gs_new

        bid_order.qty -= qty
        ask_order.qty -= qty

        if bid_order.qty <= EPS:
            bid_order.active = False
        if ask_order.qty <= EPS:
            ask_order.active = False

        # Preserve old-model semantics: after buyer/seller holdings change, any of
        # their now-infeasible resting orders are cancelled permanently rather than
        # allowed to revive later.
        self.invalidate_agent_orders(buyer)
        self.invalidate_agent_orders(seller)

        self.trade_prices.append(price)
        self.trade_qtys.append(qty)
        self.trade_times.append(int(t_now))
        return True

    def arrival(self, order_type, order, t_now: int):
        if order_type is None or order is None or order.qty <= 0:
            return

        order.price = round_to_tick(order.price)

        if order_type == "bid":
            while order.qty > EPS:
                best_ask = self.best_ask()
                if best_ask is None or order.price < best_ask.price:
                    break

                if best_ask.agent is order.agent:
                    old = self.asks.pop(0)
                    old.active = False
                    continue

                traded = self.execute_trade_mh(order, best_ask, "bid", t_now=t_now)
                if not traded:
                    break

                if best_ask.qty <= EPS:
                    best_ask.active = False
                    self.asks.discard(best_ask)

            if order.qty > EPS:
                self.add_bid(order)

        else:  # ask
            while order.qty > EPS:
                best_bid = self.best_bid()
                if best_bid is None or order.price > best_bid.price:
                    break

                if best_bid.agent is order.agent:
                    old = self.bids.pop()
                    old.active = False
                    continue

                traded = self.execute_trade_mh(best_bid, order, "ask", t_now=t_now)
                if not traded:
                    break

                if best_bid.qty <= EPS:
                    best_bid.active = False
                    self.bids.discard(best_bid)

            if order.qty > EPS:
                self.add_ask(order)


# =========================================================
# SHIP INTERNAL DYNAMICS: YOUR BETA SPLIT
# =========================================================

def pick_two_distinct(n: int):
    i = random.randrange(n)
    j = random.randrange(n - 1)
    if j >= i:
        j += 1
    return i, j


def ship_internal_trade_beta(s_i: Agent, s_j: Agent):
    Mtot = s_i.M + s_j.M
    Gtot = s_i.G + s_j.G
    if Mtot <= EPS and Gtot <= EPS:
        return

    xM = random.betavariate(max(s_i.alpha, EPS), max(s_j.alpha, EPS)) if Mtot > EPS else 0.5
    xG = random.betavariate(max(s_i.nu, EPS), max(s_j.nu, EPS)) if Gtot > EPS else 0.5

    s_i.M = xM * Mtot
    s_j.M = (1.0 - xM) * Mtot
    s_i.G = xG * Gtot
    s_j.G = (1.0 - xG) * Gtot


# =========================================================
# YOUR EXACT NEW WEAK epsilon-CONTACT JOIN
# =========================================================

def weak_contact_step(a: Agent, s: Agent):
    dM = random.uniform(-EPS_M, EPS_M)
    dG = random.uniform(-EPS_G, EPS_G)

    Ma_new = a.M - dM
    Ga_new = a.G - dG
    Ms_new = s.M + dM
    Gs_new = s.G + dG

    if Ma_new <= EPS or Ga_new <= EPS or Ms_new <= EPS or Gs_new <= EPS:
        return False

    log_before = a.log_utility() + s.log_utility()
    log_after = a.log_utility_at(Ma_new, Ga_new) + s.log_utility_at(Ms_new, Gs_new)
    logR = log_after - log_before

    if logR >= 0.0 or random.random() < math.exp(logR):
        if METER_ONLY:
            s.M, s.G = Ms_new, Gs_new
        else:
            a.M, a.G = Ma_new, Ga_new
            s.M, s.G = Ms_new, Gs_new
        return True

    return False


# =========================================================
# WORLD: NEW LOB + NEW WEAK JOIN + SHIP METER
# =========================================================

class CDLOBEntropyWorld:
    def __init__(self, agents_A: list[Agent], agents_S: list[Agent], seed: int):
        self.A = agents_A
        self.S = agents_S

        random.seed(seed)
        np.random.seed(seed)

        global_money_cap = sum(a.M for a in self.A)
        self.lob = LOB(global_money_cap=global_money_cap)

        self.t = 0

        self.MS_hist = deque(maxlen=METER_AVG_WINDOW)
        self.GS_hist = deque(maxlen=METER_AVG_WINDOW)

        # Only keep the final window of midprices, not the whole trajectory.
        self.mid_window = deque(maxlen=METER_AVG_WINDOW)

    def totals(self):
        MA = float(sum(a.M for a in self.A))
        GA = float(sum(a.G for a in self.A))
        MS = float(sum(s.M for s in self.S))
        GS = float(sum(s.G for s in self.S))
        return MA, GA, MS, GS

    def current_midprice(self):
        bb = self.lob.best_bid()
        ba = self.lob.best_ask()
        if bb is None or ba is None:
            return float("nan")
        return 0.5 * (bb.price + ba.price)

    def step_uncoupled(self, record_ship: bool = False):
        global CURRENT_TIME

        self.t += 1
        CURRENT_TIME = self.t

        self.lob.cancel_stale_orders()

        if random.random() < (len(self.A) / (len(self.A) + len(self.S))):
            a = random.choice(self.A)
            side, order = self.lob.agent_generates_order(a, arrival_time=self.t, t=self.t)
            self.lob.arrival(side, order, t_now=self.t)
        else:
            if len(self.S) >= 2:
                i, j = pick_two_distinct(len(self.S))
                ship_internal_trade_beta(self.S[i], self.S[j])

        if record_ship:
            _, _, MS, GS = self.totals()
            self.MS_hist.append(MS)
            self.GS_hist.append(GS)
            self.mid_window.append(self.current_midprice())

    def step_coupled(self, record_ship: bool = True):
        global CURRENT_TIME

        self.t += 1
        CURRENT_TIME = self.t

        self.lob.cancel_stale_orders()

        if random.random() < P_CONTACT:
            a = random.choice(self.A)
            s = random.choice(self.S)
            weak_contact_step(a, s)
        else:
            if random.random() < (len(self.A) / (len(self.A) + len(self.S))):
                a = random.choice(self.A)
                side, order = self.lob.agent_generates_order(a, arrival_time=self.t, t=self.t)
                self.lob.arrival(side, order, t_now=self.t)
            else:
                if len(self.S) >= 2:
                    i, j = pick_two_distinct(len(self.S))
                    ship_internal_trade_beta(self.S[i], self.S[j])

        if record_ship:
            #_, _, MS, GS = self.totals()
            MS, GS = self.ship_totals()
            self.MS_hist.append(MS)
            self.GS_hist.append(GS)
            self.mid_window.append(self.current_midprice())

    def run_uncoupled(self, n_steps: int):
        for _ in range(n_steps):
            self.step_uncoupled(record_ship=False)

    def run_coupled(self, n_steps: int):
        for _ in range(n_steps):
            self.step_coupled(record_ship=True)

    def ship_totals(self):
        MS = float(sum(s.M for s in self.S))
        GS = float(sum(s.G for s in self.S))
        return MS, GS


# =========================================================
# FIXED MAINLAND POPULATION
# =========================================================

def build_fixed_mainland_population(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    agents = [
        Agent(
            i,
            MAIN_BASE_MONEY,
            MAIN_BASE_GOODS,
            alpha=None,
            nu=None
        )
        for i in range(MAIN_N)
    ]
    return agents


def rescale_mainland_population(base_agents: list[Agent], m_fac: float, g_fac: float):
    scaled = []
    for a in base_agents:
        scaled.append(
            Agent(
                agent_id=a.id,
                money=MAIN_BASE_MONEY * m_fac,
                goods=MAIN_BASE_GOODS * g_fac,
                alpha=a.alpha,
                nu=a.nu
            )
        )
    return scaled


# =========================================================
# METER READ-OFF FROM SHIP
# =========================================================

def read_beta_phi_from_ship(MS_series, GS_series):
    MS_series = np.asarray(MS_series, dtype=float)
    GS_series = np.asarray(GS_series, dtype=float)

    if len(MS_series) == 0 or len(GS_series) == 0:
        return float("nan"), float("nan")

    w = int(min(METER_AVG_WINDOW, len(MS_series)))
    MS_bar = float(np.mean(MS_series[-w:]))
    GS_bar = float(np.mean(GS_series[-w:]))

    mbar = MS_bar / max(SHIP_N, 1)
    gbar = GS_bar / max(SHIP_N, 1)

    beta = float(ALPHA_SHIP / max(mbar, EPS))
    phi = float(NU_SHIP / max(gbar, EPS))
    return beta, phi


def read_avg_midprice(mid_window):
    mid_arr = np.asarray(mid_window, dtype=float)
    if len(mid_arr) == 0:
        return float("nan")

    finite = mid_arr[np.isfinite(mid_arr)]
    if len(finite) == 0:
        return float("nan")
    return float(np.mean(finite))


# =========================================================
# MEASUREMENT AT ONE GRIDPOINT
# =========================================================

def run_meter_at_macrostate(m_fac: float, g_fac: float, seed: int, base_agents_A: list[Agent]):
    random.seed(seed)
    np.random.seed(seed)

    agents_A = rescale_mainland_population(base_agents_A, m_fac, g_fac)

    agents_S = [
        Agent(
            10_000 + i,
            SHIP_BASE_MONEY,
            SHIP_BASE_GOODS,
            alpha=ALPHA_SHIP,
            nu=NU_SHIP
        )
        for i in range(SHIP_N)
    ]

    world = CDLOBEntropyWorld(agents_A, agents_S, seed=seed)

    world.run_uncoupled(EQ_STEPS_UNCOUPLED)
    world.run_coupled(STEPS_COUPLED)

    beta, phi = read_beta_phi_from_ship(world.MS_hist, world.GS_hist)
    avg_mid = read_avg_midprice(world.mid_window)

    MA, GA, MS, GS = world.totals()
    return beta, phi, avg_mid, (MA, GA, MS, GS), world


# =========================================================
# BUILD BETA / PHI GRIDS
# =========================================================

def build_beta_phi_grids(base_agents_A):
    nG = len(G_FACTORS)
    nM = len(M_FACTORS)

    beta_grid = np.zeros((nG, nM), dtype=float)
    phi_grid = np.zeros((nG, nM), dtype=float)

    for ig, gf in enumerate(G_FACTORS):
        for im, mf in enumerate(M_FACTORS):
            betas = []
            phis = []
            mids = []

            for k in range(N_SEEDS_PER_GRIDPOINT):
                beta, phi, avg_mid, totals, world = run_meter_at_macrostate(
                    mf, gf, seed=30_000 + k, base_agents_A=base_agents_A
                )
                betas.append(beta)
                phis.append(phi)
                mids.append(avg_mid)

            beta_hat = float(np.mean(betas))
            phi_hat = float(np.mean(phis))
            mid_hat = float(np.nanmean(mids))

            beta_grid[ig, im] = beta_hat
            phi_grid[ig, im] = phi_hat

            M_tot = MAIN_N * MAIN_BASE_MONEY * mf
            G_tot = MAIN_N * MAIN_BASE_GOODS * gf
            T_hat = 1.0 / max(beta_hat, EPS)
            mu_hat = phi_hat / max(beta_hat, EPS)

            print(
                f"[meter] M={M_tot:.1f} G={G_tot:.1f} | "
                f"beta={beta_hat:.6g} phi={phi_hat:.6g} | "
                f"T={T_hat:.6g} mu={mu_hat:.6g} mid={mid_hat:.6g}"
            )

    return beta_grid, phi_grid


# =========================================================
# ENTROPY: CURL + PATH INTEGRAL + LS FIT
# =========================================================

def integrability_curl(beta_grid, phi_grid, M_totals, G_totals):
    beta = np.asarray(beta_grid, dtype=float)
    phi = np.asarray(phi_grid, dtype=float)

    curls = []
    circulations = []

    for i in range(len(G_totals) - 1):
        for j in range(len(M_totals) - 1):
            dM = float(M_totals[j + 1] - M_totals[j])
            dG = float(G_totals[i + 1] - G_totals[i])

            beta_bot = 0.5 * (beta[i, j] + beta[i, j + 1])
            beta_top = 0.5 * (beta[i + 1, j] + beta[i + 1, j + 1])

            phi_left = 0.5 * (phi[i, j] + phi[i + 1, j])
            phi_right = 0.5 * (phi[i, j + 1] + phi[i + 1, j + 1])

            circulation = (beta_bot - beta_top) * dM + (phi_right - phi_left) * dG
            curl = circulation / max(dM * dG, EPS)

            circulations.append(circulation)
            curls.append(curl)

    curls = np.array(curls)
    circulations = np.array(circulations)

    print(
        "curl mean:", curls.mean(),
        "std:", curls.std(),
        "max abs:", np.max(np.abs(curls))
    )


def build_entropy_surface_from_beta_phi(beta_grid, phi_grid, M_totals, G_totals, S0=0.0):
    beta = np.asarray(beta_grid, dtype=float)
    phi = np.asarray(phi_grid, dtype=float)

    S = np.zeros_like(beta, dtype=float)
    S[0, 0] = float(S0)

    for j in range(1, len(M_totals)):
        dM = float(M_totals[j] - M_totals[j - 1])
        beta_bar = 0.5 * (beta[0, j] + beta[0, j - 1])
        S[0, j] = S[0, j - 1] + beta_bar * dM

    for i in range(1, len(G_totals)):
        dG = float(G_totals[i] - G_totals[i - 1])
        for j in range(len(M_totals)):
            phi_bar = 0.5 * (phi[i, j] + phi[i - 1, j])
            S[i, j] = S[i - 1, j] + phi_bar * dG

    return S


def ls_entropy_fit_from_beta_phi(beta_grid, phi_grid, M_totals, G_totals, gauge_fix=True):
    beta = np.asarray(beta_grid, dtype=float)
    phi = np.asarray(phi_grid, dtype=float)
    M = np.asarray(M_totals, dtype=float)
    G = np.asarray(G_totals, dtype=float)

    nG, nM = beta.shape
    assert phi.shape == (nG, nM)

    n_nodes = nG * nM
    n_h = nG * (nM - 1)
    n_v = (nG - 1) * nM
    n_edges = n_h + n_v

    def idx(i, j):
        return i * nM + j

    n_rows = n_edges + (1 if gauge_fix else 0)
    A = np.zeros((n_rows, n_nodes), dtype=float)
    b = np.zeros((n_rows,), dtype=float)

    row = 0

    for i in range(nG):
        for j in range(nM - 1):
            dM = float(M[j + 1] - M[j])
            beta_bar = 0.5 * (beta[i, j] + beta[i, j + 1])
            A[row, idx(i, j + 1)] = 1.0
            A[row, idx(i, j)] = -1.0
            b[row] = beta_bar * dM
            row += 1

    for i in range(nG - 1):
        for j in range(nM):
            dG = float(G[i + 1] - G[i])
            phi_bar = 0.5 * (phi[i, j] + phi[i + 1, j])
            A[row, idx(i + 1, j)] = 1.0
            A[row, idx(i, j)] = -1.0
            b[row] = phi_bar * dG
            row += 1

    if gauge_fix:
        A[row, idx(0, 0)] = 1.0
        b[row] = 0.0

    s_hat, *_ = np.linalg.lstsq(A, b, rcond=None)

    if gauge_fix:
        A_edges = A[:-1, :]
        b_edges = b[:-1]
    else:
        A_edges = A
        b_edges = b

    residuals = A_edges @ s_hat - b_edges
    RSS = float(np.dot(residuals, residuals))
    TSS = float(np.dot(b_edges, b_edges)) + 1e-12
    ratio = RSS / TSS

    S_fit = s_hat.reshape((nG, nM))
    return S_fit, RSS, TSS, ratio, residuals


# =========================================================
# LS DIAGNOSTICS
# =========================================================

def ls_entropy_fit_with_edge_residuals(beta_grid, phi_grid, M_totals, G_totals, gauge_fix=True):
    beta = np.asarray(beta_grid, dtype=float)
    phi = np.asarray(phi_grid, dtype=float)
    M = np.asarray(M_totals, dtype=float)
    G = np.asarray(G_totals, dtype=float)

    nG, nM = beta.shape
    assert phi.shape == (nG, nM)

    n_nodes = nG * nM
    n_h = nG * (nM - 1)
    n_v = (nG - 1) * nM
    n_edges = n_h + n_v

    def idx(i, j):
        return i * nM + j

    n_rows = n_edges + (1 if gauge_fix else 0)
    A = np.zeros((n_rows, n_nodes), dtype=float)
    b = np.zeros((n_rows,), dtype=float)

    row = 0

    for i in range(nG):
        for j in range(nM - 1):
            dM = float(M[j + 1] - M[j])
            beta_bar = 0.5 * (beta[i, j] + beta[i, j + 1])
            A[row, idx(i, j + 1)] = 1.0
            A[row, idx(i, j)] = -1.0
            b[row] = beta_bar * dM
            row += 1

    for i in range(nG - 1):
        for j in range(nM):
            dG = float(G[i + 1] - G[i])
            phi_bar = 0.5 * (phi[i, j] + phi[i + 1, j])
            A[row, idx(i + 1, j)] = 1.0
            A[row, idx(i, j)] = -1.0
            b[row] = phi_bar * dG
            row += 1

    if gauge_fix:
        A[row, idx(0, 0)] = 1.0
        b[row] = 0.0

    s_hat, *_ = np.linalg.lstsq(A, b, rcond=None)
    S_fit = s_hat.reshape((nG, nM))

    if gauge_fix:
        A_edges = A[:-1, :]
        b_edges = b[:-1]
    else:
        A_edges = A
        b_edges = b

    residuals = A_edges @ s_hat - b_edges
    res_h = residuals[:n_h].copy()
    res_v = residuals[n_h:].copy()

    RSS_h = float(np.dot(res_h, res_h))
    RSS_v = float(np.dot(res_v, res_v))
    RSS = float(np.dot(residuals, residuals))

    b0 = b_edges - float(b_edges.mean())
    TSS = float(np.dot(b0, b0)) + 1e-12
    ratio = RSS / TSS

    return {
        "S_fit": S_fit,
        "residuals_all": residuals,
        "res_h": res_h,
        "res_v": res_v,
        "RSS": RSS,
        "RSS_h": RSS_h,
        "RSS_v": RSS_v,
        "TSS": TSS,
        "RSS_over_TSS": ratio,
        "n_h": n_h,
        "n_v": n_v,
        "nG": nG,
        "nM": nM,
    }


def residual_maps_from_edge_residuals(beta_grid, phi_grid, M_totals, G_totals, gauge_fix=True):
    out = ls_entropy_fit_with_edge_residuals(beta_grid, phi_grid, M_totals, G_totals, gauge_fix=gauge_fix)
    nG, nM = out["nG"], out["nM"]

    res_h = out["res_h"]
    res_v = out["res_v"]

    Rm = res_h.reshape((nG, nM - 1))
    Rg = res_v.reshape((nG - 1, nM))

    max_h = float(np.max(np.abs(Rm))) if Rm.size else 0.0
    max_v = float(np.max(np.abs(Rg))) if Rg.size else 0.0

    loc_h = np.unravel_index(int(np.argmax(np.abs(Rm))), Rm.shape) if Rm.size else None
    loc_v = np.unravel_index(int(np.argmax(np.abs(Rg))), Rg.shape) if Rg.size else None

    def edge_midpoint_M(i, j):
        M_mid = 0.5 * (M_totals[j] + M_totals[j + 1])
        G_here = G_totals[i]
        return float(M_mid), float(G_here)

    def edge_midpoint_G(i, j):
        G_mid = 0.5 * (G_totals[i] + G_totals[i + 1])
        M_here = M_totals[j]
        return float(M_here), float(G_mid)

    max_h_at = edge_midpoint_M(*loc_h) if loc_h is not None else None
    max_v_at = edge_midpoint_G(*loc_v) if loc_v is not None else None

    return {
        "ls": out,
        "Rm_M_edges": Rm,
        "Rg_G_edges": Rg,
        "max_abs_M_edge": max_h,
        "max_abs_G_edge": max_v,
        "max_abs_M_edge_loc": loc_h,
        "max_abs_G_edge_loc": loc_v,
        "max_abs_M_edge_at_(Mmid,G)": max_h_at,
        "max_abs_G_edge_at_(M,Gmid)": max_v_at,
    }


# =========================================================
# THEORETICAL HETEROGENEOUS-CD BENCHMARK
# =========================================================

def theoretical_cd_entropy_surface_per_agent_mean(M_totals, G_totals, base_agents_A):
    """
    Per-agent mean heterogeneous CD entropy benchmark:

        S_CD(M,G) = A_sum * log(M/N) + B_sum * log(G/N)

    with
        A_sum = sum_i (alpha_i)
        B_sum = sum_i (nu_i)

    for the fixed mainland population.
    """
    A_sum = float(sum(a.alpha for a in base_agents_A))
    B_sum = float(sum(a.nu for a in base_agents_A))
    N = float(len(base_agents_A))

    M = np.asarray(M_totals, dtype=float)
    G = np.asarray(G_totals, dtype=float)

    S = np.zeros((len(G), len(M)), dtype=float)
    for i, Gtot in enumerate(G):
        for j, Mtot in enumerate(M):
            S[i, j] = A_sum * np.log(max(Mtot / N, EPS)) + B_sum * np.log(max(Gtot / N, EPS))
    return S, A_sum, B_sum


def ols_entropy_agreement(S_num, S_theory, name_num="S_num", name_theory="S_theory"):
    """
    OLS comparison between measured entropy surface and theoretical entropy surface.
    Allows for affine mismatch via intercept + slope.

    Returns model, RSS/TSS, and residual heatmap.
    """
    y = np.array(S_num, dtype=float).reshape(-1)
    x = np.array(S_theory, dtype=float).reshape(-1)

    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()

    print(f"\nOLS agreement: {name_num} ~ const + slope * {name_theory}")
    print(model.summary())

    residuals = model.resid
    RSS = float(np.sum(residuals ** 2))
    TSS = float(np.sum((y - np.mean(y)) ** 2)) + 1e-12
    ratio = RSS / TSS

    print("\nAdditional diagnostics:")
    print(f"RSS      = {RSS:.6e}")
    print(f"TSS      = {TSS:.6e}")
    print(f"RSS/TSS  = {ratio:.6e}")
    print(f"1 - R^2  = {(1 - model.rsquared):.6e}")

    return {
        "model": model,
        "RSS": RSS,
        "TSS": TSS,
        "RSS_over_TSS": ratio,
        "residuals": residuals.reshape(np.array(S_num).shape),
        "fitted": model.fittedvalues.reshape(np.array(S_num).shape),
    }


# =========================================================
# PLOTTING
# =========================================================

def plot_heatmap(Z, M_mesh, G_mesh, title, cbar_label):
    plt.figure(figsize=(7, 5))
    plt.imshow(
        Z,
        origin="lower",
        extent=[M_mesh.min(), M_mesh.max(), G_mesh.min(), G_mesh.max()],
        aspect="auto"
    )
    plt.colorbar(label=cbar_label)
    plt.xlabel("Total Money M (A)")
    plt.ylabel("Total Goods G (A)")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_surface(Z, M_mesh, G_mesh, title, zlabel):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(G_mesh, M_mesh, Z, alpha=0.85)
    ax.set_xlabel("Total Goods G (A)")
    ax.set_ylabel("Total Money M (A)")
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_edge_residual_heatmaps(Rm, Rg, M_totals, G_totals, title_prefix="LS edge residuals"):
    Rm = np.asarray(Rm, dtype=float)
    Rg = np.asarray(Rg, dtype=float)
    M = np.asarray(M_totals, dtype=float)
    G = np.asarray(G_totals, dtype=float)

    M_mid = 0.5 * (M[:-1] + M[1:])
    G_mid = 0.5 * (G[:-1] + G[1:])

    plt.figure(figsize=(7, 5))
    plt.imshow(
        Rm,
        origin="lower",
        extent=[M_mid.min(), M_mid.max(), G.min(), G.max()],
        aspect="auto"
    )
    plt.colorbar(label="Residual on M-edges (ΔS - β̄ΔM)")
    plt.xlabel("M edge midpoint")
    plt.ylabel("G node value")
    plt.title(f"{title_prefix}: horizontal (M) edges")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 5))
    plt.imshow(
        Rg,
        origin="lower",
        extent=[M.min(), M.max(), G_mid.min(), G_mid.max()],
        aspect="auto"
    )
    plt.colorbar(label="Residual on G-edges (ΔS - φ̄ΔG)")
    plt.xlabel("M node value")
    plt.ylabel("G edge midpoint")
    plt.title(f"{title_prefix}: vertical (G) edges")
    plt.tight_layout()
    plt.show()


def plot_ols_residual_heatmap(residual_grid, M_mesh, G_mesh, title):
    plt.figure(figsize=(7, 5))
    plt.imshow(
        np.asarray(residual_grid, dtype=float),
        origin="lower",
        extent=[M_mesh.min(), M_mesh.max(), G_mesh.min(), G_mesh.max()],
        aspect="auto"
    )
    plt.colorbar(label="OLS residual")
    plt.xlabel("Total Money M (A)")
    plt.ylabel("Total Goods G (A)")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def print_ls_split_summary(ls_out: dict):
    RSS = ls_out["RSS"]
    RSS_h = ls_out["RSS_h"]
    RSS_v = ls_out["RSS_v"]
    TSS = ls_out["TSS"]
    ratio = ls_out["RSS_over_TSS"]

    print(
        "[LS split] RSS =", f"{RSS:.6g}",
        "| RSS_h (M-edges) =", f"{RSS_h:.6g}",
        "| RSS_v (G-edges) =", f"{RSS_v:.6g}"
    )
    if RSS > 0:
        print(
            "[LS split] fractions:",
            "M-edges =", f"{(RSS_h / RSS):.3f}",
            "G-edges =", f"{(RSS_v / RSS):.3f}"
        )
    print("[LS split] TSS =", f"{TSS:.6g}", "| RSS/TSS =", f"{ratio:.6g}")

    r = ls_out["residuals_all"]
    print(
        "[LS split] residuals: mean", f"{r.mean():.3g}",
        "std", f"{r.std():.3g}",
        "maxabs", f"{np.max(np.abs(r)):.3g}"
    )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    print("Parameters:")
    print("  Mainland A = LOG-UTILITY MH-gated LOB")
    print("  Ship meter = homogeneous Cobb-Douglas")
    print(f"  Ship exponents: alpha_ship={ALPHA_SHIP}, nu_ship={NU_SHIP}")
    print(f"  Weak join: EPS_M={EPS_M}, EPS_G={EPS_G}, P_CONTACT={P_CONTACT}")
    print(f"  Measurement mode: METER_ONLY={METER_ONLY}")
    print(f"  LOB tick={TICK_SIZE}, K_MAX={K_MAX}, SEED_STEPS={SEED_STEPS}, Q_TRADE={Q_TRADE}")
    print(f"  BETA_ACTION={BETA_ACTION}, LAMBDA_FILL={LAMBDA_FILL}")
    print(f"  EQ_STEPS_UNCOUPLED={EQ_STEPS_UNCOUPLED}, STEPS_COUPLED={STEPS_COUPLED}")
    print(f"  N_SEEDS_PER_GRIDPOINT={N_SEEDS_PER_GRIDPOINT}")

    # Fixed mainland population across the whole grid
    base_agents_A = build_fixed_mainland_population(seed=GLOBAL_SEED)

    A_sum = float(sum(a.alpha - 1.0 for a in base_agents_A))
    B_sum = float(sum(a.nu - 1.0 for a in base_agents_A))
    print(f"Fixed mainland population coefficients:")
    print(f"  A_sum = Σ_i (alpha_i) = {A_sum:.6g}")
    print(f"  B_sum = Σ_i (nu_i ) = {B_sum:.6g}")

    M_totals = M_FACTORS * MAIN_N * MAIN_BASE_MONEY
    G_totals = G_FACTORS * MAIN_N * MAIN_BASE_GOODS
    M_mesh, G_mesh = np.meshgrid(M_totals, G_totals)

    beta_grid, phi_grid = build_beta_phi_grids(base_agents_A)

    T_grid = 1.0 / np.maximum(beta_grid, EPS)
    mu_grid = phi_grid / np.maximum(beta_grid, EPS)

    plot_heatmap(beta_grid, M_mesh, G_mesh, "β(M,G) from CD ship meter", "β")
    plot_heatmap(phi_grid, M_mesh, G_mesh, "φ(M,G)=μ/T from CD ship meter", "φ")

    plot_heatmap(T_grid, M_mesh, G_mesh, "T(M,G)=1/β (derived)", "T")
    plot_heatmap(mu_grid, M_mesh, G_mesh, "μ(M,G)=φ/β (derived)", "μ")

    if SHOW_3D:
        plot_surface(beta_grid, M_mesh, G_mesh, "β(M,G) surface", "β")
        plot_surface(phi_grid, M_mesh, G_mesh, "φ(M,G) surface", "φ")

    integrability_curl(beta_grid, phi_grid, M_totals, G_totals)

    # Entropy from integrability fit
    S_path = build_entropy_surface_from_beta_phi(beta_grid, phi_grid, M_totals, G_totals, S0=0.0)
    plot_heatmap(S_path, M_mesh, G_mesh, "Entropy proxy S_path(M,G) (path integration)", "S_path (relative)")

    S_ls, RSS, TSS, ratio, residuals = ls_entropy_fit_from_beta_phi(
        beta_grid, phi_grid, M_totals, G_totals, gauge_fix=True
    )
    print("[LS S-fit] RSS =", f"{RSS:.6g}", "| TSS =", f"{TSS:.6g}", "| RSS/TSS =", f"{ratio:.6g}")
    print(
        "[LS S-fit] residuals: mean", f"{residuals.mean():.3g}",
        "std", f"{residuals.std():.3g}",
        "maxabs", f"{np.max(np.abs(residuals)):.3g}"
    )

    plot_heatmap(
        S_ls, M_mesh, G_mesh,
        f"Entropy proxy S_ls(M,G) (least squares) | RSS/TSS={ratio:.3g}",
        "S_ls (relative)"
    )

    diag = residual_maps_from_edge_residuals(beta_grid, phi_grid, M_totals, G_totals, gauge_fix=True)

    print_ls_split_summary(diag["ls"])
    print(
        "Max |residual| on M-edges =", diag["max_abs_M_edge"],
        "at (i,j) =", diag["max_abs_M_edge_loc"],
        "approx (Mmid,G) =", diag["max_abs_M_edge_at_(Mmid,G)"]
    )
    print(
        "Max |residual| on G-edges =", diag["max_abs_G_edge"],
        "at (i,j) =", diag["max_abs_G_edge_loc"],
        "approx (M,Gmid) =", diag["max_abs_G_edge_at_(M,Gmid)"]
    )

    plot_edge_residual_heatmaps(
        diag["Rm_M_edges"],
        diag["Rg_G_edges"],
        M_totals,
        G_totals,
        title_prefix="LS edge residuals"
    )

    # =====================================================
    # Theoretical heterogeneous-CD benchmark comparison
    # =====================================================
    S_cd_theory, A_sum, B_sum = theoretical_cd_entropy_surface_per_agent_mean(
        M_totals, G_totals, base_agents_A
    )

    plot_heatmap(
        S_cd_theory,
        M_mesh,
        G_mesh,
        "Theoretical heterogeneous-CD entropy surface (per-agent mean form)",
        "S_CD_theory (relative)"
    )

    if SHOW_3D:
        plot_surface(
            S_cd_theory,
            M_mesh,
            G_mesh,
            "Theoretical heterogeneous-CD entropy surface",
            "S_CD_theory"
        )

    cd_agreement = ols_entropy_agreement(
        S_ls,
        S_cd_theory,
        name_num="S_ls",
        name_theory="S_CD_theory"
    )

    plot_ols_residual_heatmap(
        cd_agreement["residuals"],
        M_mesh,
        G_mesh,
        f"OLS residuals: S_ls vs theoretical CD entropy | RSS/TSS={cd_agreement['RSS_over_TSS']:.3g}"
    )

    if SHOW_3D:
        plot_surface(S_path, M_mesh, G_mesh, "Entropy proxy surface S_path(M,G)", "S_path")
        plot_surface(S_ls, M_mesh, G_mesh, f"Entropy proxy surface S_ls(M,G) | RSS/TSS={ratio:.3g}", "S_ls")
