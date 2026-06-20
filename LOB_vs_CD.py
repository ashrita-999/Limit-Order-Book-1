import random
import heapq
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import math

try:
    from sortedcontainers import SortedList
except ImportError as exc:
    raise ImportError(
        "This fast LOB version requires sortedcontainers. Install it with: pip install sortedcontainers"
    ) from exc

# -----------------------------
# Optional: make runs reproducible
# -----------------------------
random.seed(20)
np.random.seed(20)

CURRENT_TIME = 0
EPS = 1e-9

# =========================================================
# CONFIG
# =========================================================

TICK_SIZE = 0.01
P_MIN = TICK_SIZE

P_FILL_MIN = 1e-9
K_MAX = int(math.ceil(-math.log(P_FILL_MIN)))

SEED_STEPS = 100
BETA_ACTION = 1

ORDER_EXP_LAM = 0.001
Q_TRADE = 1.0
LAMBDA_FILL = 0.1

DIAG_EVERY = 1_000_000
MID_MEAN_WINDOW = 1000

# If True: store best bid/ask and incoming order time series and plot them.
# If False: do not store those large arrays and skip the bid/ask plot.
DO_BID_ASK_PLOT = False

# Agent-level snapshot diagnostic settings:
# Only collect snapshots after SNAPSHOT_START_FRAC of the run has elapsed.
# Then the errorbar plot uses the final N_FINAL_SNAPSHOTS_FOR_ERRORBARS of those late-time snapshots.
SNAPSHOT_START_FRAC = 0.0
SNAPSHOT_EVERY = 100_000
N_FINAL_SNAPSHOTS_FOR_ERRORBARS = 10  # use final 10 late-time snapshots for mean ± 1 std plots


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


def group_cv(values):
    values = np.asarray(values, dtype=float)
    mean = np.mean(values)
    if abs(mean) < EPS:
        return np.nan
    return float(np.std(values) / mean)


# =========================================================
# AGENT-LEVEL CD COMPARISON DIAGNOSTIC
# =========================================================

# x-axis = theoretical CD stationary mean for each agent
# y-axis = measured LOB mean over final snapshots
# y-error = one standard deviation across final snapshots

def plot_final_agent_snapshot_errorbars(agents, M_snapshots, G_snapshots, n_final=10):
    if len(M_snapshots) < 2 or len(G_snapshots) < 2:
        print("\n[agent errorbar diagnostic skipped] Need at least 2 snapshots.")
        return

    n_use = min(int(n_final), len(M_snapshots), len(G_snapshots))

    M_snaps = np.asarray(M_snapshots[-n_use:], dtype=float)
    G_snaps = np.asarray(G_snapshots[-n_use:], dtype=float)

    alpha = np.array([a.alpha for a in agents], dtype=float)
    nu = np.array([a.nu for a in agents], dtype=float)

    # Conserved totals; using snapshot totals makes the diagnostic robust to tiny numerical drift.
    Mtot = float(M_snaps[0].sum())
    Gtot = float(G_snaps[0].sum())

    M_cd = Mtot * alpha / alpha.sum()
    G_cd = Gtot * nu / nu.sum()

    M_mean = M_snaps.mean(axis=0)
    G_mean = G_snaps.mean(axis=0)

    M_std = M_snaps.std(axis=0, ddof=1)
    G_std = G_snaps.std(axis=0, ddof=1)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))

    ax[0].errorbar(
        M_cd,
        M_mean,
        yerr=M_std,
        fmt="o",
        markersize=3,
        alpha=0.6,
        elinewidth=0.7,
        capsize=1,
    )
    lo = min(float(M_cd.min()), float(M_mean.min()))
    hi = max(float(M_cd.max()), float(M_mean.max()))
    ax[0].plot([lo, hi], [lo, hi], "k--", linewidth=1, label="perfect CD agreement")
    ax[0].set_xlabel(r"Theoretical CD mean $M_i$")
    ax[0].set_ylabel(r"Measured LOB mean $M_i$")
    ax[0].set_title(f"Agent-level money: final {n_use} snapshots, ±1 std")
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    ax[1].errorbar(
        G_cd,
        G_mean,
        yerr=G_std,
        fmt="o",
        markersize=3,
        alpha=0.6,
        elinewidth=0.7,
        capsize=1,
    )
    lo = min(float(G_cd.min()), float(G_mean.min()))
    hi = max(float(G_cd.max()), float(G_mean.max()))
    ax[1].plot([lo, hi], [lo, hi], "k--", linewidth=1, label="perfect CD agreement")
    ax[1].set_xlabel(r"Theoretical CD mean $G_i$")
    ax[1].set_ylabel(r"Measured LOB mean $G_i$")
    ax[1].set_title(f"Agent-level goods: final {n_use} snapshots, ±1 std")
    ax[1].legend()
    ax[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 70)
    print("[AGENT-LEVEL CD COMPARISON]")
    print(f"  Snapshots used: {n_use}")
    print(f"  Money correlation: {np.corrcoef(M_cd, M_mean)[0, 1]:.6f}")
    print(f"  Goods  correlation: {np.corrcoef(G_cd, G_mean)[0, 1]:.6f}")
    print(f"  Money RMSE: {np.sqrt(np.mean((M_mean - M_cd) ** 2)):.6f}")
    print(f"  Goods  RMSE: {np.sqrt(np.mean((G_mean - G_cd) ** 2)):.6f}")
    print("  Error bars: one standard deviation across final snapshots")
    print("=" * 70)


# =========================================================
# CORE: ORDER / AGENT
# =========================================================

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


class Agent:
    def __init__(self, agent_id, money, goods):
        self.id = agent_id
        self.M = float(money)
        self.G = float(goods)

        self.alpha = random.uniform(1.0 + 1e-6, 30.0)
        self.nu = random.uniform(1.0 + 1e-6, 30.0)

        self.a = (self.alpha - 1.0) / (self.alpha + self.nu - 2.0)

        # Fast per-agent resting order indices.
        # These are cleaned lazily inside LOB.invalidate_agent_orders().
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


# =========================================================
# FAST LOG-UTILITY LOB MODEL
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

        self.mh_attempts = 0
        self.mh_accepts = 0
        self.mh_rejects = 0
        self.mh_logR = []
        self.mh_attempt_times = []
        self.mh_accept_times = []

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

    def _touch_or_synthetic(self, agent: Agent):
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
        - Store candidate actions in preallocated NumPy buffers.
        - Sample the final action using np.cumsum + np.searchsorted.

        Intended action probabilities are unchanged except that this fast version,
        matching your provided fast LOB, samples directly from trade actions and does
        not include a do-nothing action.
        """
        if t < SEED_STEPS:
            side = "ask" if random.random() < agent.a else "bid"

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
        n_max = 2 + 2 * K_MAX
        prices_buf = np.empty(n_max, dtype=float)
        logw_buf = np.empty(n_max, dtype=float)
        sides_buf = np.empty(n_max, dtype=np.int8)  # 1 = bid, 0 = ask
        n_actions = 0

        bb = self.best_bid()
        ba = self.best_ask()

        # Market buy
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

        # Market sell
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

        # Passive bids / asks
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

        # Sample directly from trade actions.
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

        qty = min(
            bid_order.qty,
            ask_order.qty,
            seller.G,
            buyer.M / max(price, EPS),
        )

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
        #self.mh_attempt_times.append(int(t_now)) (Disabled for long runs)

        log_u_before = buyer.log_utility() + seller.log_utility()
        log_u_after = buyer.log_utility_at(Mb_new, Gb_new) + seller.log_utility_at(Ms_new, Gs_new)
        logR = log_u_after - log_u_before
        #self.mh_logR.append(float(logR)) (Disabled for long runs)

        if logR >= 0.0:
            accept = True
        else:
            #accept = random.random() < math.exp(logR)
            accept = True

        if not accept:
            self.mh_rejects += 1
            return False

        self.mh_accepts += 1
        #self.mh_accept_times.append(int(t_now)) (Disabled for long runs)

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

        #self.trade_prices.append(price) (Disabled for long runs)
        #self.trade_qtys.append(qty) (Disabled for long runs)
        #self.trade_times.append(int(t_now)) (Disabled for long runs)
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

        else:
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
# DIAGNOSTIC HELPERS
# =========================================================

def compute_midprice(bb, ba):
    if bb is None or ba is None:
        return np.nan
    return 0.5 * (float(bb.price) + float(ba.price))


def safe_mean_last_finite(arr, window):
    if len(arr) == 0:
        return np.nan

    tail = np.asarray(arr, dtype=float)
    tail = tail[-window:]
    tail = tail[np.isfinite(tail)]

    if len(tail) == 0:
        return np.nan

    return float(np.mean(tail))


def downsample_xy(xs, ys, max_n):
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    n = len(xs)

    if n <= max_n:
        return xs, ys

    idx = np.linspace(0, n - 1, max_n).astype(int)
    return xs[idx], ys[idx]


# =========================================================
# RUN CHAIN
# =========================================================

def run_chain(
    steps=200_000,
    burn_in=2000,
    max_plot_points=5000,
    t_start=500,
    t_end=200_000,
    CONSERVATION_WATCHDOG=True,
):
    global CURRENT_TIME

    agents1 = [Agent(agent_id=i, money=300.0, goods=300.0) for i in range(0, 50)]       # poor
    agents2 = [Agent(agent_id=i, money=300.0, goods=300.0) for i in range(50, 500)]   # rich
    agents = agents1 + agents2

    global_money_cap = sum(a.M for a in agents)

    init_money = [a.M for a in agents]
    init_goods = [a.G for a in agents]

    M0 = sum(a.M for a in agents)
    G0 = sum(a.G for a in agents)

    poor_M_prev = sum(a.M for a in agents1)
    poor_G_prev = sum(a.G for a in agents1)
    rich_M_prev = sum(a.M for a in agents2)
    rich_G_prev = sum(a.G for a in agents2)

    prev_mean_mid = None

    lob = LOB(global_money_cap=global_money_cap)

    if DO_BID_ASK_PLOT:
        times = []
        bb_series_full = []
        ba_series_full = []
        limit_bid_t, limit_bid_p = [], []
        limit_ask_t, limit_ask_p = [], []
    else:
        times = None
        bb_series_full = None
        ba_series_full = None
        limit_bid_t, limit_bid_p = None, None
        limit_ask_t, limit_ask_p = None, None

    # Only keep the last MID_MEAN_WINDOW midprices needed for diagnostics.
    # This avoids storing a 100M-length midprice array.
    mid_series_full = deque(maxlen=MID_MEAN_WINDOW)

    diag_times = []

    diag_d_rich_M = []
    diag_d_rich_G = []
    diag_d_poor_M = []
    diag_d_poor_G = []

    diag_mean_mid = []
    diag_d_mid = []

    # CV diagnostic storage
    diag_cv_rich_M = []
    diag_cv_rich_G = []
    diag_cv_poor_M = []
    diag_cv_poor_G = []

    # Agent-level snapshot storage for final-10 mean ± 1 std errorbar plot.
    # We only start collecting these after snapshot_start, so early/non-equilibrated
    # states are not included in the final snapshot pool.
    M_snapshots = []
    G_snapshots = []
    snapshot_times = []
    snapshot_start = int(SNAPSHOT_START_FRAC * steps)

    for t in range(steps):
        CURRENT_TIME = t
        agent = random.choice(agents)

        side, order = lob.agent_generates_order(agent, arrival_time=t, t=t)
        lob.arrival(side, order, t_now=t)
        lob.cancel_stale_orders()

        if CONSERVATION_WATCHDOG:
            Mt = sum(a.M for a in agents)
            Gt = sum(a.G for a in agents)

            if abs(Mt - M0) > 1e-6 or abs(Gt - G0) > 1e-6:
                print("CONSERVATION BROKEN at t =", t)
                print("M0,G0 =", M0, G0)
                print("Mt,Gt =", Mt, Gt)
                raise RuntimeError("Conservation violated")

        # Save only late-time agent-level snapshots for final-snapshot comparison.
        # Also force the final state to be saved, so short runs still get a final snapshot.
        if t >= snapshot_start and ((t + 1) % SNAPSHOT_EVERY == 0 or t == steps - 1):
            M_snapshots.append([a.M for a in agents])
            G_snapshots.append([a.G for a in agents])
            snapshot_times.append(t + 1)

        bb = lob.best_bid()
        ba = lob.best_ask()
        mid = compute_midprice(bb, ba)
        mid_series_full.append(mid)

        if DO_BID_ASK_PLOT:
            times.append(t)
            bb_series_full.append(bb.price if bb is not None else np.nan)
            ba_series_full.append(ba.price if ba is not None else np.nan)

            if side is not None and order is not None:
                if side == "bid":
                    limit_bid_t.append(t)
                    limit_bid_p.append(order.price)
                elif side == "ask":
                    limit_ask_t.append(t)
                    limit_ask_p.append(order.price)

        if (t + 1) % DIAG_EVERY == 0:
            poor_M_now = sum(a.M for a in agents1)
            poor_G_now = sum(a.G for a in agents1)
            rich_M_now = sum(a.M for a in agents2)
            rich_G_now = sum(a.G for a in agents2)

            d_poor_M = poor_M_now - poor_M_prev
            d_poor_G = poor_G_now - poor_G_prev
            d_rich_M = rich_M_now - rich_M_prev
            d_rich_G = rich_G_now - rich_G_prev

            mean_mid_recent = safe_mean_last_finite(mid_series_full, MID_MEAN_WINDOW)

            if prev_mean_mid is None or not np.isfinite(prev_mean_mid) or not np.isfinite(mean_mid_recent):
                d_mid = np.nan
            else:
                d_mid = mean_mid_recent - prev_mean_mid

            # CVs at checkpoint
            cv_rich_M = group_cv([a.M for a in agents2])
            cv_rich_G = group_cv([a.G for a in agents2])
            cv_poor_M = group_cv([a.M for a in agents1])
            cv_poor_G = group_cv([a.G for a in agents1])

            print("\n" + "=" * 70)
            print(f"[DIAG] t = {t + 1:,}")
            print(f"  Rich group  ΔM = {d_rich_M: .6f} | ΔG = {d_rich_G: .6f}")
            print(f"  Poor group  ΔM = {d_poor_M: .6f} | ΔG = {d_poor_G: .6f}")
            print(f"  Mean mid (last {MID_MEAN_WINDOW:,}) = {mean_mid_recent: .6f}")
            print(f"  Δ mean mid (since last diag) = {d_mid: .6f}")
            print(f"  CV Rich: M = {cv_rich_M: .6f} | G = {cv_rich_G: .6f}")
            print(f"  CV Poor: M = {cv_poor_M: .6f} | G = {cv_poor_G: .6f}")
            print("=" * 70)

            diag_times.append(t + 1)

            diag_d_rich_M.append(d_rich_M)
            diag_d_rich_G.append(d_rich_G)
            diag_d_poor_M.append(d_poor_M)
            diag_d_poor_G.append(d_poor_G)

            diag_mean_mid.append(mean_mid_recent)
            diag_d_mid.append(d_mid)

            diag_cv_rich_M.append(cv_rich_M)
            diag_cv_rich_G.append(cv_rich_G)
            diag_cv_poor_M.append(cv_poor_M)
            diag_cv_poor_G.append(cv_poor_G)

            poor_M_prev = poor_M_now
            poor_G_prev = poor_G_now
            rich_M_prev = rich_M_now
            rich_G_prev = rich_G_now
            prev_mean_mid = mean_mid_recent

    total_M = sum(a.M for a in agents)
    total_G = sum(a.G for a in agents)

    print("Total money:", total_M)
    print("Total goods:", total_G)
    print("Seed steps:", SEED_STEPS, "| Tick:", TICK_SIZE, "| K_MAX:", K_MAX, "| P_MAX:", lob.P_MAX, "| Q_TRADE:", Q_TRADE)
    print("BETA_ACTION:", BETA_ACTION)
    print("DO_BID_ASK_PLOT:", DO_BID_ASK_PLOT)
    print("Agent-level snapshots collected:", len(M_snapshots))

    acc_rate = (lob.mh_accepts / lob.mh_attempts) if lob.mh_attempts > 0 else float("nan")
    print(f"MH attempts={lob.mh_attempts} accepts={lob.mh_accepts} rejects={lob.mh_rejects} acc_rate={acc_rate:.6f}")

    if len(lob.mh_logR) > 0:
        logR_arr = np.asarray(lob.mh_logR, dtype=float)
        tol = 1e-6
        frac_pos = float(np.mean(logR_arr > tol))
        frac_neg = float(np.mean(logR_arr < -tol))
        frac_zero = float(np.mean(np.abs(logR_arr) <= tol))

        print(f"logR: mean={np.mean(logR_arr):.4g}, median={np.median(logR_arr):.4g}")
        print(f"logR sign fractions: pos={frac_pos:.4g}, neg={frac_neg:.4g}, near0={frac_zero:.4g} (tol={tol})")
        print(f"logR frac(logR>0)={float(np.mean(logR_arr > 0)):.4g}")

    final_money = [a.M for a in agents]
    final_goods = [a.G for a in agents]

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.hist(init_money, bins=40, density=True, alpha=0.8)
    plt.title("INITIAL Money Distribution")
    plt.xlabel("Money M")
    plt.ylabel("Density")
    plt.grid(alpha=0.3)

    plt.subplot(2, 2, 2)
    plt.hist(final_money, bins=40, density=True, alpha=0.8)
    plt.title("FINAL Money Distribution")
    plt.xlabel("Money M")
    plt.ylabel("Density")
    plt.grid(alpha=0.3)

    plt.subplot(2, 2, 3)
    plt.hist(init_goods, bins=40, density=True, alpha=0.8)
    plt.title("INITIAL Goods Distribution")
    plt.xlabel("Goods G")
    plt.ylabel("Density")
    plt.grid(alpha=0.3)

    plt.subplot(2, 2, 4)
    plt.hist(final_goods, bins=40, density=True, alpha=0.8)
    plt.title("FINAL Goods Distribution")
    plt.xlabel("Goods G")
    plt.ylabel("Density")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Agent-level theoretical CD mean vs measured LOB mean, final snapshots, ±1 std
    plot_final_agent_snapshot_errorbars(
        agents,
        M_snapshots,
        G_snapshots,
        n_final=N_FINAL_SNAPSHOTS_FOR_ERRORBARS,
    )

    # Multi-window agent comparison.
    # These windows are over collected snapshots, not raw time steps.
    if len(M_snapshots) >= 30:
        n = len(M_snapshots)
        windows = [
            ("early collected snapshots (5-10%)", n // 20, n // 10),
            ("mid collected snapshots (45-50%)", int(0.45 * n), int(0.5 * n)),
            ("late collected snapshots (95-100%)", int(0.95 * n), n),
        ]
        for label, lo, hi in windows:
            if hi - lo >= 2:
                print(f"\n--- Agent-level comparison: {label} ---")
                plot_final_agent_snapshot_errorbars(
                    agents,
                    M_snapshots[lo:hi],
                    G_snapshots[lo:hi],
                    n_final=hi - lo,
                )

    if DO_BID_ASK_PLOT:
        times_arr = np.array(times)
        bb_arr = np.array(bb_series_full)
        ba_arr = np.array(ba_series_full)

        wmask = (times_arr >= t_start) & (times_arr <= t_end)

        times_w = times_arr[wmask]
        bb_w = bb_arr[wmask]
        ba_w = ba_arr[wmask]

        if len(times_w) > max_plot_points:
            idx = np.linspace(0, len(times_w) - 1, max_plot_points).astype(int)
            times_w = times_w[idx]
            bb_w = bb_w[idx]
            ba_w = ba_w[idx]

        bid_t, bid_p = downsample_xy(limit_bid_t, limit_bid_p, max_plot_points)
        ask_t, ask_p = downsample_xy(limit_ask_t, limit_ask_p, max_plot_points)

        bid_mask = (bid_t >= t_start) & (bid_t <= t_end)
        ask_mask = (ask_t >= t_start) & (ask_t <= t_end)

        bid_t, bid_p = bid_t[bid_mask], bid_p[bid_mask]
        ask_t, ask_p = ask_t[ask_mask], ask_p[ask_mask]

        plt.figure(figsize=(14, 5))
        plt.plot(times_w, bb_w, label="Best Bid", linewidth=2)
        plt.plot(times_w, ba_w, label="Best Ask", linewidth=2)
        plt.scatter(bid_t, bid_p, marker="^", s=5, label="Incoming order (bid)")
        plt.scatter(ask_t, ask_p, marker="v", s=5, label="Incoming order (ask)")
        plt.xlabel("Time step")
        plt.ylabel("Price")
        plt.title("Best bid/ask and incoming prices (zoomed)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    if len(diag_times) > 0:
        diag_times_arr = np.asarray(diag_times, dtype=float)

        plt.figure(figsize=(14, 6))
        plt.plot(diag_times_arr, diag_d_rich_M, marker="o", label="Rich ΔM")
        plt.plot(diag_times_arr, diag_d_rich_G, marker="o", label="Rich ΔG")
        plt.plot(diag_times_arr, diag_d_poor_M, marker="o", label="Poor ΔM")
        plt.plot(diag_times_arr, diag_d_poor_G, marker="o", label="Poor ΔG")
        plt.axhline(0.0, color="black", linewidth=1, alpha=0.6)
        plt.xlabel("Time")
        plt.ylabel(f"Change over last {DIAG_EVERY:,} steps")
        plt.title("Rich/Poor group inventory changes at diagnostic checkpoints")
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(14, 5))
        plt.plot(diag_times_arr, diag_d_mid, marker="o", label="Δ mean mid")
        plt.axhline(0.0, color="black", linewidth=1, alpha=0.6)
        plt.xlabel("Time")
        plt.ylabel("Change in printed mean mid")
        plt.title("Change in mean midprice at diagnostic checkpoints")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # CV plot
        plt.figure(figsize=(14, 6))
        plt.plot(diag_times_arr, diag_cv_rich_M, marker="o", label="Rich CV(M)")
        plt.plot(diag_times_arr, diag_cv_poor_M, marker="o", label="Poor CV(M)")
        plt.plot(diag_times_arr, diag_cv_rich_G, marker="o", label="Rich CV(G)")
        plt.plot(diag_times_arr, diag_cv_poor_G, marker="o", label="Poor CV(G)")
        plt.xlabel("Time")
        plt.ylabel("Coefficient of variation")
        plt.title("Within-group coefficient of variation over time")
        plt.legend()
        plt.tight_layout()
        plt.show()

    if DO_BID_ASK_PLOT:
        best_bid_series = np.array([x for x in bb_series_full if np.isfinite(x)])
        best_ask_series = np.array([x for x in ba_series_full if np.isfinite(x)])

        if burn_in < len(best_bid_series):
            best_bid_series = best_bid_series[burn_in:]
        if burn_in < len(best_ask_series):
            best_ask_series = best_ask_series[burn_in:]
    else:
        best_bid_series = np.array([])
        best_ask_series = np.array([])

    # Kept same return signature as your original code.
    return best_bid_series, best_ask_series, lob


best_bids, best_asks, lob = run_chain(
    steps=100_000_000,
    burn_in=2000,
    t_start=500,
    t_end=100_000_000,
    CONSERVATION_WATCHDOG=False, # disabled for long runs 
)
