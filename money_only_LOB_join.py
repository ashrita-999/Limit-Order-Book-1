import random
import heapq
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
# Optional: reproducibility
# -----------------------------
# random.seed(0)
# np.random.seed(0)

CURRENT_TIME = 0
EPS = 1e-9  # numerical safety

# =========================================================
# CONFIG
# =========================================================

# Tick grid
TICK_SIZE = 0.01
P_MIN = TICK_SIZE  # minimum admissible price

# Fill window
P_FILL_MIN = 1e-9
K_MAX = int(math.ceil(-math.log(P_FILL_MIN)))

# Bootstrap / seeding phase
SEED_STEPS = 100

# Contact timing
INDEP_EQ_STEPS = 10_000_000
CONTACT_TIME = SEED_STEPS + INDEP_EQ_STEPS

# Log-utility LOB params
BETA_ACTION = 1
DO_NOTHING_FLOOR_FRAC = 0
LAMBDA_FILL = 0.1

ORDER_EXP_LAM = 0.001
Q_TRADE = 1.0

# Run
STEPS = 20_000_000
TRACE_EVERY = 50
MAX_PLOT_POINTS = 6000

# Record plotting/diagnostic time series only every RECORD_EVERY steps to avoid huge memory use.
# Set this larger, e.g. 1000, for long 20M-step runs.
RECORD_EVERY = 1000

# Gate conservation checks for speed. During debugging, leave enabled and check
# periodically; for production runs, set CHECK_CONSERVATION = False.
CHECK_CONSERVATION = True
CONSERVATION_CHECK_EVERY = 10_000

# Affiliate pool sizes
AFFILIATE_FRAC_IN_A = 1.00
AFFILIATE_FRAC_IN_B = 1.00


def round_to_tick(p: float) -> float:
    if not np.isfinite(p):
        return P_MIN
    return max(P_MIN, round(p / TICK_SIZE) * TICK_SIZE)


def floor_to_tick(p: float) -> float:
    return max(P_MIN, math.floor(p / TICK_SIZE) * TICK_SIZE)


def price_to_tick(p: float) -> int:
    return int(round(float(p) / TICK_SIZE))


def tick_to_price(tk: int) -> float:
    return float(tk) * TICK_SIZE


def downsample_xy(xs, ys, max_n):
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    n = len(xs)
    if n <= max_n:
        return xs, ys
    idx = np.linspace(0, n - 1, max_n).astype(int)
    return xs[idx], ys[idx]


def downsample_triplet(t, p, flag, max_n):
    t = np.asarray(t)
    p = np.asarray(p)
    flag = np.asarray(flag, dtype=bool)
    n = len(t)
    if n <= max_n:
        return t, p, flag
    idx = np.linspace(0, n - 1, max_n).astype(int)
    return t[idx], p[idx], flag[idx]


# =========================================================
# CORE TYPES
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
    """
    Heterogeneous Cobb–Douglas exponents:
        u = m^(alpha-1) * g^(nu-1)
    with alpha, nu > 1.
    """
    def __init__(self, agent_id, money, goods, home_book: str):
        self.id = agent_id
        self.M = float(money)
        self.G = float(goods)             # PRINCIPAL (home) goods for utility
        self.home_book = str(home_book)   # "A" or "B"

        self.alpha = random.uniform(1.0 + 1e-6, 30.0)
        self.nu = random.uniform(1.0 + 1e-6, 30.0)
        self.a = (self.alpha - 1.0) / (self.alpha + self.nu - 2.0)

        # Fast per-agent resting order indices. Orders are cleaned lazily.
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
# AFFILIATE MAP (money-only contact)
# =========================================================
class AffiliateMap:
    """
    For foreign trading:
      - B principal trading in venue A uses an A affiliate (holder of venue-A goods)
      - A principal trading in venue B uses a B affiliate (holder of venue-B goods)

    The reverse maps are only for fast invalidation of foreign resting asks whose
    deliverable goods are backed by a holder/affiliate whose goods just changed.
    """
    def __init__(self, agentsA, agentsB, frac_in_A=1.0, frac_in_B=1.0):
        kA = max(1, int(round(frac_in_A * len(agentsA))))
        kB = max(1, int(round(frac_in_B * len(agentsB))))
        kA = min(kA, len(agentsA))
        kB = min(kB, len(agentsB))

        self.poolA = agentsA[:kA]
        self.poolB = agentsB[:kB]

        self.map_B_to_aff_in_A = {}
        self.map_A_to_aff_in_B = {}

        # holder.id -> list of foreign principals using that holder in the venue
        self.rev_A_holder_to_B_principals = {a.id: [] for a in self.poolA}
        self.rev_B_holder_to_A_principals = {b.id: [] for b in self.poolB}

        for i, b in enumerate(agentsB):
            holder = self.poolA[i % len(self.poolA)]
            self.map_B_to_aff_in_A[b.id] = holder
            self.rev_A_holder_to_B_principals.setdefault(holder.id, []).append(b)

        for i, a in enumerate(agentsA):
            holder = self.poolB[i % len(self.poolB)]
            self.map_A_to_aff_in_B[a.id] = holder
            self.rev_B_holder_to_A_principals.setdefault(holder.id, []).append(a)

    def holder(self, principal: Agent, venue_name: str) -> Agent:
        if principal.home_book == venue_name:
            return principal
        if venue_name == "A":
            return self.map_B_to_aff_in_A[principal.id]
        else:
            return self.map_A_to_aff_in_B[principal.id]

    def deliverable_goods(self, principal: Agent, venue_name: str) -> float:
        return float(self.holder(principal, venue_name).G)

    def principals_backed_by_holder(self, holder: Agent, venue_name: str):
        """Principals whose resting asks in venue_name are backed by holder.G."""
        out = []

        # Local principal's own asks in their home venue are backed by their own goods.
        if holder.home_book == venue_name:
            out.append(holder)

        # Foreign principals mapped to this holder for this venue.
        if venue_name == "A":
            out.extend(self.rev_A_holder_to_B_principals.get(holder.id, []))
        else:
            out.extend(self.rev_B_holder_to_A_principals.get(holder.id, []))

        # De-duplicate while preserving order.
        seen = set()
        unique = []
        for p in out:
            if p.id not in seen:
                seen.add(p.id)
                unique.append(p)
        return unique


# =========================================================
# LOB
# =========================================================
class LOB:
    def __init__(self, name: str, global_money_cap: float, affiliate_map: AffiliateMap | None):
        self.name = str(name)  # "A" or "B"
        self.aff = affiliate_map

        # Fast sorted books. Best bid is self.bids[-1], best ask is self.asks[0].
        self.bids = SortedList(key=lambda o: o.price)
        self.asks = SortedList(key=lambda o: o.price)

        # Cheap expiry cancellation. Filled/removed orders may still appear in
        # heaps, but active=False makes those heap entries harmless.
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
        self.mh_attempt_times = []
        self.mh_accept_times = []

    def deliverable_goods(self, principal: Agent) -> float:
        if self.aff is None:
            return float(principal.G)
        return self.aff.deliverable_goods(principal, self.name)

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
            and self.deliverable_goods(order.agent) + 1e-12 >= order.qty
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

    def _invalidate_bid_orders_for_principal(self, principal: Agent):
        live_bids = []
        for order in principal.resting_bids:
            if not order.active:
                continue
            if order.agent.M + 1e-12 < order.price * order.qty:
                self._remove_bid_order(order)
            else:
                live_bids.append(order)
        principal.resting_bids = live_bids

    def _invalidate_ask_orders_for_principal(self, principal: Agent):
        live_asks = []
        for order in principal.resting_asks:
            if not order.active:
                continue
            if self.deliverable_goods(order.agent) + 1e-12 < order.qty:
                self._remove_ask_order(order)
            else:
                live_asks.append(order)
        principal.resting_asks = live_asks

    def invalidate_agent_orders(self, agent: Agent):
        """
        Fast equivalent of the old full-book affordability/inventory cancellation
        for orders directly placed by this principal.
        """
        self._invalidate_bid_orders_for_principal(agent)
        self._invalidate_ask_orders_for_principal(agent)

    def invalidate_orders_after_trade(self, buyerP: Agent, sellerP: Agent, buyerH: Agent, sellerH: Agent):
        """
        Preserve the slow-book semantics after holdings change, without scanning the
        whole book.

        - Money changed for buyerP and sellerP, so their resting bids may become
          unaffordable.
        - Goods changed for buyerH and sellerH, so asks backed by those holders may
          become unbacked. For foreign asks, the principal who placed the order is
          not the holder, so we use AffiliateMap reverse maps to find those principals.
        """
        for p in {buyerP, sellerP}:
            self._invalidate_bid_orders_for_principal(p)

        if self.aff is None:
            affected_ask_principals = {buyerH, sellerH}
        else:
            affected_ask_principals = set()
            for h in (buyerH, sellerH):
                for p in self.aff.principals_backed_by_holder(h, self.name):
                    affected_ask_principals.add(p)

        for p in affected_ask_principals:
            self._invalidate_ask_orders_for_principal(p)

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
        Cheap expiry cancellation. The old version scanned the whole book every
        step; this pops only expired orders and lazily cleans invalid touch orders.
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

    # -----------------------------------------------------
    # Utility-after helpers used by router for action weights
    # -----------------------------------------------------
    def principal_log_utility_after_trade(self, principal: Agent, side: str, price: float, qty: float) -> float:
        """
        Original money-only contact rule for utility:
          - utility always uses principal's OWN goods principal.G
          - if principal is local to this book: goods change in utility
          - if foreign: goods do NOT change in utility

        This is kept so local-trade behaviour and any old diagnostics remain unchanged.
        """
        is_foreign = (principal.home_book != self.name)
        if side == "bid":
            M2 = principal.M - price * qty
            G2 = principal.G if is_foreign else (principal.G + qty)
        else:  # ask
            M2 = principal.M + price * qty
            G2 = principal.G if is_foreign else (principal.G - qty)
        return principal.log_utility_at(M2, G2)

    def joint_delta_log_utility_after_trade(self, principal: Agent, side: str, price: float, qty: float) -> float:
        """
        Action-selection utility weight.

        LOCAL trade:
            exactly the original single-principal rule, so local behaviour is unaffected.

        FOREIGN trade:
            use the joint rule
                ΔlogU = ΔlogU_principal + ΔlogU_affiliate
            where the principal receives/pays money but its own goods do not change,
            and the affiliate/holder receives/delivers the venue goods but its money does not change.
        """
        holder = principal if (self.aff is None) else self.aff.holder(principal, self.name)

        # Local trade: holder is principal, so DO NOT add an affiliate term.
        # This returns exactly the old router dlogU.
        if holder is principal:
            return (
                self.principal_log_utility_after_trade(principal, side, price, qty)
                - principal.log_utility()
            )

        # Foreign trade: principal money changes; principal's own goods do not.
        # Affiliate/holder goods change; affiliate/holder money does not.
        if side == "bid":
            principal_M2 = principal.M - price * qty
            principal_G2 = principal.G

            affiliate_M2 = holder.M
            affiliate_G2 = holder.G + qty
        else:  # ask
            principal_M2 = principal.M + price * qty
            principal_G2 = principal.G

            affiliate_M2 = holder.M
            affiliate_G2 = holder.G - qty

        dlogU_principal = (
            principal.log_utility_at(principal_M2, principal_G2)
            - principal.log_utility()
        )
        dlogU_affiliate = (
            holder.log_utility_at(affiliate_M2, affiliate_G2)
            - holder.log_utility()
        )

        return dlogU_principal + dlogU_affiliate

    # -----------------------------
    # Trading with money-only contact settlement
    # -----------------------------
    def _principal_logU_after(self, principal: Agent, dM: float, dG_if_local: float) -> float:
        M2 = principal.M + dM
        if principal.home_book == self.name:
            G2 = principal.G + dG_if_local
        else:
            G2 = principal.G
        return principal.log_utility_at(M2, G2)

    def execute_trade_mh(self, bid_order, ask_order, aggressor_side, t_now: int):
        price = ask_order.price if aggressor_side == "bid" else bid_order.price

        buyerP = bid_order.agent
        sellerP = ask_order.agent

        if buyerP is sellerP:
            return False

        buyerH = buyerP if (self.aff is None) else self.aff.holder(buyerP, self.name)
        sellerH = sellerP if (self.aff is None) else self.aff.holder(sellerP, self.name)

        if buyerH is sellerH:
            return False

        max_by_orders = min(bid_order.qty, ask_order.qty)
        max_by_seller = sellerH.G
        max_by_buyer = buyerP.M / max(price, EPS)

        qty = min(max_by_orders, max_by_seller, max_by_buyer)
        if qty <= EPS:
            return False

        cost = price * qty

        self.mh_attempts += 1
        self.mh_attempt_times.append(int(t_now))

        log_u_before = buyerP.log_utility() + sellerP.log_utility()
        log_u_after = (
            self._principal_logU_after(buyerP,  dM=-cost, dG_if_local=+qty) +
            self._principal_logU_after(sellerP, dM=+cost, dG_if_local=-qty)
        )
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
        self.mh_accept_times.append(int(t_now))

        # money transfers between principals
        buyerP.M -= cost
        sellerP.M += cost

        # goods transfer between local holders
        sellerH.G -= qty
        buyerH.G += qty

        bid_order.qty -= qty
        ask_order.qty -= qty

        if bid_order.qty <= EPS:
            bid_order.active = False
        if ask_order.qty <= EPS:
            ask_order.active = False

        self.invalidate_orders_after_trade(buyerP, sellerP, buyerH, sellerH)

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

                if self.aff is not None:
                    if self.aff.holder(best_ask.agent, self.name) is self.aff.holder(order.agent, self.name):
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
                    old = self.bids.pop(-1)
                    old.active = False
                    continue

                if self.aff is not None:
                    if self.aff.holder(best_bid.agent, self.name) is self.aff.holder(order.agent, self.name):
                        old = self.bids.pop(-1)
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
# ROUTER / ACTION SELECTION (2 books, money-only contact utility)
# =========================================================
def choose_action(agent: Agent, lobA: LOB, lobB: LOB, t: int, allow_cross: bool, do_seed: bool):
    home = lobA if agent.home_book == "A" else lobB
    books_to_consider = (lobA, lobB) if allow_cross else (home,)

    # (a) seeding: quote implied price into HOME venue only
    if do_seed:
        side = "ask" if random.random() < agent.a else "bid"
        p = round_to_tick(max(agent.implied_price(), P_MIN))
        p = min(max(p, P_MIN), home.P_MAX)

        if side == "ask" and agent.G < Q_TRADE:
            return None, None, None
        if side == "bid" and agent.M < p * Q_TRADE:
            return None, None, None

        return home, side, Order(agent, p, Q_TRADE, t)

    actions = []  # (book, kind, side, price, weight)

    def add_actions_for_book(book: LOB):
        bb = book.best_bid()
        ba = book.best_ask()

        # market buy
        if ba is not None:
            p_exec = ba.price
            if agent.M >= p_exec * Q_TRADE:
                dlogU = book.joint_delta_log_utility_after_trade(agent, "bid", p_exec, Q_TRADE)
                w = math.exp(BETA_ACTION * dlogU)
                actions.append((book, "market", "bid", float(p_exec), float(w)))

        # market sell
        if bb is not None:
            p_exec = bb.price
            if book.deliverable_goods(agent) >= Q_TRADE:
                dlogU = book.joint_delta_log_utility_after_trade(agent, "ask", p_exec, Q_TRADE)
                w = math.exp(BETA_ACTION * dlogU)
                actions.append((book, "market", "ask", float(p_exec), float(w)))

        # passive window
        b_ref, a_ref = book._touch_or_synthetic(agent)
        a_tk = price_to_tick(a_ref)
        b_tk = price_to_tick(b_ref)

        # passive bids
        if agent.M > EPS:
            lo = max(book.MIN_TICK, a_tk - K_MAX)
            hi = min(book.MAX_TICK, a_tk - 1)
            for pt in range(lo, hi + 1):
                p = float(tick_to_price(pt))
                if agent.M < p * Q_TRADE:
                    continue
                k = a_tk - pt
                if k <= 0:
                    continue
                p_fill = book._p_fill_from_k(k)
                dlogU = book.joint_delta_log_utility_after_trade(agent, "bid", p, Q_TRADE)
                w = p_fill * math.exp(BETA_ACTION * dlogU)
                actions.append((book, "passive", "bid", p, float(w)))

        # passive asks
        if book.deliverable_goods(agent) >= Q_TRADE:
            lo = max(book.MIN_TICK, b_tk + 1)
            hi = min(book.MAX_TICK, b_tk + K_MAX)
            for pt in range(lo, hi + 1):
                p = float(tick_to_price(pt))
                k = pt - b_tk
                if k <= 0:
                    continue
                p_fill = book._p_fill_from_k(k)
                dlogU = book.joint_delta_log_utility_after_trade(agent, "ask", p, Q_TRADE)
                w = p_fill * math.exp(BETA_ACTION * dlogU)
                actions.append((book, "passive", "ask", p, float(w)))

    for bk in books_to_consider:
        add_actions_for_book(bk)

    if not actions:
        return None, None, None

    total_trade_w = sum(a[4] for a in actions)
    if total_trade_w <= 0.0 or not np.isfinite(total_trade_w):
        return None, None, None

    do_nothing_w = DO_NOTHING_FLOOR_FRAC * max(1.0, total_trade_w)
    total_w = do_nothing_w + total_trade_w

    r = random.random() * total_w
    if r < do_nothing_w:
        return None, None, None
    r -= do_nothing_w

    cum = 0.0
    chosen = None
    for book, kind, side, p, w in actions:
        cum += w
        if r <= cum:
            chosen = (book, side, p)
            break

    if chosen is None:
        return None, None, None

    book, side, price = chosen
    price = min(max(round_to_tick(price), P_MIN), book.P_MAX)

    if side == "bid" and agent.M < price * Q_TRADE:
        return None, None, None
    if side == "ask" and book.deliverable_goods(agent) < Q_TRADE:
        return None, None, None

    return book, side, Order(agent, price, Q_TRADE, t)


# =========================================================
# RUN
# =========================================================
def run_two_book_money_only_contact():
    global CURRENT_TIME

    agentsA = [Agent(i, money=1000.0, goods=1000.0, home_book="A") for i in range(0, 100)]
    agentsB = [Agent(10_000 + i, money=1000.0, goods=100.0, home_book="B") for i in range(0, 100)]
    agents = agentsA + agentsB

    # Make B's exponents identical to A's so contact asymmetry comes from macrostate only.
    #for a, b in zip(agentsA, agentsB):
    #    b.alpha = a.alpha
    #    b.nu = a.nu
    #    b.a = a.a  # recompute the seeding bid/ask probability

    # Make B's exponents identical to A's so contact asymmetry comes from macrostate only. but affiliate exponents =/= principal
    for a, b in zip(agentsA, reversed(agentsB)):
        b.alpha = a.alpha
        b.nu = a.nu 
        b.a = a.a  # recompute the seeding bid/ask probability

    M0 = sum(a.M for a in agents)
    G0 = sum(a.G for a in agents)

    aff = AffiliateMap(agentsA, agentsB, frac_in_A=AFFILIATE_FRAC_IN_A, frac_in_B=AFFILIATE_FRAC_IN_B)

    global_cap = sum(a.M for a in agents)
    lobA = LOB("A", global_cap, affiliate_map=aff)
    lobB = LOB("B", global_cap, affiliate_map=aff)

    times = []
    bbA, baA = [], []
    bbB, baB = [], []

    in_t, in_p, in_book, in_is_foreign = [], [], [], []

    MA_series, MB_series = [], []
    GA_series, GB_series = [], []

    trace_agents = random.sample(agents, k=min(10, len(agents)))
    trace_ids = [a.id for a in trace_agents]
    trace_t = []
    trace_M = {a.id: [] for a in trace_agents}
    trace_G = {a.id: [] for a in trace_agents}

    for t in range(STEPS):
        CURRENT_TIME = t

        do_seed = (t < SEED_STEPS)
        allow_cross = (t >= CONTACT_TIME)

        agent = random.choice(agents)
        target, side, order = choose_action(agent, lobA, lobB, t, allow_cross=allow_cross, do_seed=do_seed)

        record_now = (t % RECORD_EVERY) == 0

        if target is not None:
            target.arrival(side, order, t_now=t)

            # Plot-only incoming-order diagnostics are sampled to avoid huge memory use.
            if record_now:
                in_t.append(t)
                in_p.append(order.price)
                in_book.append(target.name)
                in_is_foreign.append(agent.home_book != target.name)

        lobA.cancel_stale_orders()
        lobB.cancel_stale_orders()

        check_now = CHECK_CONSERVATION and (
            (t % CONSERVATION_CHECK_EVERY) == 0 or t == STEPS - 1
        )

        # Totals are only summed when needed: for sampled plotting or gated
        # conservation checks. This avoids an O(N) pass through all agents every step.
        if record_now or check_now:
            MA = sum(a.M for a in agentsA)
            MB = sum(a.M for a in agentsB)
            GA = sum(a.G for a in agentsA)
            GB = sum(a.G for a in agentsB)

            if check_now:
                if abs((MA + MB) - M0) > 1e-6 or abs((GA + GB) - G0) > 1e-6:
                    print("CONSERVATION BROKEN at t =", t)
                    print("M0,G0 =", M0, G0)
                    print("Mt,Gt =", (MA + MB), (GA + GB))
                    raise RuntimeError("Conservation violated")

            if record_now:
                bba = lobA.best_bid()
                baa = lobA.best_ask()
                bbb = lobB.best_bid()
                bab = lobB.best_ask()

                times.append(t)
                bbA.append(bba.price if bba is not None else np.nan)
                baA.append(baa.price if baa is not None else np.nan)
                bbB.append(bbb.price if bbb is not None else np.nan)
                baB.append(bab.price if bab is not None else np.nan)

                MA_series.append(MA)
                MB_series.append(MB)
                GA_series.append(GA)
                GB_series.append(GB)

        if (t % TRACE_EVERY) == 0:
            trace_t.append(t)
            for a in trace_agents:
                trace_M[a.id].append(a.M)
                trace_G[a.id].append(a.G)
        
        if (t%1_000_000) == 0:
            print(f"  step {t:,} / {STEPS:,}")

    final_MA = sum(a.M for a in agentsA)
    final_MB = sum(a.M for a in agentsB)
    final_GA = sum(a.G for a in agentsA)
    final_GB = sum(a.G for a in agentsB)

    print("\nFinal totals (principals):")
    print("  Total money:", final_MA + final_MB)
    print("  Total goods:", final_GA + final_GB)
    print("  Home A money/goods:", final_MA, final_GA)
    print("  Home B money/goods:", final_MB, final_GB)
    print("Seed steps:", SEED_STEPS, "| Indep-eq:", INDEP_EQ_STEPS, "| Contact time:", CONTACT_TIME)
    print("Recorded plot series every", RECORD_EVERY, "steps")
    if CHECK_CONSERVATION:
        print("Conservation check: every", CONSERVATION_CHECK_EVERY, "steps plus final step")
    else:
        print("Conservation check: disabled")

    for lob in (lobA, lobB):
        acc_rate = (lob.mh_accepts / lob.mh_attempts) if lob.mh_attempts > 0 else float("nan")
        print(f"\nBook {lob.name} MH: attempts={lob.mh_attempts} accepts={lob.mh_accepts} rejects={lob.mh_rejects} acc_rate={acc_rate:.6f}")

    times_arr = np.array(times)

    tds, MA_ds = downsample_xy(times_arr, np.array(MA_series), MAX_PLOT_POINTS)
    _,   MB_ds = downsample_xy(times_arr, np.array(MB_series), MAX_PLOT_POINTS)
    _,   GA_ds = downsample_xy(times_arr, np.array(GA_series), MAX_PLOT_POINTS)
    _,   GB_ds = downsample_xy(times_arr, np.array(GB_series), MAX_PLOT_POINTS)

    plt.figure(figsize=(14, 4))
    plt.plot(tds, MA_ds, label="Money (home A principals)")
    plt.plot(tds, MB_ds, label="Money (home B principals)")
    plt.axvline(CONTACT_TIME, linestyle="--", label="contact")
    plt.title("Money totals by home population")
    plt.xlabel("time")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 4))
    plt.plot(tds, GA_ds, label="Goods (home A principals)")
    plt.plot(tds, GB_ds, label="Goods (home B principals)")
    plt.axvline(CONTACT_TIME, linestyle="--", label="contact")
    plt.title("Goods totals by home population (principal goods used in utility)")
    plt.xlabel("time")
    plt.legend()
    plt.tight_layout()
    plt.show()

    bbA_arr = np.array(bbA)
    baA_arr = np.array(baA)
    bbB_arr = np.array(bbB)
    baB_arr = np.array(baB)

    tA_ds, bbA_ds = downsample_xy(times_arr, bbA_arr, MAX_PLOT_POINTS)
    _,    baA_ds = downsample_xy(times_arr, baA_arr, MAX_PLOT_POINTS)
    tB_ds, bbB_ds = downsample_xy(times_arr, bbB_arr, MAX_PLOT_POINTS)
    _,    baB_ds = downsample_xy(times_arr, baB_arr, MAX_PLOT_POINTS)

    plt.figure(figsize=(14, 4))
    plt.plot(tA_ds, bbA_ds, label="A best bid")
    plt.plot(tA_ds, baA_ds, label="A best ask")
    plt.axvline(CONTACT_TIME, linestyle="--", label="contact")
    plt.title("Book A best bid/ask")
    plt.xlabel("time")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 4))
    plt.plot(tB_ds, bbB_ds, label="B best bid")
    plt.plot(tB_ds, baB_ds, label="B best ask")
    plt.axvline(CONTACT_TIME, linestyle="--", label="contact")
    plt.title("Book B best bid/ask")
    plt.xlabel("time")
    plt.legend()
    plt.tight_layout()
    plt.show()

    in_t_arr = np.array(in_t)
    in_p_arr = np.array(in_p)
    in_book_arr = np.array(in_book)
    in_foreign_arr = np.array(in_is_foreign, dtype=bool)

    mA = (in_book_arr == "A")
    tA = in_t_arr[mA]
    pA = in_p_arr[mA]
    fA = in_foreign_arr[mA]
    tA, pA, fA = downsample_triplet(tA, pA, fA, MAX_PLOT_POINTS)

    plt.figure(figsize=(14, 4))
    plt.plot(tA_ds, bbA_ds, label="A best bid")
    plt.plot(tA_ds, baA_ds, label="A best ask")
    plt.scatter(tA[~fA], pA[~fA], s=10, marker="o", label="incoming to A (local)")
    plt.scatter(tA[fA],  pA[fA],  s=18, marker="x", label="incoming to A (foreign)")
    plt.axvline(CONTACT_TIME, linestyle="--", label="contact")
    plt.title("Book A: incoming orders local vs foreign")
    plt.xlabel("time")
    plt.legend()
    plt.tight_layout()
    plt.show()

    mB = (in_book_arr == "B")
    tB = in_t_arr[mB]
    pB = in_p_arr[mB]
    fB = in_foreign_arr[mB]
    tB, pB, fB = downsample_triplet(tB, pB, fB, MAX_PLOT_POINTS)

    plt.figure(figsize=(14, 4))
    plt.plot(tB_ds, bbB_ds, label="B best bid")
    plt.plot(tB_ds, baB_ds, label="B best ask")
    plt.scatter(tB[~fB], pB[~fB], s=10, marker="o", label="incoming to B (local)")
    plt.scatter(tB[fB],  pB[fB],  s=18, marker="x", label="incoming to B (foreign)")
    plt.axvline(CONTACT_TIME, linestyle="--", label="contact")
    plt.title("Book B: incoming orders local vs foreign")
    plt.xlabel("time")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 4))
    plt.subplot(1, 2, 1)
    if len(lobA.trade_prices) > 0:
        plt.hist(lobA.trade_prices, bins=60, density=True, alpha=0.7)
    plt.title("Trade price distribution (Book A)")
    plt.xlabel("price")
    plt.ylabel("density")

    plt.subplot(1, 2, 2)
    if len(lobB.trade_prices) > 0:
        plt.hist(lobB.trade_prices, bins=60, density=True, alpha=0.7)
    plt.title("Trade price distribution (Book B)")
    plt.xlabel("price")
    plt.ylabel("density")

    plt.tight_layout()
    plt.show()

    trace_t_arr = np.asarray(trace_t, dtype=float)
    plt.figure(figsize=(14, 4))
    for aid in trace_ids:
        plt.plot(trace_t_arr, trace_M[aid], linewidth=1)
    plt.title("Money trajectories (sample principals)")
    plt.xlabel("time")
    plt.ylabel("M")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 4))
    for aid in trace_ids:
        plt.plot(trace_t_arr, trace_G[aid], linewidth=1)
    plt.title("Goods trajectories (sample principals, i.e. utility goods)")
    plt.xlabel("time")
    plt.ylabel("G")
    plt.tight_layout()
    plt.show()

    return lobA, lobB, agentsA, agentsB, aff


if __name__ == "__main__":
    run_two_book_money_only_contact()
