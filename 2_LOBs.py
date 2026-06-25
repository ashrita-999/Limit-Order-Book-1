import random
import matplotlib.pyplot as plt
import numpy as np
import math

# -----------------------------
# Optional reproducibility
# -----------------------------
# random.seed(0)
# np.random.seed(0)

CURRENT_TIME = 0
EPS = 1e-9

# =========================================================
# CONFIG
# =========================================================
TICK_SIZE = 0.01
P_MIN = TICK_SIZE

P_FILL_MIN = 1e-12
K_MAX = int(math.ceil(-math.log(P_FILL_MIN)))

# Phase lengths
SEED_STEPS = 100
INDEP_EQ_STEPS = 5_000_000
CONTACT_TIME = SEED_STEPS + INDEP_EQ_STEPS

DO_NOTHING_FLOOR_FRAC = 0

ORDER_EXP_LAM = 0.001
Q_TRADE = 1.0

# NEW: log-utility action selection
BETA_ACTION = 1
LAMBDA_FILL = 0.1

# Run + plotting window
STEPS = 35_000_000
T_START = 0
T_END = STEPS
MAX_PLOT_POINTS = 6000

INITIAL_MONEY_A = 1000.0
INITIAL_GOODS_A = 1000.0
INITIAL_MONEY_B = 100.0
INITIAL_GOODS_B = 100.0

N_AGENTS_A = 100
N_AGENTS_B = 100

# Logging / diagnostics
LOG_EVERY = 1_000_000
FLOW_SIGN_CONVENTION = "positive means money flowed B->A (buyer in B, seller in A)"

# =========================================================
# HELPERS
# =========================================================
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

def downsample_triplet(t, p, foreign, max_n):
    t = np.asarray(t)
    p = np.asarray(p)
    foreign = np.asarray(foreign, dtype=bool)
    n = len(t)
    if n <= max_n:
        return t, p, foreign
    idx = np.linspace(0, n - 1, max_n).astype(int)
    return t[idx], p[idx], foreign[idx]

# =========================================================
# CORE TYPES
# =========================================================
class Order:
    def __init__(self, agent, price, qty, arrival_time, lam=ORDER_EXP_LAM):
        self.agent = agent
        self.price = float(price)
        self.qty = float(qty)
        self.arrival_time = int(arrival_time)
        lifetime = np.random.exponential(scale=1 / lam) if lam > 0 else float("inf")
        self.expiration_time = self.arrival_time + lifetime

class Agent:
    """
    Heterogeneous Cobb–Douglas exponents:
        u = m^(alpha-1) * g^(nu-1)
    with alpha, nu ~ Uniform(1,30), strictly > 1.
    """
    def __init__(self, agent_id, money, goods, home_book: str, alphas_array, nus_array, pref_index: int):
        self.id = agent_id
        self.M = float(money)
        self.G = float(goods)
        self.home_book = str(home_book)  # "A" or "B"

        self.alpha = float(alphas_array[pref_index])
        self.nu    = float(nus_array[pref_index])

        denom = max(self.alpha + self.nu - 2.0, EPS)
        self.a = (self.alpha - 1.0) / denom

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
# LOB (MH-gated execution + flow + matrices)
# =========================================================
class LOB:
    def __init__(self, name: str, global_money_cap: float):
        self.name = str(name)  # "A" or "B"
        self.bids = []  # ascending by price
        self.asks = []  # ascending by price
        self.trade_prices = []
        self.trade_qtys = []

        self.P_MAX = floor_to_tick(global_money_cap)
        self.MIN_TICK = price_to_tick(P_MIN)
        self.MAX_TICK = price_to_tick(self.P_MAX)

        # cross-home diagnostics for THIS BOOK
        self.cross_fills = 0
        self.cross_money_flow_signed = 0.0  # + means B->A under sign convention

        self.total_fills = 0
        self.total_money_traded = 0.0

        # trade composition diagnostics
        self.hmap = {"A": 0, "B": 1}
        self.trade_matrix_total = np.zeros((2, 2), dtype=np.int64)
        self.trade_matrix_window = np.zeros((2, 2), dtype=np.int64)

        # MH diagnostics
        self.mh_attempts = 0
        self.mh_accepts = 0
        self.mh_rejects = 0
        self.mh_logR = []
        self.mh_attempt_times = []
        self.mh_accept_times = []

    def best_bid(self):
        return self.bids[-1] if self.bids else None

    def best_ask(self):
        return self.asks[0] if self.asks else None

    def add_bid(self, order):
        self.bids.append(order)
        self.bids.sort(key=lambda o: o.price)

    def add_ask(self, order):
        self.asks.append(order)
        self.asks.sort(key=lambda o: o.price)

    def cancel_stale_orders(self):
        global CURRENT_TIME
        self.bids = [
            o for o in self.bids
            if (CURRENT_TIME < o.expiration_time)
            and (o.qty > EPS)
            and (o.agent.M + 1e-12 >= o.price * o.qty)
        ]
        self.asks = [
            o for o in self.asks
            if (CURRENT_TIME < o.expiration_time)
            and (o.qty > EPS)
            and (o.agent.G + 1e-12 >= o.qty)
        ]

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

    # -----------------------------
    # MH-gated trade execution
    # -----------------------------
    def execute_trade_mh(self, bid_order, ask_order, aggressor_side: str, t_now: int):
        price = ask_order.price if aggressor_side == "bid" else bid_order.price

        buyer = bid_order.agent
        seller = ask_order.agent

        # forbid self-trade
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

        # MH attempt
        self.mh_attempts += 1
        self.mh_attempt_times.append(int(t_now))

        log_u_before = buyer.log_utility() + seller.log_utility()
        log_u_after  = buyer.log_utility_at(Mb_new, Gb_new) + seller.log_utility_at(Ms_new, Gs_new)
        logR = float(log_u_after - log_u_before)
        self.mh_logR.append(logR)

        if logR >= 0.0:
            accept = True
        else:
            #accept = (random.random() < math.exp(logR))
            accept = True

        if not accept:
            self.mh_rejects += 1
            return False

        # accept
        self.mh_accepts += 1
        self.mh_accept_times.append(int(t_now))

        # settlement
        buyer.M = Mb_new
        buyer.G = Gb_new
        seller.M = Ms_new
        seller.G = Gs_new

        bid_order.qty -= qty
        ask_order.qty -= qty

        # book stats
        self.trade_prices.append(price)
        self.trade_qtys.append(qty)
        self.total_fills += 1
        self.total_money_traded += cost

        # trade composition matrix
        bi = self.hmap.get(buyer.home_book, None)
        si = self.hmap.get(seller.home_book, None)
        if bi is not None and si is not None:
            self.trade_matrix_total[bi, si] += 1
            self.trade_matrix_window[bi, si] += 1

        # cross-home diagnostics (signed)
        if buyer.home_book != seller.home_book:
            self.cross_fills += 1
            if buyer.home_book == "B" and seller.home_book == "A":
                self.cross_money_flow_signed += cost
            elif buyer.home_book == "A" and seller.home_book == "B":
                self.cross_money_flow_signed -= cost

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

                # if top ask is own order: remove it and keep going
                if best_ask.agent is order.agent:
                    self.asks.pop(0)
                    continue

                traded = self.execute_trade_mh(order, best_ask, "bid", t_now=t_now)
                if not traded:
                    break
                if best_ask.qty <= EPS:
                    self.asks.pop(0)

            if order.qty > EPS:
                self.add_bid(order)

        else:  # ask
            while order.qty > EPS:
                best_bid = self.best_bid()
                if best_bid is None or order.price > best_bid.price:
                    break

                # if top bid is own order: remove it and keep going
                if best_bid.agent is order.agent:
                    self.bids.pop()
                    continue

                traded = self.execute_trade_mh(best_bid, order, "ask", t_now=t_now)
                if not traded:
                    break
                if best_bid.qty <= EPS:
                    self.bids.pop()

            if order.qty > EPS:
                self.add_ask(order)

# =========================================================
# DECISION RULE (UPDATED TO LOG-UTILITY)
# =========================================================
def choose_action(agent: Agent, lobA: LOB, lobB: LOB, t: int, allow_cross: bool, do_seed: bool):
    home = lobA if agent.home_book == "A" else lobB

    # -------------------------
    # Seed phase
    # -------------------------
    if do_seed:
        side = "ask" if random.random() < agent.a else "bid"
        p = round_to_tick(max(agent.implied_price(), P_MIN))
        p = min(max(p, P_MIN), home.P_MAX)

        if side == "ask" and agent.G < Q_TRADE:
            return None, None, None
        if side == "bid" and agent.M < p * Q_TRADE:
            return None, None, None

        return home, side, Order(agent, p, Q_TRADE, t)

    books_to_consider = (lobA, lobB) if allow_cross else (home,)

    logU0 = agent.log_utility()
    actions = []
    # action: (target_lob, kind, side, price, weight)

    def add_actions_for_book(book: LOB):
        bb = book.best_bid()
        ba = book.best_ask()

        # Market buy
        if ba is not None:
            p_exec = ba.price
            if agent.M >= p_exec * Q_TRADE:
                logU_after = agent.log_utility_at(agent.M - p_exec * Q_TRADE, agent.G + Q_TRADE)
                dlogU = logU_after - logU0
                w = math.exp(BETA_ACTION * dlogU)
                actions.append((book, "market", "bid", float(p_exec), float(w)))

        # Market sell
        if bb is not None:
            p_exec = bb.price
            if agent.G >= Q_TRADE:
                logU_after = agent.log_utility_at(agent.M + p_exec * Q_TRADE, agent.G - Q_TRADE)
                dlogU = logU_after - logU0
                w = math.exp(BETA_ACTION * dlogU)
                actions.append((book, "market", "ask", float(p_exec), float(w)))

        # Passive window
        b_ref, a_ref = book._touch_or_synthetic(agent)
        a_tk = price_to_tick(a_ref)
        b_tk = price_to_tick(b_ref)

        # Passive bids
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
                logU_after = agent.log_utility_at(agent.M - p * Q_TRADE, agent.G + Q_TRADE)
                dlogU = logU_after - logU0
                w = p_fill * math.exp(BETA_ACTION * dlogU)
                actions.append((book, "passive", "bid", p, float(w)))

        # Passive asks
        if agent.G >= Q_TRADE:
            lo = max(book.MIN_TICK, b_tk + 1)
            hi = min(book.MAX_TICK, b_tk + K_MAX)
            for pt in range(lo, hi + 1):
                p = float(tick_to_price(pt))
                k = pt - b_tk
                if k <= 0:
                    continue

                p_fill = book._p_fill_from_k(k)
                logU_after = agent.log_utility_at(agent.M + p * Q_TRADE, agent.G - Q_TRADE)
                dlogU = logU_after - logU0
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
    if side == "ask" and agent.G < Q_TRADE:
        return None, None, None

    return book, side, Order(agent, price, Q_TRADE, t)

# =========================================================
# RUN + PLOTS
# =========================================================
def run():
    global CURRENT_TIME

    # shared preference list for "equivalent books" test
    alphas = [random.uniform(1.0 + 1e-6, 30.0) for _ in range(N_AGENTS_A)]
    nus    = [random.uniform(1.0 + 1e-6, 30.0) for _ in range(N_AGENTS_A)]

    agentsA = [
        Agent(i, INITIAL_MONEY_A, INITIAL_GOODS_A, home_book="A",
              alphas_array=alphas, nus_array=nus, pref_index=i)
        for i in range(N_AGENTS_A)
    ]
    agentsB = [
        Agent(10_000 + i, INITIAL_MONEY_B, INITIAL_GOODS_B, home_book="B",
              alphas_array=alphas, nus_array=nus, pref_index=i)
        for i in range(N_AGENTS_B)
    ]
    agents = agentsA + agentsB

    global_cap = sum(a.M for a in agents)
    lobA = LOB("A", global_cap)
    lobB = LOB("B", global_cap)

    times = []
    bbA, baA = [], []
    bbB, baB = [], []

    in_t, in_p, in_book, in_is_foreign = [], [], [], []

    totalM_all, totalG_all = [], []
    totalM_homeA, totalG_homeA = [], []
    totalM_homeB, totalG_homeB = [], []

    cross_flow_cum = []
    cross_flow_window = []
    cross_fills_window = []
    cross_t = []

    mix_t = []
    within_share_A = []
    within_share_B = []

    last_total_cross_fills = 0
    last_total_cross_flow = 0.0

    def get_total_cross():
        fills = lobA.cross_fills + lobB.cross_fills
        flow = lobA.cross_money_flow_signed + lobB.cross_money_flow_signed
        return fills, flow

    def within_share(mat_2x2: np.ndarray) -> float:
        tot = int(mat_2x2.sum())
        if tot <= 0:
            return np.nan
        within = int(mat_2x2[0, 0] + mat_2x2[1, 1])
        return within / tot

    for t in range(STEPS):
        CURRENT_TIME = t

        do_seed = (t < SEED_STEPS)
        allow_cross = (t >= CONTACT_TIME)

        agent = random.choice(agents)
        target, side, order = choose_action(agent, lobA, lobB, t, allow_cross=allow_cross, do_seed=do_seed)

        if target is not None:
            target.arrival(side, order, t_now=t)
            in_t.append(t)
            in_p.append(order.price)
            in_book.append(target.name)
            in_is_foreign.append(agent.home_book != target.name)

        lobA.cancel_stale_orders()
        lobB.cancel_stale_orders()

        # record best bid/ask by book
        bba = lobA.best_bid()
        baa = lobA.best_ask()
        bbb = lobB.best_bid()
        bab = lobB.best_ask()

        times.append(t)
        bbA.append(bba.price if bba is not None else np.nan)
        baA.append(baa.price if baa is not None else np.nan)
        bbB.append(bbb.price if bbb is not None else np.nan)
        baB.append(bab.price if bab is not None else np.nan)

        # record totals
        MA = sum(a.M for a in agentsA)
        GA = sum(a.G for a in agentsA)
        MB = sum(a.M for a in agentsB)
        GB = sum(a.G for a in agentsB)
        totalM_homeA.append(MA); totalG_homeA.append(GA)
        totalM_homeB.append(MB); totalG_homeB.append(GB)
        totalM_all.append(MA + MB); totalG_all.append(GA + GB)

        if (t > 0) and (t % LOG_EVERY == 0):
            fills_now, flow_now = get_total_cross()
            d_fills = fills_now - last_total_cross_fills
            d_flow  = flow_now  - last_total_cross_flow

            accA = (lobA.mh_accepts / lobA.mh_attempts) if lobA.mh_attempts > 0 else float("nan")
            accB = (lobB.mh_accepts / lobB.mh_attempts) if lobB.mh_attempts > 0 else float("nan")

            print(f"[t={t}] CONTACT={'ON' if allow_cross else 'OFF'} | "
                  f"cross_fills(last {LOG_EVERY})={d_fills} | "
                  f"cross_flow(last {LOG_EVERY})={d_flow:.6g} ({FLOW_SIGN_CONVENTION}) | "
                  f"cross_flow_cum={flow_now:.6g}")
            print(f"        MH acc: BookA={accA:.4f} (att={lobA.mh_attempts}) | BookB={accB:.4f} (att={lobB.mh_attempts})")

            cross_t.append(t)
            cross_fills_window.append(d_fills)
            cross_flow_window.append(d_flow)
            cross_flow_cum.append(flow_now)

            print(f"  Book A trade_matrix_window (buyer x seller) last {LOG_EVERY}:")
            print(lobA.trade_matrix_window)
            print(f"  Book B trade_matrix_window (buyer x seller) last {LOG_EVERY}:")
            print(lobB.trade_matrix_window)

            mix_t.append(t)
            within_share_A.append(within_share(lobA.trade_matrix_window))
            within_share_B.append(within_share(lobB.trade_matrix_window))

            lobA.trade_matrix_window[:] = 0
            lobB.trade_matrix_window[:] = 0

            last_total_cross_fills = fills_now
            last_total_cross_flow  = flow_now

    print("\nFinal totals:")
    print("  Total money (final):", totalM_all[-1])
    print("  Total goods (final):", totalG_all[-1])
    print("  Home-A money/goods (final):", totalM_homeA[-1], totalG_homeA[-1])
    print("  Home-B money/goods (final):", totalM_homeB[-1], totalG_homeB[-1])
    print("Unified caps: P_MAX(A)=", lobA.P_MAX, "P_MAX(B)=", lobB.P_MAX, "| global_cap=", global_cap)
    print("Seed steps:", SEED_STEPS, "| Indep-eq steps:", INDEP_EQ_STEPS, "| Contact time:", CONTACT_TIME)
    print("BETA_ACTION:", BETA_ACTION, "| LAMBDA_FILL:", LAMBDA_FILL)

    fills_now, flow_now = (lobA.cross_fills + lobB.cross_fills,
                           lobA.cross_money_flow_signed + lobB.cross_money_flow_signed)
    print("Cross-home totals:")
    print("  cross_fills_total =", fills_now)
    print("  cross_money_flow_signed_total =", flow_now, f"({FLOW_SIGN_CONVENTION})")

    print("\nFinal trade matrices (TOTAL, buyer x seller):")
    print("Book A trade_matrix_total:")
    print(lobA.trade_matrix_total)
    print("Book B trade_matrix_total:")
    print(lobB.trade_matrix_total)

    # MH summary
    accA = (lobA.mh_accepts / lobA.mh_attempts) if lobA.mh_attempts > 0 else float("nan")
    accB = (lobB.mh_accepts / lobB.mh_attempts) if lobB.mh_attempts > 0 else float("nan")
    print("\nMH summary:")
    print(f"  Book A: attempts={lobA.mh_attempts} accepts={lobA.mh_accepts} rejects={lobA.mh_rejects} acc_rate={accA:.6f}")
    print(f"  Book B: attempts={lobB.mh_attempts} accepts={lobB.mh_accepts} rejects={lobB.mh_rejects} acc_rate={accB:.6f}")

    # -------------------------
    # Plots
    # -------------------------
    times_arr = np.array(times)

    def plot_totals(title, y1, y2, lab1, lab2):
        tw, y1w = downsample_xy(times_arr, np.array(y1), MAX_PLOT_POINTS)
        _,  y2w = downsample_xy(times_arr, np.array(y2), MAX_PLOT_POINTS)
        plt.figure(figsize=(14, 4))
        plt.plot(tw, y1w, label=lab1, linewidth=2)
        plt.plot(tw, y2w, label=lab2, linewidth=2)
        plt.axvline(CONTACT_TIME, linestyle="--", linewidth=2, label="contact")
        plt.xlabel("time")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.show()

    plot_totals("TOTALS (ALL AGENTS): Money and Goods over time",
                totalM_all, totalG_all, "Total Money (all)", "Total Goods (all)")
    plot_totals("TOTALS (HOME A POP): Money and Goods over time",
                totalM_homeA, totalG_homeA, "Money (home A)", "Goods (home A)")
    plot_totals("TOTALS (HOME B POP): Money and Goods over time",
                totalM_homeB, totalG_homeB, "Money (home B)", "Goods (home B)")

    if len(cross_t) > 0:
        ct = np.array(cross_t)
        cfill = np.array(cross_fills_window, dtype=float)
        cflow = np.array(cross_flow_window, dtype=float)
        ccum  = np.array(cross_flow_cum, dtype=float)

        tw, cfloww = downsample_xy(ct, cflow, MAX_PLOT_POINTS)
        _,  cfillw = downsample_xy(ct, cfill, MAX_PLOT_POINTS)
        _,  ccumw  = downsample_xy(ct, ccum,  MAX_PLOT_POINTS)

        plt.figure(figsize=(14, 4))
        plt.plot(tw, cfillw, linewidth=2)
        plt.axvline(CONTACT_TIME, linestyle="--", linewidth=2)
        plt.xlabel("time")
        plt.title(f"Cross-home FILLS per {LOG_EVERY} steps")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(14, 4))
        plt.plot(tw, cfloww, linewidth=2)
        plt.axvline(CONTACT_TIME, linestyle="--", linewidth=2)
        plt.xlabel("time")
        plt.title(f"Cross-home MONEY FLOW per {LOG_EVERY} steps (signed; {FLOW_SIGN_CONVENTION})")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(14, 4))
        plt.plot(tw, ccumw, linewidth=2)
        plt.axvline(CONTACT_TIME, linestyle="--", linewidth=2)
        plt.xlabel("time")
        plt.title(f"Cumulative cross-home MONEY FLOW (signed; {FLOW_SIGN_CONVENTION})")
        plt.tight_layout()
        plt.show()
    else:
        print("No cross-home logs collected (LOG_EVERY too large or run too short).")

    if len(mix_t) > 0:
        mt = np.array(mix_t, dtype=float)
        wa = np.array(within_share_A, dtype=float)
        wb = np.array(within_share_B, dtype=float)

        tw, waw = downsample_xy(mt, wa, MAX_PLOT_POINTS)
        _,  wbw = downsample_xy(mt, wb, MAX_PLOT_POINTS)

        plt.figure(figsize=(14, 4))
        plt.plot(tw, waw, linewidth=2, label="Within-home share (Book A)")
        plt.plot(tw, wbw, linewidth=2, label="Within-home share (Book B)")
        plt.axvline(CONTACT_TIME, linestyle="--", linewidth=2, label="contact")
        plt.ylim(-0.05, 1.05)
        plt.xlabel("time")
        plt.ylabel("within-home fraction of trades")
        plt.title(f"Trade mixing by book (window = {LOG_EVERY} steps)")
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        print("No mixing logs collected (LOG_EVERY too large or run too short).")

    bbA_arr = np.array(bbA)
    baA_arr = np.array(baA)
    bbB_arr = np.array(bbB)
    baB_arr = np.array(baB)

    in_t_arr = np.array(in_t)
    in_p_arr = np.array(in_p)
    in_book_arr = np.array(in_book)
    in_foreign_arr = np.array(in_is_foreign, dtype=bool)

    t0, t1 = T_START, T_END

    def window(xs, ys):
        m = (xs >= t0) & (xs <= t1)
        return xs[m], ys[m]

    # Book A plot
    twA, bbAw = window(times_arr, bbA_arr)
    _, baAw = window(times_arr, baA_arr)
    if len(twA) > MAX_PLOT_POINTS:
        idx = np.linspace(0, len(twA) - 1, MAX_PLOT_POINTS).astype(int)
        twA, bbAw, baAw = twA[idx], bbAw[idx], baAw[idx]

    mA = in_book_arr == "A"
    tA = in_t_arr[mA]
    pA = in_p_arr[mA]
    fA = in_foreign_arr[mA]
    mm = (tA >= t0) & (tA <= t1)
    tA, pA, fA = tA[mm], pA[mm], fA[mm]
    tA, pA, fA = downsample_triplet(tA, pA, fA, MAX_PLOT_POINTS)

    localA = ~fA
    foreignA = fA

    plt.figure(figsize=(14, 5))
    plt.plot(twA, bbAw, label="LOB A best bid", linewidth=2)
    plt.plot(twA, baAw, label="LOB A best ask", linewidth=2)
    plt.scatter(tA[localA], pA[localA], s=10, marker="o", label="incoming to A (local)")
    plt.scatter(tA[foreignA], pA[foreignA], s=18, marker="x", label="incoming to A (foreign)")
    plt.axvline(CONTACT_TIME, linestyle="--", linewidth=2, label="contact")
    plt.xlabel("time")
    plt.ylabel("price")
    plt.title("Book A: best bid/ask + incoming orders (local vs foreign)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Book B plot
    twB, bbBw = window(times_arr, bbB_arr)
    _, baBw = window(times_arr, baB_arr)
    if len(twB) > MAX_PLOT_POINTS:
        idx = np.linspace(0, len(twB) - 1, MAX_PLOT_POINTS).astype(int)
        twB, bbBw, baBw = twB[idx], bbBw[idx], baBw[idx]

    mB = in_book_arr == "B"
    tB = in_t_arr[mB]
    pB = in_p_arr[mB]
    fB = in_foreign_arr[mB]
    mm = (tB >= t0) & (tB <= t1)
    tB, pB, fB = tB[mm], pB[mm], fB[mm]
    tB, pB, fB = downsample_triplet(tB, pB, fB, MAX_PLOT_POINTS)

    localB = ~fB
    foreignB = fB

    plt.figure(figsize=(14, 5))
    plt.plot(twB, bbBw, label="LOB B best bid", linewidth=2)
    plt.plot(twB, baBw, label="LOB B best ask", linewidth=2)
    plt.scatter(tB[localB], pB[localB], s=6, marker="o", label="incoming to B (local)")
    plt.scatter(tB[foreignB], pB[foreignB], s=6, marker="x", label="incoming to B (foreign)")
    plt.axvline(CONTACT_TIME, linestyle="--", linewidth=2, label="contact")
    plt.xlabel("time")
    plt.ylabel("price")
    plt.title("Book B: best bid/ask + incoming orders (local vs foreign)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Trade price histograms
    plt.figure(figsize=(14, 5))

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

    return lobA, lobB, agents

if __name__ == "__main__":
    run()
