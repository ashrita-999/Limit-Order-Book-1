import random
import matplotlib.pyplot as plt
import numpy as np
import math

# -----------------------------
# Optional: make runs reproducible
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

# Action-choice sensitivity in exp(beta_action * Delta log U)
BETA_ACTION = 1

# Do-nothing floor as a fraction of total trade weight
DO_NOTHING_FLOOR_FRAC = 0

ORDER_EXP_LAM = 0.001

Q_TRADE = 1.0

LAMBDA_FILL = 0.1

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
    with alpha, nu > 1.
    """
    def __init__(self, agent_id, money, goods):
        self.id = agent_id
        self.M = float(money)
        self.G = float(goods)

        self.alpha = random.uniform(1.0 + 1e-6, 100.0)
        self.nu = random.uniform(1.0 + 1e-6, 100.0)

        # Seed-stage ask probability
        self.a = (self.alpha - 1.0) / (self.alpha + self.nu - 2.0)

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


class LOB:
    def __init__(self, global_money_cap: float):
        self.bids = []   # sorted ascending by price
        self.asks = []   # sorted ascending by price

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
        return float(math.exp(-k*LAMBDA_FILL))

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
    # Agent action selection
    # -----------------------------
    def agent_generates_order(self, agent: Agent, arrival_time: int, t: int):
        # -------------------------
        # Seeding phase
        # -------------------------
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

        # -------------------------
        # Post-seeding phase
        # -------------------------
        logU0 = agent.log_utility()
        actions = []  # (kind, side, price, weight)

        bb = self.best_bid()
        ba = self.best_ask()

        # Market buy
        if ba is not None:
            p_exec = ba.price
            if agent.M >= p_exec * Q_TRADE:
                logU_after = agent.log_utility_at(agent.M - p_exec * Q_TRADE, agent.G + Q_TRADE)
                dlogU = logU_after - logU0
                w = math.exp(BETA_ACTION * dlogU)  # p_fill = 1 for market
                actions.append(("market", "bid", float(p_exec), float(w)))

        # Market sell
        if bb is not None:
            p_exec = bb.price
            if agent.G >= Q_TRADE:
                logU_after = agent.log_utility_at(agent.M + p_exec * Q_TRADE, agent.G - Q_TRADE)
                dlogU = logU_after - logU0
                w = math.exp(BETA_ACTION * dlogU)  # p_fill = 1 for market
                actions.append(("market", "ask", float(p_exec), float(w)))

        # Passive window
        b_ref, a_ref = self._touch_or_synthetic(agent)
        a_tk = price_to_tick(a_ref)
        b_tk = price_to_tick(b_ref)

        # Passive bids
        if agent.M > EPS:
            lo = max(self.MIN_TICK, a_tk - K_MAX)
            hi = min(self.MAX_TICK, a_tk - 1)
            for pt in range(lo, hi + 1):
                p = float(tick_to_price(pt))
                if agent.M < p * Q_TRADE:
                    continue
                k = a_tk - pt
                if k <= 0:
                    continue
                p_fill = self._p_fill_from_k(k)
                logU_after = agent.log_utility_at(agent.M - p * Q_TRADE, agent.G + Q_TRADE)
                dlogU = logU_after - logU0
                w = p_fill * math.exp(BETA_ACTION * dlogU)
                actions.append(("passive", "bid", p, float(w)))

        # Passive asks
        if agent.G >= Q_TRADE:
            lo = max(self.MIN_TICK, b_tk + 1)
            hi = min(self.MAX_TICK, b_tk + K_MAX)
            for pt in range(lo, hi + 1):
                p = float(tick_to_price(pt))
                k = pt - b_tk
                if k <= 0:
                    continue
                p_fill = self._p_fill_from_k(k)
                logU_after = agent.log_utility_at(agent.M + p * Q_TRADE, agent.G - Q_TRADE)
                dlogU = logU_after - logU0
                w = p_fill * math.exp(BETA_ACTION * dlogU)
                actions.append(("passive", "ask", p, float(w)))

        if not actions:
            return None, None

        total_trade_w = sum(a[3] for a in actions)
        if total_trade_w <= 0.0 or not np.isfinite(total_trade_w):
            return None, None

        do_nothing_w = DO_NOTHING_FLOOR_FRAC * max(1.0, total_trade_w)
        total_w = do_nothing_w + total_trade_w

        r = random.random() * total_w
        if r < do_nothing_w:
            return None, None
        r -= do_nothing_w

        cum = 0.0
        chosen_side = None
        chosen_price = None
        for kind, side, p, w in actions:
            cum += w
            if r <= cum:
                chosen_side, chosen_price = side, p
                break

        if chosen_side is None or chosen_price is None:
            return None, None

        return chosen_side, Order(agent, chosen_price, Q_TRADE, arrival_time)

    # -----------------------------
    # Trading
    # -----------------------------
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
        self.mh_attempt_times.append(int(t_now))

        log_u_before = buyer.log_utility() + seller.log_utility()
        log_u_after = buyer.log_utility_at(Mb_new, Gb_new) + seller.log_utility_at(Ms_new, Gs_new)
        logR = log_u_after - log_u_before
        self.mh_logR.append(float(logR))

        if logR >= 0.0:
            accept = True
        else:
            #accept = (random.random() < math.exp(logR))
            accept = True #removing MH gating

        if not accept:
            self.mh_rejects += 1
            return False

        self.mh_accepts += 1
        self.mh_accept_times.append(int(t_now))

        buyer.M = Mb_new
        buyer.G = Gb_new
        seller.M = Ms_new
        seller.G = Gs_new

        bid_order.qty -= qty
        ask_order.qty -= qty

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
# Extra diagnostics: detailed balance on midprice bins
# =========================================================
def compute_midprice(bb, ba):
    if bb is None or ba is None:
        return np.nan
    return 0.5 * (float(bb.price) + float(ba.price))


def transition_matrix_from_series(values, edges):
    values = np.asarray(values, dtype=float)
    idx = np.digitize(values, edges) - 1
    B = len(edges) - 1
    C = np.zeros((B, B), dtype=np.int64)

    for t in range(len(idx) - 1):
        i = idx[t]
        j = idx[t + 1]
        if not (0 <= i < B and 0 <= j < B):
            continue
        if np.isfinite(values[t]) and np.isfinite(values[t + 1]):
            C[i, j] += 1
    return C


def detailed_balance_report(mid_series, burn_in_steps=50000, n_bins=25, plot=True):
    mids = np.asarray(mid_series, dtype=float)
    if burn_in_steps >= len(mids) - 10:
        print("[DB] burn_in_steps too large; skipping detailed-balance check.")
        return None

    mids_ss = mids[burn_in_steps:]
    mids_ss = mids_ss[np.isfinite(mids_ss)]
    if len(mids_ss) < 1000:
        print("[DB] not enough finite mid samples; skipping detailed-balance check.")
        return None

    lo = float(np.quantile(mids_ss, 0.01))
    hi = float(np.quantile(mids_ss, 0.99))
    if hi <= lo + 1e-12:
        print("[DB] mid range too narrow; skipping detailed-balance check.")
        return None

    edges = np.linspace(lo, hi, n_bins + 1)

    mids_seg = mids[burn_in_steps:]
    C = transition_matrix_from_series(mids_seg, edges)
    CT = C.T
    denom = (C + CT).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_asym = np.where(denom > 0, np.abs(C - CT) / denom, 0.0)

    asym_score = float(np.sum(np.abs(C - CT)) / max(np.sum(C + CT), 1.0))
    print(f"[DB] bins={n_bins}, burn_in={burn_in_steps}")
    print(f"[DB] asym_score = sum|C-CT| / sum(C+CT) = {asym_score:.6g}  (0 is perfect DB symmetry)")

    pairs = []
    B = C.shape[0]
    for i in range(B):
        for j in range(i + 1, B):
            s = C[i, j] + C[j, i]
            if s >= 50:
                pairs.append((abs(int(C[i, j]) - int(C[j, i])) / s, i, j, int(C[i, j]), int(C[j, i]), int(s)))
    pairs.sort(reverse=True)
    if pairs:
        top = pairs[:10]
        print("[DB] top asymmetric bin-pairs (only where total>=50 transitions):")
        for frac, i, j, cij, cji, s in top:
            print(f"     bins ({i}->{j} vs {j}->{i}): {cij} vs {cji}  |diff|/sum={frac:.3f}  sum={s}")
    else:
        print("[DB] no bin pairs with total>=50 transitions (increase steps or reduce bins).")

    if plot:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        C_plot = np.log1p(C).astype(float)
        plt.imshow(C_plot, aspect="auto", origin="lower")
        plt.title("log(1 + C) mid-bin transitions")
        plt.xlabel("to bin j")
        plt.ylabel("from bin i")
        plt.colorbar()

        plt.subplot(1, 2, 2)
        plt.imshow(rel_asym, aspect="auto", origin="lower", vmin=0.0, vmax=1.0)
        plt.title("|C-CT| / (C+CT) per bin-pair")
        plt.xlabel("to bin j")
        plt.ylabel("from bin i")
        plt.colorbar()
        plt.tight_layout()
        plt.show()

    return {"edges": edges, "C": C, "rel_asym": rel_asym, "asym_score": asym_score}


# =========================================================
# Rolling utilities
# =========================================================
def rolling_rate(event_times, t_max, window=5000, step=1000):
    event_times = np.asarray(event_times, dtype=int)
    centers = []
    rates = []
    if t_max <= window:
        return np.array([]), np.array([])
    for a in range(0, t_max - window + 1, step):
        b = a + window
        cnt = np.sum((event_times >= a) & (event_times < b))
        centers.append(a + window / 2)
        rates.append(cnt / window)
    return np.asarray(centers), np.asarray(rates)


def run_chain(
    steps=200_000,
    burn_in=2000,
    max_plot_points=5000,
    t_start=500,
    t_end=200_000,
    trace_every=50,
    rolling_window=5000,
    rolling_step=1000,
    db_burn_in_steps=50_000,
    db_bins=25,
    CONSERVATION_WATCHDOG=True,
):
    global CURRENT_TIME

    agents1 = [Agent(agent_id=i, money=100.0, goods=100.0) for i in range(0, 100)]
    agents2 = [Agent(agent_id=i, money=100.0, goods=100.0) for i in range(100, 200)]
    agents = agents1 + agents2
    global_money_cap = sum(a.M for a in agents)

    init_money = [a.M for a in agents]
    init_goods = [a.G for a in agents]

    M0 = sum(a.M for a in agents)
    G0 = sum(a.G for a in agents)

    lob = LOB(global_money_cap=global_money_cap)

    trace_agents = random.sample(agents, k=min(10, len(agents)))
    trace_ids = [a.id for a in trace_agents]
    trace_t = []
    trace_M = {a.id: [] for a in trace_agents}
    trace_G = {a.id: [] for a in trace_agents}

    times = []
    bb_series_full = []
    ba_series_full = []
    mid_series_full = []

    limit_bid_t, limit_bid_p = [], []
    limit_ask_t, limit_ask_p = [], []

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

        bb = lob.best_bid()
        ba = lob.best_ask()
        mid = compute_midprice(bb, ba)

        times.append(t)
        bb_series_full.append(bb.price if bb is not None else np.nan)
        ba_series_full.append(ba.price if ba is not None else np.nan)
        mid_series_full.append(mid)

        if side is not None and order is not None:
            if side == "bid":
                limit_bid_t.append(t)
                limit_bid_p.append(order.price)
            elif side == "ask":
                limit_ask_t.append(t)
                limit_ask_p.append(order.price)

        if (t % trace_every) == 0:
            trace_t.append(t)
            for a in trace_agents:
                trace_M[a.id].append(a.M)
                trace_G[a.id].append(a.G)

    total_M = sum(a.M for a in agents)
    total_G = sum(a.G for a in agents)
    print("Total money:", total_M)
    print("Total goods:", total_G)
    print("Seed steps:", SEED_STEPS, "| Tick:", TICK_SIZE, "| K_MAX:", K_MAX, "| P_MAX:", lob.P_MAX, "| Q_TRADE:", Q_TRADE)
    print("BETA_ACTION:", BETA_ACTION)

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

    final_bids = [o.price for o in lob.bids]
    final_asks = [o.price for o in lob.asks]
    trade_prices = lob.trade_prices

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    if len(trade_prices) > 0:
        plt.hist(trade_prices, bins=40, density=True, alpha=0.7)
    plt.xlabel("Trade prices")
    plt.ylabel("Density")
    plt.title("Trade price distribution")

    plt.subplot(1, 2, 2)
    if len(final_bids) > 0:
        plt.hist(final_bids, bins=40, density=True, alpha=0.6, label="Bids")
    if len(final_asks) > 0:
        plt.hist(final_asks, bins=40, density=True, alpha=0.6, label="Asks")
    plt.axvline(x = final_bids[-1], color = 'r', label = 'p_bid')
    plt.axvline(x = final_asks[0], color = 'g', label = 'p_ask')
    plt.xlabel("Price")
    plt.title("Resting LOB price distribution")
    plt.legend()
    plt.tight_layout()
    plt.show()

    def downsample_xy(xs, ys, max_n):
        xs = np.asarray(xs)
        ys = np.asarray(ys)
        n = len(xs)
        if n <= max_n:
            return xs, ys
        idx = np.linspace(0, n - 1, max_n).astype(int)
        return xs[idx], ys[idx]

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

    trade_centers, trade_rates = rolling_rate(
        lob.trade_times, t_max=steps, window=rolling_window, step=rolling_step
    )
    plt.figure(figsize=(14, 4))
    if len(trade_centers) > 0:
        plt.plot(trade_centers, trade_rates * 1000.0)
    plt.xlabel("Time")
    plt.ylabel(f"Trades per 1000 steps (window={rolling_window})")
    plt.title("Rolling trade rate")
    plt.tight_layout()
    plt.show()

    mh_centers, mh_attempt_rate = rolling_rate(
        lob.mh_attempt_times, t_max=steps, window=rolling_window, step=rolling_step
    )
    _, mh_accept_rate = rolling_rate(
        lob.mh_accept_times, t_max=steps, window=rolling_window, step=rolling_step
    )
    if len(mh_centers) > 0:
        with np.errstate(divide="ignore", invalid="ignore"):
            roll_acc = np.where(mh_attempt_rate > 0, mh_accept_rate / mh_attempt_rate, np.nan)
        plt.figure(figsize=(14, 4))
        plt.plot(mh_centers, roll_acc)
        plt.xlabel("Time")
        plt.ylabel(f"Rolling MH acceptance (window={rolling_window})")
        plt.title("Rolling MH acceptance rate")
        plt.tight_layout()
        plt.show()

    if len(lob.mh_logR) > 0:
        plt.figure(figsize=(10, 4))
        plt.hist(np.asarray(lob.mh_logR), bins=60, density=True, alpha=0.8)
        plt.xlabel("logR")
        plt.title("MH log acceptance ratio distribution (attempted crossings)")
        plt.tight_layout()
        plt.show()

    trace_t_arr = np.asarray(trace_t, dtype=float)

    plt.figure(figsize=(14, 5))
    for aid in trace_ids:
        plt.plot(trace_t_arr, trace_M[aid], linewidth=1)
    plt.xlabel("Time")
    plt.ylabel("Money M")
    plt.title("Money trajectories (10 agents, downsampled)")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 5))
    for aid in trace_ids:
        plt.plot(trace_t_arr, trace_G[aid], linewidth=1)
    plt.xlabel("Time")
    plt.ylabel("Goods G")
    plt.title("Goods trajectories (10 agents, downsampled)")
    plt.tight_layout()
    plt.show()

    db_result = detailed_balance_report(
        mid_series_full,
        burn_in_steps=db_burn_in_steps,
        n_bins=db_bins,
        plot=True,
    )

    best_bid_series = np.array([x for x in bb_series_full if np.isfinite(x)])
    best_ask_series = np.array([x for x in ba_series_full if np.isfinite(x)])
    if burn_in < len(best_bid_series):
        best_bid_series = best_bid_series[burn_in:]
    if burn_in < len(best_ask_series):
        best_ask_series = best_ask_series[burn_in:]

    return best_bid_series, best_ask_series, lob, db_result


# -----------------------------
# Run
# -----------------------------
best_bids, best_asks, lob, db_result = run_chain(
    steps=5_000_000,
    burn_in=2000,
    t_start=500,
    t_end=5_000_000,
    trace_every=50,
    rolling_window=5000,
    rolling_step=1000,
    db_burn_in_steps=50_000,
    db_bins=25,
    CONSERVATION_WATCHDOG=True,
)
