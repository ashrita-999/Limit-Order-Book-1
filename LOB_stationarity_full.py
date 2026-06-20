"""
LOB  (LOG-UTILITY action selection) + FULL stationarity diagnostics


All stationarity checks:
   - Mid/spread/imbalance time series
   - Batch means
   - KS + Wasserstein window distances (mid/spread/imb)
   - Time-averaged money/goods for 10 agents
   - Snapshot window distances for money/goods
   - Multi-chain cross comparisons (money/goods)
   - Resting LOB + trade price hist at end
   - Conservation watchdog
   - Initial vs final money/goods distributions
   - Best bid/ask + incoming order prices scatter (zoomed)
   - Rolling trade rate
   - Rolling MH acceptance rate
   - logR histogram
   - Detailed balance report on mid-bin transitions

Run as: python this_file.py
"""

import random
import math
import numpy as np
import matplotlib.pyplot as plt

CURRENT_TIME = 0
EPS = 1e-9  # numerical safety

# =========================================================
# CONFIG (LOG-UTILITY MODEL)
# =========================================================

# Tick grid
TICK_SIZE = 0.01
P_MIN = TICK_SIZE  # minimum admissible price

# Fill window
P_FILL_MIN = 1e-9
K_MAX = int(math.ceil(-math.log(P_FILL_MIN)))  # exp(-K_MAX) ~ P_FILL_MIN

# Bootstrap / seeding phase
SEED_STEPS = 100

# Action-choice sensitivity in exp(beta_action * Delta log U)
BETA_ACTION = 1

# Do-nothing floor as a fraction of total trade weight
DO_NOTHING_FLOOR_FRAC = 0

ORDER_EXP_LAM = 0.001
Q_TRADE = 1.0

# Passive fill decay
LAMBDA_FILL = 0.1


def round_to_tick(p: float) -> float:
    if not np.isfinite(p):
        return P_MIN
    return max(P_MIN, round(float(p) / TICK_SIZE) * TICK_SIZE)


def floor_to_tick(p: float) -> float:
    return max(P_MIN, math.floor(float(p) / TICK_SIZE) * TICK_SIZE)


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
    with alpha, nu ~ Uniform(1,100), strictly > 1.
    """

    def __init__(self, agent_id, money, goods):
        self.id = agent_id
        self.M = float(money)
        self.G = float(goods)

        self.alpha = random.uniform(1.0 + 1e-6, 100.0)
        self.nu = random.uniform(1.0 + 1e-6, 100.0)

        # probability an agent "wants to sell" in seed phase
        self.a = (self.alpha - 1.0) / (self.alpha + self.nu - 2.0)

    def utility_at(self, M: float, G: float) -> float:
        M = max(float(M), EPS)
        G = max(float(G), EPS)
        return (M ** (self.alpha - 1.0)) * (G ** (self.nu - 1.0))

    def utility(self) -> float:
        return self.utility_at(self.M, self.G)

    # numerically stable log-utility
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
    # Agent action selection (LOG-UTILITY VERSION)
    # -----------------------------
    def agent_generates_order(self, agent: Agent, arrival_time: int, t: int):
        # seed phase: quote at implied price with side bias a
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
                w = math.exp(BETA_ACTION * dlogU)
                actions.append(("market", "bid", float(p_exec), float(w)))

        # Market sell
        if bb is not None:
            p_exec = bb.price
            if agent.G >= Q_TRADE:
                logU_after = agent.log_utility_at(agent.M + p_exec * Q_TRADE, agent.G - Q_TRADE)
                dlogU = logU_after - logU0
                w = math.exp(BETA_ACTION * dlogU)
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
    # Trading (MH-gated)
    # -----------------------------
    def execute_trade_mh(self, bid_order, ask_order, aggressor_side, t_now: int):
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

        # accept
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

                # if top ask is own order: cancel it and keep matching
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

                # if top bid is own order: cancel it and keep matching
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
# Diagnostics helpers (OLD suite + NEW suite)
# =========================================================

def compute_midprice(bb, ba):
    if bb is None or ba is None:
        return np.nan
    return 0.5 * (float(bb.price) + float(ba.price))


def _drop_nan(x):
    x = np.asarray(x, dtype=float)
    return x[np.isfinite(x)]


def ks_statistic(x, y):
    x = np.sort(_drop_nan(x))
    y = np.sort(_drop_nan(y))
    if len(x) == 0 or len(y) == 0:
        return np.nan
    data_all = np.sort(np.concatenate([x, y]))
    cx = np.searchsorted(x, data_all, side="right") / len(x)
    cy = np.searchsorted(y, data_all, side="right") / len(y)
    return float(np.max(np.abs(cx - cy)))


def wasserstein_1d(x, y):
    x = np.sort(_drop_nan(x))
    y = np.sort(_drop_nan(y))
    if len(x) == 0 or len(y) == 0:
        return np.nan
    n = min(len(x), len(y))
    if n < 2:
        return np.nan
    q = np.linspace(0.0, 1.0, n, endpoint=True)
    xq = np.interp(q, np.linspace(0.0, 1.0, len(x), endpoint=True), x)
    yq = np.interp(q, np.linspace(0.0, 1.0, len(y), endpoint=True), y)
    return float(np.mean(np.abs(xq - yq)))


def window_distances_1dseries(x, window=5000, step=2000):
    x = np.asarray(x, dtype=float)
    ks_list, w1_list, pos = [], [], []
    i = 0
    while i + window + step <= len(x):
        wA = x[i:i + window]
        wB = x[i + step:i + step + window]
        ks_list.append(ks_statistic(wA, wB))
        w1_list.append(wasserstein_1d(wA, wB))
        pos.append(i)
        i += step
    return np.array(pos), np.array(ks_list), np.array(w1_list)


def window_distances_snapshots(snapshots, window=50, step=20):
    ks_list, w1_list, pos = [], [], []
    i = 0
    n = len(snapshots)
    while i + window + step <= n:
        A = np.concatenate(snapshots[i:i + window])
        B = np.concatenate(snapshots[i + step:i + step + window])
        ks_list.append(ks_statistic(A, B))
        w1_list.append(wasserstein_1d(A, B))
        pos.append(i)
        i += step
    return np.array(pos), np.array(ks_list), np.array(w1_list)


def batch_means(x, batch=5000):
    x = _drop_nan(x)
    if len(x) < batch:
        return np.array([])
    nb = len(x) // batch
    return x[:nb * batch].reshape(nb, batch).mean(axis=1)


def depth_within_H(lob: LOB, H=50):
    bb = lob.best_bid()
    ba = lob.best_ask()
    if bb is None or ba is None:
        return np.nan, np.nan

    b0 = price_to_tick(bb.price)
    a0 = price_to_tick(ba.price)

    bid_depth = 0.0
    for o in lob.bids:
        dt = b0 - price_to_tick(o.price)
        if 0 <= dt <= H:
            bid_depth += o.qty

    ask_depth = 0.0
    for o in lob.asks:
        dt = price_to_tick(o.price) - a0
        if 0 <= dt <= H:
            ask_depth += o.qty

    return bid_depth, ask_depth


def rolling_rate(event_times, t_max, window=5000, step=1000):
    event_times = np.asarray(event_times, dtype=int)
    centers, rates = [], []
    if t_max <= window:
        return np.array([]), np.array([])
    for a in range(0, t_max - window + 1, step):
        b = a + window
        cnt = np.sum((event_times >= a) & (event_times < b))
        centers.append(a + window / 2)
        rates.append(cnt / window)
    return np.asarray(centers), np.asarray(rates)


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


def detailed_balance_report(mid_series, burn_in_steps=50_000, n_bins=25, plot=True):
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
        print("[DB] top asymmetric bin-pairs (only where total>=50):")
        for frac, i, j, cij, cji, s in pairs[:10]:
            print(f"     bins ({i}->{j} vs {j}->{i}): {cij} vs {cji}  |diff|/sum={frac:.3f}  sum={s}")
    else:
        print("[DB] no bin pairs with total>=50 transitions (increase steps or reduce bins).")

    if plot:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(np.log1p(C).astype(float), aspect="auto", origin="lower")
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
# Plotting (OLD suite)
# =========================================================

def _downsample_xy(x, y, max_n):
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) <= max_n:
        return x, y
    idx = np.linspace(0, len(x) - 1, max_n).astype(int)
    return x[idx], y[idx]


def plot_lob_timeseries(chain, t_start=0, t_end=None, max_points=6000):
    t = chain["t_series"]
    if len(t) == 0:
        print("No LOB series recorded.")
        return
    if t_end is None:
        t_end = int(t[-1])

    mask = (t >= t_start) & (t <= t_end)
    tt = t[mask]
    mid = chain["mid_series"][mask]
    spr = chain["spread_series"][mask]
    imb = chain["imb_series"][mask]

    tt, mid = _downsample_xy(tt, mid, max_points)
    _, spr = _downsample_xy(np.arange(len(spr)), spr, max_points)
    _, imb = _downsample_xy(np.arange(len(imb)), imb, max_points)

    plt.figure(figsize=(14, 8))

    plt.subplot(3, 1, 1)
    plt.plot(tt, mid, linewidth=1.2)
    plt.title("Midprice against time")
    plt.xlabel("time")
    plt.ylabel("Price against time")
    plt.grid(alpha=0.3)

    plt.subplot(3, 1, 2)
    plt.plot(tt, spr, linewidth=1.2)
    plt.title("Spread against time")
    plt.xlabel("time")
    plt.ylabel("Spread")
    plt.grid(alpha=0.3)

    plt.subplot(3, 1, 3)
    plt.plot(tt, imb, linewidth=1.2)
    plt.title("Order-book imbalance within H ticks")
    plt.xlabel("time")
    plt.ylabel("imbalance")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_batch_means_mid_spread_imb(chain, batch=5000):
    burn = chain["burn_in_steps"]
    t = chain["t_series"]
    burn_idx = np.searchsorted(t, burn, side="left")

    plt.figure(figsize=(14, 8))

    for i, (x, title) in enumerate(
        [
            (chain["mid_series"][burn_idx:], "Batch means: mid"),
            (chain["spread_series"][burn_idx:], "Batch means: spread"),
            (chain["imb_series"][burn_idx:], "Batch means: imbalance"),
        ],
        start=1,
    ):
        bm = batch_means(x, batch=batch)
        plt.subplot(3, 1, i)
        if len(bm) == 0:
            plt.text(0.5, 0.5, "Not enough samples for batch means", ha="center", va="center")
            plt.axis("off")
        else:
            plt.plot(bm, marker="o", linewidth=1.2)
            plt.title(title)
            plt.xlabel("batch index")
            plt.ylabel("mean")
            plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_window_distances_mid_spread_imb(chain, window=5000, step=2000):
    burn = chain["burn_in_steps"]
    t = chain["t_series"]
    burn_idx = np.searchsorted(t, burn, side="left")

    keys = [
        ("mid_series", "Midprice"),
        ("spread_series", "Spread"),
        ("imb_series", "Imbalance"),
    ]

    plt.figure(figsize=(14, 9))
    for r, (k, name) in enumerate(keys, start=1):
        x = chain[k][burn_idx:]
        pos, ks_vals, w1_vals = window_distances_1dseries(x, window=window, step=step)

        plt.subplot(3, 2, 2 * r - 1)
        plt.plot(pos, ks_vals, marker="o", linewidth=1.0)
        plt.title(f"KS across windows: {name}")
        plt.xlabel("index (post burn-in)")
        plt.ylabel("KS")
        plt.grid(alpha=0.3)

        plt.subplot(3, 2, 2 * r)
        plt.plot(pos, w1_vals, marker="o", linewidth=1.0)
        plt.title(f"Wasserstein-1 across windows: {name}")
        plt.xlabel("Index")
        plt.ylabel("Wasserstein-1")
        plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_time_averaged_MG(chain, t_start=0, t_end=None, max_points=6000):
    t = chain["avg_t"]
    if t_end is None:
        t_end = int(t[-1])
    mask = (t >= t_start) & (t <= t_end)
    tt = t[mask]

    if len(tt) > max_points:
        idx = np.linspace(0, len(tt) - 1, max_points).astype(int)
        tt = tt[idx]
    else:
        idx = None

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    for aid, series in chain["avgM_series"].items():
        x = np.asarray(series)[mask]
        if idx is not None:
            x = x[idx]
        plt.plot(tt, x, linewidth=1.2, label=f"agent {aid}")
    plt.title(f"Time-Averaged Money: {len(chain['avgM_series'])} Agents")
    plt.xlabel("time")
    plt.ylabel(r'$\overline{M}_i(t)$')
    plt.grid(alpha=0.3)
    plt.legend(ncol=2, fontsize=8)

    plt.subplot(1, 2, 2)
    for aid, series in chain["avgG_series"].items():
        x = np.asarray(series)[mask]
        if idx is not None:
            x = x[idx]
        plt.plot(tt, x, linewidth=1.2, label=f"agent {aid}")
    plt.title(f"Time-Averaged Goods: {len(chain['avgG_series'])} Agents")
    plt.xlabel("time")
    plt.ylabel(r'$\overline{G}_i(t)$')
    plt.grid(alpha=0.3)
    plt.legend(ncol=2, fontsize=8)

    plt.tight_layout()
    plt.show()


def plot_snapshot_window_distances_money_goods(chain, window=50, step=20):
    posM, ksM, w1M = window_distances_snapshots(chain["money_snaps"], window=window, step=step)
    posG, ksG, w1G = window_distances_snapshots(chain["goods_snaps"], window=window, step=step)

    plt.figure(figsize=(14, 7))

    plt.subplot(2, 2, 1)
    plt.plot(posM, ksM, marker="o", linewidth=1.2)
    plt.title("KS across snapshot windows: money")
    plt.xlabel("snapshot index")
    plt.ylabel("KS")
    plt.grid(alpha=0.3)

    plt.subplot(2, 2, 2)
    plt.plot(posM, w1M, marker="o", linewidth=1.2)
    plt.title("Wasserstein-1 across snapshot windows: money")
    plt.xlabel("snapshot index")
    plt.ylabel("W1")
    plt.grid(alpha=0.3)

    plt.subplot(2, 2, 3)
    plt.plot(posG, ksG, marker="o", linewidth=1.2)
    plt.title("KS across snapshot windows: goods")
    plt.xlabel("snapshot index")
    plt.ylabel("KS")
    plt.grid(alpha=0.3)

    plt.subplot(2, 2, 4)
    plt.plot(posG, w1G, marker="o", linewidth=1.2)
    plt.title("Wasserstein-1 across snapshot windows: goods")
    plt.xlabel("snapshot index")
    plt.ylabel("W1")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def cross_chain_money_goods_report(chains, tail_snapshots=80):
    print("\n=== Cross-chain comparison (money/goods, pooled tail snapshots) ===")
    for i in range(len(chains)):
        for j in range(i + 1, len(chains)):
            ci, cj = chains[i], chains[j]
            Mi = np.concatenate(ci["money_snaps"][-tail_snapshots:])
            Mj = np.concatenate(cj["money_snaps"][-tail_snapshots:])
            Gi = np.concatenate(ci["goods_snaps"][-tail_snapshots:])
            Gj = np.concatenate(cj["goods_snaps"][-tail_snapshots:])
            print(
                f"chain {i} (seed={ci['seed']}) vs {j} (seed={cj['seed']}): "
                f"KS(M)={ks_statistic(Mi, Mj):.4f}, W1(M)={wasserstein_1d(Mi, Mj):.4g} | "
                f"KS(G)={ks_statistic(Gi, Gj):.4f}, W1(G)={wasserstein_1d(Gi, Gj):.4g}"
            )


def overlay_histograms_money_goods(chains, which="money", bins=60, tail_snaps=80):
    key = "money_snaps" if which == "money" else "goods_snaps"
    plt.figure(figsize=(12, 4))
    for ch in chains:
        x = np.concatenate(ch[key][-tail_snaps:])
        plt.hist(x, bins=bins, density= False, alpha=0.35, label=f"seed={ch['seed']}")
    plt.title(f"Overlay stationary distribution: {which}")
    plt.xlabel("value")
    plt.ylabel("density")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_resting_lob_and_trades(chain):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    if len(chain["trade_prices"]) > 0:
        plt.hist(chain["trade_prices"], bins=60, density=False, alpha=0.7)
    plt.title("Trade price distribution")
    plt.xlabel("price")
    plt.ylabel("density")
    plt.grid(alpha=0.25)

    plt.subplot(1, 2, 2)
    if len(chain["final_bids"]) > 0:
        plt.hist(chain["final_bids"], bins=60, density=False, alpha=0.5, label="bids")
    if len(chain["final_asks"]) > 0:
        plt.hist(chain["final_asks"], bins=60, density=False, alpha=0.5, label="asks")
    plt.title("Resting LOB price distribution (final)")
    plt.xlabel("price")
    plt.ylabel("density")
    plt.grid(alpha=0.25)
    plt.legend()

    plt.tight_layout()
    plt.show()


# =========================================================
# Plotting (NEW suite)
# =========================================================

def plot_initial_vs_final_money_goods(init_money, final_money, init_goods, final_goods, bins=40):
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.hist(init_money, bins=bins, density=False, alpha=0.8)
    plt.title("Initial Money Distribution")
    plt.xlabel("Money M")
    plt.ylabel("Density")
    plt.grid(alpha=0.3)

    plt.subplot(2, 2, 2)
    plt.hist(final_money, bins=bins, density=False, alpha=0.8)
    plt.title("Final Money Distribution")
    plt.xlabel("Money M")
    plt.ylabel("Density")
    plt.grid(alpha=0.3)

    plt.subplot(2, 2, 3)
    plt.hist(init_goods, bins=bins, density=False, alpha=0.8)
    plt.title("Initial Goods Distribution")
    plt.xlabel("Goods G")
    plt.ylabel("Density")
    plt.grid(alpha=0.3)

    plt.subplot(2, 2, 4)
    plt.hist(final_goods, bins=bins, density=False, alpha=0.8)
    plt.title("Final Goods Distribution")
    plt.xlabel("Goods G")
    plt.ylabel("Density")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_best_bid_ask_and_incoming(chain, t_start=0, t_end=None, max_plot_points=5000):
    times = chain["bb_ba_times"]
    bb = chain["bb_series_full"]
    ba = chain["ba_series_full"]

    if t_end is None:
        t_end = int(times[-1]) if len(times) else 0

    times = np.asarray(times)
    bb = np.asarray(bb, dtype=float)
    ba = np.asarray(ba, dtype=float)

    wmask = (times >= t_start) & (times <= t_end)
    times_w = times[wmask]
    bb_w = bb[wmask]
    ba_w = ba[wmask]

    if len(times_w) > max_plot_points:
        idx = np.linspace(0, len(times_w) - 1, max_plot_points).astype(int)
        times_w = times_w[idx]
        bb_w = bb_w[idx]
        ba_w = ba_w[idx]

    bid_t, bid_p = _downsample_xy(chain["limit_bid_t"], chain["limit_bid_p"], max_plot_points)
    ask_t, ask_p = _downsample_xy(chain["limit_ask_t"], chain["limit_ask_p"], max_plot_points)

    bid_t = np.asarray(bid_t)
    bid_p = np.asarray(bid_p)
    ask_t = np.asarray(ask_t)
    ask_p = np.asarray(ask_p)

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


def plot_rolling_trade_rate(chain, rolling_window=5000, rolling_step=1000):
    steps = chain["steps"]
    lob = chain["lob"]

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


def plot_rolling_mh_acceptance(chain, rolling_window=5000, rolling_step=1000):
    steps = chain["steps"]
    lob = chain["lob"]

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


def plot_logR_hist(chain, bins=60):
    lob = chain["lob"]
    if len(lob.mh_logR) > 0:
        plt.figure(figsize=(10, 4))
        plt.hist(np.asarray(lob.mh_logR), bins=bins, density=False, alpha=0.8)
        plt.xlabel("logR")
        plt.title("MH log acceptance ratio distribution (attempted crossings)")
        plt.tight_layout()
        plt.show()


# =========================================================
# Unified runner (NEW LOB + ALL diagnostics data)
# =========================================================

#def make_agents(n=50, base_money=100.0, base_goods=100.0):
#    return [Agent(agent_id=i, money=base_money, goods=base_goods) for i in range(n)]

#def make_agents(n=50):
#    return [
#        Agent(
#            agent_id=i,
#            money=random.uniform(1, 100),
#            goods=random.uniform(1, 100),
#        )
#        for i in range(25)
#    ]

def make_agents(n=50):
   half = n // 2
   return [
       Agent(
           agent_id=i,
           money=100.0 if i < half else 1000.0,
           goods=100.0 if i < half else 1000.0,
       )
       for i in range(n)
   ]


def run_chain_new(
    seed=0,
    steps=5_000_000,
    burn_in_steps=2_000,
    depth_H=50,
    sample_every=1,
    snapshot_every=2000,
    track_n_agents=10,
    CONSERVATION_WATCHDOG=True,
):
    global CURRENT_TIME
    random.seed(seed)
    np.random.seed(seed)

    # agents = make_agents(n=50, base_money=random.uniform(1, 100), base_goods=random.uniform(1, 100))
    agents = make_agents(n=50)
    lob = LOB(global_money_cap=sum(a.M for a in agents))

    # baseline totals for watchdog
    M0 = sum(a.M for a in agents)
    G0 = sum(a.G for a in agents)

    init_money = np.array([a.M for a in agents], dtype=float)
    init_goods = np.array([a.G for a in agents], dtype=float)

    # old suite: mid/spread/imb time series
    t_series, mid_series, spr_series, imb_series = [], [], [], []

    # old suite: time-averages for track_n_agents
    #track_ids = list(range(min(track_n_agents, len(agents))))

    # Track 5 poor agents and 5 rich agents
    poor_ids = [a.id for a in agents if a.M == 100.0 and a.G == 100.0]
    rich_ids = [a.id for a in agents if a.M == 1000.0 and a.G == 1000.0]
    track_ids = poor_ids[:5] + rich_ids[:5]
    
    cum_M = {i: 0.0 for i in track_ids}
    cum_G = {i: 0.0 for i in track_ids}
    avgM_series = {i: [] for i in track_ids}
    avgG_series = {i: [] for i in track_ids}
    avg_t = []

    # old suite: snapshots
    money_snaps, goods_snaps, snap_t = [], [], []

    # new suite: best bid/ask series + incoming orders
    bb_ba_times, bb_series_full, ba_series_full, mid_series_full = [], [], [], []
    limit_bid_t, limit_bid_p = [], []
    limit_ask_t, limit_ask_p = [], []

    for t in range(steps):
        CURRENT_TIME = t
        a = random.choice(agents)

        side, order = lob.agent_generates_order(a, arrival_time=t, t=t)
        lob.arrival(side, order, t_now=t)
        lob.cancel_stale_orders()

        # record incoming orders (for scatter)
        if side is not None and order is not None:
            if side == "bid":
                limit_bid_t.append(t)
                limit_bid_p.append(order.price)
            elif side == "ask":
                limit_ask_t.append(t)
                limit_ask_p.append(order.price)

        # watchdog
        if CONSERVATION_WATCHDOG:
            Mt = sum(x.M for x in agents)
            Gt = sum(x.G for x in agents)
            if abs(Mt - M0) > 1e-6 or abs(Gt - G0) > 1e-6:
                print("CONSERVATION BROKEN at t =", t)
                print("M0,G0 =", M0, G0)
                print("Mt,Gt =", Mt, Gt)
                raise RuntimeError("Conservation violated")

        # time averages (old)
        for i in track_ids:
            ai = agents[i]
            cum_M[i] += ai.M
            cum_G[i] += ai.G
            avgM_series[i].append(cum_M[i] / (t + 1))
            avgG_series[i].append(cum_G[i] / (t + 1))
        avg_t.append(t)

        # mid/bid/ask full (new + DB)
        bb = lob.best_bid()
        ba = lob.best_ask()
        mid = compute_midprice(bb, ba)
        bb_ba_times.append(t)
        bb_series_full.append(bb.price if bb is not None else np.nan)
        ba_series_full.append(ba.price if ba is not None else np.nan)
        mid_series_full.append(mid)

        # old suite sampled series (mid/spread/imb)
        if (t % sample_every) == 0:
            if bb is not None and ba is not None:
                spr = float(ba.price - bb.price)
            else:
                spr = np.nan

            bd, ad = depth_within_H(lob, H=depth_H)
            if np.isfinite(bd) and np.isfinite(ad) and (bd + ad) > 0:
                imb = float((bd - ad) / (bd + ad))
            else:
                imb = np.nan

            t_series.append(t)
            mid_series.append(mid)
            spr_series.append(spr)
            imb_series.append(imb)

        # snapshots
        if (t % snapshot_every) == 0:
            money_snaps.append(np.array([x.M for x in agents], dtype=float))
            goods_snaps.append(np.array([x.G for x in agents], dtype=float))
            snap_t.append(t)

    final_money = np.array([a.M for a in agents], dtype=float)
    final_goods = np.array([a.G for a in agents], dtype=float)

    # summary prints
    acc_rate = (lob.mh_accepts / lob.mh_attempts) if lob.mh_attempts > 0 else float("nan")
    print(
        f"[seed={seed}] total M={final_money.sum():.6f} | total G={final_goods.sum():.6f} "
        f"| trades={len(lob.trade_prices)} | MH acc_rate={acc_rate:.6f}"
    )
    print(
        f"    SEED_STEPS={SEED_STEPS} | TICK_SIZE={TICK_SIZE} | K_MAX={K_MAX} "
        f"| P_MAX={lob.P_MAX} | Q_TRADE={Q_TRADE}"
    )
    print(f"    BETA_ACTION={BETA_ACTION} | LAMBDA_FILL={LAMBDA_FILL}")

    if len(lob.mh_logR) > 0:
        logR_arr = np.asarray(lob.mh_logR, dtype=float)
        tol = 1e-6
        frac_pos = float(np.mean(logR_arr > tol))
        frac_neg = float(np.mean(logR_arr < -tol))
        frac_zero = float(np.mean(np.abs(logR_arr) <= tol))
        print(f"    logR mean={np.mean(logR_arr):.4g}, median={np.median(logR_arr):.4g}")
        print(f"    logR sign fractions: pos={frac_pos:.4g}, neg={frac_neg:.4g}, near0={frac_zero:.4g} (tol={tol})")

    return dict(
        seed=seed,
        steps=steps,
        burn_in_steps=burn_in_steps,
        depth_H=depth_H,
        lob=lob,
        # distributions
        init_money=init_money,
        init_goods=init_goods,
        final_money=final_money,
        final_goods=final_goods,
        # old suite series
        t_series=np.array(t_series, dtype=int),
        mid_series=np.array(mid_series, dtype=float),
        spread_series=np.array(spr_series, dtype=float),
        imb_series=np.array(imb_series, dtype=float),
        # old suite time averages + snapshots
        avg_t=np.array(avg_t, dtype=int),
        avgM_series=avgM_series,
        avgG_series=avgG_series,
        money_snaps=money_snaps,
        goods_snaps=goods_snaps,
        snap_t=np.array(snap_t, dtype=int),
        # for end plots
        trade_prices=np.array(lob.trade_prices, dtype=float),
        final_bids=np.array([o.price for o in lob.bids], dtype=float),
        final_asks=np.array([o.price for o in lob.asks], dtype=float),
        # new suite series
        bb_ba_times=np.array(bb_ba_times, dtype=int),
        bb_series_full=np.array(bb_series_full, dtype=float),
        ba_series_full=np.array(ba_series_full, dtype=float),
        mid_series_full=np.array(mid_series_full, dtype=float),
        limit_bid_t=np.array(limit_bid_t, dtype=int),
        limit_bid_p=np.array(limit_bid_p, dtype=float),
        limit_ask_t=np.array(limit_ask_t, dtype=int),
        limit_ask_p=np.array(limit_ask_p, dtype=float),
    )


# =========================================================
# MAIN RUN (1 chain + all diagnostics + multi-chain)
# =========================================================

if __name__ == "__main__":
    # ---------- single chain ----------
    chain0 = run_chain_new(
        seed=0,
        steps=20_000_000,
        burn_in_steps=2_000,
        depth_H=50,
        sample_every=1,
        snapshot_every=2000,
        track_n_agents=10,
        CONSERVATION_WATCHDOG=True,
    )

    # (NEW) initial vs final distributions
    plot_initial_vs_final_money_goods(
        chain0["init_money"], chain0["final_money"], chain0["init_goods"], chain0["final_goods"], bins=40
    )

    # (OLD) LOB convergence diagnostics
    plot_lob_timeseries(chain0, t_start=0, t_end=20_000_000, max_points=6000)
    plot_batch_means_mid_spread_imb(chain0, batch=5000)
    plot_window_distances_mid_spread_imb(chain0, window=5000, step=2000)

    # (OLD) Money/goods time averages + snapshot window distances
    plot_time_averaged_MG(chain0, t_start=0, t_end=20_000_000, max_points=6000)
    plot_snapshot_window_distances_money_goods(chain0, window=50, step=20)

    # (NEW) best bid/ask + incoming prices (zoomed)
    plot_best_bid_ask_and_incoming(chain0, t_start=500, t_end=20_000_000, max_plot_points=5000)

    # (NEW) rolling trade rate + rolling MH acceptance + logR hist
    plot_rolling_trade_rate(chain0, rolling_window=5000, rolling_step=1000)
    plot_rolling_mh_acceptance(chain0, rolling_window=5000, rolling_step=1000)
    plot_logR_hist(chain0, bins=60)

    # (NEW) detailed balance check on mid bins
    #_ = detailed_balance_report(
    #    chain0["mid_series_full"],
    #    burn_in_steps=50_000,
    #    n_bins=25,
    #    plot=True,
    #)

    # (OLD) resting LOB at end
    plot_resting_lob_and_trades(chain0)

    # ---------- multi-chain checks (money/goods stationarity cross-chain) ----------
    #chains = [
    #    run_chain_new(seed=1, steps=10_000_000, burn_in_steps=2_000, snapshot_every=2000, track_n_agents=10),
     #   run_chain_new(seed=2, steps=10_000_000, burn_in_steps=2_000, snapshot_every=2000, track_n_agents=10),
      #  run_chain_new(seed=3, steps=10_000_000, burn_in_steps=2_000, snapshot_every=2000, track_n_agents=10),
    #]
    #cross_chain_money_goods_report(chains, tail_snapshots=80)
    #overlay_histograms_money_goods(chains, which="money", bins=60, tail_snaps=80)
    #overlay_histograms_money_goods(chains, which="goods", bins=60, tail_snaps=80)
