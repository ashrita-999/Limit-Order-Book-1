import random
import matplotlib.pyplot as plt
import numpy as np
import math


CURRENT_TIME = 0
EPS = 1e-9

AGENT_SEED = 123
agent_rng = random.Random(AGENT_SEED)

# =========================================================
# CONFIG
# =========================================================

TICK_SIZE = 0.01
P_MIN = TICK_SIZE

P_FILL_MIN = 1e-9
K_MAX = int(math.ceil(-math.log(P_FILL_MIN)))

SEED_STEPS = 100
BETA_ACTION = 1
DO_NOTHING_FLOOR_FRAC = 0

ORDER_EXP_LAM = 0.001
Q_TRADE = 1.0
LAMBDA_FILL = 0.1

DIAG_EVERY = 1_000_000
MID_MEAN_WINDOW = 1000


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


def group_cv(values):
    values = np.asarray(values, dtype=float)
    mean = np.mean(values)
    if abs(mean) < EPS:
        return np.nan
    return float(np.std(values) / mean)


class Order:
    def __init__(self, agent, price, qty, arrival_time, lam=ORDER_EXP_LAM):
        self.agent = agent
        self.price = float(price)
        self.qty = float(qty)
        self.arrival_time = int(arrival_time)
        lifetime = np.random.exponential(scale=1 / lam) if lam > 0 else float("inf")
        self.expiration_time = self.arrival_time + lifetime


class Agent:
    def __init__(self, agent_id, money, goods):
        self.id = agent_id
        self.M = float(money)
        self.G = float(goods)


        self.alpha = agent_rng.uniform(1.0 + 1e-6, 30.0)
        self.nu = agent_rng.uniform(1.0 + 1e-6, 30.0)
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
        self.bids = []
        self.asks = []

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

    def _touch_or_synthetic(self, agent: Agent):
        bb = self.best_bid()
        ba = self.best_ask()

        if bb is not None and ba is not None:
            return bb.price, ba.price

        p_star = round_to_tick(max(agent.implied_price(), P_MIN))

        if bb is None and ba is None:
            return max(P_MIN, p_star - TICK_SIZE), max(P_MIN, p_star)

        if bb is None and ba is not None:
            a_ref = ba.price
            return max(P_MIN, a_ref - TICK_SIZE), a_ref

        b_ref = bb.price
        return b_ref, b_ref + TICK_SIZE

    def agent_generates_order(self, agent: Agent, arrival_time: int, t: int):
        if t < SEED_STEPS:
            side = "ask" if random.random() < agent.a else "bid"

            p = round_to_tick(max(agent.implied_price(), P_MIN))
            p = min(max(p, P_MIN), self.P_MAX)

            if side == "ask" and agent.G < Q_TRADE:
                return None, None
            if side == "bid" and agent.M < p * Q_TRADE:
                return None, None

            return side, Order(agent, p, Q_TRADE, arrival_time)

        logU0 = agent.log_utility()
        actions = []

        bb = self.best_bid()
        ba = self.best_ask()

        if ba is not None:
            p_exec = ba.price
            if agent.M >= p_exec * Q_TRADE:
                logU_after = agent.log_utility_at(agent.M - p_exec * Q_TRADE, agent.G + Q_TRADE)
                w = math.exp(BETA_ACTION * (logU_after - logU0))
                actions.append(("market", "bid", float(p_exec), float(w)))

        if bb is not None:
            p_exec = bb.price
            if agent.G >= Q_TRADE:
                logU_after = agent.log_utility_at(agent.M + p_exec * Q_TRADE, agent.G - Q_TRADE)
                w = math.exp(BETA_ACTION * (logU_after - logU0))
                actions.append(("market", "ask", float(p_exec), float(w)))

        b_ref, a_ref = self._touch_or_synthetic(agent)
        a_tk = price_to_tick(a_ref)
        b_tk = price_to_tick(b_ref)

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
                w = p_fill * math.exp(BETA_ACTION * (logU_after - logU0))
                actions.append(("passive", "bid", p, float(w)))

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
                w = p_fill * math.exp(BETA_ACTION * (logU_after - logU0))
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

        for kind, side, p, w in actions:
            cum += w
            if r <= cum:
                return side, Order(agent, p, Q_TRADE, arrival_time)

        return None, None

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
        self.mh_attempt_times.append(int(t_now))

        log_u_before = buyer.log_utility() + seller.log_utility()
        log_u_after = buyer.log_utility_at(Mb_new, Gb_new) + seller.log_utility_at(Ms_new, Gs_new)
        logR = log_u_after - log_u_before

        self.mh_logR.append(float(logR))

        accept = logR >= 0.0 or random.random() < math.exp(logR)

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

        else:
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


def compute_midprice(bb, ba):
    if bb is None or ba is None:
        return np.nan
    return 0.5 * (float(bb.price) + float(ba.price))


def safe_mean_last_finite(arr, window):
    if len(arr) == 0:
        return np.nan

    tail = np.asarray(arr[-window:], dtype=float)
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

def make_agents(n):
    money_weights = [random.uniform(1, 100) for _ in range(n)]
    goods_weights = [random.uniform(1, 100) for _ in range(n)]

    money_total = sum(money_weights)
    goods_total = sum(goods_weights)

    return [
        Agent(
            agent_id=i,
            money=money_weights[i] / money_total * 750_000,
            goods=goods_weights[i] / goods_total * 750_000
        )
        for i in range(n)
    ]
    

def run_chain(
    steps=200_000,
    burn_in=2000,
    max_plot_points=5000,
    t_start=500,
    t_end=200_000,
    CONSERVATION_WATCHDOG=True,
):
    global CURRENT_TIME

    #rich poor grouping:

    agents1 = [Agent(agent_id=i, money=1000.0, goods=1000.0) for i in range(0, 250)]       # poor
    agents2 = [Agent(agent_id=i, money=2000.0, goods=2000.0) for i in range(250, 500)]   # rich
    agents = agents1 + agents2

    #random agent endowments:
    #agents = make_agents(500)
    #agents1 = agents[0:250]
    #agents2 = agents[250:500]

    #unimodal:
    #agents1 = [Agent(agent_id=i, money=1500.0, goods=1500.0) for i in range(0, 250)]       # poor
    #agents2 = [Agent(agent_id=i, money=1500.0, goods=1500.0) for i in range(250, 500)]   # rich
    #agents = agents1 + agents2

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

    times = []
    bb_series_full = []
    ba_series_full = []
    mid_series_full = []

    limit_bid_t, limit_bid_p = [], []
    limit_ask_t, limit_ask_p = [], []

    diag_times = []

    diag_d_rich_M = []
    diag_d_rich_G = []
    diag_d_poor_M = []
    diag_d_poor_G = []

    diag_mean_mid = []
    diag_d_mid = []

    # NEW: CV diagnostic storage
    diag_cv_rich_M = []
    diag_cv_rich_G = []
    diag_cv_poor_M = []
    diag_cv_poor_G = []

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

            # NEW: CVs at checkpoint
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

    final_distribution_data = np.column_stack(([a.id for a in agents], final_money, final_goods))
    np.savetxt(
        "final_M_G_distributions_bimodal_2.csv",
        final_distribution_data,
        delimiter=",",
        header="agent_id,final_M,final_G",
        comments="",
    )
    print("Saved final M and G distributions to final_M_G_distributions_bimodal_2.csv")

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

    #times_arr = np.array(times)
    #bb_arr = np.array(bb_series_full)
    #ba_arr = np.array(ba_series_full)

    #wmask = (times_arr >= t_start) & (times_arr <= t_end)

    #times_w = times_arr[wmask]
    #bb_w = bb_arr[wmask]
    #ba_w = ba_arr[wmask]

    #if len(times_w) > max_plot_points:
    #    idx = np.linspace(0, len(times_w) - 1, max_plot_points).astype(int)
    #    times_w = times_w[idx]
    #    bb_w = bb_w[idx]
    #    ba_w = ba_w[idx]

    #bid_t, bid_p = downsample_xy(limit_bid_t, limit_bid_p, max_plot_points)
    #ask_t, ask_p = downsample_xy(limit_ask_t, limit_ask_p, max_plot_points)

    #bid_mask = (bid_t >= t_start) & (bid_t <= t_end)
    #ask_mask = (ask_t >= t_start) & (ask_t <= t_end)

    #bid_t, bid_p = bid_t[bid_mask], bid_p[bid_mask]
    #ask_t, ask_p = ask_t[ask_mask], ask_p[ask_mask]

    #plt.figure(figsize=(14, 5))
    #plt.plot(times_w, bb_w, label="Best Bid", linewidth=2)
    #plt.plot(times_w, ba_w, label="Best Ask", linewidth=2)
    #plt.scatter(bid_t, bid_p, marker="^", s=5, label="Incoming order (bid)")
    #plt.scatter(ask_t, ask_p, marker="v", s=5, label="Incoming order (ask)")
    #plt.xlabel("Time step")
    #plt.ylabel("Price")
    #plt.title("Best bid/ask and incoming prices (zoomed)")
    #plt.legend()
    #plt.tight_layout()
    #plt.show()

    #if len(diag_times) > 0:
    #    diag_times_arr = np.asarray(diag_times, dtype=float)

    #    plt.figure(figsize=(14, 6))
    #    plt.plot(diag_times_arr, diag_d_rich_M, marker="o", label="Rich ΔM")
    #    plt.plot(diag_times_arr, diag_d_rich_G, marker="o", label="Rich ΔG")
    #    plt.plot(diag_times_arr, diag_d_poor_M, marker="o", label="Poor ΔM")
    #    plt.plot(diag_times_arr, diag_d_poor_G, marker="o", label="Poor ΔG")
    #    plt.axhline(0.0, color="black", linewidth=1, alpha=0.6)
    #    plt.xlabel("Time")
    #    plt.ylabel(f"Change over last {DIAG_EVERY:,} steps")
    #    plt.title("Rich/Poor group inventory changes at diagnostic checkpoints")
    #    plt.legend()
    #    plt.tight_layout()
    #    plt.show()

    #    plt.figure(figsize=(14, 5))
    #    plt.plot(diag_times_arr, diag_d_mid, marker="o", label="Δ mean mid")
    #    plt.axhline(0.0, color="black", linewidth=1, alpha=0.6)
    #    plt.xlabel("Time")
    #    plt.ylabel("Change in printed mean mid")
    #    plt.title("Change in mean midprice at diagnostic checkpoints")
    #    plt.legend()
    #    plt.tight_layout()
    #    plt.show()

        # NEW: CV plot
    #    plt.figure(figsize=(14, 6))
    #    plt.plot(diag_times_arr, diag_cv_rich_M, marker="o", label="Rich CV(M)")
    #    plt.plot(diag_times_arr, diag_cv_poor_M, marker="o", label="Poor CV(M)")
    #    plt.plot(diag_times_arr, diag_cv_rich_G, marker="o", label="Rich CV(G)")
    #    plt.plot(diag_times_arr, diag_cv_poor_G, marker="o", label="Poor CV(G)")
    #    plt.xlabel("Time")
    #    plt.ylabel("Coefficient of variation")
    #    plt.title("Within-group coefficient of variation over time")
    #    plt.legend()
    #    plt.tight_layout()
    #    plt.show()

    best_bid_series = np.array([x for x in bb_series_full if np.isfinite(x)])
    best_ask_series = np.array([x for x in ba_series_full if np.isfinite(x)])

    if burn_in < len(best_bid_series):
        best_bid_series = best_bid_series[burn_in:]
    if burn_in < len(best_ask_series):
        best_ask_series = best_ask_series[burn_in:]

    return best_bid_series, best_ask_series, lob


best_bids, best_asks, lob = run_chain(
    steps= 100_000_000,
    burn_in=2000,
    t_start=500,
    t_end= 100_000_000,
    CONSERVATION_WATCHDOG=True,
)
