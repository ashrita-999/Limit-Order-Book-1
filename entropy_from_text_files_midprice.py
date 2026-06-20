import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

EPS = 1e-9
SHOW_3D = True

MAIN_N = 500

A_SUM_OVERRIDE = 23374.7 + 500
B_SUM_OVERRIDE = 24724.0 + 500

CSV_FILES = [
    "lob_entropy_grid_seed_10000_1.csv",
    "lob_entropy_grid_seed_20000_1.csv",
    "lob_entropy_grid_seed_30000_1.csv",
]


# =========================================================
# HELPERS
# =========================================================

def mean_and_se(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) > 1:
        se = float(values.std(ddof=1) / np.sqrt(len(values)))
    else:
        se = 0.0
    return mean, se


# =========================================================
# LOAD CSV / BUILD GRIDS
# =========================================================

def build_grids_from_df(df):
    required = ["M", "G", "beta", "phi"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    df = df.sort_values(["G", "M"]).reset_index(drop=True)

    M_totals = np.sort(df["M"].unique())
    G_totals = np.sort(df["G"].unique())

    nM = len(M_totals)
    nG = len(G_totals)

    if len(df) != nM * nG:
        raise ValueError(
            f"Grid incomplete: got {len(df)} rows but expected {nM*nG}."
        )

    beta_grid = np.zeros((nG, nM))
    phi_grid = np.zeros((nG, nM))
    mid_grid = np.full((nG, nM), np.nan)

    for _, row in df.iterrows():
        i = np.where(G_totals == row["G"])[0][0]
        j = np.where(M_totals == row["M"])[0][0]

        beta_grid[i, j] = row["beta"]
        phi_grid[i, j] = row["phi"]

        if "mid" in df.columns:
            mid_grid[i, j] = row["mid"]

    return M_totals, G_totals, beta_grid, phi_grid, mid_grid


def load_one_csv(filename):
    path = os.path.join(os.path.dirname(__file__), filename)
    df = pd.read_csv(path)
    return build_grids_from_df(df)


# =========================================================
# ENTROPY FIT
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

            curls.append(curl)
            circulations.append(circulation)

    curls = np.asarray(curls)
    circulations = np.asarray(circulations)

    if len(curls) == 0:
        print("\n[Curl diagnostics] Not available for 1D grid.")
        return curls, circulations

    print("\n[Curl diagnostics]")
    print("curl mean:", curls.mean())
    print("curl std:", curls.std())
    print("curl max abs:", np.max(np.abs(curls)))
    print("circulation mean:", circulations.mean())
    print("circulation std:", circulations.std())
    print("circulation max abs:", np.max(np.abs(circulations)))

    return curls, circulations


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

    TSS = float(np.dot(b_edges, b_edges)) + 1e-12
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


# =========================================================
# CD BENCHMARK + OLS
# =========================================================

def theoretical_cd_entropy_surface(M_totals, G_totals, A_sum, B_sum, N=MAIN_N):
    M = np.asarray(M_totals, dtype=float)
    G = np.asarray(G_totals, dtype=float)

    S = np.zeros((len(G), len(M)), dtype=float)

    for i, Gtot in enumerate(G):
        for j, Mtot in enumerate(M):
            S[i, j] = (
                A_sum * np.log(max(Mtot / N, EPS))
                + B_sum * np.log(max(Gtot / N, EPS))
            )

    return S


def ols_entropy_agreement(
    S_num,
    S_theory,
    name_num="S_LOB_mean",
    name_theory="S_CD",
    print_summary=True,
):
    y = np.asarray(S_num, dtype=float).reshape(-1)
    x = np.asarray(S_theory, dtype=float).reshape(-1)

    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()

    residuals = model.resid
    RSS = float(np.sum(residuals ** 2))
    TSS = float(np.sum((y - np.mean(y)) ** 2)) + 1e-12
    ratio = RSS / TSS

    if print_summary:
        print(f"\n[OLS agreement: {name_num} ~ const + slope * {name_theory}]")
        print(model.summary())

        print("\nAdditional OLS diagnostics:")
        print(f"RSS      = {RSS:.6e}")
        print(f"TSS      = {TSS:.6e}")
        print(f"RSS/TSS  = {ratio:.6e}")
        print(f"1 - R^2  = {(1 - model.rsquared):.6e}")

    return {
        "model": model,
        "RSS": RSS,
        "TSS": TSS,
        "RSS_over_TSS": ratio,
        "residuals": residuals.reshape(np.asarray(S_num).shape),
        "fitted": model.fittedvalues.reshape(np.asarray(S_num).shape),
    }


def residual_se_against_mean_cd_fit(S_surfaces, S_cd, model):
    const = float(model.params[0])
    slope = float(model.params[1])

    fitted_cd = const + slope * S_cd

    residual_surfaces = S_surfaces - fitted_cd[None, :, :]
    residual_mean = residual_surfaces.mean(axis=0)

    n = S_surfaces.shape[0]
    if n > 1:
        residual_se = residual_surfaces.std(axis=0, ddof=1) / np.sqrt(n)
    else:
        residual_se = np.zeros_like(residual_mean)

    return residual_mean, residual_se, fitted_cd, residual_surfaces


# =========================================================
# MIDPRICE VS MU REGRESSION
# =========================================================

def plot_mid_vs_mu_regression(mu_points, mid_points):
    mu_points = np.asarray(mu_points, dtype=float)
    mid_points = np.asarray(mid_points, dtype=float)

    mask = np.isfinite(mu_points) & np.isfinite(mid_points)
    x = mu_points[mask]
    y = mid_points[mask]

    if len(x) < 2:
        print("\n[Midprice vs mu regression] Not enough finite points.")
        return None

    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()

    intercept = float(model.params[0])
    slope = float(model.params[1])
    r2 = float(model.rsquared)

    residuals = model.resid
    RSS = float(np.sum(residuals ** 2))
    TSS = float(np.sum((y - np.mean(y)) ** 2)) + 1e-12
    ratio = RSS / TSS

    print("\n[Midprice vs μ regression: all seed/grid points]")
    print(model.summary())
    print("\nAdditional diagnostics:")
    print(f"N        = {len(x)}")
    print(f"intercept = {intercept:.6g}")
    print(f"slope     = {slope:.6g}")
    print(f"R^2       = {r2:.6g}")
    print(f"RSS       = {RSS:.6e}")
    print(f"TSS       = {TSS:.6e}")
    print(f"RSS/TSS   = {ratio:.6e}")

    x_line = np.linspace(np.min(x), np.max(x), 300)
    y_fit = intercept + slope * x_line

    lo = min(np.min(x), np.min(y))
    hi = max(np.max(x), np.max(y))
    ref_line = np.linspace(lo, hi, 300)

    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, alpha=0.65, label="seed/grid points")
    plt.plot(x_line, y_fit, 'r--', label="OLS fit")
    plt.plot(ref_line, ref_line, 'k--', label="mid = μ")
    plt.xlabel("Thermodynamic Price, μ")
    plt.ylabel("LOB midprice")
    plt.title(
        f"LOB midprice vs μ across all runs | "
        f"slope={slope:.3g}, R²={r2:.4f}"
    )
    plt.legend()
    plt.tight_layout()
    plt.show()

    return {
        "model": model,
        "intercept": intercept,
        "slope": slope,
        "r2": r2,
        "RSS": RSS,
        "TSS": TSS,
        "RSS_over_TSS": ratio,
        "n_points": len(x),
    }


# =========================================================
# PLOTTING
# =========================================================

def plot_heatmap(Z, M_mesh, G_mesh, title, cbar_label):
    plt.figure(figsize=(7, 5))
    plt.imshow(
        np.asarray(Z, dtype=float),
        origin="lower",
        extent=[M_mesh.min(), M_mesh.max(), G_mesh.min(), G_mesh.max()],
        aspect="auto",
    )
    plt.colorbar(label=cbar_label)
    plt.xlabel("Total Money M")
    plt.ylabel("Total Goods G")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_surface(Z, M_mesh, G_mesh, title, zlabel):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(G_mesh, M_mesh, np.asarray(Z, dtype=float), alpha=0.85)
    ax.set_xlabel("Total Goods G")
    ax.set_ylabel("Total Money M")
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def print_ls_summary(name, ls_out):
    RSS = ls_out["RSS"]
    TSS = ls_out["TSS"]
    ratio = ls_out["RSS_over_TSS"]
    RSS_h = ls_out["RSS_h"]
    RSS_v = ls_out["RSS_v"]

    print(f"\n[LS entropy fit: {name}]")
    print(f"RSS      = {RSS:.6g}")
    print(f"TSS      = {TSS:.6g}")
    print(f"RSS/TSS  = {ratio:.6g}")
    print(f"RSS_h M-edges = {RSS_h:.6g}")
    print(f"RSS_v G-edges = {RSS_v:.6g}")

    if RSS > 0:
        print(f"M-edge fraction = {RSS_h / RSS:.3f}")
        print(f"G-edge fraction = {RSS_v / RSS:.3f}")

    r = ls_out["residuals_all"]
    print(f"Residual mean   = {r.mean():.3g}")
    print(f"Residual std    = {r.std():.3g}")
    print(f"Residual maxabs = {np.max(np.abs(r)):.3g}")


# =========================================================
# SAVE OUTPUTS
# =========================================================

def save_grid_csv(filename, grid, M_totals, G_totals):
    rows = []
    for i, G in enumerate(G_totals):
        for j, M in enumerate(M_totals):
            rows.append({"M": M, "G": G, "value": grid[i, j]})
    pd.DataFrame(rows).to_csv(filename, index=False)


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)

    S_surfaces = []
    beta_grids = []
    phi_grids = []
    mid_grids = []
    ls_outputs = []

    all_mu_points = []
    all_mid_points = []

    M_ref = None
    G_ref = None

    for filename in CSV_FILES:
        M_totals, G_totals, beta_grid, phi_grid, mid_grid = load_one_csv(filename)

        if M_ref is None:
            M_ref = M_totals
            G_ref = G_totals
        else:
            if not np.allclose(M_totals, M_ref) or not np.allclose(G_totals, G_ref):
                raise ValueError(
                    f"{filename} has a different M/G grid from the first CSV."
                )

        print(f"\nLoaded {filename}")
        print(f"Grid shape: {len(G_totals)} G-values × {len(M_totals)} M-values")

        mu_grid_seed = phi_grid / np.maximum(beta_grid, EPS)
        all_mu_points.extend(mu_grid_seed.reshape(-1))
        all_mid_points.extend(mid_grid.reshape(-1))

        integrability_curl(beta_grid, phi_grid, M_totals, G_totals)

        out = ls_entropy_fit_with_edge_residuals(
            beta_grid,
            phi_grid,
            M_totals,
            G_totals,
            gauge_fix=True,
        )

        print_ls_summary(filename, out)

        S_surfaces.append(out["S_fit"])
        beta_grids.append(beta_grid)
        phi_grids.append(phi_grid)
        mid_grids.append(mid_grid)
        ls_outputs.append(out)

    M_totals = M_ref
    G_totals = G_ref
    M_mesh, G_mesh = np.meshgrid(M_totals, G_totals)

    S_surfaces = np.asarray(S_surfaces, dtype=float)
    beta_grids = np.asarray(beta_grids, dtype=float)
    phi_grids = np.asarray(phi_grids, dtype=float)
    mid_grids = np.asarray(mid_grids, dtype=float)

    n_runs = S_surfaces.shape[0]

    S_mean = S_surfaces.mean(axis=0)

    if n_runs > 1:
        S_se = S_surfaces.std(axis=0, ddof=1) / np.sqrt(n_runs)
        beta_mean = beta_grids.mean(axis=0)
        phi_mean = phi_grids.mean(axis=0)
        beta_se = beta_grids.std(axis=0, ddof=1) / np.sqrt(n_runs)
        phi_se = phi_grids.std(axis=0, ddof=1) / np.sqrt(n_runs)
    else:
        S_se = np.zeros_like(S_mean)
        beta_mean = beta_grids[0]
        phi_mean = phi_grids[0]
        beta_se = np.zeros_like(beta_mean)
        phi_se = np.zeros_like(phi_mean)

    mid_mean = np.nanmean(mid_grids, axis=0)

    T_mean = 1.0 / np.maximum(beta_mean, EPS)
    mu_mean = phi_mean / np.maximum(beta_mean, EPS)

    # NEW plot: all 300 seed/grid points, midprice vs μ
    mid_mu_regression = plot_mid_vs_mu_regression(all_mu_points, all_mid_points)

    # Mean-field LS diagnostics for plot headline number
    mean_ls_out = ls_entropy_fit_with_edge_residuals(
        beta_mean,
        phi_mean,
        M_totals,
        G_totals,
        gauge_fix=True,
    )

    print_ls_summary("MEAN FIELD", mean_ls_out)

    mean_ls_ratio = mean_ls_out["RSS_over_TSS"]

    individual_ls_ratios = [out["RSS_over_TSS"] for out in ls_outputs]
    individual_ls_mean, individual_ls_se = mean_and_se(individual_ls_ratios)

    print("\n[Per-seed entropy reconstruction RSS/TSS]")
    for filename, ratio in zip(CSV_FILES, individual_ls_ratios):
        print(f"{filename}: RSS/TSS = {ratio:.6g}")

    print(
        "Average individual entropy LS RSS/TSS = "
        f"{individual_ls_mean:.6g} ± {individual_ls_se:.6g}"
    )

    print("\n==============================")
    print("Across-run entropy summary")
    print("==============================")
    print(f"Number of runs: {n_runs}")
    print(f"S_mean min/max: {np.min(S_mean):.6g}, {np.max(S_mean):.6g}")
    print(f"S_se mean/max: {np.mean(S_se):.6g}, {np.max(S_se):.6g}")
    print(f"Mean-field LS RSS/TSS: {mean_ls_ratio:.6g}")

    # Mean-field plots
    plot_heatmap(beta_mean, M_mesh, G_mesh, "Mean β(M,G) across runs", "mean β")
    plot_heatmap(phi_mean, M_mesh, G_mesh, "Mean φ(M,G) across runs", "mean φ")
    plot_heatmap(T_mean, M_mesh, G_mesh, "T(M,G)=1/mean β", "T")
    plot_heatmap(mu_mean, M_mesh, G_mesh, "μ(M,G)=mean φ / mean β", "μ")

    plot_heatmap(
        S_mean,
        M_mesh,
        G_mesh,
        f"Mean reconstructed entropy S_ls(M,G) | LS RSS/TSS={mean_ls_ratio:.3g}",
        "mean S_ls",
    )

    plot_heatmap(
        S_se,
        M_mesh,
        G_mesh,
        "Standard error of reconstructed entropy S_ls(M,G)",
        "SE(S_ls)",
    )

    if np.isfinite(mid_mean).any():
        plot_heatmap(mid_mean, M_mesh, G_mesh, "Mean LOB midprice across runs", "mean mid")

    if SHOW_3D:
        plot_surface(
            S_mean,
            M_mesh,
            G_mesh,
            f"Mean reconstructed entropy surface | LS RSS/TSS={mean_ls_ratio:.3g}",
            "mean S_ls",
        )

    # CD benchmark using mean LOB entropy
    if A_SUM_OVERRIDE is not None and B_SUM_OVERRIDE is not None:
        S_cd = theoretical_cd_entropy_surface(
            M_totals,
            G_totals,
            A_sum=float(A_SUM_OVERRIDE),
            B_sum=float(B_SUM_OVERRIDE),
            N=MAIN_N,
        )

        plot_heatmap(
            S_cd,
            M_mesh,
            G_mesh,
            "Theoretical heterogeneous-CD entropy surface",
            "S_CD",
        )

        cd_agreement = ols_entropy_agreement(
            S_mean,
            S_cd,
            name_num="mean S_LOB",
            name_theory="S_CD",
            print_summary=True,
        )

        individual_cd_ratios = []

        print("\n[Per-seed CD vs LOB entropy RSS/TSS]")
        for filename, S_seed in zip(CSV_FILES, S_surfaces):
            cd_seed_agreement = ols_entropy_agreement(
                S_seed,
                S_cd,
                name_num=f"S_LOB {filename}",
                name_theory="S_CD",
                print_summary=False,
            )
            individual_cd_ratios.append(cd_seed_agreement["RSS_over_TSS"])
            print(f"{filename}: CD vs LOB RSS/TSS = {cd_seed_agreement['RSS_over_TSS']:.6g}")

        individual_cd_mean, individual_cd_se = mean_and_se(individual_cd_ratios)

        print(
            "Average individual CD vs LOB RSS/TSS = "
            f"{individual_cd_mean:.6g} ± {individual_cd_se:.6g}"
        )

        residual_mean, residual_se, fitted_cd, residual_surfaces = residual_se_against_mean_cd_fit(
            S_surfaces,
            S_cd,
            cd_agreement["model"],
        )

        plot_heatmap(
            cd_agreement["residuals"],
            M_mesh,
            G_mesh,
            f"OLS residuals: mean S_LOB vs fitted S_CD | RSS/TSS={cd_agreement['RSS_over_TSS']:.3g}",
            "mean OLS residual",
        )

        plot_heatmap(
            residual_se,
            M_mesh,
            G_mesh,
            "Standard error of CD-vs-LOB OLS residuals",
            "SE(residual)",
        )

        if SHOW_3D:
            plot_surface(S_cd, M_mesh, G_mesh, "Theoretical CD entropy surface", "S_CD")
            plot_surface(
                cd_agreement["residuals"],
                M_mesh,
                G_mesh,
                f"Mean OLS residual surface: S_mean - fitted S_CD | RSS/TSS={cd_agreement['RSS_over_TSS']:.3g}",
                "residual",
            )

        save_grid_csv("S_cd_theory.csv", S_cd, M_totals, G_totals)
        save_grid_csv("S_cd_fitted_to_mean_lob.csv", fitted_cd, M_totals, G_totals)
        save_grid_csv("OLS_residual_mean_lob_vs_cd.csv", cd_agreement["residuals"], M_totals, G_totals)
        save_grid_csv("OLS_residual_se_lob_vs_cd.csv", residual_se, M_totals, G_totals)

    else:
        print("\nSkipping CD benchmark because A_SUM_OVERRIDE/B_SUM_OVERRIDE are not set.")

    # Save mean / SE outputs
    save_grid_csv("S_mean_from_runs.csv", S_mean, M_totals, G_totals)
    save_grid_csv("S_se_from_runs.csv", S_se, M_totals, G_totals)

    save_grid_csv("beta_mean_from_runs.csv", beta_mean, M_totals, G_totals)
    save_grid_csv("phi_mean_from_runs.csv", phi_mean, M_totals, G_totals)
    save_grid_csv("beta_se_from_runs.csv", beta_se, M_totals, G_totals)
    save_grid_csv("phi_se_from_runs.csv", phi_se, M_totals, G_totals)

    save_grid_csv("T_mean_from_runs.csv", T_mean, M_totals, G_totals)
    save_grid_csv("mu_mean_from_runs.csv", mu_mean, M_totals, G_totals)

    print("\nSaved output CSVs:")
    print("  S_mean_from_runs.csv")
    print("  S_se_from_runs.csv")
    print("  beta_mean_from_runs.csv")
    print("  phi_mean_from_runs.csv")
    print("  beta_se_from_runs.csv")
    print("  phi_se_from_runs.csv")
    print("  T_mean_from_runs.csv")
    print("  mu_mean_from_runs.csv")
    if A_SUM_OVERRIDE is not None and B_SUM_OVERRIDE is not None:
        print("  S_cd_theory.csv")
        print("  S_cd_fitted_to_mean_lob.csv")
        print("  OLS_residual_mean_lob_vs_cd.csv")
        print("  OLS_residual_se_lob_vs_cd.csv")
