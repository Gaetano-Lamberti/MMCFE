"""
Lab7_GlobalFit_4Params.py
Lecture 6 - Parameter estimation, Case history 4.15.4 (guiding example, Section 5.5)

One-stage (global) estimation of the four Arrhenius parameters p = (k1ref, Ea1, k2ref, Ea2), centred at
T0 = 900 K, directly from all the pressure and selectivity data of the three runs (N = 912), by
Levenberg-Marquardt with the analytic Jacobian obtained by the chain rule. Start deliberately generic.
"""
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- settings (same in Labs 4-8)
R_GAS = 8.314e-3                       # kJ/(mol K)
T0 = 900.0                             # K, reference temperature
P_TRUE = np.array([0.7705, 90.98, 0.0764, 207.99])   # k1ref [1/s], Ea1 [kJ/mol], k2ref [1/s], Ea2 [kJ/mol]
T_RUNS = np.array([850.0, 900.0, 923.0])             # K
P0 = 100.0                             # kPa, initial pressure (pure EB), all runs
T_P = np.linspace(0.0, 15.0, 300)      # s, pressure sampling (300 points per run)
T_GC = np.array([2.0, 5.0, 9.0, 13.0]) # s, GC sampling (4 per run)
SIG_P = 0.20                           # kPa, transducer noise
SIG_S = 0.01                           # -, GC selectivity noise
SEED = 20261012


def prng_normal(n, state):
    """n standard normal numbers. Park-Miller minimal standard generator + Box-Muller (cosine branch).
    Returns (z, new_state). Identical sequence in MATLAB and Python."""
    z = np.empty(n)
    for i in range(n):
        state = (16807.0 * state) % 2147483647.0
        u1 = state / 2147483647.0
        state = (16807.0 * state) % 2147483647.0
        u2 = state / 2147483647.0
        z[i] = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
    return z, state


def k_arrhenius(p, T):
    """k1, k2 at temperature T from p = (k1ref, Ea1, k2ref, Ea2), centred at T0."""
    tau = 1.0 / T - 1.0 / T0
    return p[0] * np.exp(-p[1] / R_GAS * tau), p[2] * np.exp(-p[3] / R_GAS * tau)


def generate_data():
    """Pseudo-data for the three runs: list of dicts with T, P (300), S (4) and the true values."""
    state = float(SEED)
    runs = []
    for T in T_RUNS:
        k1, k2 = k_arrhenius(P_TRUE, T)
        K = k1 + k2
        P_true = P0 * (2.0 - np.exp(-K * T_P))
        S_true = k1 / K
        e, state = prng_normal(T_P.size, state)
        P_meas = P_true + SIG_P * e
        e, state = prng_normal(T_GC.size, state)
        S_meas = S_true + SIG_S * e
        runs.append(dict(T=T, k1=k1, k2=k2, P=P_meas, S=S_meas, P_true=P_true, S_true=S_true))
    return runs




def lm(res, jac, p0, lam=1e-3, tol=1e-10, maxit=200):
    """Levenberg-Marquardt from scratch (Section 4.9.3), Marquardt diagonal scaling.
    res(p): weighted residual vector; jac(p): its Jacobian. Returns p, SSE, iterations, SSE history."""
    p = np.array(p0, float)
    r = res(p); sse = r @ r; hist = [sse]
    for it in range(1, maxit + 1):
        J = jac(p); A = J.T @ J; g = J.T @ r
        while True:
            delta = np.linalg.solve(A + lam * np.diag(np.diag(A)), -g)
            r_new = res(p + delta); sse_new = r_new @ r_new
            if sse_new < sse:                     # accept the step
                lam = max(lam / 10.0, 1e-15)
                break
            lam *= 10.0                           # reject: more damping
            if lam > 1e15:
                return p, sse, it, hist
        p = p + delta; r = r_new
        conv = np.max(np.abs(delta) / np.maximum(np.abs(p), 1e-12)) < tol or (sse - sse_new) < tol * sse_new
        sse = sse_new; hist.append(sse)
        if conv:
            break
    return p, sse, it, hist


# ---------------------------------------------------------------- model of one run, p = (k1, k2)
def residuals_run(p, run):
    k1, k2 = p
    K = k1 + k2
    rP = (run["P"] - P0 * (2.0 - np.exp(-K * T_P))) / SIG_P
    rS = (run["S"] - k1 / K) / SIG_S
    return np.concatenate([rP, rS])


def jacobian_run(p, run):
    """Analytic Jacobian of the weighted residuals (guiding example, Section 5.3)."""
    k1, k2 = p
    K = k1 + k2
    dP = -P0 * T_P * np.exp(-K * T_P) / SIG_P            # identical for k1 and k2: the P block cannot separate them
    JP = np.column_stack([dP, dP])
    JS = np.tile([-k2 / (SIG_S * K ** 2), k1 / (SIG_S * K ** 2)], (T_GC.size, 1))
    return np.vstack([JP, JS])


def initial_guess(run):
    """Non-arbitrary start: ln(2 - P/P0) = -(k1+k2) t is linear in t (through the origin);
    use the points with conversion below 50 %, then split K with the mean measured selectivity."""
    y = 2.0 - run["P"] / P0
    m = (T_P > 0) & (y > 0.5)
    K0 = -np.sum(T_P[m] * np.log(y[m])) / np.sum(T_P[m] ** 2)
    S0 = run["S"].mean()
    return np.array([S0 * K0, (1.0 - S0) * K0])


def fit_all_runs(runs, verbose=False):
    out = []
    for run in runs:
        p0 = initial_guess(run)
        p, sse, it, hist = lm(lambda q: residuals_run(q, run), lambda q: jacobian_run(q, run), p0)
        J = jacobian_run(p, run)
        nu = T_P.size + T_GC.size - 2
        cov = (sse / nu) * np.linalg.inv(J.T @ J)
        out.append(dict(T=run["T"], p0=p0, p=p, sse=sse, it=it, hist=hist, se=np.sqrt(np.diag(cov)),
                        rho=cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]), nu=nu))
    return out


# ---------------------------------------------------------------- global model, p = (k1ref, Ea1, k2ref, Ea2)
def residuals_global(p, runs):
    out = []
    for run in runs:
        k1, k2 = k_arrhenius(p, run["T"])
        out.append(residuals_run([k1, k2], run))
    return np.concatenate(out)


def jacobian_global(p, runs):
    """Chain rule: d r/d p = (d r/d k_i) (d k_i/d p), with d k_i/d k_i,ref = k_i/k_i,ref and
    d k_i/d Ea_i = -k_i (1/T - 1/T0)/R. Every block depends on all four parameters through k1(T), k2(T)."""
    blocks = []
    for run in runs:
        T = run["T"]
        k1, k2 = k_arrhenius(p, T)
        Jk = jacobian_run([k1, k2], run)                   # columns: d r/d k1, d r/d k2
        tau = 1.0 / T - 1.0 / T0
        blocks.append(np.column_stack([Jk[:, 0] * k1 / p[0], Jk[:, 0] * (-k1 * tau / R_GAS),
                                       Jk[:, 1] * k2 / p[2], Jk[:, 1] * (-k2 * tau / R_GAS)]))
    return np.vstack(blocks)


def fit_global(runs, p0):
    return lm(lambda q: residuals_global(q, runs), lambda q: jacobian_global(q, runs), p0)


if __name__ == "__main__":
    runs = generate_data()
    p0 = np.array([0.5, 100.0, 0.05, 150.0])           # deliberately generic start
    p, sse, it, hist = fit_global(runs, p0)
    N = sum(T_P.size + T_GC.size for _ in runs)
    print("Lab 7 - global one-stage fit of the four Arrhenius parameters (centred at T0 = %.0f K)" % T0)
    # Jacobian check
    Ja = jacobian_global(p0, runs); Jn = np.empty_like(Ja)
    for j in range(4):
        dp = p0.copy(); h = 1e-7 * p0[j]; dp[j] += h
        Jn[:, j] = (residuals_global(dp, runs) - residuals_global(p0, runs)) / h
    print("  Jacobian check: max |analytic - finite difference| / max |J| = %.1e" % (np.max(np.abs(Ja - Jn)) / np.max(np.abs(Ja))))
    print("  start: k1ref = %.3f, Ea1 = %.1f, k2ref = %.3f, Ea2 = %.1f" % tuple(p0))
    print("  LM iterations: %d, N = %d, nu = %d, SSE_w = %.1f, s2 = %.3f" % (it, N, N - 4, sse, sse / (N - 4)))
    names = ["k1ref [1/s]", "Ea1 [kJ/mol]", "k2ref [1/s]", "Ea2 [kJ/mol]"]
    for nm, v, tv in zip(names, p, P_TRUE):
        print("    %-13s %10.4f   (true %.4f)" % (nm, v, tv))
    print("  SSE history:", np.array2string(np.array(hist), precision=1))

    # cross-check with a library LM
    try:
        from scipy.optimize import least_squares
        sol = least_squares(lambda q: residuals_global(q, runs), p0, jac=lambda q: jacobian_global(q, runs),
                            method="lm", xtol=1e-12, ftol=1e-12)
        print("  scipy least_squares: max relative difference %.1e" % np.max(np.abs(sol.x / p - 1)))
    except ImportError:
        pass

    # comparison with the two-stage approach (Labs 5-6)
    fits = fit_all_runs(runs)
    print("  per-temperature k from the global fit vs two-stage (Lab 5):")
    for f in fits:
        k1, k2 = k_arrhenius(p, f["T"])
        print("    T = %.0f K: k1 %.4f vs %.4f, k2 %.4f vs %.4f" % (f["T"], k1, f["p"][0], k2, f["p"][1]))

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    axs[0].semilogy(range(len(hist)), hist, "o-", color="#233F98")
    axs[0].set_xlabel("LM iteration"); axs[0].set_ylabel("weighted SSE")
    xx = np.linspace(1 / 940, 1 / 840, 100)
    for f, c in zip(fits, ["#233F98", "#D35400", "#2E7D32"]):
        axs[1].semilogy(1000 / f["T"], f["p"][0], "o", color="black")
        axs[1].semilogy(1000 / f["T"], f["p"][1], "s", color="black")
    k1c, k2c = k_arrhenius(p, 1 / xx)
    axs[1].semilogy(1000 * xx, k1c, color="#233F98", label="k1, global fit")
    axs[1].semilogy(1000 * xx, k2c, color="#D35400", label="k2, global fit")
    axs[1].set_xlabel("1000/T [1/K]"); axs[1].set_ylabel("k [1/s]"); axs[1].legend()
    axs[1].set_title("symbols: per-temperature estimates (Lab 5)")
    fig.tight_layout(); fig.savefig("Lab7_global.png", dpi=150)
    plt.show()
