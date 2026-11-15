"""
Lab5_BatchFitting_LM_k1k2.py
Lecture 6 - Parameter estimation, Case history 4.15.2 (guiding example, Sections 5.2-5.3)

For each temperature, k1 and k2 are estimated from the pressure curve P(t) (300 points) and from the
GC selectivity (4 points), with weighted residuals (Section 4.6.3) and Levenberg-Marquardt implemented
from scratch with the analytic Jacobian (Section 4.9). The pseudo-data are regenerated exactly as in Lab 4.
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


if __name__ == "__main__":
    runs = generate_data()
    fits = fit_all_runs(runs)

    # Jacobian check: analytic vs forward differences, at the start of the 900 K run
    run = runs[1]; p = initial_guess(run); Ja = jacobian_run(p, run); Jn = np.empty_like(Ja)
    for j in range(2):
        dp = p.copy(); h = 1e-7 * p[j]; dp[j] += h
        Jn[:, j] = (residuals_run(dp, run) - residuals_run(p, run)) / h
    print("Lab 5 - k1, k2 at each temperature (weighted LM, analytic Jacobian)")
    print("  Jacobian check: max |analytic - finite difference| / max |J| = %.1e" % (np.max(np.abs(Ja - Jn)) / np.max(np.abs(Ja))))
    print("   T [K]   start k1, k2          k1 (true)          k2 (true)          it   SSE_w   s2    rho(k1,k2)")
    for f, r in zip(fits, runs):
        print("  %6.0f   %.4f, %.4f    %.4f (%.4f)    %.4f (%.4f)    %2d   %6.1f  %.3f   %+.3f"
              % (f["T"], *f["p0"], f["p"][0], r["k1"], f["p"][1], r["k2"], f["it"], f["sse"], f["sse"] / f["nu"], f["rho"]))
    print("  standard errors:")
    for f in fits:
        print("  %6.0f   SE(k1) = %.4f  SE(k2) = %.4f   relative %.2f %%, %.2f %%"
              % (f["T"], *f["se"], 100 * f["se"][0] / f["p"][0], 100 * f["se"][1] / f["p"][1]))

    # identifiability: the P block alone gives a singular J'J
    JP = jacobian_run(fits[1]["p"], runs[1])[:T_P.size]
    print("  P block only: cond(J'J) = %.1e  (columns identical -> k1, k2 not separable)" % np.linalg.cond(JP.T @ JP))

    # cross-check with a library LM
    try:
        from scipy.optimize import least_squares
        for f, run in zip(fits, runs):
            sol = least_squares(lambda q: residuals_run(q, run), f["p0"], jac=lambda q: jacobian_run(q, run),
                                method="lm", xtol=1e-12, ftol=1e-12)
            print("  scipy least_squares, T=%.0f K: max relative difference %.1e" % (f["T"], np.max(np.abs(sol.x / f["p"] - 1))))
    except ImportError:
        pass

    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    cols = ["#233F98", "#D35400", "#2E7D32"]
    for f, run, c in zip(fits, runs, cols):
        K = f["p"].sum()
        axs[0].plot(T_P, run["P"], ".", color=c, ms=2)
        axs[0].plot(T_P, P0 * (2 - np.exp(-K * T_P)), "-", color="black", lw=1)
        axs[1].plot(T_GC, run["S"], "o", color=c, label="T = %.0f K" % f["T"])
        axs[1].axhline(f["p"][0] / K, color=c, lw=1)
        axs[2].semilogy(range(len(f["hist"])), f["hist"], "o-", color=c)
    axs[0].set_xlabel("t [s]"); axs[0].set_ylabel("P [kPa]")
    axs[1].set_xlabel("t [s]"); axs[1].set_ylabel("S"); axs[1].legend()
    axs[2].set_xlabel("LM iteration"); axs[2].set_ylabel("weighted SSE")
    fig.tight_layout(); fig.savefig("Lab5_fits.png", dpi=150)
    plt.show()
