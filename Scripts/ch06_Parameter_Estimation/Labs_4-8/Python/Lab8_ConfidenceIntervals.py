"""
Lab8_ConfidenceIntervals.py
Lecture 6 - Parameter estimation, Case history 4.15.5 (guiding example, Section 5.6)

Uncertainty of the global fit of Lab 7 (recomputed here): linearized covariance s^2 (J'J)^-1 (Eq. 45),
standard errors, 95 % marginal intervals, correlation matrix, and the joint 95 % region of the most
correlated pair compared with the rectangle of the marginal intervals.
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


def student_t_975(nu):
    """97.5 % quantile of Student's t (Cornish-Fisher expansion, accurate to 1e-4 for nu >= 3),
    written out so that the MATLAB version needs no Statistics Toolbox."""
    z = 1.959963984540054
    g1 = (z ** 3 + z) / 4
    g2 = (5 * z ** 5 + 16 * z ** 3 + 3 * z) / 96
    g3 = (3 * z ** 7 + 19 * z ** 5 + 17 * z ** 3 - 15 * z) / 384
    g4 = (79 * z ** 9 + 776 * z ** 7 + 1482 * z ** 5 - 1920 * z ** 3 - 945 * z) / 92160
    return z + g1 / nu + g2 / nu ** 2 + g3 / nu ** 3 + g4 / nu ** 4


if __name__ == "__main__":
    runs = generate_data()
    p, sse, it, _ = fit_global(runs, np.array([0.5, 100.0, 0.05, 150.0]))     # = Lab 7
    J = jacobian_global(p, runs)
    N = J.shape[0]; nu = N - 4
    s2 = sse / nu
    cov = s2 * np.linalg.inv(J.T @ J)                    # Eq. (45)
    se = np.sqrt(np.diag(cov))
    t = student_t_975(nu)
    R = cov / np.outer(se, se)
    names = ["k1ref [1/s]", "Ea1 [kJ/mol]", "k2ref [1/s]", "Ea2 [kJ/mol]"]
    print("Lab 8 - confidence intervals of the global fit (N = %d, nu = %d, s2 = %.3f, t = %.4f)" % (N, nu, s2, t))
    print("  parameter        estimate        SE           95 % interval            true   inside?")
    for j in range(4):
        lo, hi = p[j] - t * se[j], p[j] + t * se[j]
        print("  %-13s %10.4f  %10.4f   [%9.4f, %9.4f]   %8.4f   %s"
              % (names[j], p[j], se[j], lo, hi, P_TRUE[j], "yes" if lo <= P_TRUE[j] <= hi else "NO"))
    print("  correlation matrix:")
    for j in range(4):
        print("   " + "  ".join("%+.3f" % v for v in R[j]))
    i, k = int(np.argmax(np.abs(np.triu(R, 1)))) // 4, int(np.argmax(np.abs(np.triu(R, 1)))) % 4
    print("  strongest correlation: rho(%s, %s) = %+.3f" % (names[i].split()[0], names[k].split()[0], R[i, k]))
    try:
        from scipy import stats
        print("  check: t from scipy = %.4f; F(2, nu; 0.95) = %.4f" % (stats.t.ppf(0.975, nu), stats.f.ppf(0.95, 2, nu)))
    except ImportError:
        pass

    # joint 95 % region of the pair (k1ref, k2ref) vs the rectangle of the marginal intervals
    F2 = 3.0056                                   # F(2, 908; 0.95), from tables
    C2 = cov[np.ix_([0, 2], [0, 2])]
    w, V = np.linalg.eigh(C2)
    th = np.linspace(0, 2 * np.pi, 400)
    E = p[[0, 2]][:, None] + V @ (np.sqrt(w * 2 * F2)[:, None] * np.vstack([np.cos(th), np.sin(th)]))
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.plot(E[0], E[1], color="#233F98", lw=2, label="95 % joint region")
    lo = p[[0, 2]] - t * se[[0, 2]]
    ax.add_patch(plt.Rectangle(lo, 2 * t * se[0], 2 * t * se[2], fill=False, ec="#D35400", lw=2, ls="--",
                               label="95 % marginal intervals"))
    ax.plot(p[0], p[2], "+", color="black", ms=12, mew=2, label="estimate")
    ax.plot(P_TRUE[0], P_TRUE[2], "x", color="black", ms=8, mew=1.5, label="true value")
    ax.set_xlabel("k1ref [1/s]"); ax.set_ylabel("k2ref [1/s]")
    ax.set_title("rho(k1ref, k2ref) = %.2f" % R[0, 2]); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig("Lab8_ellipse.png", dpi=150)
    plt.show()
