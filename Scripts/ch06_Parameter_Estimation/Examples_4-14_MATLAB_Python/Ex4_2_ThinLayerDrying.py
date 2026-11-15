"""Ex4_2_ThinLayerDrying.py
Lecture 6 - Parameter estimation, Section 4.14.3-4.14.4, Example 4.2: thin-layer drying of apple slices

Slices 4 mm thick, dried from both faces (half thickness L = 2 mm). Moisture ratio MR every 15 min for 5 h.
Two models:
  Fick's second law (Eq. 51), one physical parameter D_eff:
      dX/dt = D d2X/dx2, dX/dx = 0 at x = 0 (symmetry), X = 0 at x = L (surface at equilibrium),
      MR(t) = (1/L) * integral_0^L X dx
  Page (Eq. 52), empirical, two parameters: MR = exp(-k t^n), t in min
Algorithms:
  Page: linearized fit ln(-ln MR) = ln k + n ln t; Levenberg-Marquardt.
  Fick: slope method (first term of the series solution at long times); LM on D with the PDE solved
        numerically by the method of lines inside the fit (Section 4.10); 1-D bounded search;
        LM with the series solution as a check.
Pseudo-data: Fick with D = 3.0e-10 m^2/s, noise sigma = 0.01 on MR (numpy generator, seed 52),
the same values as in the MATLAB script and in the slides. Requires numpy, scipy, matplotlib.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, minimize_scalar
from scipy.integrate import solve_ivp

# ---------------------------------------------------------------- settings and data
Lh = 2.0e-3                     # m, half thickness
D_TRUE = 3.0e-10                # m^2/s, used only to generate the pseudo-data
SIG_MR = 0.01                   # noise on MR
t_min = np.arange(0, 301, 15.0); t_s = 60 * t_min
NPDE = [0]                      # counter of PDE solutions


def mr_series(t, D, nterm=60):
    """Series solution of Fick's law for a slab, uniform initial moisture, surface at equilibrium."""
    n = np.arange(nterm)[:, None]
    return np.sum(8 / ((2 * n + 1) ** 2 * np.pi ** 2)
                  * np.exp(-(2 * n + 1) ** 2 * np.pi ** 2 * D * t[None, :] / (4 * Lh ** 2)), 0)


NX = 101
xg = np.linspace(0, Lh, NX); dx = xg[1] - xg[0]


def mr_mol(t, D, rtol=1e-8):
    """Method of lines (Lecture 5): 101 nodes, unknowns at nodes 1..100 (node 101 = surface, X = 0).
    Symmetry at x = 0 by a mirror node; stiff solver (BDF); MR = average of X, trapezoidal rule."""
    NPDE[0] += 1
    def rhs(_, u):
        full = np.append(u, 0.0)
        d2 = np.empty_like(u)
        d2[0] = 2 * (full[1] - full[0]) / dx ** 2
        d2[1:] = (full[2:] - 2 * full[1:-1] + full[:-2]) / dx ** 2
        return D * d2
    sol = solve_ivp(rhs, (0, t[-1]), np.ones(NX - 1), t_eval=t, method="BDF", rtol=rtol, atol=1e-10)
    U = np.vstack([sol.y, np.zeros(len(t))])
    return np.trapezoid(U, xg, axis=0) / Lh


rng = np.random.default_rng(52)
MR = mr_series(t_s, D_TRUE) + SIG_MR * rng.standard_normal(t_s.size)
MR[0] = 1.0                                                # by definition
print("4.2 MOL vs series at D_true: max diff = %.1e" % np.max(np.abs(mr_mol(t_s, D_TRUE) - mr_series(t_s, D_TRUE))))

# ---------------------------------------------------------------- Page model
m = (t_min > 0) & (MR > 0.02) & (MR < 0.98)
b1, b0 = np.polyfit(np.log(t_min[m]), np.log(-np.log(MR[m])), 1)       # ln(-ln MR) = ln k + n ln t
pg_lin = np.array([np.exp(b0), b1])
page = lambda t, k, n: np.exp(-k * t ** n)
res_pg = lambda p: (MR - page(t_min, *p)) / SIG_MR
pg_lm = least_squares(res_pg, pg_lin, method="lm")
for name, p in [("Page linearized", pg_lin), ("Page LM", pg_lm.x)]:
    r = res_pg(p)
    print("4.2 %-18s k=%.5f  n=%.4f   SSE_w=%.2f  s2=%.2f" % (name, *p, r @ r, r @ r / (t_min.size - 2)))

# ---------------------------------------------------------------- Fick: slope method, ln MR = ln(8/pi^2) - pi^2 D t/(4 L^2)
late = (MR < 0.6) & (MR > 0.02)
sl, ic = np.polyfit(t_s[late], np.log(MR[late]), 1)
D_slope = -sl * 4 * Lh ** 2 / np.pi ** 2
print("4.2 Fick slope method: D=%.3e (intercept %.3f vs ln(8/pi^2)=%.3f)" % (D_slope, ic, np.log(8 / np.pi ** 2)))

# ---------------------------------------------------------------- Fick: LM on d = D/1e-10, PDE solved at every evaluation
res_f = lambda q: (MR - mr_mol(t_s, q[0] * 1e-10)) / SIG_MR
NPDE[0] = 0
fk = least_squares(res_f, [D_slope / 1e-10], method="lm", diff_step=1e-6)
r = fk.fun
J = (res_f(fk.x * (1 + 1e-6)) - r) / (fk.x * 1e-6)                    # 1 x 1 problem: J is a column
se_D = np.sqrt((r @ r / (t_min.size - 1)) / (J @ J)) * 1e-10          # Eqs. (45)-(46)
print("4.2 Fick LM (MOL): D=%.4e +- %.1e   SSE_w=%.2f  s2=%.2f  PDE solutions=%d"
      % (fk.x[0] * 1e-10, se_D, r @ r, r @ r / (t_min.size - 1), NPDE[0]))

# ---------------------------------------------------------------- Fick: 1-D bounded search
NPDE[0] = 0
gs = minimize_scalar(lambda d: np.sum(((MR - mr_mol(t_s, d * 1e-10)) / SIG_MR) ** 2), bounds=(0.5, 10),
                     method="bounded", options=dict(xatol=1e-6))
print("4.2 Fick 1-D bounded search: D=%.4e   PDE solutions=%d" % (gs.x * 1e-10, NPDE[0]))

# ---------------------------------------------------------------- Fick: LM with the series solution (check)
fs = least_squares(lambda q: (MR - mr_series(t_s, q[0] * 1e-10)) / SIG_MR, [D_slope / 1e-10], method="lm")
print("4.2 Fick LM (series): D=%.4e" % (fs.x[0] * 1e-10))

# ---------------------------------------------------------------- figure
fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.6))
tt = np.linspace(0, 300, 301)
axs[0].plot(t_min, MR, "ko", mfc="w", label="data")
axs[0].plot(tt, page(tt, *pg_lm.x), "--", color="#D35400", lw=2, label="Page, LM")
axs[0].plot(tt, mr_series(60 * tt, fk.x[0] * 1e-10), color="#233F98", lw=2, label="Fick, LM (MOL)")
axs[0].set_xlabel("time t [min]"); axs[0].set_ylabel("moisture ratio MR"); axs[0].legend(frameon=False)
axs[1].axhline(0, color="k", lw=0.8)
axs[1].plot(t_min, res_pg(pg_lm.x), "s", color="#D35400", label="Page")
axs[1].plot(t_min, res_f(fk.x), "o", color="#233F98", label="Fick")
axs[1].set_ylim(-4.5, 4); axs[1].set_xlabel("time t [min]"); axs[1].set_ylabel(r"$(y_i-\hat y_i)/\sigma$")
axs[1].legend(frameon=False, ncol=2, loc="lower center")
fig.tight_layout(); fig.savefig("Ex4_2_drying_py.png", dpi=150)
