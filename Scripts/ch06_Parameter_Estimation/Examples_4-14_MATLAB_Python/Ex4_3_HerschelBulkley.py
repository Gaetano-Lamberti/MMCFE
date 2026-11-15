"""Ex4_3_HerschelBulkley.py
Lecture 6 - Parameter estimation, Section 4.14.5-4.14.6, Example 4.3: Herschel-Bulkley fluid

Flow curve of a yield-stress food (e.g. a sauce): tau = tau0 + K * gdot^n (Eq. 53), 12 shear rates
from 1 to 200 1/s, relative error 3% on tau (heteroscedastic: weights w_i = 1/sigma_i^2, Section 4.6).
Algorithms:
  power law tau = K gdot^n by log-log linear regression (tau0 = 0): starting point;
  Herschel-Bulkley by Levenberg-Marquardt;
  separable least squares: for fixed n the model is linear in (tau0, K) -> 1-D search on n only.
Uncertainty and correlations from Eqs. (45)-(46).
Pseudo-data: tau0 = 15 Pa, K = 6 Pa s^n, n = 0.35 (numpy generator, seed 7): same values as the slides
and the MATLAB script. Requires numpy, scipy, matplotlib.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, minimize_scalar

TAU0, KHB, NHB, REL = 15.0, 6.0, 0.35, 0.03
gdot = np.logspace(0, np.log10(200), 12)
hb = lambda g, t0, k, n: t0 + k * g ** n
rng = np.random.default_rng(7)
tau = hb(gdot, TAU0, KHB, NHB) * (1 + REL * rng.standard_normal(gdot.size))
sig = REL * tau                                            # relative error -> weights
NEV = [0]

# ---------------------------------------------------------------- power law, log-log
pl1, pl0 = np.polyfit(np.log(gdot), np.log(tau), 1)
print("4.3 power law (log-log): K=%.2f n=%.3f" % (np.exp(pl0), pl1))
r = (tau - np.exp(pl0) * gdot ** pl1) / sig
print("    power law  SSE_w=%.1f  s2=%.1f" % (r @ r, r @ r / (gdot.size - 2)))


def res_hb(p):
    NEV[0] += 1
    return (tau - hb(gdot, *p)) / sig


# ---------------------------------------------------------------- Herschel-Bulkley by LM, from the power law
p0 = np.array([1.0, np.exp(pl0), pl1])
NEV[0] = 0
f = least_squares(res_hb, p0, method="lm")
p_lm, n_lm = f.x, NEV[0]


# ---------------------------------------------------------------- separable least squares
def linear_part(n):
    """For fixed n the model is linear in (tau0, K): weighted linear least squares, Eq. (30)."""
    A = np.column_stack([np.ones_like(gdot), gdot ** n]) / sig[:, None]
    q, *_ = np.linalg.lstsq(A, tau / sig, rcond=None)
    rr = tau / sig - A @ q
    return q, rr @ rr


cnt = [0]
def prof(n):
    cnt[0] += 1
    return linear_part(n)[1]


ns = minimize_scalar(prof, bounds=(0.05, 1.5), method="bounded", options=dict(xatol=1e-10))
q, _ = linear_part(ns.x)
p_sep = np.array([q[0], q[1], ns.x])
for name, p, n in [("LM", p_lm, n_lm), ("separable (1-D on n)", p_sep, cnt[0])]:
    rr = (tau - hb(gdot, *p)) / sig
    print("4.3 %-24s tau0=%.2f K=%.3f n=%.4f  SSE_w=%.2f  evals=%d" % (name, *p, rr @ rr, n))

# ---------------------------------------------------------------- uncertainty, Eqs. (45)-(46)
J = np.empty((gdot.size, 3)); r0 = res_hb(p_lm)
for j in range(3):
    dp = p_lm.copy(); h = 1e-7 * abs(p_lm[j]); dp[j] += h
    J[:, j] = (res_hb(dp) - r0) / h
s2 = r0 @ r0 / (gdot.size - 3)
cov = s2 * np.linalg.inv(J.T @ J); se = np.sqrt(np.diag(cov)); Rm = cov / np.outer(se, se)
print("4.3 SE: tau0 %.2f K %.2f n %.3f  s2=%.2f ; rho(tau0,K)=%.2f rho(tau0,n)=%.2f rho(K,n)=%.2f"
      % (*se, s2, Rm[0, 1], Rm[0, 2], Rm[1, 2]))

# ---------------------------------------------------------------- figure: flow curve and profile on n
nn = np.linspace(0.15, 0.7, 200); pr = np.array([linear_part(v)[1] for v in nn])
fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.6))
gg = np.logspace(0, np.log10(200), 200)
axs[0].errorbar(gdot, tau, yerr=sig, fmt="ko", mfc="w", ecolor="0.5", capsize=2, label="data")
axs[0].loglog(gg, np.exp(pl0) * gg ** pl1, "--", color="#D35400", lw=2, label="power law (log-log)")
axs[0].loglog(gg, hb(gg, *p_lm), color="#233F98", lw=2, label="Herschel-Bulkley, LM")
axs[0].set_xlabel(r"shear rate [s$^{-1}$]"); axs[0].set_ylabel("shear stress [Pa]"); axs[0].legend(frameon=False)
axs[1].semilogy(nn, pr, color="#233F98", lw=2); axs[1].semilogy(ns.x, linear_part(ns.x)[1], "o", color="#D35400")
axs[1].set_xlabel("flow index n (fixed)"); axs[1].set_ylabel(r"min SSE$_w$ over $\tau_0$, K")
fig.tight_layout(); fig.savefig("Ex4_3_herschel_bulkley_py.png", dpi=150)
