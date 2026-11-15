"""Ex4_1_Thermistor.py
Lecture 6 - Parameter estimation, Section 4.14.1-4.14.2, Example 4.1: NTC thermistor

Calibration of an NTC thermistor with the Steinhart-Hart equation (Eq. 50), R in kOhm, T in K:
    1/T = A + B*ln(R) + C*(ln R)^3
Two objectives:
  (1) error on 1/T -> linear in (A, B, C): closed form, linear least squares (Section 4.8);
  (2) error on T   -> nonlinear: Levenberg-Marquardt (Section 4.9.3), from the closed form and from
                      200 random starts. Derivative-free and global methods on this problem: Lecture 7.
Five points, three parameters (nu = 2): a calibration, not a statistical estimation.
Same numbers as the slides (numpy generator, seed 0). Requires numpy and matplotlib.
"""
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- data
TC = np.array([0.00, 10.00, 20.00, 30.16, 40.12])          # degC
T = TC + 273.15                                            # K
R = np.array([32.65, 19.90, 12.50, 8.00, 5.30])            # kOhm
L = np.log(R)
NEV = [0]                                                  # counter of model evaluations


def res_T(p):
    """Residuals on T; where the denominator vanishes or the result is not finite, a large penalty."""
    NEV[0] += 1
    with np.errstate(all="ignore"):
        r = T - 1 / (p[0] + p[1] * L + p[2] * L ** 3)
    return np.where(np.isfinite(r), r, 1e6)


def lm_fd(res, p0, lam=1e-3, tol=1e-10, maxit=300, h_rel=1e-6):
    """Levenberg-Marquardt with Marquardt diagonal scaling (Section 4.9.3), forward-difference Jacobian."""
    p = np.asarray(p0, float).copy(); r = res(p); f = r @ r
    for _ in range(maxit):
        J = np.empty((r.size, p.size))
        for j in range(p.size):
            dp = p.copy(); h = h_rel * max(abs(p[j]), 1e-12); dp[j] += h
            J[:, j] = (res(dp) - r) / h
        A, g = J.T @ J, J.T @ r
        while True:
            try:
                d = np.linalg.solve(A + lam * np.diag(np.diag(A)), -g)
            except np.linalg.LinAlgError:
                return p, False                            # singular system: failure
            rn = res(p + d); fn = rn @ rn
            if fn < f:
                lam /= 10; break                           # accept
            lam *= 10                                      # reject
            if lam > 1e14:
                return p, False
        p, r = p + d, rn
        if (f - fn) < tol * fn:
            return p, True
        f = fn
    return p, False


# ---------------------------------------------------------------- (1) closed form: least squares on 1/T
X = np.column_stack([np.ones_like(L), L, L ** 3])
p_lin = np.linalg.lstsq(X, 1 / T, rcond=None)[0]           # QR, Section 4.8
print("4.1 closed form (1/T): A=%.4e B=%.4e C=%.4e  max |T err| = %.3f K"
      % (*p_lin, np.max(np.abs(1 / (X @ p_lin) - T))))
print("    R=10 kOhm -> T = %.2f C" % (1 / (p_lin @ [1, np.log(10), np.log(10) ** 3]) - 273.15))
print("    cond(X'X) = %.1e,  corr(lnR, ln^3R) = %.3f" % (np.linalg.cond(X.T @ X), np.corrcoef(L, L ** 3)[0, 1]))

# ---------------------------------------------------------------- (2) error on T: LM from the closed form
NEV[0] = 0
p_ref, ok = lm_fd(res_T, p_lin)
print("4.1 LM from the closed form: evals %d, A=%.4e B=%.4e C=%.4e" % (NEV[0], *p_ref))

# random starts: |p_j| log-uniform in [1e-5, 1], random signs
rng = np.random.default_rng(0)
NST, succ, ev = 200, 0, []
for _ in range(NST):
    p0 = rng.choice([-1, 1], 3) * 10 ** rng.uniform(-5, 0, 3)
    NEV[0] = 0
    p, _ = lm_fd(res_T, p0)
    ev.append(NEV[0]); succ += np.allclose(p, p_ref, rtol=1e-2)
print("4.1 random starts (|p_j| 1e-5..1, random signs): LM success %3d/%d  median evals %d"
      % (succ, NST, np.median(ev)))
print("4.1 T residuals at the LM solution (mK):", np.round(1000 * res_T(p_ref), 2))

# ---------------------------------------------------------------- figure: residuals of the two objectives
fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
RR = np.logspace(np.log10(5), np.log10(33), 200); LL = np.log(RR)
axs[0].plot(R, TC, "ko", mfc="w", label="data")
axs[0].plot(RR, 1 / (np.column_stack([np.ones_like(LL), LL, LL ** 3]) @ p_ref) - 273.15, color="#233F98", lw=1.5,
            label="Steinhart-Hart, fit in T")
axs[0].set_xlabel(r"R [k$\Omega$]"); axs[0].set_ylabel(r"T [$^\circ$C]"); axs[0].legend(frameon=False)
axs[1].plot(TC, 1000 * (T - 1 / (X @ p_lin)), "s-", color="#D35400", label="fit in 1/T")
axs[1].plot(TC, 1000 * res_T(p_ref), "o-", color="#233F98", label="fit in T")
axs[1].axhline(0, color="k", lw=0.8)
axs[1].set_xlabel(r"T [$^\circ$C]"); axs[1].set_ylabel("T residual [mK]"); axs[1].legend(frameon=False)
fig.tight_layout(); fig.savefig("Ex4_1_thermistor_py.png", dpi=150)
