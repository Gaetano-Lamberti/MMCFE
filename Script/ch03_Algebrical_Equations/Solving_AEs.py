"""
Module 1, Part 1: Implicit AE, Newton-Raphson and variants
Generic example, independent of the reactor case study, illustrating
step-by-step the base method and its variants on a scalar implicit
equation. Includes convergence plots and "cobweb" visualization.

BASE CASE (clean convergence):
    f1(x) = x^3 - 2x - 5 = 0      (single real root, ~2.0946)

CRITICAL CASE (why the robust variants are needed):
    f2(x) = x^3 - 2x + 2 = 0      (single real root, ~-1.7693)
    Pure Newton-Raphson from x0=0 enters an EXACT cycle: 0 -> 1 -> 0 -> ...
    Classic textbook case, not an implementation bug.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend: figures are saved to file, not shown
import matplotlib.pyplot as plt

# ============================================================
#  BASE CASE: f1(x) = x^3 - 2x - 5
# ============================================================
f1 = lambda x: x**3 - 2*x - 5
f1p = lambda x: 3*x**2 - 2

tol = 1e-10
maxiter = 50

# --- Newton-Raphson, x0 = 3 ---
print('=== BASE CASE: Newton-Raphson, x0=3 ===')
print('it        x                f(x)')
x = 3.0
hist_NR_x = [x]
hist_NR_f = [f1(x)]
for it in range(maxiter + 1):
    fx = f1(x)
    print(f'{it:2d}   {x:.10f}   {fx:.3e}')
    if abs(fx) < tol:
        break
    x = x - fx / f1p(x)
    hist_NR_x.append(x)
    hist_NR_f.append(f1(x))
print(f'Root: x = {x:.10f}\n')

# --- Secant method, x0=2, x1=3 (no analytical derivative) ---
print('=== Secant method, x0=2, x1=3 ===')
print('it        x                f(x)')
x0, x1 = 2.0, 3.0
f0, fval1 = f1(x0), f1(x1)
hist_SEC_x = [x0, x1]
hist_SEC_f = [f0, fval1]
print(f'{0:2d}   {x0:.10f}   {f0:.3e}')
print(f'{1:2d}   {x1:.10f}   {fval1:.3e}')
x2 = x1
for it in range(2, maxiter + 1):
    x2 = x1 - fval1 * (x1 - x0) / (fval1 - f0)
    f2 = f1(x2)
    print(f'{it:2d}   {x2:.10f}   {f2:.3e}')
    hist_SEC_x.append(x2)
    hist_SEC_f.append(f2)
    if abs(f2) < tol:
        break
    x0, f0 = x1, fval1
    x1, fval1 = x2, f2
print(f'Root: x = {x2:.10f}  (superlinear convergence, ~1.618: slower than Newton)\n')

# --- Modified Newton, x0=2 (Jacobian fixed, never recompQuted) ---
print("=== Modified Newton, x0=2 (f'(x0) fixed) ===")
print('it        x                f(x)')
x = 2.0
fp_fixed = f1p(x)  # computed only once
hist_MOD_x = [x]
hist_MOD_f = [f1(x)]
it = 0
for it in range(maxiter + 1):
    fx = f1(x)
    print(f'{it:2d}   {x:.10f}   {fx:.3e}')
    if abs(fx) < tol:
        break
    x = x - fx / fp_fixed  # same denominator every iteration
    hist_MOD_x.append(x)
    hist_MOD_f.append(f1(x))
print(f'Root: x = {x:.10f}  (linear convergence: {it} iterations instead of 4)\n')

# ============================================================
#  CRITICAL CASE: f2(x) = x^3 - 2x + 2  (real root ~ -1.7693)
# ============================================================
f2fun = lambda x: x**3 - 2*x + 2
f2p = lambda x: 3*x**2 - 2

# --- Pure Newton-Raphson, x0=0: EXACT CYCLE ---
print('=== CRITICAL CASE: pure Newton-Raphson, x0=0 ===')
print('it        x                f(x)')
x = 0.0
hist_CYC_x = [x]
hist_CYC_f = [f2fun(x)]
for it in range(8):
    fx = f2fun(x)
    print(f'{it:2d}   {x:.6f}   {fx:.6f}')
    x = x - fx / f2p(x)
    hist_CYC_x.append(x)
    hist_CYC_f.append(f2fun(x))
print('--> Never converges: exact cycle 0 -> 1 -> 0 -> 1 -> ...')
print('    (provable analytically, not an implementation error)\n')

# --- "Naive" damped Newton (backtracking on |f| only) ---
# Breaks the cycle, but can get STUCK at a spurious stationary point
# (where f'(x)=0 but f(x)!=0): illustrates the LIMIT of naive damping,
# not just its benefit.
print('=== "Naive" damped Newton (backtracking on |f| only), x0=0 ===')
print('it        x                f(x)')
x = 0.0
hist_DAMP_x = [x]
hist_DAMP_f = [f2fun(x)]
for it in range(15):
    fx = f2fun(x)
    print(f'{it:2d}   {x:.8f}   {fx:.6e}')
    if abs(fx) < 1e-10:
        break
    fpx = f2p(x)
    dx = -fx / fpx
    lam = 1.0
    while abs(f2fun(x + lam * dx)) >= abs(fx) and lam > 1e-8:
        lam = lam / 2
    x = x + lam * dx
    hist_DAMP_x.append(x)
    hist_DAMP_f.append(f2fun(x))
print('--> Gets stuck near x=0.8165 (critical point, f\'(x)=0), NOT a root')
print('    (f is about 0.911 there, not zero): limit of naive damping.\n')

# --- Hybrid Newton-bisection (safeguarded): converges correctly ---
# Maintains a bracket [a,b] with f(a)*f(b)<0. At each iteration it tries
# the Newton step; if the result falls outside the bracket, it uses a
# bisection step instead. ALWAYS guarantees progress toward a true root
# (bisection), with Newton's acceleration whenever possible.
print('=== Hybrid Newton-bisection (safeguarded), initial bracket [-2, 0] ===')
print('it        x                f(x)             bracket')
a, b = -2.0, 0.0
fa, fb = f2fun(a), f2fun(b)
if fa * fb >= 0:
    raise ValueError('Invalid bracket: f(a) and f(b) must have opposite signs.')
x = 0.0  # same "bad" guess as before
hist_HYB_x = [x]
hist_HYB_f = [f2fun(x)]
for it in range(15):
    fx = f2fun(x)
    print(f'{it:2d}   {x:.10f}   {fx:.6e}   [{a:.6f}, {b:.6f}]')
    if abs(fx) < 1e-12:
        break
    # update the bracket
    if fa * fx < 0:
        b, fb = x, fx
    else:
        a, fa = x, fx
    # try the Newton step; accept it only if it stays inside the bracket
    fpx = f2p(x)
    accepted = False
    if fpx != 0:
        x_newton = x - fx / fpx
        if a < x_newton < b:
            x = x_newton
            accepted = True
    if not accepted:
        x = 0.5 * (a + b)  # fallback: bisection step
    hist_HYB_x.append(x)
    hist_HYB_f.append(f2fun(x))
print(f'--> Converges correctly to the true root: x = {x:.10f}')
print('    This is exactly the role of bisection as a "safety net" (S16.2):')
print('    it does not replace Newton, it guarantees progress when Newton fails.\n')

# ============================================================
#  PLOTS
# ============================================================

# --- Figure 1: base case - cobweb + convergence ---
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig1.canvas.manager.set_window_title('Base case: f1(x) = x^3 - 2x - 5')

xg = np.linspace(1.5, 3.2, 300)
ax1.plot(xg, f1(xg), 'b-', linewidth=1.8, label='f_1(x)')
ax1.plot(xg, np.zeros_like(xg), 'k-', label='f_1(x)=0')
ax1.plot(2.0945514815, 0, marker='*', markersize=14, markerfacecolor='y',
          markeredgecolor='k', linestyle='None', label='root')
ax1.plot(hist_NR_x, hist_NR_f, '--o', color='r', linewidth=1.2,
          markerfacecolor='r', label='Newton-Raphson')
ax1.plot(hist_SEC_x, hist_SEC_f, '--s', color='g', linewidth=1.2,
          markerfacecolor='g', label='Secant')
ax1.plot(hist_MOD_x, hist_MOD_f, '--^', color='m', linewidth=1.0,
          markerfacecolor='m', markersize=4, label='Modified Newton')
ax1.set_xlabel('x')
ax1.set_ylabel('f_1(x)')
ax1.set_title('f_1(x)=x^3-2x-5: iteration path')
ax1.legend(loc='lower right')
ax1.grid(True)

ax2.semilogy(range(len(hist_NR_f)), np.abs(hist_NR_f), '-o', color='r',
             linewidth=1.5, markerfacecolor='r',
             label=f'Newton-Raphson ({len(hist_NR_f)-1} it.)')
ax2.semilogy(range(len(hist_SEC_f)), np.abs(hist_SEC_f), '-s', color='g',
             linewidth=1.5, markerfacecolor='g',
             label=f'Secant ({len(hist_SEC_f)-1} it.)')
ax2.semilogy(range(len(hist_MOD_f)), np.abs(hist_MOD_f), '-^', color='m',
             linewidth=1.5, markerfacecolor='m',
             label=f'Modified Newton ({len(hist_MOD_f)-1} it.)')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('|f_1(x)|  (log scale)')
ax2.set_title('Convergence comparison')
ax2.legend(loc='upper right')
ax2.grid(True)

fig1.tight_layout()
fig1.savefig('fig1_base_case_py.png', dpi=150)

# --- Figure 2: critical case - cycle, naive damping, hybrid ---
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 5))
fig2.canvas.manager.set_window_title('Critical case: f2(x) = x^3 - 2x + 2')

xg2 = np.linspace(-2.2, 1.5, 300)
ax3.plot(xg2, f2fun(xg2), 'b-', linewidth=1.8, label='f_2(x)')
ax3.plot(xg2, np.zeros_like(xg2), 'k-', label='f_2(x)=0')
ax3.plot(-1.7692923542, 0, marker='*', markersize=14, markerfacecolor='y',
          markeredgecolor='k', linestyle='None', label='true root')
ax3.plot(hist_CYC_x, hist_CYC_f, '--x', color='r', linewidth=1.5,
          markersize=10, label='pure NR (cycle)')
ax3.plot(hist_DAMP_x, hist_DAMP_f, '--^', color='c', linewidth=1.2,
          markerfacecolor='c', markersize=5, label='naive damping (stuck)')
ax3.plot(hist_HYB_x, hist_HYB_f, '--o', color=(0, 0.6, 0), linewidth=1.2,
          markerfacecolor=(0, 0.6, 0), label='hybrid Newton-bisection (converges)')
ax3.set_xlabel('x')
ax3.set_ylabel('f_2(x)')
ax3.set_title('f_2(x)=x^3-2x+2: pure Newton cycles, "naive" gets stuck, hybrid converges')
ax3.legend(loc='lower right')
ax3.grid(True)

ax4.plot(range(len(hist_CYC_x)), hist_CYC_x, '-x', color='r', linewidth=1.5,
         markersize=8, label='pure NR (0/1 cycle)')
ax4.plot(range(len(hist_DAMP_x)), hist_DAMP_x, '-^', color='c', linewidth=1.5,
         markerfacecolor='c', label='naive damping (stalls ~0.82)')
ax4.plot(range(len(hist_HYB_x)), hist_HYB_x, '-o', color=(0, 0.6, 0), linewidth=1.5,
         markerfacecolor=(0, 0.6, 0), label='hybrid (converges to -1.769)')
yline_val = -1.7692923542
ax4.plot(ax4.get_xlim(), [yline_val, yline_val], 'k:', linewidth=1, label='true root')
ax4.set_xlabel('Iteration')
ax4.set_ylabel('x_n')
ax4.set_title('Trajectory of x_n: cycle vs stagnation vs convergence')
ax4.legend(loc='center right')
ax4.grid(True)

fig2.tight_layout()
fig2.savefig('fig2_critical_case_py.png', dpi=150)

print('Figures saved: fig1_base_case_py.png, fig2_critical_case_py.png')
