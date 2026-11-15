"""Lecture 4, Section 2.11 - Boundary value problem: steady conduction in a wall
with temperature-dependent conductivity and internal heat generation (dimensionless).

    d/dxi [ (1 + beta*theta) dtheta/dxi ] + G = 0,   theta(0) = 1,  theta(1) = 0
    beta = 1, G = 4

Exact solution (Kirchhoff transform u = theta + beta*theta^2/2 gives u'' = -G):
    u(xi) = -G xi^2/2 + (G/2 - u0) xi + u0,   u0 = 1 + beta/2
    theta = (-1 + sqrt(1 + 2 beta u)) / beta

Three numerical approaches: shooting (RK45 + secant), finite differences (Newton), SciPy solve_bvp.
"""
import numpy as np
from scipy.integrate import solve_ivp, solve_bvp
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 15, 'axes.linewidth': 1.0,
                     'mathtext.fontset': 'cm'})

beta, G = 1.0, 4.0
u0 = 1 + beta / 2

def theta_exact(xi):
    u = -G * xi**2 / 2 + (G / 2 - u0) * xi + u0
    return (-1 + np.sqrt(1 + 2 * beta * u)) / beta

# first-order system: y1 = theta, y2 = (1 + beta*theta) dtheta/dxi  (dimensionless heat flux, sign changed)
def rhs(xi, y):
    return [y[1] / (1 + beta * y[0]), -G]

# ---------------------------------------------------------------- 1) shooting
def theta_at_1(s):                     # s = dtheta/dxi at xi = 0 (unknown initial slope)
    sol = solve_ivp(rhs, (0, 1), [1.0, (1 + beta) * s], rtol=1e-10, atol=1e-12)
    return sol.y[0, -1]

F = lambda s: theta_at_1(s) - 0.0      # residual of the right boundary condition
s0, s1 = 0.0, 1.0                      # two starting guesses for the secant method
print('Shooting (secant on s = dtheta/dxi at 0):')
print(f'  k={0}: s = {s0: .10f}  F = {F(s0): .3e}')
print(f'  k={1}: s = {s1: .10f}  F = {F(s1): .3e}')
shots = [s0, s1]
for k in range(2, 12):
    F0, F1 = F(s0), F(s1)
    s2 = s1 - F1 * (s1 - s0) / (F1 - F0)
    s0, s1 = s1, s2
    shots.append(s2)
    print(f'  k={k}: s = {s2: .10f}  F = {F(s2): .3e}')
    if abs(F(s2)) < 1e-12:
        break
s_exact = (G / 2 - u0) / (1 + beta)    # u'(0) = (1 + beta*theta(0)) theta'(0)
print(f'  exact s = {s_exact:.10f}')

# ---------------------------------------------------------------- 2) finite differences + Newton
# expanded form: (1 + beta*theta) theta'' + beta theta'^2 + G = 0, central differences (order 2)
def fd_solve(N):
    xi = np.linspace(0, 1, N + 1); d = 1.0 / N
    th = 1 - xi                                         # initial guess: linear profile (constant-k solution)
    for it in range(30):
        c = 1 + beta * th[1:-1]
        lap = (th[2:] - 2 * th[1:-1] + th[:-2]) / d**2  # second derivative
        grd = (th[2:] - th[:-2]) / (2 * d)              # first derivative
        r = c * lap + beta * grd**2 + G                 # residuals at the interior nodes
        n = N - 1
        J = (np.diag(beta * lap - 2 * c / d**2)         # tridiagonal Jacobian dr/dtheta
             + np.diag((c / d**2 + beta * grd / d)[:-1], 1)
             + np.diag((c / d**2 - beta * grd / d)[1:], -1))
        dth = np.linalg.solve(J, -r)
        th[1:-1] += dth
        if np.max(np.abs(dth)) < 1e-12:
            break
    return xi, th, it + 1

print('Finite differences (central, expanded form) + Newton:')
errs = []
for N in [5, 10, 20, 40, 80]:
    xi, th, its = fd_solve(N)
    e = np.max(np.abs(th - theta_exact(xi)))
    errs.append(e)
    print(f'  N = {N:3d}: Newton iterations = {its}, max error = {e:.3e}')
print('  observed order:', [f'{np.log2(errs[i] / errs[i + 1]):.2f}' for i in range(len(errs) - 1)])

# ---------------------------------------------------------------- 3) SciPy solve_bvp (collocation)
def rhs_vec(xi, y):                    # vectorized right-hand side, as solve_bvp requires
    return np.vstack([y[1] / (1 + beta * y[0]), -G * np.ones_like(y[0])])

def bc(ya, yb):                        # residuals of the boundary conditions
    return [ya[0] - 1.0, yb[0] - 0.0]

xg = np.linspace(0, 1, 5)
yg = np.vstack([1 - xg, np.zeros_like(xg)])            # initial guess on a coarse mesh
res = solve_bvp(rhs_vec, bc, xg, yg, tol=1e-8)
xx = np.linspace(0, 1, 201)
print(f'solve_bvp: status = {res.status}, mesh nodes = {len(res.x)}, '
      f'max error = {np.max(np.abs(res.sol(xx)[0] - theta_exact(xx))):.3e}')

# ---------------------------------------------------------------- figure: shooting trajectories
fig, ax = plt.subplots(figsize=(7.4, 4.3))
greys = ['#C8CDD6', '#A9B1BE', '#8A94A6', '#6B778D', '#D35400']
for k, s_k in enumerate(shots[:5]):
    sol = solve_ivp(rhs, (0, 1), [1.0, (1 + beta) * s_k], dense_output=True, rtol=1e-10, atol=1e-12)
    xe = xx[xx <= sol.t[-1]]
    ax.plot(xe, sol.sol(xe)[0], color=greys[k], lw=1.6, label=f'shot {k}: $s={s_k:.3f}$')
sol = solve_ivp(rhs, (0, 1), [1.0, (1 + beta) * shots[-1]], dense_output=True, rtol=1e-10, atol=1e-12)
ax.plot(xx, sol.sol(xx)[0], color='#1F3A93', lw=2.6, label=f'converged: $s={shots[-1]:.4f}$')
ax.plot(xx, theta_exact(xx), 'k:', lw=1.5, label='exact')
ax.plot([0, 1], [1, 0], 'ks', ms=7)
ax.annotate('target $\\theta(1)=0$', xy=(1, 0), xytext=(0.66, -0.55), arrowprops=dict(arrowstyle='->'), fontsize=12)
ax.set_xlabel(r'$\xi$'); ax.set_ylabel(r'$\theta$')
ax.legend(loc='lower left', frameon=False, fontsize=11)
for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig('fig_2_11_shooting.png', dpi=220, bbox_inches='tight', pad_inches=0.08)
