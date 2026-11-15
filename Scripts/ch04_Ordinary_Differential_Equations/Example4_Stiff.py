"""Lecture 4, Example 4 - Stiffness: explicit vs implicit methods, then built-in solvers.

    dy/dt = A y,  A = diag(-1, -1000),  y(0) = (1, 1),  t in [0, 1]
Exact: y1 = exp(-t), y2 = exp(-1000 t); stiffness ratio 1000.
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 14, 'axes.linewidth': 1.0,
                     'mathtext.fontset': 'cm'})

A = np.diag([-1.0, -1000.0]); I = np.eye(2)
y0 = np.array([1.0, 1.0]); tf = 1.0
y_exact = lambda t: np.array([np.exp(-t), np.exp(-1000*t)])

def run(method, h):
    N = round(tf/h); Y = np.empty((N + 1, 2)); Y[0] = y0
    if method == 'implicit Euler':           # (I - hA) y_(n+1) = y_n
        Mi = np.linalg.inv(I - h*A)
    elif method == 'Crank-Nicolson':         # (I - h/2 A) y_(n+1) = (I + h/2 A) y_n
        Mi = np.linalg.solve(I - h/2*A, I + h/2*A)
    for n in range(N):
        if method == 'explicit Euler':
            Y[n + 1] = Y[n] + h*(A @ Y[n])
        else:
            Y[n + 1] = Mi @ Y[n]
    return np.linspace(0, tf, N + 1), Y

t, Y = run('explicit Euler', 0.1)
print('explicit Euler, h = 0.1 (limit 2/1000 = 0.002):')
for n in range(4):
    print(f'  t = {t[n]:.1f}   y1 = {Y[n, 0]:.6f}   y2 = {Y[n, 1]:.6f}')
t, Y = run('explicit Euler', 0.0015)
print(f'explicit Euler, h = 0.0015: {len(t) - 1} steps, '
      f'error y1(1) = {abs(Y[-1, 0] - np.exp(-1)):.3e}')
for m in ['implicit Euler', 'Crank-Nicolson']:
    t, Y = run(m, 0.1)
    print(f'{m:15s} h = 0.1 (10 steps): error y1(1) = {abs(Y[-1, 0] - np.exp(-1)):.3e}')
    print(f'{"":15s} y2 after 1, 2, 3 steps = {Y[1, 1]:.4f}, {Y[2, 1]:.4f}, {Y[3, 1]:.4f}')
print('fast-mode amplification, h = 0.1: IE %.4f, CN %.4f' % (1/(1 + 100), (1 - 50)/(1 + 50)))

for m in ['RK45', 'BDF']:
    s = solve_ivp(lambda t, y: A @ y, (0, tf), y0, method=m, rtol=1e-6, atol=1e-9)
    print(f'solve_ivp {m:4s}: steps = {len(s.t) - 1:4d}, f-evals = {s.nfev:5d}, '
          f'error y1(1) = {abs(s.y[0, -1] - np.exp(-1)):.2e}')

# figure: fast component y2 with implicit Euler and Crank-Nicolson (h = 0.1)
fig, ax = plt.subplots(figsize=(7.4, 4.0))
tt = np.linspace(0, tf, 2000)
ax.plot(tt, np.exp(-1000*tt), color='k', lw=1.5, ls=':', label='exact $y_2=e^{-1000t}$')
for m, c, mk in [('implicit Euler', '#27864A', 's'), ('Crank-Nicolson', '#8E44AD', 'o')]:
    t, Y = run(m, 0.1)
    ax.plot(t, Y[:, 1], '-' + mk, color=c, lw=1.4, ms=6, label=m)
ax.axhline(0, color='k', lw=0.6)
ax.set_xlabel('$t$'); ax.set_ylabel('$y_2$ (fast component)'); ax.set_title('$h=0.1$', fontsize=13)
ax.legend(frameon=False, fontsize=12, loc='upper left', bbox_to_anchor=(1.0, 1.0))
for s in ('top', 'right'): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig('ex4_stiff.png', dpi=220, bbox_inches='tight', pad_inches=0.08)
