"""Lecture 4, Example 3-bis - Undamped oscillator: purely imaginary eigenvalues.

    y'' + omega^2 y = 0  ->  y1' = y2,  y2' = -omega^2 y1,   omega = 1,  y(0) = 1, y'(0) = 0
Exact: y1 = cos t, y2 = -sin t; the amplitude sqrt(y1^2 + y2^2) stays equal to 1.
Eigenvalues +/- i: Euler and Heun amplify at every step, RK4 (h < 2.83) damps very slightly.
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 14, 'axes.linewidth': 1.0,
                     'mathtext.fontset': 'cm'})

w, h, tf = 1.0, 0.1, 20.0
A = np.array([[0.0, 1.0], [-w**2, 0.0]])
f = lambda y: A @ y
N = round(tf/h)

def step_euler(y): return y + h*f(y)
def step_heun(y):
    k1 = f(y); k2 = f(y + h*k1); return y + h/2*(k1 + k2)
def step_rk4(y):
    k1 = f(y); k2 = f(y + h/2*k1); k3 = f(y + h/2*k2); k4 = f(y + h*k3)
    return y + h/6*(k1 + 2*k2 + 2*k3 + k4)

paths = {}
for name, step in [('Euler', step_euler), ('Heun', step_heun), ('RK4', step_rk4)]:
    Y = np.empty((N + 1, 2)); Y[0] = [1.0, 0.0]
    for n in range(N):
        Y[n + 1] = step(Y[n])
    paths[name] = Y
    print(f'{name:5s}: amplitude at t = {tf:g}: {np.hypot(*Y[-1]):.6f}')
z = 1j*w*h
R = [1 + z, 1 + z + z**2/2, 1 + z + z**2/2 + z**3/6 + z**4/24]
print('|R(i h w)|: Euler %.8f  Heun %.8f  RK4 %.8f' % tuple(abs(r) for r in R))

t = np.linspace(0, tf, N + 1)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.0, 4.4), gridspec_kw=dict(width_ratios=[1, 1.3]))
cols = {'Euler': '#D35400', 'Heun': '#27864A', 'RK4': '#1F3A93'}
th = np.linspace(0, 2*np.pi, 300)
a1.plot(np.cos(th), -np.sin(th), 'k:', lw=1.4, label='exact')
for name, Y in paths.items():
    a1.plot(Y[:, 0], Y[:, 1], color=cols[name], lw=1.2, label=name)
a1.set_aspect('equal'); a1.set_xlabel('$y_1=y$'); a1.set_ylabel("$y_2=y'$")
a1.set_title('phase plane, $t\\in[0,20]$', fontsize=13)
for name, Y in paths.items():
    a2.plot(t, np.hypot(Y[:, 0], Y[:, 1]), color=cols[name], lw=2.0, label=name)
a2.axhline(1, color='k', ls=':', lw=1.2)
a2.set_xlabel('$t$'); a2.set_ylabel('amplitude $\\sqrt{y_1^2+y_2^2}$')
a2.set_title(f'$h={h}$, exact amplitude = 1', fontsize=13)
a2.legend(frameon=False, fontsize=12, loc='upper left')
for a in (a1, a2):
    for s in ('top', 'right'): a.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig('ex3bis_oscillator.png', dpi=220, bbox_inches='tight', pad_inches=0.08)
