"""Lecture 4, Example 3 - A non-stiff 2x2 linear system: Euler and RK4 in vector form.

    dy/dt = A y,  A = [[-2, 1], [1, -2]],  y(0) = (2, 0),  t in [0, 2]
Eigenvalues -1 and -3 (stiffness ratio 3). Exact: y1 = exp(-t) + exp(-3t), y2 = exp(-t) - exp(-3t)
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 14, 'axes.linewidth': 1.0,
                     'mathtext.fontset': 'cm'})

A = np.array([[-2.0, 1.0], [1.0, -2.0]])
f = lambda t, y: A @ y                         # vector right-hand side
y0 = np.array([2.0, 0.0]); tf = 2.0
y_exact = lambda t: np.array([np.exp(-t) + np.exp(-3*t), np.exp(-t) - np.exp(-3*t)])

def euler(h):
    N = round(tf/h); t, y = 0.0, y0.copy()
    for n in range(N):
        y = y + h*f(t, y); t += h
    return y

def rk4(h):
    N = round(tf/h); t, y = 0.0, y0.copy()
    for n in range(N):
        k1 = f(t, y)                           # all components of k1 first ...
        k2 = f(t + h/2, y + h/2*k1)            # ... then k2, and so on
        k3 = f(t + h/2, y + h/2*k2)
        k4 = f(t + h, y + h*k3)
        y = y + h/6*(k1 + 2*k2 + 2*k3 + k4); t += h
    return y

print('eigenvalues of A:', np.linalg.eigvals(A))
hs = [0.2, 0.1, 0.05, 0.025, 0.0125]
eE = [np.linalg.norm(euler(h) - y_exact(tf)) for h in hs]
eR = [np.linalg.norm(rk4(h) - y_exact(tf)) for h in hs]
print('     h     ||e|| Euler    ||e|| RK4')
for h, a, b in zip(hs, eE, eR):
    print(f'{h:6.4f} {a:12.3e} {b:12.3e}')
print('observed order: Euler', [f'{np.log2(eE[i]/eE[i+1]):.3f}' for i in range(4)])
print('                RK4  ', [f'{np.log2(eR[i]/eR[i+1]):.3f}' for i in range(4)])

# figure: components with h = 0.2
h = 0.2; N = round(tf/h); tn = np.linspace(0, tf, N + 1)
YE = [y0]; YR = [y0]
for n in range(N):
    YE.append(YE[-1] + h*f(0, YE[-1]))
    y = YR[-1]; k1 = f(0, y); k2 = f(0, y + h/2*k1); k3 = f(0, y + h/2*k2); k4 = f(0, y + h*k3)
    YR.append(y + h/6*(k1 + 2*k2 + 2*k3 + k4))
YE, YR = np.array(YE), np.array(YR)
tt = np.linspace(0, tf, 300); ex = y_exact(tt)
fig, ax = plt.subplots(figsize=(7.4, 4.2))
for i, c in enumerate(['#1F3A93', '#8E44AD']):
    ax.plot(tt, ex[i], color=c, lw=2.0, label=f'$y_{i+1}$ exact')
    ax.plot(tn, YE[:, i], 'o', color='#D35400', ms=5, label='Euler' if i == 0 else None)
    ax.plot(tn, YR[:, i], 's', mfc='none', mec='#27864A', ms=7, mew=1.5, label='RK4' if i == 0 else None)
ax.set_xlabel('$t$'); ax.set_ylabel(r'$y_1,\;y_2$'); ax.legend(frameon=False, fontsize=12)
for s in ('top', 'right'): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig('ex3_system.png', dpi=220, bbox_inches='tight', pad_inches=0.08)
