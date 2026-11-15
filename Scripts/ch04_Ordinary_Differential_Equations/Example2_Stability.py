"""Lecture 4, Example 2 - Absolute stability of explicit Euler (and RK4).

Test equation:  dy/dt = lam*y,  lam = -10,  y(0) = 1,  t in [0, 2];  exact y = exp(lam*t)
Euler is stable for 0 < h < 2/|lam| = 0.2; RK4 for 0 < h < 2.785/|lam| = 0.2785.
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 14, 'axes.linewidth': 1.0,
                     'mathtext.fontset': 'cm'})

lam, y0, tf = -10.0, 1.0, 2.0

def euler_path(h):
    N = round(tf/h); y = np.empty(N + 1); y[0] = y0
    for n in range(N):
        y[n + 1] = y[n] + h*lam*y[n]          # y_(n+1) = (1 + h*lam) y_n
    return np.linspace(0, tf, N + 1), y

def rk4_path(h):
    N = round(tf/h); y = np.empty(N + 1); y[0] = y0
    z = h*lam
    R = 1 + z + z**2/2 + z**3/6 + z**4/24      # stability function of RK4
    for n in range(N):
        y[n + 1] = R*y[n]
    return np.linspace(0, tf, N + 1), y

print('   h     z = h*lam   R_Euler   |y_N| Euler   R_RK4    |y_N| RK4')
for h in [0.05, 0.15, 0.2, 0.25]:
    z = h*lam
    tE, yE = euler_path(h); tR, yR = rk4_path(h)
    RR = 1 + z + z**2/2 + z**3/6 + z**4/24
    print(f'{h:5.2f} {z:9.2f} {1 + z:10.3f} {abs(yE[-1]):12.3e}'
          f' {RR:9.4f} {abs(yR[-1]):12.3e}')

fig, axs = plt.subplots(2, 2, figsize=(9.0, 5.4), sharex=True)
tt = np.linspace(0, tf, 400)
for ax, h in zip(axs.flat, [0.05, 0.15, 0.2, 0.25]):
    t, y = euler_path(h)
    ax.plot(tt, np.exp(lam*tt), color='#1F3A93', lw=2.0, label='exact')
    ax.plot(t, y, 'o-', color='#D35400', ms=4, lw=1.2, label='Euler')
    if h == 0.25:
        tR, yR = rk4_path(h)
        ax.plot(tR, yR, 's--', color='#27864A', ms=4, lw=1.0, label='RK4')
        ax.set_ylim(-12, 12)
    else:
        ax.set_ylim(-1.2, 1.2)
    ax.axhline(0, color='k', lw=0.6)
    ax.set_title(f'$h={h}$,  $z=h\\lambda={h*lam:.1f}$', fontsize=13)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
axs[0, 0].legend(loc='upper right', frameon=False, fontsize=11)
axs[1, 1].legend(loc='upper left', frameon=False, fontsize=11)
for ax in axs[1]: ax.set_xlabel('$t$')
fig.tight_layout(); fig.savefig('ex2_stability.png', dpi=220, bbox_inches='tight', pad_inches=0.08)
