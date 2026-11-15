"""Lecture 4, Example 1 - Convergence: Euler, Heun and RK4 written from scratch.

Test problem (Burden & Faires):  dy/dt = y - t^2 + 1,  y(0) = 0.5,  t in [0, 2]
Exact solution:                  y(t) = (t + 1)^2 - 0.5 exp(t)
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 14, 'axes.linewidth': 1.0,
                     'mathtext.fontset': 'cm'})

f = lambda t, y: y - t**2 + 1                  # right-hand side dy/dt = f(t, y)
y_exact = lambda t: (t + 1)**2 - 0.5*np.exp(t)
t0, tf, y0 = 0.0, 2.0, 0.5

def euler(f, t0, y0, h, N):
    t, y = t0, y0
    for n in range(N):
        y = y + h*f(t, y)
        t = t + h
    return y

def heun(f, t0, y0, h, N):
    t, y = t0, y0
    for n in range(N):
        k1 = f(t, y)
        k2 = f(t + h, y + h*k1)
        y = y + h/2*(k1 + k2)
        t = t + h
    return y

def rk4(f, t0, y0, h, N):
    t, y = t0, y0
    for n in range(N):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2*k1)
        k3 = f(t + h/2, y + h/2*k2)
        k4 = f(t + h, y + h*k3)
        y = y + h/6*(k1 + 2*k2 + 2*k3 + k4)
        t = t + h
    return y

methods = {'Euler': euler, 'Heun': heun, 'RK4': rk4}
hs = [0.2, 0.1, 0.05, 0.025, 0.0125]
err = {m: [] for m in methods}
for h in hs:
    N = round((tf - t0)/h)                     # number of steps
    for m, method in methods.items():
        err[m].append(abs(method(f, t0, y0, h, N) - y_exact(tf)))

print('global error at t = 2')
print('     h       Euler        Heun         RK4')
for i, h in enumerate(hs):
    print(f'{h:6.4f}' + ''.join(f'{err[m][i]:12.3e}' for m in methods))
print('observed order p = log2(e_h / e_h/2)')
for i in range(len(hs) - 1):
    print(f'{hs[i]:6.4f}->{hs[i+1]:6.4f}' + ''.join(
        f'{np.log2(err[m][i]/err[m][i+1]):8.3f}' for m in methods))

# figure: global error vs h (log-log)
fig, ax = plt.subplots(figsize=(7.0, 4.3))
for m, c, mk in zip(methods, ['#D35400', '#27864A', '#1F3A93'], ['o', '^', 's']):
    ax.loglog(hs, err[m], '-' + mk, color=c, lw=1.8, ms=7, label=m)
for p, c0 in [(1, 2.2), (2, 1.9), (4, 0.07)]:
    hh = np.array([hs[-1], hs[0]])
    ax.loglog(hh, c0*hh**p*0.45, 'k--', lw=0.8)
    ax.text(hs[-1]*0.82, c0*hs[-1]**p*0.45, f'slope {p}', fontsize=11, ha='right', va='center')
ax.set_xlabel('step size $h$'); ax.set_ylabel('global error at $t=2$')
ax.set_xlim(0.006, 0.3)
ax.legend(loc='lower right', frameon=False)
for s in ('top', 'right'): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig('ex1_order.png', dpi=220, bbox_inches='tight', pad_inches=0.08)
