"""Lecture 4, Section 2.12 - Differential-algebraic equations: batch reactor with a fast equilibrium.

    A <=> B   (fast, equilibrium constant K = kf/kb)      B -> C   (slow, k2)

ODE model (stiff when kb is large):
    dA/dt = -kf A + kb B,   dB/dt = kf A - kb B - k2 B,   dC/dt = k2 B
DAE model (equilibrium imposed, kb -> infinity), unknowns A, B, C:
    d(A + B)/dt = -k2 B,    0 = B - K A,    dC/dt = k2 B        (mass-matrix form, M singular)
Exact DAE solution, with consistent initial values A0 = 1/(1+K), B0 = K/(1+K), C0 = 0:
    S = A + B = exp(-k_eff t),  k_eff = k2 K/(1+K),  A = S/(1+K),  B = K S/(1+K),  C = 1 - S

SciPy solve_ivp has no DAE solver: the index-1 DAE is solved by eliminating the algebraic
unknown (A, B from S), which gives an ODE in S and C.
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 15, 'axes.linewidth': 1.0,
                     'mathtext.fontset': 'cm'})
K, k2, tf = 2.0, 1.0, 5.0
keff = k2 * K / (1 + K)

def dae_exact(t):
    S = np.exp(-keff * t)
    return np.array([S / (1 + K), K * S / (1 + K), 1 - S])

def ode_model(kb):
    kf = K * kb
    def f(t, y):
        A, B, C = y
        return [-kf * A + kb * B, kf * A - kb * B - k2 * B, k2 * B]
    def jac(t, y):
        return [[-kf, kb, 0], [kf, -kb - k2, 0], [0, k2, 0]]
    return f, jac

# ---------------------------------------------------------------- DAE by elimination (index 1)
def reduced(t, x):                       # x = (S, C): A and B follow from the constraint B = K A
    S, C = x
    B = K * S / (1 + K)
    return [-k2 * B, k2 * B]
red = solve_ivp(reduced, (0, tf), [1.0, 0.0], method='RK45', rtol=1e-10, atol=1e-12, dense_output=True)
tt = np.linspace(0, tf, 501)
S, C = red.sol(tt)
err = np.max(np.abs(np.vstack([S / (1 + K), K * S / (1 + K), C]) - dae_exact(tt)))
print(f'DAE by elimination (RK45): nfev = {red.nfev}, max error vs exact = {err:.2e}')

# ---------------------------------------------------------------- stiff ODE models, A0 = 1 (not consistent)
print('Stiff ODE model, A(0) = 1, B(0) = 0, C(0) = 0; deviation from the DAE for t >= 0.1:')
late = tt >= 0.1
sols = {}
for kb in [1e1, 1e2, 1e3, 1e4]:
    f, jac = ode_model(kb)
    rows = []
    for m in ['RK45', 'BDF']:
        kw = dict(jac=jac) if m == 'BDF' else {}
        s = solve_ivp(f, (0, tf), [1.0, 0.0, 0.0], method=m, rtol=1e-8, atol=1e-10, dense_output=True, **kw)
        dev = np.max(np.abs(s.sol(tt)[:, late] - dae_exact(tt)[:, late]))
        rows.append(f'{m}: nfev = {s.nfev:7d}')
        if m == 'BDF':
            sols[kb] = s; devb = dev
    print(f'  kb = {kb:7.0e}: max deviation = {devb:.2e}   ' + '   '.join(rows))

# ---------------------------------------------------------------- figure: ODE (kb = 100) vs DAE, log time axis
tl = np.logspace(-4, np.log10(tf), 600)
fig, ax = plt.subplots(figsize=(8.0, 4.4))
cols = ['#D35400', '#27864A', '#1F3A93']
ex = dae_exact(tl); od = sols[1e2].sol(tl)
for i, nm in enumerate(['A', 'B', 'C']):
    ax.semilogx(tl, od[i], color=cols[i], lw=2.2, label=f'{nm}: stiff ODE ($k_b=100$)')
    ax.semilogx(tl, ex[i], color=cols[i], lw=1.4, ls='--', label=f'{nm}: DAE (equilibrium)')
ax.axvspan(1e-4, 3e-2, color='0.92', zorder=0)
ax.text(2.2e-3, 0.10, 'fast transient\n(ODE only)', fontsize=11.5, color='0.3')
ax.set_xlabel('$t$ (log scale)'); ax.set_ylabel('concentration')
ax.set_xlim(1e-4, tf); ax.set_ylim(-0.02, 1.02)
ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=11)
for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig('fig_2_12_dae.png', dpi=220, bbox_inches='tight', pad_inches=0.08)
print('consistent initial values: A0 = %.4f, B0 = %.4f' % (1 / (1 + K), K / (1 + K)))
