"""Example 3.12.2 - Thermal sterilization of a canned, conduction-heating food.
    dT/dt = alpha*[ (1/r) d/dr (r dT/dr) + d2T/dz2 ]   on 0<=r<=R, 0<=z<=Hh (half can, z from the mid-plane)
    BCs: dT/dr = 0 at r = 0 (axis), dT/dz = 0 at z = 0 (mid-plane), T = T_w(t) at r = R and z = Hh
    T_w = T_R (retort, condensing steam) for t < t_h, then T_C (cooling water).   IC: T = T0.
    Parabolic in t only: in (r, z) the operator is ELLIPTIC -> the method of lines discretizes BOTH r and z.
    Lethality at the cold point (the centre, r = 0, z = 0):  F0 = int 10^((T - 121.1)/z_D) dt   [min]
Exact solution (verification): double series (Bessel in r x cosine in z) + superposition of the BC step."""
import numpy as np, scipy.sparse as sp, json, time
from scipy.special import j0, j1, jn_zeros
from scipy.integrate import solve_ivp

R, H = 0.038, 0.100; Hh = H/2                # can radius and half height [m]
alpha = 1.5e-7                               # thermal diffusivity [m2/s]
T0, TR, TC = 40.0, 121.1, 20.0               # initial, retort and cooling temperatures [C]
t_h, t_end = 80*60.0, 110*60.0               # end of heating, end of cooling [s]
zD, Tref = 10.0, 121.1                       # z-value and reference temperature [C]

# ---------------------------------------------------------------- exact solution at the cold point
Mt = 80
beta = jn_zeros(0, Mt); cr = 2/(beta*j1(beta))                    # 1 = sum cr_m J0(beta_m r/R)
gam = (2*np.arange(Mt) + 1)*np.pi/(2*Hh); cz = 4*(-1)**np.arange(Mt)/((2*np.arange(Mt) + 1)*np.pi)   # 1 = sum cz_n cos(gam_n z)
lam = alpha*((beta[:, None]/R)**2 + gam[None, :]**2); C = cr[:, None]*cz[None, :]
def T_exact_centre(t):
    t = np.atleast_1d(t); out = np.empty_like(t)
    for k, tt in enumerate(t):
        if tt <= t_h: out[k] = TR + (T0 - TR)*np.sum(C*np.exp(-lam*tt))
        else: out[k] = TC + (T0 - TR)*np.sum(C*np.exp(-lam*tt)) + (TR - TC)*np.sum(C*np.exp(-lam*(tt - t_h)))
    return out

# ---------------------------------------------------------------- method of lines in (r, z)
def build(Nr, Nz):
    """Unknowns: nodes i = 0..Nr-1 (r_i = i dr), j = 0..Nz-1 (z_j = j dz); wall/lid nodes (i = Nr or j = Nz) carry T_w(t)."""
    dr, dz = R/Nr, Hh/Nz; r = np.arange(Nr)*dr
    # radial operator (conservative form, Eq. 19-21) with axis ghost; last column couples to the wall
    Ar = np.zeros((Nr, Nr)); br = np.zeros(Nr)
    Ar[0, 0] = -4/dr**2; Ar[0, 1] = 4/dr**2
    for i in range(1, Nr):
        rm, rp = r[i] - dr/2, r[i] + dr/2
        Ar[i, i-1] = rm/(r[i]*dr**2); Ar[i, i] = -(rm + rp)/(r[i]*dr**2)
        if i < Nr - 1: Ar[i, i+1] = rp/(r[i]*dr**2)
        else: br[i] = rp/(r[i]*dr**2)                        # times T_w
    Az = np.zeros((Nz, Nz)); bz = np.zeros(Nz)
    Az[0, 0] = -2/dz**2; Az[0, 1] = 2/dz**2                  # mid-plane ghost
    for j in range(1, Nz):
        Az[j, j-1] = 1/dz**2; Az[j, j] = -2/dz**2
        if j < Nz - 1: Az[j, j+1] = 1/dz**2
        else: bz[j] = 1/dz**2
    # unknown k = i*Nz + j
    A = alpha*(sp.kron(sp.csr_matrix(Ar), sp.eye(Nz)) + sp.kron(sp.eye(Nr), sp.csr_matrix(Az))).tocsr()
    b = alpha*(np.kron(br, np.ones(Nz)) + np.kron(np.ones(Nr), bz))
    return A, b

def run(Nr, Nz, rtol=1e-8):
    A, b = build(Nr, Nz); n = A.shape[0]
    Tw = lambda t: TR if t < t_h else TC
    def f(t, y):
        T = y[:n]
        return np.append(A @ T + b*Tw(t), 10**((T[0] - Tref)/zD)/60.0)   # last equation: dF0/dt [min/s]
    pat = sp.bmat([[(A != 0).astype(float), None], [sp.csr_matrix(([1.0], ([0], [0])), shape=(1, n)), sp.csr_matrix((1, 1))]])
    y0 = np.append(T0*np.ones(n), 0.0); ts = np.linspace(0, t_end, 661)
    s1 = solve_ivp(f, [0, t_h], y0, method='BDF', rtol=rtol, atol=1e-8, jac_sparsity=pat, t_eval=ts[ts <= t_h])
    s2 = solve_ivp(f, [t_h, t_end], s1.y[:, -1], method='BDF', rtol=rtol, atol=1e-8, jac_sparsity=pat, t_eval=ts[ts > t_h])
    t = np.concatenate([s1.t, s2.t]); Tc = np.concatenate([s1.y[0], s2.y[0]]); F = np.concatenate([s1.y[-1], s2.y[-1]])
    return t, Tc, F, n, s1.t.size + s2.t.size, len(s1.t), A

out = {}
tt = np.linspace(0, t_end, 661); Tex = T_exact_centre(tt)
Fex = np.concatenate([[0], np.cumsum(0.5*(10**((Tex[1:] - Tref)/zD) + 10**((Tex[:-1] - Tref)/zD))*np.diff(tt))/60])
# finer exact F0 (trapezoid on 0.5 s grid) for a reliable reference
tf = np.linspace(0, t_end, 13201); Tf = T_exact_centre(tf); F0_ex = np.trapezoid(10**((Tf - Tref)/zD), tf)/60
F0_heat_ex = np.trapezoid(10**((Tf[tf <= t_h] - Tref)/zD), tf[tf <= t_h])/60
print('Exact: T_centre(t_h) = %.3f C, max T_centre = %.3f C at t = %.1f min, F0 = %.4f min (heating part %.4f)'
      % (T_exact_centre(t_h)[0], Tf.max(), tf[np.argmax(Tf)]/60, F0_ex, F0_heat_ex))
out['exact'] = dict(Tc_th=float(T_exact_centre(t_h)[0]), Tmax=float(Tf.max()), t_Tmax=float(tf[np.argmax(Tf)]/60), F0=float(F0_ex), F0_heat=float(F0_heat_ex))
grid = []
for Nr, Nz in ((5, 6), (10, 13), (20, 26), (40, 52)):
    t0 = time.time(); t, Tc, F, n, _, _, A = run(Nr, Nz)
    e = np.max(np.abs(Tc[1:] - Tex[1:])); eF = abs(F[-1] - F0_ex)
    grid.append(dict(Nr=Nr, Nz=Nz, n=n, errT=float(e), F0=float(F[-1]), errF=float(eF), cpu=time.time() - t0))
    print(f'Nr={Nr:3d} Nz={Nz:3d} n={n:5d}  max|T_c - exact|={e:.3e} C   F0={F[-1]:.4f} (err {eF:.2e})  cpu {time.time()-t0:.1f}s')
    if (Nr, Nz) == (20, 26): np.savez('ex312b.npz', t=t, Tc=Tc, F=F, Tex=Tex, tt=tt, Fex=Fex)
out['grid'] = grid
out['orders_T'] = [float(np.log2(grid[k-1]['errT']/grid[k]['errT'])) for k in range(1, len(grid))]
print('orders (T centre):', out['orders_T'])
# stiffness of the semi-discrete system (Nr=20, Nz=26)
A, b = build(20, 26); ev = np.linalg.eigvals(A.toarray()).real
out['eig'] = dict(min=float(-ev.max()), max=float(-ev.min()), ratio=float(ev.min()/ev.max()))
print('eigenvalues: slowest %.3e, fastest %.3e 1/s, stiffness ratio %.0f; explicit Euler step limit %.2f s' % (-ev.max(), -ev.min(), ev.min()/ev.max(), 2/(-ev.min())))
print('slowest mode exact: %.3e 1/s' % lam[0, 0])
out['lam11'] = float(lam[0, 0])
# heating time needed for F0 = 3 min (12D botulinum cook), exact solution, cooling kept 30 min
def F0_for(th):
    global t_h
    old = t_h; t_h = th; tg = np.linspace(0, th + 30*60, 8001); Tg = T_exact_centre(tg); t_h = old
    return np.trapezoid(10**((Tg - Tref)/zD), tg)/60
for th_min in (50, 60, 70, 80):
    print('heating %d min -> F0 = %.2f min' % (th_min, F0_for(th_min*60)))
out['F0_vs_th'] = {th: float(F0_for(th*60)) for th in (50, 60, 70, 80)}
json.dump(out, open('ex312b.json', 'w'), indent=1)

# ---------------------------------------------------------------- numbers quoted in the detailed slides 3.12.2
A, b = build(20, 26); n = A.shape[0]
print('\n2D operator 20 x 26: %d unknowns, %d non-zeros (%.2f %%)' % (n, A.nnz, 100*A.nnz/n**2))
print('check A*1 + b = 0: max |.| = %.1e' % np.max(np.abs(A @ np.ones(n) + b)))
dr = R/20; r = np.arange(20)*dr
w = [(1 - 1/(2*i), 1 + 1/(2*i)) for i in (1, 2, 19)]
print('radial weights (1 - 1/2i, 1 + 1/2i): i=1 (%.3f, %.3f), i=2 (%.3f, %.3f), i=19 (%.3f, %.3f)' % (*w[0], *w[1], *w[2]))
print('dr = %.2f mm, dz = %.2f mm' % (1e3*dr, 1e3*Hh/26))
