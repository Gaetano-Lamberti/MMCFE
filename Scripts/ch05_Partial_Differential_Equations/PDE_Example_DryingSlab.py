"""Example 3.12.3 - Drying of a food slab with moisture-dependent diffusivity (nonlinear parabolic PDE).
    dX/dt = d/dx( D(X) dX/dx ),  0 <= x <= L (x = 0: centre, symmetry; x = L: surface)
    D(X) = D_d*exp(b*X)  (D grows with moisture: from 1e-10 m2/s at X_e to 2e-9 m2/s at X_0)
    BCs: dX/dx = 0 at x = 0;  X = X_e at x = L (negligible external resistance).  IC: X = X_0.
FVM (cell-centred, face diffusivity from the face moisture), method of lines + stiff BDF.
Comparison with the constant-D model (Crank's series) with the best-fit effective diffusivity."""
import numpy as np, scipy.sparse as sp, json
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

L = 5e-3                                  # half thickness [m]
X0, Xe = 4.0, 0.1                         # initial and equilibrium moisture [kg water / kg dry solid]
D_lo, D_hi = 1e-10, 2e-9                  # D at Xe and at X0 [m2/s]
b = np.log(D_hi/D_lo)/(X0 - Xe); D_d = D_lo/np.exp(b*Xe)
D = lambda X: D_d*np.exp(b*X)
t_end = 24*3600.0
ts = np.linspace(0, t_end, 289)           # every 5 min

def fvm(N, face='mean', rtol=1e-8):
    h = L/N; xc = (np.arange(N) + 0.5)*h
    def rhs(t, X):
        Xf = 0.5*(X[1:] + X[:-1])                                  # interior faces
        Df = D(Xf) if face == 'mean' else 2*D(X[1:])*D(X[:-1])/(D(X[1:]) + D(X[:-1]))
        J = -Df*(X[1:] - X[:-1])/h                                 # fluxes at interior faces
        Js = -D(0.5*(X[-1] + Xe))*(Xe - X[-1])/(h/2)               # surface face (X = Xe at distance h/2)
        dX = np.zeros(N)
        dX[:-1] -= J; dX[1:] += J; dX[-1] -= Js                     # centre face: zero flux
        return dX/h
    pat = sp.diags([np.ones(N-1), np.ones(N), np.ones(N-1)], [-1, 0, 1])
    s = solve_ivp(rhs, [0, t_end], X0*np.ones(N), method='BDF', rtol=rtol, atol=1e-10, jac_sparsity=pat, t_eval=ts)
    Xm = s.y.mean(axis=0)                                          # mean moisture (uniform cells)
    return xc, s.y, Xm, s.t.size

out = {}
ref = fvm(800)[2]                                                  # fine-grid reference
for N in (10, 20, 40, 80, 160):
    xc, Y, Xm, _ = fvm(N)
    e = np.max(np.abs(Xm - ref)); e1 = np.max(np.abs(Xm - ref)[ts >= 3600]); out[N] = float(e); out[f'{N}_after1h'] = float(e1)
    print(f'N={N:4d}  max|Xmean - ref| = {e:.3e}   (t >= 1 h: {e1:.3e})')
print('orders, all t   ', [round(np.log2(out[a]/out[c]), 2) for a, c in ((10, 20), (20, 40), (40, 80), (80, 160))])
print('orders, t >= 1 h', [round(np.log2(out[f'{a}_after1h']/out[f'{c}_after1h']), 2) for a, c in ((10, 20), (20, 40), (40, 80), (80, 160))])
xc, Y, Xm, _ = fvm(80); np.savez('ex312c.npz', x=xc, Y=Y, Xm=Xm, t=ts, ref=ref)
xh, Yh, Xmh, _ = fvm(80, face='harmonic'); print('face mean vs harmonic, max diff in Xmean: %.2e' % np.max(np.abs(Xmh - Xm)))

# constant-D model: Crank, slab, Dirichlet surface
def crank(Dc, t):
    n = np.arange(200)[:, None]; k = (2*n + 1)*np.pi/(2*L)
    return Xe + (X0 - Xe)*np.sum(8/((2*n + 1)**2*np.pi**2)*np.exp(-Dc*k**2*t[None, :]), axis=0)
fit = minimize_scalar(lambda lg: np.sum((crank(10**lg, ts) - ref)**2), bounds=(-11, -8), method='bounded')
Deff = 10**fit.x; cr = crank(Deff, ts)
emax = np.max(np.abs(cr - ref)); imax = np.argmax(np.abs(cr - ref))
# time to reach X = 0.5 (a typical end point) with both models
def t_at(Xm, target):
    i = np.argmax(Xm <= target); return np.interp(target, [Xm[i], Xm[i-1]], [ts[i], ts[i-1]])
t_nl, t_c = t_at(ref, 1.0), t_at(cr, 1.0)
print('X_mean at 24 h: nonlinear %.3f, constant D %.3f' % (ref[-1], cr[-1])); out['Xm24'] = [float(ref[-1]), float(cr[-1])]
print('best-fit D_eff = %.3e m2/s; max |Crank - nonlinear| = %.3f kg/kg at t = %.1f h' % (Deff, emax, ts[imax]/3600))
print('time to X_mean = 1.0: nonlinear %.2f h, constant D %.2f h (%.0f %%)' % (t_nl/3600, t_c/3600, 100*(t_c - t_nl)/t_nl))
# surface 'crust': moisture at the cell next to the surface and D there after 1 h
i1 = np.argmin(np.abs(ts - 3600)); print('after 1 h: X near surface %.2f, at centre %.2f; D ratio centre/surface %.1f' % (Y[-1, i1], Y[0, i1], D(Y[0, i1])/D(Y[-1, i1])))
out.update(dict(Deff=float(Deff), emax=float(emax), t_emax=float(ts[imax]/3600), t_nl=float(t_nl/3600), t_c=float(t_c/3600),
                b=float(b), D_d=float(D_d), X_surf_1h=float(Y[-1, i1]), X_c_1h=float(Y[0, i1])))
np.save('ex312c_crank.npy', cr)
json.dump(out, open('ex312c.json', 'w'), indent=1)

# ---------------------------------------------------------------- numbers quoted in the detailed slides 3.12.3
N = 80; h = L/N
def rhs80(t, X):
    Xf = 0.5*(X[1:] + X[:-1]); J = -D(Xf)*(X[1:] - X[:-1])/h; Js = -D(0.5*(X[-1] + Xe))*(Xe - X[-1])/(h/2)
    dX = np.zeros(N); dX[:-1] -= J; dX[1:] += J; dX[-1] -= Js; return dX/h
def jac(X, eps=1e-7):                                          # finite-difference Jacobian (dense, for eigenvalues)
    f0 = rhs80(0, X); return np.array([(rhs80(0, X + eps*np.eye(N)[j]) - f0)/eps for j in range(N)]).T
pat = sp.diags([np.ones(N-1), np.ones(N), np.ones(N-1)], [-1, 0, 1])
s = solve_ivp(rhs80, [0, t_end], X0*np.ones(N), method='BDF', rtol=1e-8, atol=1e-10, jac_sparsity=pat, dense_output=True)
print('\nh = %.1f um; b = %.4f, D_d = %.3e m2/s' % (1e6*h, b, D_d))
print('BDF: %d steps, %d evaluations of f, %d Jacobians' % (s.t.size - 1, s.nfev, s.njev))
for th in (0, 1, 24):
    X = X0*np.ones(N) if th == 0 else s.sol(th*3600.0)
    ev = np.linalg.eigvals(jac(X)).real
    print('t = %2d h: max|lambda| = %.3g, min|lambda| = %.3g 1/s, ratio %.2g, explicit Euler dt < %.3g s' % (th, -ev.min(), -ev.max(), ev.min()/ev.max(), 2/(-ev.min())))
print('4 D(X0)/h^2 = %.3g 1/s' % (4*D(X0)/h**2))
tg = np.linspace(0, t_end, 200)
lmax = [-np.linalg.eigvals(jac(X0*np.ones(N) if t == 0 else s.sol(t))).real.min() for t in tg]
print('explicit Euler, steps needed for 24 h (integral of |lambda|max/2): about %.0f' % np.trapezoid(np.array(lmax)/2, tg))
