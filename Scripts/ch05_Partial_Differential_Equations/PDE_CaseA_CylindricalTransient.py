"""Test case (a): transient conduction in a solid cylinder (Python port of PDE_CaseA_CylindricalTransient.m).

    dT/dt = alpha*(d2T/dr2 + (1/r) dT/dr),   0 <= r <= R
    BC: dT/dr = 0 at r = 0 (symmetry),  T(R, t) = 0 (Dirichlet)
    IC: T(r, 0) = T0*J0(lambda1*r), a single Bessel mode, so that the exact solution has one term:
        T(r, t) = T0*J0(lambda1*r)*exp(-t/tau),  lambda1 = j01/R,  tau = 1/(alpha*lambda1^2)

Method of lines in r (central differences, axis formula 4*(T1 - T0)/dr^2), integrated in t
  1) with explicit RK4 at a fixed step h = 0.3*h_lim, h_lim = 0.5*dr^2/alpha;
  2) with an implicit, adaptive BDF method (solve_ivp), what MATLAB's pdepe does inside with ode15s
     (SciPy has no pdepe).
Also: the exact step limit of RK4 from the eigenvalues, an instability demonstration and the order of
convergence on five grids. Python arrays start at 0: node 0 is the axis, node N-1 the surface.
"""
import numpy as np
from scipy.special import j0, jn_zeros
from scipy.integrate import solve_ivp
from scipy.linalg import expm
import matplotlib.pyplot as plt

# --- Parameters ---------------------------------------------------------------------------------
R = 0.05                 # cylinder radius [m]
alpha = 1.2e-5           # thermal diffusivity [m^2/s]
T0 = 100.0               # amplitude of the initial mode [K]
j01 = jn_zeros(0, 1)[0]  # first zero of J0
lam1 = j01/R
tau = 1/(alpha*lam1**2)  # decay time of the mode [s]
T_exact = lambda r, t: T0*j0(lam1*r)*np.exp(-alpha*lam1**2*t)
print(f"j01 = {j01:.6f}, lambda1 = {lam1:.2f} 1/m, tau = {tau:.1f} s")


def mol_rhs(T, r, dr):
    """dT/dt at the nodes 0..N-2 (the surface node N-1 is fixed at 0)."""
    d = np.empty_like(T)
    d[0] = 4*alpha/dr**2*(T[1] - T[0])                       # axis: L'Hopital + ghost node
    Tp = np.append(T, 0.0)                                   # append the surface value
    i = np.arange(1, T.size)
    d[1:] = alpha*((Tp[i+1] - 2*Tp[i] + Tp[i-1])/dr**2 + (Tp[i+1] - Tp[i-1])/(2*dr*r[i]))
    return d


def rk4_step(f, T, h):
    k1 = f(T); k2 = f(T + h/2*k1); k3 = f(T + h/2*k2); k4 = f(T + h*k3)
    return T + h/6*(k1 + 2*k2 + 2*k3 + k4)


def jacobian(r, dr):
    n = r.size - 1; J = np.zeros((n, n))
    for j in range(n):
        e = np.zeros(n); e[j] = 1.0; J[:, j] = mol_rhs(e, r, dr)   # the system is linear
    return J


# --- Grid and step limits -----------------------------------------------------------------------
N = 41
r = np.linspace(0, R, N); dr = r[1]
h_lim = 0.5*dr**2/alpha
lam = np.linalg.eigvals(jacobian(r, dr)).real
lmax, lmin = np.max(abs(lam)), np.min(abs(lam))
print(f"N = {N}, dr = {dr*1e3:.2f} mm, h_lim = 0.5 dr^2/alpha = {h_lim:.4f} s")
print(f"|lambda|max = {lmax:.1f} 1/s = {lmax*dr**2/alpha:.2f} alpha/dr^2; |lambda|min = {lmin:.4f} 1/s (1/tau = {1/tau:.4f})")
print(f"stiffness ratio {lmax/lmin:.0f}; limits: Euler 2/|lambda|max = {2/lmax/h_lim:.2f} h_lim, "
      f"RK4 2.785/|lambda|max = {2.785/lmax/h_lim:.2f} h_lim")

# --- Instability demonstration -------------------------------------------------------------------
for frac, nst in ((1.3, 60), (2.5, 12)):
    T = T_exact(r[:-1], 0)
    for _ in range(nst):
        T = rk4_step(lambda u: mol_rhs(u, r, dr), T, frac*h_lim)
    print(f"h = {frac} h_lim: |T(0)| = {abs(T[0]):.1e} K after {nst} steps")

# --- Stable run with RK4 up to 0.5 tau -----------------------------------------------------------
tend = 0.5*tau
nsteps = int(round(tend/(0.3*h_lim))); h = tend/nsteps
T = T_exact(r[:-1], 0)
for _ in range(nsteps):
    T = rk4_step(lambda u: mol_rhs(u, r, dr), T, h)
T_mol = np.append(T, 0.0)
err_rk4 = np.max(abs(T_mol - T_exact(r, tend)))
print(f"RK4: {nsteps} steps ({4*nsteps} evaluations), max error at 0.5 tau = {err_rk4:.4e} K "
      f"(axis: {T_mol[0]:.3f} vs {T_exact(0, tend):.3f} K)")
# the same ODEs integrated exactly in time: only the spatial error is left
T_ex_time = np.append(expm(jacobian(r, dr)*tend) @ T_exact(r[:-1], 0), 0.0)
print(f"exact time integration (expm): max error {np.max(abs(T_ex_time - T_exact(r, tend))):.4e} K, "
      f"RK4 adds {np.max(abs(T_mol - T_ex_time)):.1e} K")

# --- Implicit, adaptive BDF ----------------------------------------------------------------------
sol = solve_ivp(lambda t, u: mol_rhs(u, r, dr), (0, tend), T_exact(r[:-1], 0), method='BDF',
                rtol=1e-6, atol=1e-8)
err_bdf = np.max(abs(np.append(sol.y[:, -1], 0.0) - T_exact(r, tend)))
print(f"BDF: {sol.t.size - 1} steps, {sol.nfev} evaluations, max error {err_bdf:.4e} K")

# --- Order of convergence ------------------------------------------------------------------------
print("\nN      dr [mm]   max error at 0.3 tau   order")
prev = None
for Nk in (11, 21, 41, 81, 161):
    rk = np.linspace(0, R, Nk); drk = rk[1]; te = 0.3*tau
    ns = int(round(te/(0.3*0.5*drk**2/alpha))); hk = te/ns
    Tk = T_exact(rk[:-1], 0)
    for _ in range(ns):
        Tk = rk4_step(lambda u: mol_rhs(u, rk, drk), Tk, hk)
    e = np.max(abs(np.append(Tk, 0.0) - T_exact(rk, te)))
    print(f"{Nk:4d}  {drk*1e3:8.4f}   {e:.3e}            {'' if prev is None else f'{np.log2(prev/e):.2f}'}")
    prev = e

# --- Plot ------------------------------------------------------------------------------------------
rf = np.linspace(0, R, 200)
for f in (0, 0.1, 0.25, 0.5):
    plt.plot(rf*1e3, T_exact(rf, f*tau), label=f"t = {f} tau")
plt.plot(r*1e3, T_mol, 'ko', ms=3, label="method of lines, 0.5 tau")
plt.xlabel("r (mm)"); plt.ylabel("T (K)"); plt.legend(); plt.grid(True); plt.show()
