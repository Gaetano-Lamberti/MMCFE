"""Canned food sterilization solved by finite elements: Python port of PDE_Example_CanSterilization_solvepde.m.

Same problem as PDE_Example_CanSterilization.py (quarter of the axial section, 0 <= r <= R, 0 <= z <= Hh,
z from the mid-plane):
    dT/dt = alpha * [ (1/r) d/dr( r dT/dr ) + d2T/dz2 ]
MATLAB's PDE Toolbox (solvepde) works in Cartesian coordinates, so the axisymmetric PDE is multiplied by r:
    r dT/dt = div( alpha r grad T )      (coefficient form: m = 0, d = r, c = alpha*r, a = f = 0)
SciPy has no solvepde: this port uses scikit-fem (pip install scikit-fem) with quadratic triangles (P2), the
default of the PDE Toolbox, and the same weak form:
    M dT/dt + K T = 0,   M_ij = int r phi_i phi_j dA,   K_ij = int alpha r grad phi_i . grad phi_j dA
BCs: wall (r = R) and lid (z = Hh) at T_w(t), Dirichlet; axis and mid-plane: nothing (zero flux, natural BC).
Time integration: BDF2 with a constant step (first step implicit Euler); the matrix is factorized once per phase.
Expected (exact series): centre 116.56 C at the end of heating, maximum 117.42 C, F0 = 8.89 min.
"""
import numpy as np
import skfem
from skfem.helpers import dot, grad
from scipy.sparse.linalg import splu
from scipy.integrate import cumulative_trapezoid

# --- Parameters (as in PDE_Example_CanSterilization.py) ------------------------------------------------
R, Hh = 0.038, 0.050                 # can radius and half height [m]
alpha = 1.5e-7                       # thermal diffusivity [m^2/s]
T0, TR, TC = 40.0, 121.1, 20.0       # initial, retort and cooling-water temperatures [C]
t_h, t_end = 80*60.0, 110*60.0       # end of heating, end of cooling [s]
zD, Tref = 10.0, 121.1               # z-value and reference temperature [C]
dt = 5.0                             # time step [s]

# --- Mesh (x = r, y = z) and P2 basis -------------------------------------------------------------------
mesh = skfem.MeshTri.init_tensor(np.linspace(0, R, 20), np.linspace(0, Hh, 26))
basis = skfem.Basis(mesh, skfem.ElementTriP2())


@skfem.BilinearForm
def stiffness(u, v, w):              # c = alpha*r
    return alpha*w.x[0]*dot(grad(u), grad(v))


@skfem.BilinearForm
def mass(u, v, w):                   # d = r
    return w.x[0]*u*v


K = stiffness.assemble(basis)
M = mass.assemble(basis)
D = basis.get_dofs(lambda x: np.isclose(x[0], R) | np.isclose(x[1], Hh)).all()   # wall and lid
I = basis.complement_dofs(D)                                                      # free unknowns
xd = basis.doflocs
centre = np.argmin(xd[0]**2 + xd[1]**2)
print(f"Mesh: {mesh.p.shape[1]} vertices, {mesh.t.shape[1]} triangles, {basis.N} P2 unknowns "
      f"({I.size} free)")


def march(T, Tw, t0, t1):
    """BDF2 in time from t0 to t1 with the wall/lid at Tw; returns times and centre temperatures."""
    n = int(round((t1 - t0)/dt))
    T = T.copy(); T[D] = Tw
    Mi, Ki = M[I][:, I], K[I][:, I]
    bD = lambda c: -(c[0]*M[I][:, D] + c[1]*K[I][:, D]) @ (Tw*np.ones(D.size))   # Dirichlet part, constant
    lu1 = splu((Mi + dt*Ki).tocsc())                 # implicit Euler (first step)
    lu2 = splu((1.5*Mi + dt*Ki).tocsc())             # BDF2
    out_t, out_T = [t0], [T[centre]]
    Tprev = T.copy()
    for k in range(n):
        if k == 0:
            rhs = Mi @ T[I] + bD((1.0, dt)) + (M[I][:, D] @ T[D])
            Tn = T.copy(); Tn[I] = lu1.solve(rhs)
        else:
            rhs = Mi @ (2*T[I] - 0.5*Tprev[I]) + bD((1.5, dt)) + M[I][:, D] @ (2*T[D] - 0.5*Tprev[D])
            Tn = T.copy(); Tn[I] = lu2.solve(rhs)
        Tprev, T = T, Tn
        out_t.append(t0 + (k + 1)*dt); out_T.append(T[centre])
    return T, np.array(out_t), np.array(out_T)


T = T0*np.ones(basis.N)
T1, t1, c1 = march(T, TR, 0.0, t_h)                 # heating: the Dirichlet value jumps to TR at t = 0
T2, t2, c2 = march(T1, TC, t_h, t_end)              # cooling: the last heating state, wall at TC
t = np.concatenate([t1, t2[1:]]); Tc = np.concatenate([c1, c2[1:]])
F0 = cumulative_trapezoid(10**((Tc - Tref)/zD), t, initial=0)/60
k = np.argmax(Tc)
print(f"Centre at end of heating: {c1[-1]:.2f} C   (exact 116.56 C)")
print(f"Centre maximum: {Tc[k]:.2f} C at t = {t[k]/60:.1f} min   (exact 117.42 C)")
print(f"F0 = {F0[-1]:.3f} min   (exact 8.890 min)")
