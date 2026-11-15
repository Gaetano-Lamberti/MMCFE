"""
Example 4.1 LTR isothermal - comparison of solution methods (Python version)
============================================================================
A guiding example: the styrene reactor
4. The PDE(s) problem(s)
4.1 Isothermal steady-state Laminar Tubular Reactor (LTR)   (slides 21-26)

NOTATION (as in the slides)
    A = ethylbenzene (EB), B = styrene (ST); also H2, BZ (benzene),
    C2H4 (ethylene) and W (steam, inert).
        R1:  EB -> ST + H2        rate constant k1
        R2:  EB -> BZ + C2H4      rate constant k2
    n_i = v_z*C_i : local axial molar flux of species i   [kmol/(m^2 s)]
    X_A(r,z) = 1 - n_A(r,z)/n_A0(r)     local CONVERSION of EB
    Y_B(r,z) = n_B(r,z)/n_A0(r)         local YIELD of styrene

PROBLEM (4.1): one nonlinear, PARABOLIC PDE in (r, z)

    2*vbar0*(1 - (r/R)^2) * dX_A/dz =
         = D*(1/r)*d/dr( r*dX_A/dr ) + (k1 + k2)*(1 - X_A)/(1 + epsA*X_A)

    BC1  @ r = 0, for all z :  dX_A/dr = 0   (symmetry)
    BC2  @ r = R, for all z :  dX_A/dr = 0   (impermeable wall)
    BC3  @ z = 0, for all r :  X_A = 0       (feed)

    z plays the role of time: the solution is MARCHED from z = 0 to z = L.

ONE SPATIAL DISCRETIZATION, THREE WAYS OF MARCHING IN z
    The radius is discretized with N_r = 40 intervals and second-order finite
    differences (METHOD OF LINES): the PDE becomes a system of ODEs in z, one
    per radial node. The same discretized system is then marched in z by:
      1) FTCS  - Forward-Time (explicit Euler in z), Centred-Space.
                 Simplest possible scheme; limited by STABILITY.
      2) IMEX  - IMplicit-EXplicit Euler: radial diffusion (the stiff part)
                 implicit, reaction explicit. One tridiagonal LINEAR system per
                 step, with a constant matrix factorized once.
                 Limited by ACCURACY (first order in z).
      3) BDF   - scipy.integrate.solve_ivp(method='BDF'): variable-order
                 (1 to 5) backward differentiation formulas with adaptive step,
                 the Python counterpart of MATLAB's ode15s. Used as REFERENCE.
    Finally, from the BDF solution, the concentrations of ALL species are
    computed along the reactor and plotted.

HOW TO RUN
    python example_4_1_ltr_methods_comparison.py
    (requires numpy, scipy, matplotlib)

UNITS
    Lengths in m, velocities in m/s, D in m^2/s, rate constants in 1/s,
    molar flows in kmol/s, concentrations in kmol/m^3.

INDEXING NOTE
    Python arrays start at 0: node 0 is the axis (r = 0) and node N_r is the
    wall (r = R). The MATLAB version of this script uses 1 ... N_r+1.
"""

import time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from scipy.integrate import solve_ivp, trapezoid
import matplotlib.pyplot as plt

# =============================================================================
# 1) PARAMETERS
# =============================================================================

# --- Kinetic parameters (Arrhenius, Sec. 1.5) ---
Rg   = 8.314        # gas constant (NB: R is the tube radius)      [kJ/(kmol K)]
k0_1 = 1.47e5       # pre-exponential factor, R1                   [1/s]
Ea1  = 90.98e3      # activation energy, R1                        [kJ/kmol]
k0_2 = 9.02e10      # pre-exponential factor, R2                   [1/s]
Ea2  = 207.99e3     # activation energy, R2                        [kJ/kmol]

# --- Operating conditions (Sec. 1.6), same as Sections 2-3 ---
T0     = 900.0                     # temperature, FIXED (isothermal)  [K]
P      = 1e5                       # pressure (isobaric)              [Pa]
F_A0   = 1/3600                    # inlet molar flow of EB, 1 kmol/h [kmol/s]
F_W0   = 9/3600                    # inlet molar flow of steam        [kmol/s]
F_tot0 = F_A0 + F_W0               # total inlet molar flow           [kmol/s]
y_A0   = F_A0/F_tot0               # inlet mole fraction of EB (0.10) [-]
epsA   = y_A0*1                    # expansion factor (delta_nu = +1) [-]
Q0     = F_tot0*1e3*Rg*T0/P        # total inlet volumetric flow      [m^3/s]
C_tot  = P/(1e3*Rg*T0)             # total molar concentration P/RT   [kmol/m^3]
C_A0   = y_A0*C_tot                # inlet concentration of EB        [kmol/m^3]
V_tot  = 0.30                      # total volume (as the PFR, Ex. 3.1) [m^3]

k1   = k0_1*np.exp(-Ea1/(Rg*T0))   # [1/s]
k2   = k0_2*np.exp(-Ea2/(Rg*T0))   # [1/s]
ksum = k1 + k2                     # [1/s]
S    = k1/ksum                     # selectivity to ST, constant at fixed T [-]

# --- Geometry of the multitubular reactor ---
R     = 0.015                      # tube radius                      [m]
vbar0 = 1.0                        # mean inlet velocity per tube     [m/s]
tau   = V_tot/Q0                   # mean residence time              [s]
L     = vbar0*tau                  # tube length                      [m]
N_t   = Q0/(np.pi*R**2*vbar0)      # number of tubes                  [-]
D     = 1e-4                       # diffusivity (all species)        [m^2/s]

# --- Radial grid (common to ALL methods) ---
N_r  = 40                          # number of radial INTERVALS
r    = np.linspace(0.0, R, N_r + 1)   # nodes: r[0] = 0 (axis) ... r[N_r] = R (wall)
dr   = R/N_r
v0   = 2*vbar0*(1 - (r/R)**2)      # Poiseuille profile at the nodes
v0[-1] = 0.0                       # exactly zero at the wall (no slip)
n_in = N_r                         # number of MARCHED nodes (all but the wall)

print(f"Isothermal LTR: N_t = {N_t:.0f} tubes, R = {R:.3f} m, "
      f"L = {L:.3f} m, N_r = {N_r}\n")

# =============================================================================
# LOCAL HELPER FUNCTIONS
# =============================================================================

def build_laplacian(r):
    """Sparse finite-difference matrix of (1/r) d/dr (r d/dr) on the nodes
    r[0] = 0 ... r[-1] = R, with zero-gradient conditions at the axis (BC1)
    and at the wall (BC2), both imposed with a ghost node."""
    n  = r.size
    dr = r[1] - r[0]
    lo = np.zeros(n - 1)      # sub-diagonal:   entry (i, i-1)
    di = np.zeros(n)          # main diagonal:  entry (i, i)
    up = np.zeros(n - 1)      # super-diagonal: entry (i, i+1)
    # Axis: (1/r) dX/dr -> d2X/dr2 (L'Hopital), so the Laplacian is 2*d2X/dr2;
    # with the symmetric ghost node X(-dr) = X(+dr) it becomes 4*(X1 - X0)/dr^2
    di[0] = -4/dr**2
    up[0] =  4/dr**2
    # Interior nodes: central differences for d2X/dr2 + (1/r) dX/dr
    i = np.arange(1, n - 1)
    lo[i - 1] = 1/dr**2 - 1/(2*r[i]*dr)
    di[i]     = -2/dr**2
    up[i]     = 1/dr**2 + 1/(2*r[i]*dr)
    # Wall: ghost node X(R+dr) = X(R-dr) -> 2*(X_{N-1} - X_N)/dr^2
    lo[n - 2] = 2/dr**2
    di[n - 1] = -2/dr**2
    return sp.diags([lo, di, up], [-1, 0, 1], format="csr")


def wall_value(XN):
    """Wall conversion from its ALGEBRAIC equation (v0(R) = 0):
        0 = a*(XN - Xw) + ksum*(1 - Xw)/(1 + epsA*Xw),   a = 2*D/dr^2
    (XN = neighbouring node). Multiplying by (1 + epsA*Xw) gives the quadratic
        a*epsA*Xw^2 + b*Xw - c = 0,  b = a + ksum - a*epsA*XN,  c = a*XN + ksum
    whose positive root is written in the numerically stable form below."""
    a = 2*D/dr**2
    b = a + ksum - a*epsA*XN
    c = a*XN + ksum
    return 2*c/(b + np.sqrt(b**2 + 4*a*epsA*c))


def src(X):
    """Reaction source term of (4.1): (k1 + k2)*(1 - X_A)/(1 + epsA*X_A)."""
    return ksum*(1 - X)/(1 + epsA*X)


def rhs_interior(z, Xi):
    """dX_i/dz at the marched nodes (all but the wall):
        f_i = [ D*(Lap X)_i + (k1+k2)*(1 - X_i)/(1 + epsA*X_i) ] / v0_i
    The wall value is appended first ("slaved" to its neighbour)."""
    X  = np.append(Xi, wall_value(Xi[-1]))
    LX = Lap @ X
    return (D*LX[:n_in] + src(Xi))/v0[:n_in]


def with_wall(Xi):
    """Append the (slaved) wall value to the marched nodes.
    Works on a single profile (1-D) or on many profiles (2-D, one per row)."""
    Xi = np.atleast_2d(Xi)
    return np.column_stack([Xi, wall_value(Xi[:, -1])])


def bulk_average(Xs):
    """Flow-averaged ("cup-mixing") value over the cross-section:
        Xb = integral(v0*X*r dr) / integral(v0*r dr)   (trapezoidal rule)
    Xs: one ROW per axial position, one COLUMN per radial node."""
    w = v0*r
    return trapezoid(np.atleast_2d(Xs)*w, r, axis=1)/trapezoid(w, r)


def numerical_jacobian(F, x):
    """Forward finite differences, one column per variable (same scheme as
    in the Newton solver of Example 2.2)."""
    F0 = F(x)
    J  = np.zeros((F0.size, x.size))
    for j in range(x.size):
        h  = 1e-7*max(1.0, abs(x[j]))
        xh = x.copy()
        xh[j] += h
        J[:, j] = (F(xh) - F0)/h
    return J


def march_ftcs(nz):
    """Explicit Euler in z on the method-of-lines system:
        X_i^{n+1} = X_i^n + dz*f_i(X^n)   (marched nodes), wall slaved.
    Returns (Xall, ok, n_blow): Xall has one ROW per axial step; the march
    stops (ok = False) as soon as the solution blows up."""
    dz = L/nz
    Xi = np.zeros(n_in)
    Xall = np.zeros((nz + 1, n_in + 1))
    Xall[0] = with_wall(Xi)[0]
    for n in range(1, nz + 1):
        Xi = Xi + dz*rhs_interior((n - 1)*dz, Xi)      # explicit Euler step
        if not np.all(np.isfinite(Xi)) or np.max(np.abs(Xi)) > 10:
            return Xall[:n], False, n                   # divergence detected
        Xall[n] = with_wall(Xi)[0]
    return Xall, True, None


def march_imex(nz):
    """IMEX Euler: implicit radial diffusion, explicit reaction.
        (diag(v0)/dz - D*Lap) X^{n+1} = diag(v0)/dz X^n + src(X^n)
    The wall row has v0 = 0: it becomes the algebraic wall equation (with the
    reaction evaluated at the previous step, an O(dz) approximation).
    The matrix does not change with n: LU-factorize ONCE (splu), then only
    forward/back substitutions (cheap: the matrix is tridiagonal)."""
    dz  = L/nz
    Mdz = sp.diags(v0/dz)
    lu  = splu((Mdz - D*Lap).tocsc())                  # factorize once
    Xk  = np.zeros(N_r + 1)
    Xall = np.zeros((nz + 1, N_r + 1))
    for k in range(1, nz + 1):
        b  = Mdz @ Xk + src(Xk)                        # explicit part (reaction)
        Xk = lu.solve(b)                               # implicit part (diffusion)
        Xall[k] = Xk
    return Xall


# =============================================================================
# 2) THE DISCRETIZED SYSTEM (method of lines)
# =============================================================================
# At each marched node i (v0 > 0):   v0_i * dX_i/dz = D*(Lap X)_i + src(X_i)
# At the wall (v0 = 0) the equation has NO dz-derivative: it is ALGEBRAIC,
#     0 = D*(Lap X)_wall + src(X_wall)
# a quadratic in X_wall, solved in closed form by wall_value(): the wall is
# "slaved" to its neighbour and is never marched.
Lap = build_laplacian(r)

# =============================================================================
# 3) STABILITY OF EXPLICIT EULER: where the FTCS limit comes from
# =============================================================================
# Explicit Euler applied to dX/dz = J X (J = Jacobian of rhs_interior)
# multiplies, at each step, every eigen-component of the error by
# (1 + dz*lambda). Stability requires |1 + dz*lambda| <= 1 for ALL
# eigenvalues lambda of J. Here they are real and negative, so
#          dz <= 2/|lambda|_max
# The largest |lambda| comes from the node next to the wall, where the
# diffusion term is divided by a very small velocity v0(r_{N-1}). A quick
# (Gershgorin) upper bound of that row is
#          |lambda|_max  <~  4*D/(v0(r_{N-1})*dr^2)
# and since v0(r_{N-1}) ~ 4*vbar0/N_r and dr = R/N_r:
#          |lambda|_max  ~  D*N_r^3/(vbar0*R^2)   ->   dz_crit ~ N_r^(-3)
# The EXACT value is computed here from the eigenvalues of J at the inlet.
J         = numerical_jacobian(lambda Xi: rhs_interior(0.0, Xi), np.zeros(n_in))
lam       = np.linalg.eigvals(J)
lam_max   = np.max(np.abs(lam))
dz_crit   = 2/lam_max                                  # exact explicit-Euler limit
lam_gersh = 4*D/(v0[n_in - 1]*dr**2)                   # quick upper bound
print(f"Stability of explicit Euler (FTCS), N_r = {N_r}")
print(f"  |lambda|_max = {lam_max:.3e} 1/m (Gershgorin bound {lam_gersh:.3e} 1/m)")
print(f"  critical step dz_crit = 2/|lambda|_max = {dz_crit:.3e} m  ->  "
      f"at least {int(np.ceil(L/dz_crit))} steps\n")

# =============================================================================
# 4) METHOD 3 FIRST: BDF (it provides the REFERENCE solution)
# =============================================================================
# jac_sparsity: the Jacobian is tridiagonal (each node talks only to its two
# neighbours), so it is approximated with 3 evaluations of f instead of n_in.
# first_step: small, because at z = 0 the wall value jumps to X_wall > 0 while
# its neighbours are still 0 (the inlet is a discontinuity).
jac_pattern = sp.diags([np.ones(n_in - 1), np.ones(n_in), np.ones(n_in - 1)],
                       [-1, 0, 1])
z = np.linspace(0.0, L, 301)                            # output grid
bdf_opts = dict(method="BDF", rtol=1e-8, atol=1e-10,
                jac_sparsity=jac_pattern, first_step=1e-6*L)
t0 = time.perf_counter()
# First run WITHOUT output grid: sol_steps.t then contains exactly the
# accepted steps chosen by the solver (to count them).
sol_steps = solve_ivp(rhs_interior, (0.0, L), np.zeros(n_in), **bdf_opts)
nsteps_bdf = sol_steps.t.size - 1                       # accepted steps
t_bdf = time.perf_counter() - t0
# Second run with the output grid t_eval, for tables and plots
sol = solve_ivp(rhs_interior, (0.0, L), np.zeros(n_in), t_eval=z, **bdf_opts)
X_bdf  = with_wall(sol.y.T)                             # rows: z, cols: r (wall appended)
Xb_bdf = bulk_average(X_bdf)                            # cup-mixing average
X_ref  = Xb_bdf[-1]

# =============================================================================
# 5) METHOD 1: FTCS (explicit Euler in z)
# =============================================================================
# (a) a step just BELOW the critical one: stable
nz_ftcs = int(np.ceil(1.05*L/dz_crit))                 # 5% more steps than the minimum
t0 = time.perf_counter()
X_ftcs, ok, _ = march_ftcs(nz_ftcs)
t_ftcs = time.perf_counter() - t0
Xb_ftcs = bulk_average(X_ftcs)                          # bulk conversion at every step
z_ftcs  = np.linspace(0.0, L, nz_ftcs + 1)
# (b) a step just ABOVE the critical one: diverges
nz_bad = int(np.floor(0.95*L/dz_crit))                 # 5% fewer steps than the minimum
_, ok_bad, n_blow = march_ftcs(nz_bad)

# =============================================================================
# 6) METHOD 2: IMEX Euler (implicit diffusion, explicit reaction)
# =============================================================================
nz_imex_list = [200, 2000]
Xb_imex, t_imex = [], []
for k, nz in enumerate(nz_imex_list):
    t0 = time.perf_counter()
    X_imex = march_imex(nz)
    t_imex.append(time.perf_counter() - t0)
    Xbz = bulk_average(X_imex)                          # bulk conversion at every step
    Xb_imex.append(Xbz[-1])
    if k == 0:
        Xbz_imex1 = Xbz                                 # coarse history, for the plot
z_imex1 = np.linspace(0.0, L, nz_imex_list[0] + 1)

# =============================================================================
# 7) COMPARISON TABLE
# =============================================================================
print(f"Comparison of marching methods (same radial grid, N_r = {N_r})")
print(f"{'Method':40s} {'z-steps':>10s} {'X_bulk(L)':>12s} {'|error|':>12s} {'CPU [s]':>10s}")
print(f"{'BDF, solve_ivp (reference, rtol 1e-8)':40s} {nsteps_bdf:10d} {X_ref:12.6f} "
      f"{'-':>12s} {t_bdf:10.3f}")
print(f"{'FTCS, 5% below dz_crit':40s} {nz_ftcs:10d} {Xb_ftcs[-1]:12.6f} "
      f"{abs(Xb_ftcs[-1] - X_ref):12.1e} {t_ftcs:10.3f}")
if not ok_bad:
    print(f"{'FTCS, 5% above dz_crit':40s} {nz_bad:10d} {'DIVERGED@' + str(n_blow):>12s} "
          f"{'-':>12s} {'-':>10s}")
for nz, xb, t in zip(nz_imex_list, Xb_imex, t_imex):
    print(f"{'IMEX Euler, ' + str(nz) + ' steps':40s} {nz:10d} {xb:12.6f} "
          f"{abs(xb - X_ref):12.1e} {t:10.3f}")
print("Notes: FTCS pays for STABILITY (the step is fixed by the near-wall nodes);\n"
      "       IMEX pays for ACCURACY (first order: error ~ 1/steps);\n"
      "       BDF adapts order and step: fewest steps for a given accuracy.\n")

# =============================================================================
# 8) CONCENTRATIONS OF ALL SPECIES ALONG THE REACTOR (from BDF)
# =============================================================================
# Local yields per mole of EB fed along the same streamline (slide 26: equal
# diffusivities -> stoichiometry holds at every point):
#   EB   : 1 - X_A
#   ST   : Y_B = S*X_A                  (R1, constant S at fixed T)
#   H2   : Y_H2   = Y_B                 (formed with ST by R1)
#   BZ   : Y_BZ   = (1 - S)*X_A         (R2)
#   C2H4 : Y_C2H4 = Y_BZ                (formed with BZ by R2)
#   W    : F_W0/F_A0 = 9                (inert)
# Total moles per mole of EB fed: 1/y_A0 + X_A (each converted EB gives
# 2 moles). Hence the local mole fraction of species i is
#   y_i = Y_i/(1/y_A0 + X_A) = y_A0*Y_i/(1 + epsA*X_A)
# and, isothermal and isobaric, the local concentration is
#   C_i = y_i*C_tot = C_A0*Y_i/(1 + epsA*X_A)
# (for EB this is the familiar C_A = C_A0*(1 - X_A)/(1 + epsA*X_A)).
names = ["EB", "ST", "H$_2$", "BZ", "C$_2$H$_4$", "W"]


def yields(X):
    """Yields of the six species (last axis) for any array of X_A values."""
    X = np.asarray(X)
    return np.stack([1 - X, S*X, S*X, (1 - S)*X, (1 - S)*X,
                     (F_W0/F_A0)*np.ones_like(X)], axis=-1)


def concentrations(X):
    """Local concentrations C_i = C_A0*Y_i/(1 + epsA*X_A)   [kmol/m^3]."""
    X = np.asarray(X)
    return C_A0*yields(X)/(1 + epsA*X)[..., None]


# (a) CUP-MIXING (flow-averaged) concentrations along z: what a sample of the
#     whole tube outlet at position z would contain. The flows add up linearly
#     over the section, so F_i(z) = F_A0*Y_i,bulk(z) and
#     C_i,cup(z) = F_i(z)/Q(z) with Q = Q0*(1 + epsA*X_bulk).
C_cup = concentrations(Xb_bdf)                          # rows: z, cols: species
# (b) LOCAL concentrations over the whole (r, z) domain
C_loc = concentrations(X_bdf)                           # z x r x species

print("Outlet flows of the whole reactor [kmol/h] and cup-mixing concentrations [mol/m^3]")
Y_out = yields(Xb_bdf[-1])
for s, name in enumerate(names):
    label = name.replace("$", "").replace("_", "")
    print(f"  {label:7s}  F = {3600*F_A0*Y_out[s]:.4f} kmol/h   "
          f"C = {1e3*C_cup[-1, s]:8.3f} mol/m^3")

# =============================================================================
# 9) PLOTS
# =============================================================================
# Figure 1: the three methods on the bulk conversion
plt.figure("Example 4.1 - methods comparison", figsize=(7, 5))
plt.plot(z, Xb_bdf, "-", lw=2.0, label="BDF (reference)")
plt.plot(z_ftcs, Xb_ftcs, "--", lw=1.2, label="FTCS")
plt.plot(z_imex1, Xbz_imex1, ":", lw=1.8, label=f"IMEX, {nz_imex_list[0]} steps")
plt.xlabel("Axial position z [m]")
plt.ylabel("Bulk conversion $X_{A,bulk}$ [-]")
plt.title(f"Isothermal LTR, $N_r$ = {N_r} (curves nearly coincide)")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()

# Figure 2: concentrations of all species (BDF solution)
fig, ax = plt.subplots(1, 3, figsize=(15, 4.5), num="Example 4.1 - species concentrations")
styles = ["-", "-", "--", "-", "--"]                    # dashed: coincident curves
for s in range(5):                                       # reacting species
    ax[0].plot(z, 1e3*C_cup[:, s], styles[s], lw=1.6, label=names[s])
ax[0].set_xlabel("Axial position z [m]")
ax[0].set_ylabel("Cup-mixing concentration [mol/m$^3$]")
ax[0].set_title("Reacting species")
ax[0].legend(loc="upper right")
ax[0].grid(True)
# NB: the ST and H2 curves coincide, and so do BZ and C2H4 (same reaction).

ax[1].plot(z, 1e3*C_cup[:, 5], lw=1.4)                  # steam
ax[1].set_xlabel("Axial position z [m]")
ax[1].set_ylabel("Cup-mixing concentration [mol/m$^3$]")
ax[1].set_title("Steam (inert): diluted by the mole increase")
ax[1].grid(True)

for s in (0, 1, 3):                                      # outlet radial profiles
    ax[2].plot(r/R, 1e3*C_loc[-1, :, s], lw=1.4, label=names[s])
ax[2].set_xlabel("Dimensionless radius r/R [-]")
ax[2].set_ylabel("Local concentration [mol/m$^3$]")
ax[2].set_title("Outlet radial profiles")
ax[2].legend(loc="center right")
ax[2].grid(True)
fig.tight_layout()

plt.show()
