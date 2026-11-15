"""
Example 4.2 LTR non-isothermal (Python version)
================================================
A guiding example: the styrene reactor
4. The PDE(s) problem(s)
4.2 Non-isothermal steady-state Laminar Tubular Reactor (LTR)   (slides 27-28)

NOTATION (as in the slides)
    A = ethylbenzene (EB), B = styrene (ST); also H2, BZ (benzene),
    C2H4 (ethylene) and W (steam, inert).
        R1:  EB -> ST + H2        rate constant k1(T), endothermic
        R2:  EB -> BZ + C2H4      rate constant k2(T), endothermic
    X_A(r,z) = 1 - n_A/n_A0   local CONVERSION of EB
    Y_B(r,z) = n_B/n_A0       local YIELD of styrene
    T(r,z)                    local temperature
    (n_i = v_z*C_i: local axial molar flux of species i)
    k     = thermal conductivity of the gas       [kW/(m K)]
    alpha = thermal diffusivity                   [m^2/s]
    h_w   = wall heat-exchange coefficient        [kW/(m^2 K)]  (h_w = 0: adiabatic)
    T_a   = service (heating) fluid temperature   [K]

PROBLEM (4.2): three coupled, nonlinear, PARABOLIC PDEs in (r, z)

    v0(r) dX_A/dz = D*Lap(X_A) + (k1(T) + k2(T)) * c(X_A,T)
    v0(r) dY_B/dz = D*Lap(Y_B) + k1(T) * c(X_A,T)
    v0(r) dT/dz   = alpha*Lap(T) + (y_A0/cp)*[(-dHr1)*k1(T) + (-dHr2)*k2(T)] * c(X_A,T)

    with  v0(r) = 2*vbar0*(1 - (r/R)^2),   Lap = (1/r) d/dr (r d/dr),
          c(X_A,T) = (1 - X_A)/((1 + epsA*X_A)*T/T0)

                         X_A            Y_B            T
    BC1 @ z = 0, all r : X_A = 0        Y_B = 0        T = T0
    BC2 @ r = 0, all z : dX_A/dr = 0    dY_B/dr = 0    dT/dr = 0
    BC3 @ r = R, all z : dX_A/dr = 0    dY_B/dr = 0    -k*dT/dr = h_w*(T - T_a)

    BC3 for T is a ROBIN condition (a mix of the value and of the derivative).

WHAT THIS SCRIPT DOES
    The radius is discretized with N_r = 40 intervals and second-order finite
    differences (METHOD OF LINES): each PDE becomes a set of ODEs in z, one per
    radial node (3 variables per node). The system is marched in z with:
      1) FTCS - explicit Euler in z: simple, slow (tiny step imposed by
                STABILITY), shown for teaching purposes;
      2) BDF  - scipy.integrate.solve_ivp(method="BDF"), the counterpart of
                MATLAB's ode15s: the REFERENCE solution.
    Two cases are solved: adiabatic wall (h_w = 0) and heated wall. Each is
    compared with its plug-flow limit (the PFR of Section 3). Finally the
    concentrations of ALL species are computed along the reactor (slide 28)
    and plotted.

TREATMENT OF THE WALL (strategy "a")
    At the wall v0(R) = 0: there is no marching equation there. The wall
    values are obtained from BC3 written with a one-sided, second-order
    difference for d/dr:
        du/dr|_R ~ (3*u_N - 4*u_{N-1} + u_{N-2}) / (2*dr)
    which gives the wall value in closed form (see wall_values below):
        X_A, Y_B :  u_N = (4*u_{N-1} - u_{N-2})/3
        T        :  T_N = [ k*(4*T_{N-1} - T_{N-2})/(2*dr) + h_w*T_a ]
                          / [ 3*k/(2*dr) + h_w ]
    Only the nodes with v0 > 0 (axis ... last node before the wall) are
    marched in z.

HOW TO RUN
    python example_4_2_ltr_nonisothermal.py     (numpy, scipy, matplotlib)

UNITS
    Lengths in m, velocities in m/s, D and alpha in m^2/s, rate constants in
    1/s, enthalpies in kJ/kmol, cp in kJ/(kmol K), k in kW/(m K), h_w in
    kW/(m^2 K), molar flows in kmol/s, concentrations in kmol/m^3.

INDEXING NOTE
    Python arrays start at 0: node 0 is the axis (r = 0) and node N_r is the
    wall (r = R). The MATLAB version of this script uses 1 ... N_r+1.
"""

import time
import numpy as np
import scipy.sparse as sp
from scipy.integrate import solve_ivp, trapezoid
import matplotlib.pyplot as plt

# =============================================================================
# 1) PARAMETERS
# =============================================================================

# --- Kinetic parameters (Arrhenius, Sec. 1.5) ---
Rg = 8.314                            # gas constant (NB: R is the tube radius) [kJ/(kmol K)]
k0_1, Ea1 = 1.47e5, 90.98e3           # R1: [1/s], [kJ/kmol]
k0_2, Ea2 = 9.02e10, 207.99e3         # R2: [1/s], [kJ/kmol]


def k1(T):
    """Rate constant of R1 as a function of T (Arrhenius)   [1/s]."""
    return k0_1*np.exp(-Ea1/(Rg*T))


def k2(T):
    """Rate constant of R2 as a function of T (Arrhenius)   [1/s]."""
    return k0_2*np.exp(-Ea2/(Rg*T))


# --- Thermal parameters (Sec. 2.2) ---
cp   = 58.0          # molar heat capacity of the feed mixture       [kJ/(kmol K)]
dHr1 = 124e3         # reaction enthalpy, R1 (> 0: endothermic)      [kJ/kmol]
dHr2 = 105e3         # reaction enthalpy, R2 (> 0: endothermic)      [kJ/kmol]

# --- Operating conditions (Sec. 1.6), same as Sections 2-3 ---
T0     = 900.0                    # feed temperature                 [K]
P      = 1e5                      # pressure (isobaric)              [Pa]
F_A0   = 1/3600                   # inlet molar flow of EB, 1 kmol/h [kmol/s]
F_W0   = 9/3600                   # inlet molar flow of steam        [kmol/s]
F_tot0 = F_A0 + F_W0              # total inlet molar flow           [kmol/s]
y_A0   = F_A0/F_tot0              # inlet mole fraction of EB (0.10) [-]
epsA   = y_A0*1                   # expansion factor (delta_nu = +1) [-]
Q0     = F_tot0*1e3*Rg*T0/P       # total inlet volumetric flow      [m^3/s]
C_tot0 = P/(1e3*Rg*T0)            # inlet total concentration P/RT0  [kmol/m^3]
C_A0   = y_A0*C_tot0              # inlet concentration of EB        [kmol/m^3]
V_tot  = 0.30                     # total volume (as the PFR, Sec. 3) [m^3]

# --- Geometry of the multitubular reactor ---
R     = 0.015                     # tube radius                      [m]
vbar0 = 1.0                       # mean inlet velocity per tube     [m/s]
tau   = V_tot/Q0                  # mean residence time              [s]
L     = vbar0*tau                 # tube length                      [m]
N_t   = Q0/(np.pi*R**2*vbar0)     # number of tubes                  [-]

# --- Transport properties ---
D      = 1e-4                     # mass diffusivity (all species)   [m^2/s]
alpha  = D                        # thermal diffusivity: Lewis number = 1 (stated
                                  # approximation, Sec. 4.4 of the document)
k_cond = alpha*C_tot0*cp          # thermal conductivity k = alpha*C_tot0*cp [kW/(m K)]
                                  # (~0.078 W/(m K), close to steam at 900 K)

# --- The two cases: adiabatic and heated wall ---
# h_w is chosen so that 2*h_w/R = 2 kW/(m^3 K): the SAME volumetric
# coefficient Ua_v used for the PFR of Section 3.2.
cases = [dict(name="Adiabatic wall", h_w=0.0,   T_a=T0),
         dict(name="Heated wall",    h_w=0.015, T_a=1000.0)]   # [kW/(m^2 K)], [K]

# --- Radial grid (common to both methods) ---
N_r = 40                          # number of radial INTERVALS
r   = np.linspace(0.0, R, N_r + 1)   # nodes: r[0] = 0 (axis) ... r[N_r] = R (wall)
dr  = R/N_r
v0  = 2*vbar0*(1 - (r/R)**2)      # Poiseuille profile at the nodes
v0[-1] = 0.0                      # exactly zero at the wall
n   = N_r                         # number of MARCHED nodes per variable

print(f"Non-isothermal LTR: N_t = {N_t:.0f} tubes, R = {R:.3f} m, "
      f"L = {L:.3f} m, N_r = {N_r}")
print(f"  thermal conductivity k = {1e3*k_cond:.3f} W/(m K), Biot number "
      f"h_w*R/k = {cases[1]['h_w']*R/k_cond:.2f} (heated case)\n")

# =============================================================================
# LOCAL HELPER FUNCTIONS
# =============================================================================

def cfac(X, T):
    """Concentration factor c = C_A/C_A0 = (1 - X_A)/((1 + epsA*X_A)*T/T0)."""
    return (1 - X)/((1 + epsA*X)*(T/T0))


def wall_values(X, Y, T, h_w, T_a):
    """Wall values from BC3 with a one-sided, second-order difference
        du/dr|_R ~ (3*u_N - 4*u_{N-1} + u_{N-2})/(2*dr)
    X_A, Y_B: du/dr = 0            ->  u_N = (4*u_{N-1} - u_{N-2})/3
    T: -k*dT/dr = h_w*(T_N - T_a)  ->  T_N = [k*(4*T_{N-1} - T_{N-2})/(2*dr) + h_w*T_a]
                                             / [3*k/(2*dr) + h_w]
    X, Y, T: values at the marched nodes (last axis); the last two entries are
    the two nodes before the wall. Works on 1-D and 2-D arrays."""
    a  = k_cond/(2*dr)
    Xw = (4*X[..., -1] - X[..., -2])/3
    Yw = (4*Y[..., -1] - Y[..., -2])/3
    Tw = (a*(4*T[..., -1] - T[..., -2]) + h_w*T_a)/(3*a + h_w)
    return Xw, Yw, Tw


def lap_fd(u):
    """Radial Laplacian (1/r) d/dr (r du/dr) at the marched nodes 0..n-1,
    given the values u at ALL nodes 0..n (wall included).
    Axis (BC2): (1/r) du/dr -> d2u/dr2 (L'Hopital), Laplacian = 2*d2u/dr2;
    with the symmetric ghost node u(-dr) = u(+dr) it is 4*(u_1 - u_0)/dr^2."""
    Lu = np.empty(n)
    Lu[0] = 4*(u[1] - u[0])/dr**2
    i = np.arange(1, n)
    Lu[i] = (u[i + 1] - 2*u[i] + u[i - 1])/dr**2 + (u[i + 1] - u[i - 1])/(2*r[i]*dr)
    return Lu


def rhs_ltr(z, U, h_w, T_a):
    """dU/dz for the marched nodes (method of lines), U = [X_A; Y_B; T]:
        v0_i dX_i/dz = D*(Lap X)_i     + (k1+k2)*c
        v0_i dY_i/dz = D*(Lap Y)_i     + k1*c
        v0_i dT_i/dz = alpha*(Lap T)_i + (y_A0/cp)*[(-dHr1)k1 + (-dHr2)k2]*c"""
    X, Y, T = U[:n], U[n:2*n], U[2*n:]
    Xw, Yw, Tw = wall_values(X, Y, T, h_w, T_a)          # BC3 (slaved wall)
    LX = lap_fd(np.append(X, Xw))
    LY = lap_fd(np.append(Y, Yw))
    LT = lap_fd(np.append(T, Tw))
    c  = cfac(X, T)
    a1, a2 = k1(T), k2(T)
    sX = (a1 + a2)*c
    sY = a1*c
    sT = (y_A0/cp)*(-dHr1*a1 - dHr2*a2)*c
    v  = v0[:n]
    return np.concatenate([(D*LX + sX)/v, (D*LY + sY)/v, (alpha*LT + sT)/v])


def full_fields(U, h_w, T_a):
    """From the marched unknowns (one ROW per z) to the full fields on all
    N_r+1 nodes, appending the wall values given by BC3."""
    U = np.atleast_2d(U)
    X, Y, T = U[:, :n], U[:, n:2*n], U[:, 2*n:]
    Xw, Yw, Tw = wall_values(X, Y, T, h_w, T_a)
    return (np.column_stack([X, Xw]), np.column_stack([Y, Yw]),
            np.column_stack([T, Tw]))


def march_ftcs(f, U0, nz):
    """Explicit Euler in z:  U^{k+1} = U^k + dz*f(U^k). Returns (U_end, ok);
    ok = False if the solution blows up."""
    dz = L/nz
    U = U0.copy()
    for k in range(nz):
        U = U + dz*f(k*dz, U)
        if not np.all(np.isfinite(U)) or np.max(np.abs(U)) > 1e4:
            return U, False
    return U, True


def numerical_jacobian(F, x):
    """Forward finite differences, one column per variable (same scheme as in
    the Newton solver of Example 2.2)."""
    F0 = F(x)
    J = np.zeros((F0.size, x.size))
    for j in range(x.size):
        h = 1e-7*max(1.0, abs(x[j]))
        xh = x.copy()
        xh[j] += h
        J[:, j] = (F(xh) - F0)/h
    return J


def bulk_average(Xs):
    """Flow-averaged ("cup-mixing") value over the cross-section, weighted by
    the local mass (and heat-capacity) flux v0*r:
        Xb = integral(v0*X*r dr) / integral(v0*r dr)   (trapezoidal rule)
    Xs: one ROW per axial position, one COLUMN per radial node."""
    w = v0*r
    return trapezoid(np.atleast_2d(Xs)*w, r, axis=1)/trapezoid(w, r)


# =============================================================================
# 2) SOLUTION OF THE TWO CASES
# =============================================================================
# State vector of the marched system (3n unknowns), stacked by variable:
#   U = [ X_A(r_0..r_{n-1}) ; Y_B(r_0..r_{n-1}) ; T(r_0..r_{n-1}) ]
z  = np.linspace(0.0, L, 301)                       # output grid
U0 = np.concatenate([np.zeros(n), np.zeros(n), T0*np.ones(n)])   # BC1: feed

# Sparsity pattern of the Jacobian: each unknown depends on the three
# variables at its own node and at the two neighbouring nodes
tri = sp.diags([np.ones(n - 1), np.ones(n), np.ones(n - 1)], [-1, 0, 1])
jac_pattern = sp.kron(np.ones((3, 3)), tri)

for case in cases:
    h_w, T_a = case["h_w"], case["T_a"]
    f = lambda zz, U: rhs_ltr(zz, U, h_w, T_a)       # dU/dz = f(U)

    # --- 2a) BDF (solve_ivp): reference solution ---
    # first_step: small, because at z = 0 the wall temperature jumps to its
    # BC3 value while the neighbouring nodes are still at T0.
    bdf_opts = dict(method="BDF", rtol=1e-8, atol=1e-10,
                    jac_sparsity=jac_pattern, first_step=1e-6*L)
    t0 = time.perf_counter()
    sol_steps = solve_ivp(f, (0.0, L), U0, **bdf_opts)          # to count the steps
    case["nsteps_bdf"] = sol_steps.t.size - 1
    sol = solve_ivp(f, (0.0, L), U0, t_eval=z, **bdf_opts)      # on the output grid
    case["t_bdf"] = time.perf_counter() - t0
    case["X"], case["Y"], case["T"] = full_fields(sol.y.T, h_w, T_a)

    # --- 2b) FTCS: explicit Euler in z ---
    # Stability limit dz <= 2/|lambda|_max, with lambda the eigenvalues of the
    # Jacobian of f (at the inlet). The largest |lambda| comes from the nodes
    # next to the wall, where v0 is very small.
    J = numerical_jacobian(lambda U: f(0.0, U), U0)
    lam_max = np.max(np.abs(np.linalg.eigvals(J)))
    case["dz_crit"] = 2/lam_max
    case["nz_ftcs"] = int(np.ceil(1.10*L/case["dz_crit"]))   # 10% margin: lambda changes along z
    t0 = time.perf_counter()
    U_end, case["ftcs_ok"] = march_ftcs(f, U0, case["nz_ftcs"])
    case["t_ftcs"] = time.perf_counter() - t0
    Xf, Yf, Tf = full_fields(U_end, h_w, T_a)
    case["ftcs"] = (bulk_average(Xf)[0], bulk_average(Tf)[0],
                    bulk_average(Yf)[0]/bulk_average(Xf)[0])

    # --- 2c) plug-flow limit (radial diffusion -> infinity) ---
    # Averaging the three PDEs over the cross-section with a flat profile gives
    # the 1-D PFR model of Section 3.2, with Ua_v = 2*h_w/R:
    #   vbar0*dX/dz = (k1+k2)*c,   vbar0*dY/dz = k1*c,
    #   vbar0*dT/dz = (y_A0/cp)*[..]*c + (2*h_w/R)*(T_a - T)/(C_tot0*cp)
    def f_pfr(zz, u):
        X, Y, T = u
        c = cfac(X, T)
        return np.array([(k1(T) + k2(T))*c,
                         k1(T)*c,
                         (y_A0/cp)*(-dHr1*k1(T) - dHr2*k2(T))*c
                         + (2*h_w/R)*(T_a - T)/(C_tot0*cp)])/vbar0
    up = solve_ivp(f_pfr, (0.0, L), [0.0, 0.0, T0], method="BDF",
                   rtol=1e-10, atol=1e-12).y[:, -1]
    case["pfr"] = (up[0], up[2], up[1]/up[0])

# =============================================================================
# 3) RESULTS TABLE
# =============================================================================
for case in cases:
    Xb, Tb, Yb = bulk_average(case["X"]), bulk_average(case["T"]), bulk_average(case["Y"])
    case["Xb"], case["Tb"], case["Yb"] = Xb, Tb, Yb
    print(f"{case['name']} (h_w = {1e3*case['h_w']:.0f} W/(m^2 K), T_a = {case['T_a']:.0f} K)")
    print(f"  {'Method':32s} {'z-steps':>9s} {'X_bulk':>10s} {'T_bulk [K]':>12s} {'S':>10s}")
    print(f"  {'BDF, solve_ivp (reference)':32s} {case['nsteps_bdf']:9d} {Xb[-1]:10.4f} "
          f"{Tb[-1]:12.2f} {Yb[-1]/Xb[-1]:10.4f}")
    if case["ftcs_ok"]:
        print(f"  {'FTCS (explicit Euler)':32s} {case['nz_ftcs']:9d} {case['ftcs'][0]:10.4f} "
              f"{case['ftcs'][1]:12.2f} {case['ftcs'][2]:10.4f}")
    else:
        print(f"  {'FTCS (explicit Euler)':32s} {case['nz_ftcs']:9d}   DIVERGED")
    print(f"  {'plug-flow limit (PFR, Sec. 3.2)':32s} {'-':>9s} {case['pfr'][0]:10.4f} "
          f"{case['pfr'][1]:12.2f} {case['pfr'][2]:10.4f}")
    print(f"  outlet T: centre {case['T'][-1, 0]:.1f} K, wall {case['T'][-1, -1]:.1f} K; "
          f"FTCS step limit dz_crit = {case['dz_crit']:.2e} m; CPU: BDF {case['t_bdf']:.2f} s, "
          f"FTCS {case['t_ftcs']:.2f} s\n")

# =============================================================================
# 4) CONCENTRATIONS OF ALL SPECIES ALONG THE REACTOR (slide 28, BDF solution)
# =============================================================================
# Local yields (stoichiometry, equal diffusivities):
#   EB: Y_A = 1 - X_A;  ST: Y_B;  H2: Y_H2 = Y_B;
#   BZ: Y_BZ = X_A - Y_B;  C2H4: Y_C2H4 = Y_BZ;  W: Y_W = F_W0/F_A0 = 9
# Local concentrations (slide 28):  C_i = C_A0*Y_i/((1 + epsA*X_A)*T/T0)
# CUP-MIXING concentration at position z (what a sample of the whole tube
# outlet at z would contain) = molar flow / volumetric flow:
#   C_i,cup = int( n_i r dr ) / int( v_z r dr )
#           = C_A0*int( v0*Y_i r dr ) / int( v0*(1 + epsA*X_A)*T/T0 r dr )
# since n_i = v0*C_A0*Y_i and v_z = v0*(1 + epsA*X_A)*T/T0 (slide 28).
names = ["EB", "ST", "H$_2$", "BZ", "C$_2$H$_4$", "W"]
w = v0*r                                             # weights v0*r
for case in cases:
    X, Y, T = case["X"], case["Y"], case["T"]
    Yi = np.stack([1 - X, Y, Y, X - Y, X - Y, (F_W0/F_A0)*np.ones_like(X)], axis=-1)
    den = trapezoid(w*(1 + epsA*X)*(T/T0), r, axis=1)            # int v_z r dr
    case["C_cup"] = C_A0*trapezoid(w[None, :, None]*Yi, r, axis=1)/den[:, None]
    Yb_out = trapezoid(w[:, None]*Yi[-1], r, axis=0)/trapezoid(w, r)   # bulk yields
    print(f"{case['name']}: outlet flows [kmol/h] and cup-mixing concentrations [mol/m^3]")
    for s, name in enumerate(names):
        label = name.replace("$", "").replace("_", "")
        print(f"  {label:6s} F = {3600*F_A0*Yb_out[s]:.4f} kmol/h   "
              f"C = {1e3*case['C_cup'][-1, s]:8.3f} mol/m^3")
    print()

# =============================================================================
# 5) PLOTS
# =============================================================================
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), num="Example 4.2 - bulk profiles")
for case in cases:
    ax[0].plot(z, case["Xb"], lw=1.6, label=case["name"])
ax[0].set_xlabel("Axial position z [m]")
ax[0].set_ylabel("Bulk conversion $X_{A,bulk}$ [-]")
ax[0].set_title("Conversion")
ax[0].legend(loc="lower right")
ax[0].grid(True)
for case, col in zip(cases, ["C0", "C1"]):
    ax[1].plot(z, case["Tb"], "-", color=col, lw=1.6, label=f"{case['name']}, bulk")
    ax[1].plot(z, case["T"][:, 0], ":", color=col, lw=1.2, label=f"{case['name']}, centre")
    ax[1].plot(z, case["T"][:, -1], "--", color=col, lw=1.2, label=f"{case['name']}, wall")
ax[1].set_xlabel("Axial position z [m]")
ax[1].set_ylabel("Temperature [K]")
ax[1].set_title("Temperature")
ax[1].legend(fontsize=8)
ax[1].grid(True)
fig.tight_layout()

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), num="Example 4.2 - radial profiles (heated wall)")
for frac in [0.1, 0.25, 0.5, 1.0]:
    i = np.argmin(np.abs(z - frac*L))
    ax[0].plot(r/R, cases[1]["T"][i], lw=1.3, label=f"z/L = {frac:.2f}")
    ax[1].plot(r/R, cases[1]["X"][i], lw=1.3, label=f"z/L = {frac:.2f}")
ax[0].set_xlabel("r/R [-]"); ax[0].set_ylabel("T [K]")
ax[0].set_title("Heated wall: radial temperature profiles"); ax[0].legend(); ax[0].grid(True)
ax[1].set_xlabel("r/R [-]"); ax[1].set_ylabel("$X_A$ [-]")
ax[1].set_title("Heated wall: radial conversion profiles"); ax[1].legend(); ax[1].grid(True)
fig.tight_layout()

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), num="Example 4.2 - species concentrations")
styles = ["-", "-", "--", "-", "--"]                  # dashed: coincident curves
for a, case in zip(ax, cases):
    for s in range(5):
        a.plot(z, 1e3*case["C_cup"][:, s], styles[s], lw=1.5, label=names[s])
    a.set_xlabel("Axial position z [m]")
    a.set_ylabel("Cup-mixing concentration [mol/m$^3$]")
    a.set_title(case["name"])
    a.legend(loc="upper right")
    a.grid(True)
# NB: ST/H2 and BZ/C2H4 coincide (same reaction); steam is printed only.
fig.tight_layout()

plt.show()
