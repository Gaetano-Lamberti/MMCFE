"""
Example 3.2 PFR non-isothermal (Python version)
================================================
A guiding example: the styrene reactor
3. The ODE(s) problem(s) - 3.2 Steady-state PFR with heat exchange
(slides 19-20)

NOTATION (as in the slides and in the MATLAB version)
    A = ethylbenzene (EB), B = styrene (ST), C = benzene (BZ)
        R1:  A -> B + H2          rate constant k1(T), endothermic
        R2:  A -> C + C2H4        rate constant k2(T), endothermic
    X_A = (nA0 - nA)/nA0 is the conversion of A.

PROBLEM (3.2): a system of two nonlinear ODEs (initial value problem)
    System's state:  x = [X_A, T]^T
        dx/dV = f(x),   x = x0 = [0, T0]  @  V = 0
    with
    f1(X_A,T) = (k1(T) + k2(T)) * (1 - X_A) / ( Vdot0*(1 + epsA*X_A)*T/T0 )
    f2(X_A,T) = { [(-dHr1)*k1(T) + (-dHr2)*k2(T)]*C_A(X_A,T)
                  - UaV*(T - Tc) } / ( ndot(X_A)*Cp )
    C_A(X_A,T) = nA0*(1 - X_A) / ( Vdot0*(1 + epsA*X_A)*T/T0 )
    ndot(X_A) = n0*(1 + epsA*X_A)

    Compared with Example 3.1:
    - k1 and k2 are now FUNCTIONS of T (Arrhenius): the temperature couples
      the two equations;
    - the volumetric flow Vdot = Vdot0*(1 + epsA*X_A)*T/T0 contains the factor T/T0
      (ideal gas at constant pressure), which was identically 1 in the
      isothermal case. T therefore enters the mass balance TWICE: through the
      kinetics and through the gas expansion.

WHAT THIS SCRIPT DOES
    1) Solves (3.2) with the classical RK4 method, fixed step, for two cases:
       adiabatic (UaV = 0) and with heat exchange (UaV > 0).
    2) Recovers the styrene flow nB by integrating dnB/dV = k1(T)*C_A along
       the computed profile. (Unlike Example 3.1, the selectivity is NOT
       constant here, because k1/(k1+k2) depends on T.)
    3) Stiffness check: eigenvalues of the Jacobian of f along the adiabatic
       profile.
    4) Step-size test: the same problem with a much larger step.
    5) Plots the profiles, with the isothermal PFR as reference.

HOW TO RUN
    python example_3_2_pfr_nonisothermal.py   (requires numpy, scipy, matplotlib)

UNITS (important for the energy balance)
    Rate constants in 1/s, volumes in m^3, molar flows in kmol/s, enthalpies
    in kJ/kmol, Cp in kJ/(kmol K), UaV in kW/(m^3 K). Then every term in the
    numerator of f2 is in kW/m^3, the denominator ndot*Cp is in kW/K, and
    dT/dV is in K/m^3 as it should be. The feed flows of Sec. 1.6 are given
    in kmol/h: divide by 3600.

PYTHON vs MATLAB
    The state x is a 1-D NumPy array: x[0] = X_A, x[1] = T (Python indices
    start at 0). MATLAB's cumtrapz -> scipy.integrate.cumulative_trapezoid,
    fzero -> scipy.optimize.brentq (both need a bracketing interval).
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import brentq
import matplotlib.pyplot as plt


# =============================================================================
# HELPER FUNCTIONS (MATLAB: local functions at the end of the script;
# Python: they must be defined BEFORE they are called)
# =============================================================================

def rk4_fixed(f, t0, tend, y0, h):
    """Classical 4th-order Runge-Kutta with a fixed step.
    Solves y' = f(t, y), y(t0) = y0, on [t0, tend]. Each step uses four
    evaluations of the slope:
        K1 = f(t,       y)
        K2 = f(t + h/2, y + h/2*K1)
        K3 = f(t + h/2, y + h/2*K2)
        K4 = f(t + h,   y + h*K3)
        y_new = y + h/6*(K1 + 2*K2 + 2*K3 + K4)
    Global error O(h^4); only f itself is needed (no derivatives of f).
    Works for a scalar ODE and for a system (y and f(t,y) 1-D arrays).
    OUTPUT: t (grid points), Y (one ROW per grid point)."""
    N = int(round((tend - t0)/h))       # number of steps (h must divide the interval)
    t = t0 + np.arange(N + 1)*h         # grid
    y = np.atleast_1d(np.array(y0, dtype=float))   # state as a 1-D array
    Y = np.zeros((N + 1, y.size))       # preallocate the solution
    Y[0] = y                            # initial condition in the first row
    for n in range(N):
        K1 = f(t[n],       y)
        K2 = f(t[n] + h/2, y + h/2*K1)
        K3 = f(t[n] + h/2, y + h/2*K2)
        K4 = f(t[n] + h,   y + h*K3)
        y  = y + h/6*(K1 + 2*K2 + 2*K3 + K4)
        Y[n + 1] = y                    # store as a row
    return t, Y


def numjac_fd(F, x):
    """Jacobian dF/dx by forward finite differences, column by column.
    Column j contains the derivatives of all components of F with respect to
    x[j]. The step is RELATIVE to |x_j| (absolute 1e-6 if |x_j| < 1), which
    matters here because X_A ~ 0.1-1 while T ~ 900 K."""
    x  = np.array(x, dtype=float)
    F0 = F(x)
    n  = x.size
    J  = np.zeros((n, n))
    for j in range(n):
        hj = 1e-6*max(1.0, abs(x[j]))   # relative step
        xh = x.copy()
        xh[j] += hj                     # perturb only component j
        J[:, j] = (F(xh) - F0)/hj       # column j = dF/dx_j
    return J


# =============================================================================
# 1) PARAMETERS
# =============================================================================

# --- Kinetic parameters (Arrhenius, Sec. 1.5) ---
R    = 8.314        # gas constant                                 [kJ/(kmol K)]
k0_1 = 1.47e5       # pre-exponential factor, R1 (A -> B)          [1/s]
Ea1  = 90.98e3      # activation energy, R1                        [kJ/kmol]
k0_2 = 9.02e10      # pre-exponential factor, R2 (A -> C)          [1/s]
Ea2  = 207.99e3     # activation energy, R2                        [kJ/kmol]


# Rate constants as FUNCTIONS of T. NumPy operations are element-wise, so
# they also accept arrays of T.
def k1(T):
    return k0_1*np.exp(-Ea1/(R*T))      # [1/s]


def k2(T):
    return k0_2*np.exp(-Ea2/(R*T))      # [1/s]
# Note: Ea2 >> Ea1, so k2 grows much faster with T than k1: higher T favours
# the side reaction (lower selectivity to styrene).


# --- Operating conditions (Sec. 1.6) ---
T0     = 900.0         # feed temperature (also the T at which Vdot0 is given) [K]
p      = 1e5           # pressure (isobaric reactor)                  [Pa]
nA0   = 1/3600        # inlet molar flow of A (EB),  1 kmol/h        [kmol/s]
nW0   = 9/3600        # inlet molar flow of steam,   9 kmol/h        [kmol/s]
n0 = nA0 + nW0   # total inlet molar flow                       [kmol/s]
y_A0   = nA0/n0   # mole fraction of A in the feed (0.10)        [-]
epsA   = y_A0*1        # expansion factor (delta_nu = +1 per mole of A) [-]
Vdot0     = n0*1e3*R*T0/p   # inlet volumetric flow, ideal gas       [m^3/s]
                             # (1e3: kmol -> mol, with R in J/(mol K))

# --- Thermal parameters ---
Cp   = 58.0            # molar heat capacity of the (steam-rich) mixture [kJ/(kmol K)]
dHr1 = 124e3           # reaction enthalpy, R1 (> 0, endothermic)     [kJ/kmol]
dHr2 = 105e3           # reaction enthalpy, R2 (> 0, endothermic)     [kJ/kmol]
# Declared simplifications: Cp constant (feed composition), reaction
# enthalpies independent of T, pressure drop neglected.

# --- Heat exchange: the two cases ---
# UaV is a VOLUMETRIC heat-transfer coefficient [kW/(m^3 K)]: a jacketed
# tube exchanges heat continuously along its length, so the exchange term is
# written per unit reactor volume.
UaV_adiab, Tc_adiab = 0.0, T0          # case 1: adiabatic (Tc irrelevant)
UaV_heat,  Tc_heat  = 2.0, 1000.0      # case 2: heated jacket, Tc > T0

# --- Numerical settings ---
V_end = 0.30                     # reactor volume to integrate over    [m^3]
h     = 0.001                    # RK4 fixed step in V (300 steps)     [m^3]
x0    = np.array([0.0, T0])      # initial state: X_A = 0 and T = T0 @ V = 0

# =============================================================================
# 2) MODEL: right-hand side f(x) of the ODE system (3.2)
# =============================================================================
# All quantities are written as functions of the state (X_A, T), exactly as
# in the slides. (In MATLAB they are anonymous functions, which "freeze" the
# parameter values when defined; Python functions read the parameters when
# they are CALLED.)

def Vdot_fun(X_A, T):
    """Local volumetric flow: expands with composition AND temperature."""
    return Vdot0*(1 + epsA*X_A)*(T/T0)                         # [m^3/s]


def CA_fun(X_A, T):
    """Local concentration of A."""
    return nA0*(1 - X_A)/Vdot_fun(X_A, T)                     # [kmol/m^3]


def ndot_fun(X_A):
    """Total molar flow: increases with conversion (1 mole A -> 2 moles)."""
    return n0*(1 + epsA*X_A)                            # [kmol/s]


def f1(X_A, T):
    """Mass balance on A written for X_A.
    From dnA/dV = -(k1 + k2)*C_A with dnA = -nA0*dX_A:
      dX_A/dV = (k1 + k2)*C_A/nA0 = (k1 + k2)*(1 - X_A)/(Vdot0*(1+epsA*X_A)*T/T0)"""
    return (k1(T) + k2(T))*CA_fun(X_A, T)/nA0              # [1/m^3]


def f2(X_A, T, UaV, Tc):
    """Energy balance. Numerator: heat "generated" by the reactions per unit
    volume (NEGATIVE, both reactions are endothermic: they consume heat) plus
    heat received from the jacket (positive if Tc > T). Denominator: heat
    capacity flow of the gas mixture. UaV and Tc are ARGUMENTS, so the same
    f2 serves both cases."""
    return (((-dHr1)*k1(T) + (-dHr2)*k2(T))*CA_fun(X_A, T)
            - UaV*(T - Tc))/(ndot_fun(X_A)*Cp)              # [K/m^3]


def f(V, x, UaV, Tc):
    """f(x): packs f1 and f2 into the array expected by the solver, mapping the
    physical notation (X_A, T) onto the state x = [x[0], x[1]]. V is kept as
    first argument (standard ODE-solver signature), although the system is
    autonomous."""
    return np.array([f1(x[0], x[1]),
                     f2(x[0], x[1], UaV, Tc)])


# =============================================================================
# 3) NUMERICAL SOLUTION: RK4 for the two cases
# =============================================================================
# rk4_fixed expects f(V, x): the case-specific parameters are fixed by
# wrapping f in a new (lambda) function of (V, x) only.
Vgrid, Xad = rk4_fixed(lambda V, x: f(V, x, UaV_adiab, Tc_adiab), 0.0, V_end, x0, h)
_,     Xht = rk4_fixed(lambda V, x: f(V, x, UaV_heat,  Tc_heat),  0.0, V_end, x0, h)
# Each solution has one ROW per grid point: column 0 = X_A, column 1 = T.

# --- Styrene flow nB along the reactor (post-processing) ---
# Mass balance on B: dnB/dV = k1(T)*C_A(X_A,T). The right-hand side is known
# at every grid point once X_A(V) and T(V) are known, so nB follows by
# numerical QUADRATURE (cumulative trapezoidal rule), with nB = 0 at V = 0.
# With h = 0.001 m^3 the quadrature error is negligible.
# (In Example 3.1 the ratio dnB/dnA = -k1/(k1+k2) was constant; here it
# depends on T, so nB cannot be obtained algebraically from X_A.)
FB_ad = cumulative_trapezoid(k1(Xad[:, 1])*CA_fun(Xad[:, 0], Xad[:, 1]), Vgrid, initial=0)
FB_ht = cumulative_trapezoid(k1(Xht[:, 1])*CA_fun(Xht[:, 0], Xht[:, 1]), Vgrid, initial=0)

# Overall selectivity to styrene at the outlet: moles of B formed per mole of
# A converted.
S_ad = FB_ad[-1]/(nA0*Xad[-1, 0])
S_ht = FB_ht[-1]/(nA0*Xht[-1, 0])

# --- Isothermal reference (Example 3.1): analytical solution at V_end ---
# -epsA*X - (1 + epsA)*ln(1 - X) = (k1 + k2)/Vdot0*V, solved by brentq
# (the Python counterpart of fzero with a bracketing interval).
kiso  = k1(T0) + k2(T0)
X_iso = brentq(lambda X: -epsA*X - (1 + epsA)*np.log(1 - X) - kiso/Vdot0*V_end, 0, 0.99)
S_iso = k1(T0)/kiso

# --- Results table ---
print(f"Non-isothermal PFR, V = {V_end:.2f} m^3, RK4 with h = {h:.3f} m^3\n")
print(f"{'Case':40s} {'X_A':>8s} {'T out [K]':>12s} {'Selectivity':>12s}")
print(f"{'Isothermal (Example 3.1)':40s} {X_iso:8.3f} {T0:12.1f} {S_iso:12.3f}")
print(f"{'Adiabatic (UaV = 0)':40s} {Xad[-1, 0]:8.3f} {Xad[-1, 1]:12.1f} {S_ad:12.3f}")
print(f"{f'Heat exchange (UaV = {UaV_heat:g}, Tc = {Tc_heat:g} K)':40s} "
      f"{Xht[-1, 0]:8.3f} {Xht[-1, 1]:12.1f} {S_ht:12.3f}\n")
# Physical reading: the adiabatic reactor cools down (endothermic reactions)
# and self-limits: lower conversion, but higher selectivity. With heat
# exchange T rises: higher conversion, lower selectivity (Ea2 >> Ea1 favours
# the side reaction at high T).

# =============================================================================
# 4) STIFFNESS CHECK: eigenvalues of the Jacobian along the adiabatic run
# =============================================================================
# A system is "stiff" when its Jacobian has eigenvalues of very different
# magnitude AND the fast one forces an explicit method to take tiny steps.
# What matters for stability of an explicit method is h*|lambda_fast|
# relative to the method's stability region (for RK4, roughly h*|lambda| < 2.8
# on the negative real axis), not the ratio alone.
print("Stiffness check (adiabatic case)")
print(f"{'V':>8s} {'T [K]':>10s} {'X_A':>8s} {'lambda_1':>12s} {'lambda_2':>12s} {'ratio':>8s}")
for Vcheck in [0.0, 0.15, 0.30]:
    i = np.argmin(np.abs(Vgrid - Vcheck))               # closest grid point
    xi = Xad[i]
    J = numjac_fd(lambda x: f(0.0, x, UaV_adiab, Tc_adiab), xi)
    lam = np.linalg.eigvals(J)
    ratio = np.max(np.abs(lam))/np.min(np.abs(lam))
    print(f"{Vgrid[i]:8.2f} {xi[1]:10.1f} {xi[0]:8.3f} {lam[0].real:12.4f} "
          f"{lam[1].real:12.4f} {ratio:8.1f}")
print()

# =============================================================================
# 5) STEP-SIZE TEST: is an explicit method adequate?
# =============================================================================
# Same adiabatic problem with a step 50 times larger (only 6 steps over the
# whole reactor). If the result stays close to the fine-step solution, RK4 is
# stable and accurate here, and no implicit method is needed.
h_big = 0.05
_, Xbig = rk4_fixed(lambda V, x: f(V, x, UaV_adiab, Tc_adiab), 0.0, V_end, x0, h_big)
print(f"Step-size test (adiabatic): h = {h:.3f} -> X_A = {Xad[-1, 0]:.5f}, T = {Xad[-1, 1]:.3f} K")
print(f"                            h = {h_big:.3f} -> X_A = {Xbig[-1, 0]:.5f}, T = {Xbig[-1, 1]:.3f} K\n")

# =============================================================================
# 6) PLOTS
# =============================================================================
# Isothermal conversion profile for reference (analytical, via brentq)
X_iso_prof = np.zeros_like(Vgrid)
for i in range(1, Vgrid.size):
    X_iso_prof[i] = brentq(lambda X: -epsA*X - (1 + epsA)*np.log(1 - X)
                           - kiso/Vdot0*Vgrid[i], 0, 0.99)

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5), num="Example 3.2 - Non-isothermal PFR")

ax[0].plot(Vgrid, X_iso_prof, "k--", lw=1.0, label="Isothermal")
ax[0].plot(Vgrid, Xad[:, 0], "-", lw=1.2, label="Adiabatic")
ax[0].plot(Vgrid, Xht[:, 0], "-", lw=1.2, label="Heat exchange")
ax[0].set_xlabel("Reactor volume V [m$^3$]")
ax[0].set_ylabel("Conversion of EB, $X_A$ [-]")
ax[0].set_title("Conversion profiles")
ax[0].legend(loc="lower right")
ax[0].grid(True)

ax[1].plot(Vgrid, T0*np.ones_like(Vgrid), "k--", lw=1.0, label="Isothermal")
ax[1].plot(Vgrid, Xad[:, 1], "-", lw=1.2, label="Adiabatic")
ax[1].plot(Vgrid, Xht[:, 1], "-", lw=1.2, label="Heat exchange")
ax[1].set_xlabel("Reactor volume V [m$^3$]")
ax[1].set_ylabel("Temperature T [K]")
ax[1].set_title("Temperature profiles")
ax[1].legend(loc="center left")
ax[1].grid(True)

fig.tight_layout()
plt.show()
