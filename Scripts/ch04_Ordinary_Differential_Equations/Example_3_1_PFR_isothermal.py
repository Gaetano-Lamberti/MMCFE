"""
Example 3.1 PFR isothermal (Python version)
============================================
A guiding example: the styrene reactor
3. The ODE(s) problem(s) - 3.1 Steady-state PFR at constant temperature
(slides 16-18)

NOTATION (as in the slides and in the MATLAB version)
    A = ethylbenzene (EB), B = styrene (ST), C = benzene (BZ)
        R1:  A -> B + H2          rate constant k1
        R2:  A -> C + C2H4        rate constant k2
    Both reactions are irreversible and first order in A.
    X_A = (nA0 - nA)/nA0 is the conversion of A.

PROBLEM (3.1): a single, nonlinear ODE (initial value problem)
        dX_A/dV = (k1 + k2) * (1 - X_A) / ( Vdot0*(1 + epsA*X_A) )
        X_A = 0  @  V = 0
    It comes from the mass balance on A, dnA/dV = -(k1 + k2)*C_A (A), with
    C_A = nA/Vdot(X_A), Vdot(X_A) = Vdot0*(1 + epsA*X_A), and nA = nA0*(1 - X_A)
    (so dnA = -nA0*dX_A).
    The temperature is FIXED (T = 900 K), so k1 and k2 are constants and the
    volumetric flow changes only with composition (no T/T0 factor).

WHAT THIS SCRIPT DOES
    1) Solves (3.1) numerically with the classical 4th-order Runge-Kutta
       method (RK4), fixed step, from V = 0 to V = 0.30 m^3.
    2) Checks it against the ANALYTICAL solution (slide 17, "Note"):
           -epsA*X_A - (1 + epsA)*ln(1 - X_A) = (k1 + k2)/Vdot0 * V
       which is implicit in X_A and is solved at each V by scalar
       Newton-Raphson (the same machinery used for the CSTR in Sec. 2.1).
    3) Recovers the styrene flow nB ALGEBRAICALLY (slide 18): from the ratio
       (B)/(A), dnB/dnA = -k1/(k1 + k2) is constant, hence
           nB = k1/(k1 + k2) * nA0 * X_A
       so only ONE ODE needs to be integrated.
    4) Plots the profiles.

HOW TO RUN
    python example_3_1_pfr_isothermal.py      (requires numpy, matplotlib)

UNITS
    Rate constants in 1/s, volumes in m^3, molar flows in kmol/s, volumetric
    flows in m^3/s. The feed flows of Sec. 1.6 are given in kmol/h and must be
    divided by 3600.
"""

import warnings
import numpy as np
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


def newton_scalar(g, dg, x0, tol, maxiter):
    """Scalar Newton-Raphson: x_{n+1} = x_n - g(x_n)/g'(x_n).
    g, dg   - the function and its derivative
    x0      - initial guess; tol - tolerance on |g|; maxiter - max steps"""
    x = x0
    for it in range(maxiter):
        gx = g(x)
        if abs(gx) < tol:
            return x                    # converged
        x = x - gx/dg(x)                # Newton update
    warnings.warn(f"newton_scalar: no convergence in {maxiter} iterations "
                  f"(|g| = {abs(g(x)):.2e})")
    return x


# =============================================================================
# 1) PARAMETERS
# =============================================================================

# --- Kinetic parameters (Arrhenius, Sec. 1.5) ---
R    = 8.314        # gas constant                                 [kJ/(kmol K)]
k0_1 = 1.47e5       # pre-exponential factor, R1 (A -> B)          [1/s]
Ea1  = 90.98e3      # activation energy, R1                        [kJ/kmol]
k0_2 = 9.02e10      # pre-exponential factor, R2 (A -> C)          [1/s]
Ea2  = 207.99e3     # activation energy, R2                        [kJ/kmol]

# --- Operating conditions (Sec. 1.6) ---
T      = 900.0         # reactor temperature, FIXED (isothermal)      [K]
p      = 1e5           # pressure (isobaric reactor)                  [Pa]
nA0   = 1/3600        # inlet molar flow of A (EB),  1 kmol/h        [kmol/s]
nW0   = 9/3600        # inlet molar flow of steam,   9 kmol/h        [kmol/s]
n0 = nA0 + nW0   # total inlet molar flow                       [kmol/s]

# Expansion factor epsA = y_A0 * delta_nu.
# Each reaction turns 1 mole of A into 2 moles of products, so
# delta_nu = +1 per mole of A converted; with y_A0 = 0.10 -> epsA = 0.10.
# The steam is inert: it dilutes the feed and damps the expansion.
y_A0 = nA0/n0     # mole fraction of A in the feed               [-]
epsA = y_A0*1          # expansion factor                             [-]

# Inlet volumetric flow from the ideal-gas law, Vdot0 = n0*R*T/p.
# The factor 1e3 converts kmol -> mol, because with R in J/(mol K) and
# p in Pa, R*T/p is the molar volume in m^3/mol.
Vdot0 = n0*1e3*R*T/p  # inlet volumetric flow                        [m^3/s]

# --- Rate constants at the fixed temperature ---
# Because T is constant, k1 and k2 are plain NUMBERS here, not functions
# (compare with Example 3.2, where they become functions of T).
k1 = k0_1*np.exp(-Ea1/(R*T))   # [1/s]  (0.7705 1/s at 900 K, slide 18)
k2 = k0_2*np.exp(-Ea2/(R*T))   # [1/s]  (0.0764 1/s at 900 K, slide 18)

# --- Numerical settings ---
V_end = 0.30           # reactor volume to integrate over             [m^3]
h     = 0.01           # RK4 fixed step in V (30 steps)               [m^3]
X_A0  = 0.0            # initial condition: X_A = 0 @ V = 0           [-]

print(f"Isothermal PFR, T = {T:.0f} K")
print(f"  k1 = {k1:.4f} 1/s,  k2 = {k2:.4f} 1/s,  k1+k2 = {k1 + k2:.4f} 1/s")
print(f"  Vdot0 = {Vdot0:.5f} m^3/s,  epsA = {epsA:.2f}\n")

# =============================================================================
# 2) MODEL: right-hand side of the ODE (3.1)
# =============================================================================
# f(V, X_A) = (k1 + k2)*(1 - X_A)/(Vdot0*(1 + epsA*X_A))
# V does not appear explicitly (the ODE is AUTONOMOUS), but it is kept as the
# first argument, as is standard for ODE solvers (same signature as MATLAB's
# ode45 and SciPy's solve_ivp: f(t, y)). NumPy operations are element-wise,
# so f also works on arrays.

def f(V, X_A):
    return (k1 + k2)*(1 - X_A)/(Vdot0*(1 + epsA*X_A))


# =============================================================================
# 3) NUMERICAL SOLUTION: classical RK4 with fixed step
# =============================================================================
# rk4_fixed returns the grid of V values and the solution, one ROW per grid
# point; here the state has one component, so column 0 is X_A.
Vgrid, Y = rk4_fixed(f, 0.0, V_end, X_A0, h)
X_rk4 = Y[:, 0]

print(f"RK4 (h = {h:.3f} m^3, {Vgrid.size - 1} steps): X_A at V = {V_end:.2f} m^3 "
      f"is {X_rk4[-1]:.4f} ({100*X_rk4[-1]:.2f}%)")

# =============================================================================
# 4) CHECK AGAINST THE ANALYTICAL SOLUTION (slide 17, "Note")
# =============================================================================
# The ODE is SEPARABLE. With the partial-fraction split
#       (1 + epsA*X_A)/(1 - X_A) = -epsA + (1 + epsA)/(1 - X_A)
# it integrates in closed form (X_A = 0 at V = 0) to
#       -epsA*X_A - (1 + epsA)*ln(1 - X_A) = (k1 + k2)/Vdot0 * V
# This relation is IMPLICIT in X_A: for each V we solve g(X_A) = 0 with
#       g(X_A)  = -epsA*X_A - (1 + epsA)*ln(1 - X_A) - (k1 + k2)/Vdot0*V
#       g'(X_A) = -epsA + (1 + epsA)/(1 - X_A)   (> 0 on [0,1): unique root)
# by scalar Newton-Raphson. This reference uses NO ODE solver, so it cannot
# share any blind spot with RK4.

def g(X_A, V):
    return -epsA*X_A - (1 + epsA)*np.log(1 - X_A) - (k1 + k2)/Vdot0*V


def dg(X_A):
    return -epsA + (1 + epsA)/(1 - X_A)


X_exact = np.zeros_like(Vgrid)
for i, Vi in enumerate(Vgrid):
    # Initial guess: the RK4 value (very close, Newton converges in 1-3 steps).
    # "lambda X_A, Vi=Vi: ..." freezes the current Vi inside the function.
    X_exact[i] = newton_scalar(lambda X_A, Vi=Vi: g(X_A, Vi), dg, X_rk4[i], 1e-14, 50)

max_err = np.max(np.abs(X_rk4 - X_exact))
print(f"Max |X_A(RK4) - X_A(analytical)| over the grid: {max_err:.1e}")
print("  (RK4 global error is O(h^4): halving h reduces it by about 16)\n")

# =============================================================================
# 5) STYRENE AND BENZENE: obtained algebraically (slide 18)
# =============================================================================
# Mass balance on B: dnB/dV = k1*C_A   (B)
# Ratio (B)/(A): dnB/dnA = k1*C_A / ( -(k1 + k2)*C_A ) = -k1/(k1 + k2)
# C_A cancels, the ratio is CONSTANT (T is fixed), and integrating from the
# inlet (nA = nA0, nB = 0):
#       nB = k1/(k1 + k2) * (nA0 - nA) = k1/(k1 + k2) * nA0 * X_A
# The same argument for C gives nC = k2/(k1 + k2) * nA0 * X_A.
S_B = k1/(k1 + k2)                 # selectivity to styrene (constant)  [-]
nA = nA0*(1 - X_rk4)             # EB molar flow profile              [kmol/s]
nB = S_B*nA0*X_rk4               # ST molar flow profile              [kmol/s]
nC = (1 - S_B)*nA0*X_rk4         # BZ molar flow profile              [kmol/s]

print(f"Selectivity to styrene k1/(k1+k2) = {S_B:.4f} ({100*S_B:.1f}%), constant along V")
print(f"Outlet flows [kmol/h]: EB {3600*nA[-1]:.4f}, ST {3600*nB[-1]:.4f}, "
      f"BZ {3600*nC[-1]:.4f}")

# =============================================================================
# 6) PLOTS
# =============================================================================
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5), num="Example 3.1 - Isothermal PFR")

ax[0].plot(Vgrid, X_rk4, "o", ms=4, label="RK4")
ax[0].plot(Vgrid, X_exact, "-", lw=1.2, label="Analytical")
ax[0].set_xlabel("Reactor volume V [m$^3$]")
ax[0].set_ylabel("Conversion of EB, $X_A$ [-]")
ax[0].set_title("Conversion profile, T = 900 K")
ax[0].legend(loc="lower right")
ax[0].grid(True)

ax[1].plot(Vgrid, 3600*nA, "-", lw=1.2, label="EB ($\dot{n}_A$)")
ax[1].plot(Vgrid, 3600*nB, "-", lw=1.2, label="ST ($\dot{n}_B$)")
ax[1].plot(Vgrid, 3600*nC, "-", lw=1.2, label="BZ ($\dot{n}_C$)")
ax[1].set_xlabel("Reactor volume V [m$^3$]")
ax[1].set_ylabel("Molar flow [kmol/h]")
ax[1].set_title("Molar flow profiles")
ax[1].legend(loc="center right")
ax[1].grid(True)

fig.tight_layout()
plt.show()
