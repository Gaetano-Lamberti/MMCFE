"""
Example 2.2 CSTR with heat exchange (Python version)
=====================================================
A guiding example: the styrene reactor
2. The Algebraic Equation(s) problem(s)
2.2 Steady-state CSTR with heat exchange (1. adiabatic / 2. heated)
(slides 13-15)

NOTATION (as in the slides and in the MATLAB version)
    A = ethylbenzene (EB)
        R1:  EB -> ST + H2        r1 = k1(T)*C_A     (endothermic)
        R2:  EB -> BZ + C2H4      r2 = k2(T)*C_A     (endothermic)
    X_A = (F_A0 - F_A)/F_A0 is the conversion of A.

A NOTE ON TERMINOLOGY
    A steady-state CSTR is ALWAYS isothermal: perfect mixing makes the
    temperature uniform throughout the vessel and equal to the outlet
    temperature T. What distinguishes this example from Example 2.1 is that
    T is no longer FIXED at the feed value T0: it is an UNKNOWN, set by the
    energy balance (heats of reaction plus heat exchange with the
    surroundings, Ua = 0 being the adiabatic limit). Hence "CSTR with heat
    exchange", not "non-isothermal CSTR".

WHAT CHANGES WITH RESPECT TO EXAMPLE 2.1 (slide 13)
    1. The volumetric flow changes with conversion AND temperature
       (ideal gas at constant pressure):   Q(X_A,T) = Q0*(1 + epsA*X_A)*T/T0
    2. Hence the concentration of A too:   C_A(X_A,T) = F_A0*(1 - X_A)/Q(X_A,T)
    3. The kinetic constants depend on T:  k_i(T) = k_i0*exp(-Ea_i/(R*T))
    4. Hence the Damkohler number too:     N_Da(T) = (k1(T) + k2(T))*V/Q0

PROBLEM (2.2): a system of two nonlinear algebraic equations
    System's state:  x = [X_A, T]^T   (column vector)
    Statement:       F(x) = 0,  with x and F in R^2

    F1(X_A,T) = X_A - N_Da(T)*(1 - X_A)/((1 + epsA*X_A)*T/T0) = 0   (2.2.1)
        mass balance on A (dimensionless)

    F2(X_A,T) = (1 - T/T0)
              + { V*[(-dHr1)*k1(T) + (-dHr2)*k2(T)]*C_A(X_A,T)
                  - Ua*(T - Ta) } / (F_tot0*cp*T0)             = 0   (2.2.2)
        energy balance, made dimensionless by dividing the dimensional
        balance F2_dim (a heat flow in kW) by the reference heat flow
        F_tot0*cp*T0 (slide 14). Ua = 0: adiabatic; Ua != 0: heated.

    The two equations are GENUINELY COUPLED: F1 depends on T (through N_Da(T)
    and T/T0) and F2 depends on X_A (through C_A). They must be solved
    SIMULTANEOUSLY, not one after the other.

WHAT THIS SCRIPT DOES
    1) Solves F(x) = 0 by multivariate Newton-Raphson with a numerical
       Jacobian, for the adiabatic and the heated reactor.
    2) Prints the iterates (quadratic convergence) and the heat duties.
    3) Compares with the CSTR at fixed temperature T = T0 of Example 2.1.
    4) Shows the effect of the T/T0 factor in Q (same problem without it).

HOW TO RUN
    python example_2_2_cstr_heat_exchange.py      (requires numpy)

UNITS (important for the energy balance)
    Rate constants in 1/s, V in m^3, so V*k*C_A is in kmol/s: ALL molar flows
    must be in kmol/s (not kmol/h). With enthalpies in kJ/kmol and cp in
    kJ/(kmol K), every term of F2_dim is in kJ/s = kW, and Ua must be in kW/K.

PYTHON vs MATLAB
    The state x is a 1-D NumPy array np.array([X_A, T]); x[0] = X_A and
    x[1] = T (Python indices start at 0, MATLAB's at 1). The linear system
    J*dx = -F, solved in MATLAB with the backslash J\\(-F), is solved here with
    np.linalg.solve(J, -F).
"""

import warnings
import numpy as np


# =============================================================================
# HELPER FUNCTIONS (MATLAB: local functions at the end of the script;
# Python: they must be defined BEFORE they are called)
# =============================================================================

def newton_system(Fvec, x0, Ua, Ta, tol, maxiter):
    """Multivariate Newton-Raphson with a numerical Jacobian (forward finite
    differences). At each iteration it solves the LINEAR system J*dx = -F to
    obtain the increment dx.
    INPUT   Fvec    - residual vector F(x, Ua, Ta), returns a 1-D array
            x0      - initial guess of the unknowns, e.g. [X_A, T]
            Ua, Ta  - physical parameters passed through to Fvec
            tol     - tolerance on max(abs(F)) (stopping criterion)
            maxiter - maximum number of iterations
    OUTPUT  x       - final estimate
            hist    - history of the iterates, one ROW per iteration (row 0 = x0)
            iters   - number of Newton iterations actually performed"""
    x = np.array(x0, dtype=float)       # a (copied) float array, whatever x0 is
    hist = [x.copy()]                   # store the initial guess as the first row
    n = x.size                          # number of unknowns (size of J: n x n)

    for it in range(1, maxiter + 1):
        F0 = Fvec(x, Ua, Ta)            # residuals at the current point (reused below)

        if np.max(np.abs(F0)) < tol:    # converged: largest residual below tolerance
            return x, np.array(hist), it - 1   # point produced by the previous iteration

        # Numerical Jacobian, one COLUMN at a time (forward differences):
        # column j = derivatives of all residuals with respect to x[j]
        J = np.zeros((n, n))
        for j in range(n):
            h = 1e-6*max(1.0, abs(x[j]))    # relative step (absolute if |x_j| < 1):
                                            # X_A ~ 0.5 and T ~ 900 K differ by
                                            # three orders of magnitude
            xh = x.copy()
            xh[j] += h                      # perturb only component j
            J[:, j] = (Fvec(xh, Ua, Ta) - F0)/h   # dF/dx_j

        dx = np.linalg.solve(J, -F0)    # solve J*dx = -F0 (LU factorization;
                                        # never compute inv(J))
        x = x + dx                      # full Newton step (no damping)
        hist.append(x.copy())           # store the new iterate as a row

    # Only reached if the loop ended WITHOUT meeting the stopping criterion
    warnings.warn(f"newton_system: no convergence in {maxiter} iterations "
                  f"(max|F| = {np.max(np.abs(Fvec(x, Ua, Ta))):.3e})")
    return x, np.array(hist), maxiter


def print_history(title_str, hist, Fvec, Ua, Ta):
    """Prints the Newton iterates and the residual norm max|F_i|."""
    print(title_str)
    print(f"{'it':>4s} {'X_A':>12s} {'T [K]':>12s} {'max|F_i|':>12s}")
    for k, xk in enumerate(hist):
        Fk = Fvec(xk, Ua, Ta)
        print(f"{k:4d} {xk[0]:12.6f} {xk[1]:12.3f} {np.max(np.abs(Fk)):12.2e}")
    print()


# =============================================================================
# 1) PARAMETERS
# =============================================================================

# --- Kinetic parameters (Arrhenius, Sec. 1.5) ---
R    = 8.314        # gas constant                                 [kJ/(kmol K)]
k0_1 = 1.47e5       # pre-exponential factor, R1 (A -> ST)         [1/s]
Ea1  = 90.98e3      # activation energy, R1                        [kJ/kmol]
k0_2 = 9.02e10      # pre-exponential factor, R2 (A -> BZ)         [1/s]
Ea2  = 207.99e3     # activation energy, R2                        [kJ/kmol]


# Rate constants as FUNCTIONS of T (slide 13, point 3). NumPy operations are
# element-wise, so they also accept arrays of T.
def k1(T):
    return k0_1*np.exp(-Ea1/(R*T))      # [1/s]


def k2(T):
    return k0_2*np.exp(-Ea2/(R*T))      # [1/s]


# --- Operating conditions (Sec. 1.6) ---
T0     = 900.0         # feed temperature (also the T at which Q0 is given) [K]
P      = 1e5           # pressure (isobaric reactor)                  [Pa]
F_A0   = 1/3600        # inlet molar flow of A (EB),  1 kmol/h        [kmol/s]
F_W0   = 9/3600        # inlet molar flow of steam,   9 kmol/h        [kmol/s]
F_tot0 = F_A0 + F_W0   # total inlet molar flow                       [kmol/s]
y_A0   = F_A0/F_tot0   # mole fraction of A in the feed (0.10)        [-]
epsA   = y_A0*1        # expansion factor (delta_nu = +1 per mole of A) [-]
Q0     = F_tot0*1e3*R*T0/P   # inlet volumetric flow at T0, ideal gas [m^3/s]
                             # (1e3: kmol -> mol, with R in J/(mol K))
V      = 0.250         # reactor volume (same as Example 2.1)         [m^3]

# --- Thermal parameters ---
cp   = 58.0            # molar heat capacity of the (steam-rich) feed  [kJ/(kmol K)]
dHr1 = 124e3           # reaction enthalpy, R1 (> 0, endothermic)     [kJ/kmol]
dHr2 = 105e3           # reaction enthalpy, R2 (> 0, endothermic)     [kJ/kmol]
# Declared simplifications: cp constant (feed composition); the sensible heat
# term uses the inlet molar flow F_tot0 (the moles actually increase with
# X_A); reaction enthalpies independent of T.

# --- Heat exchange: the two cases (slide title: 1. adiabatic / 2. heated) ---
# Ua = U*a: heat-exchange coefficient times heat-exchange surface area.
Ua_adiab, Ta_adiab = 0.0, T0          # case 1: adiabatic (Ta irrelevant)
Ua_heat,  Ta_heat  = 0.20, 1000.0     # case 2: heated, Ta > T0        [kW/K], [K]

# --- Numerical settings ---
tol     = 1e-10                  # tolerance on max|F_i| (both residuals are O(1))
maxiter = 50                     # maximum number of Newton iterations
x0      = np.array([0.49, 900.0])   # initial guess [X_A, T]: the solution at T = T0

# =============================================================================
# 2) MODEL: the residual vector F(x), with x = [X_A, T]
# =============================================================================
# In MATLAB these are anonymous functions (@), which "freeze" the parameter
# values when they are defined. Python functions instead read the global
# parameters when they are CALLED: if a parameter is changed afterwards, the
# functions see the new value.

def Q_fun(X_A, T):
    """Slide 13, point 1: volumetric flow, expanded by composition AND
    temperature. In Example 2.1 the factor T/T0 was identically 1; here T is
    an unknown."""
    return Q0*(1 + epsA*X_A)*(T/T0)                         # [m^3/s]


def CA_fun(X_A, T):
    """Slide 13, point 2: concentration of A."""
    return F_A0*(1 - X_A)/Q_fun(X_A, T)                     # [kmol/m^3]


def NDa_fun(T):
    """Slide 13, point 4: Damkohler number, referred to the INLET flow Q0.
    The expansion effects are NOT inside N_Da: they appear explicitly in F1."""
    return (k1(T) + k2(T))*V/Q0                             # [-]


def F1(X_A, T):
    """(2.2.1) Mass balance on A, dimensionless residual, O(1).
    Derived from F_A0*X_A = (k1 + k2)*C_A*V, substituting C_A and dividing by
    F_A0. T enters TWICE: through N_Da(T) (kinetics) and through T/T0 (gas
    expansion)."""
    return X_A - NDa_fun(T)*(1 - X_A)/((1 + epsA*X_A)*(T/T0))


def F2_dim(X_A, T, Ua, Ta):
    """Energy balance, DIMENSIONAL form (slide 14), every term in kW:
      F_tot0*cp*(T0 - T)   sensible heat released by the feed cooling from
                           T0 to T (positive if T < T0)
      V*[...]*C_A          heat "generated" by the reactions: NEGATIVE, since
                           both are endothermic (they CONSUME heat)
      -Ua*(T - Ta)         heat received from the heating medium (positive
                           if Ta > T); Ua = 0 -> adiabatic
    Ua and Ta are ARGUMENTS, so the same function serves both cases."""
    return (F_tot0*cp*(T0 - T)
            + V*((-dHr1)*k1(T) + (-dHr2)*k2(T))*CA_fun(X_A, T)
            - Ua*(T - Ta))                                   # [kW]


# (2.2.2) Energy balance, DIMENSIONLESS form (slide 14).
# F1 is O(1), while F2_dim is O(10) kW: without scaling, the stopping test
# max|F_i| < tol would be dominated by the energy balance, and tol would mean
# different things for the two equations. Dividing by the reference heat flow
# F_tot0*cp*T0 gives F2 = (1 - T/T0) + ..., i.e. the energy balance expressed
# as a fraction of T0: |F2| < tol means an error of about tol*T0 on T.
# Scaling changes neither the solution nor (in exact arithmetic) the Newton
# iterates; it changes the meaning of the stopping criterion.
Qref = F_tot0*cp*T0                                          # [kW]


def F2(X_A, T, Ua, Ta):
    return F2_dim(X_A, T, Ua, Ta)/Qref                       # [-]


def Fvec(x, Ua, Ta):
    """(2.2) Residual vector F(x): packs F1 and F2 into the array expected by
    the solver and maps the physical notation (X_A, T) onto the state
    x = [x[0], x[1]]."""
    return np.array([F1(x[0], x[1]),
                     F2(x[0], x[1], Ua, Ta)])


# =============================================================================
# 3) SOLUTION: multivariate Newton-Raphson, two cases
# =============================================================================
# The scalar update of Example 2.1 generalizes to
#   J(x_n)*dx = -F(x_n),   x_{n+1} = x_n + dx
# where J is the 2x2 Jacobian matrix of partial derivatives dF_i/dx_j.
# Each iteration solves a 2x2 LINEAR system. J is evaluated by forward
# finite differences inside newton_system (differentiating F2 by hand through
# k1(T), k2(T) is tedious and error-prone, with no benefit).

# --- Case 1: adiabatic reactor (Ua = 0) ---
xA, histA, _ = newton_system(Fvec, x0, Ua_adiab, Ta_adiab, tol, maxiter)
print_history("Case 1 - adiabatic (Ua = 0)", histA, Fvec, Ua_adiab, Ta_adiab)

# --- Case 2: heated reactor (Ua > 0, Ta > T0) ---
xH, histH, _ = newton_system(Fvec, x0, Ua_heat, Ta_heat, tol, maxiter)
print_history(f"Case 2 - heated (Ua = {Ua_heat:.2f} kW/K, Ta = {Ta_heat:.0f} K)",
              histH, Fvec, Ua_heat, Ta_heat)

# --- Heat duties at the solution of Case 2 (diagnostics, kW) ---
# At the solution the three terms of F2_dim must sum to zero.
Xs, Ts = xH
q_sens  = F_tot0*cp*(T0 - Ts)
q_react = V*((-dHr1)*k1(Ts) + (-dHr2)*k2(Ts))*CA_fun(Xs, Ts)
q_exch  = -Ua_heat*(Ts - Ta_heat)
print(f"Case 2 heat duties [kW]: sensible {q_sens:+.3f}, reaction {q_react:+.3f}, "
      f"exchange {q_exch:+.3f}, sum {q_sens + q_react + q_exch:.1e}\n")

# --- Comparison with the CSTR at fixed temperature (Example 2.1) ---
# Reference at T = T0: analytical root of epsA*X^2 + (1 + N_Da)*X - N_Da = 0
NDa0  = NDa_fun(T0)
X_iso = (-(1 + NDa0) + np.sqrt((1 + NDa0)**2 + 4*epsA*NDa0))/(2*epsA)

print(f"{'Case':26s} {'X_A [%]':>10s} {'T [K]':>10s}")
print(f"{'T = T0 (Example 2.1)':26s} {100*X_iso:10.2f} {T0:10.1f}")
print(f"{'Adiabatic':26s} {100*xA[0]:10.2f} {xA[1]:10.1f}")
print(f"{'Heated':26s} {100*xH[0]:10.2f} {xH[1]:10.1f}\n")
# Physical reading: without an external heat supply the endothermic reactions
# cool the reactor down and self-limit (conversion drops from about 49% to
# about 29% at the SAME volume). With heat supplied (Ta > T0), temperature and
# conversion recover above the value at T = T0.

# =============================================================================
# 4) WHY THE T/T0 FACTOR MATTERS: the same problem WITHOUT it
# =============================================================================
# For comparison only: residuals re-written as if the volumetric flow did not
# change with temperature (Q = Q0*(1 + epsA*X_A)). Names end in "_noT" so that
# the correct model above is left untouched.

def CA_noT(X_A):
    return F_A0*(1 - X_A)/(Q0*(1 + epsA*X_A))


def F1_noT(X_A, T):
    return X_A - NDa_fun(T)*(1 - X_A)/(1 + epsA*X_A)


def F2_noT(X_A, T, Ua, Ta):
    return (F_tot0*cp*(T0 - T)
            + V*((-dHr1)*k1(T) + (-dHr2)*k2(T))*CA_noT(X_A) - Ua*(T - Ta))/Qref


def Fvec_noT(x, Ua, Ta):
    return np.array([F1_noT(x[0], x[1]), F2_noT(x[0], x[1], Ua, Ta)])


xA_noT, _, _ = newton_system(Fvec_noT, x0, Ua_adiab, Ta_adiab, tol, maxiter)
xH_noT, _, _ = newton_system(Fvec_noT, x0, Ua_heat,  Ta_heat,  tol, maxiter)

print("Effect of T/T0 in Q(X_A,T)")
print(f"{'Case':12s} {'X_A no T/T0 [%]':>16s} {'X_A with [%]':>14s} "
      f"{'T no T/T0 [K]':>16s} {'T with [K]':>14s}")
print(f"{'Adiabatic':12s} {100*xA_noT[0]:16.2f} {100*xA[0]:14.2f} "
      f"{xA_noT[1]:16.2f} {xA[1]:14.2f}")
print(f"{'Heated':12s} {100*xH_noT[0]:16.2f} {100*xH[0]:14.2f} "
      f"{xH_noT[1]:16.2f} {xH[1]:14.2f}")
# In the adiabatic case the correction RAISES the conversion: the cooled gas
# contracts, the volumetric flow falls, and both the residence time and C_A
# increase. In the heated case T > T0 and the effect reverses (smaller,
# because T departs from T0 by only about 7 K).
