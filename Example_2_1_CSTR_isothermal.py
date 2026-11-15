"""
Example 2.1 CSTR isothermal (Python version)
=============================================
A guiding example: the styrene reactor
2. The Algebraic Equation(s) problem(s)
2.1 Steady-state CSTR at fixed temperature   (slides 11-12)

NOTATION (as in the slides and in the MATLAB version)
    A = ethylbenzene (EB)
        R1:  EB -> ST + H2        r1 = k1*C_A
        R2:  EB -> BZ + C2H4      r2 = k2*C_A
    Both reactions are irreversible and first order in A.
    X_A = (F_A0 - F_A)/F_A0 is the conversion of A, so F_A = F_A0*(1 - X_A).
    Q = Q0*(1 + epsA*X_A) is the outlet volumetric flow (the gas expands,
    because each mole of A converted gives two moles of products), so
        C_A = F_A/Q = (F_A0/Q0) * (1 - X_A)/(1 + epsA*X_A)

PROBLEM (2.1): a single, nonlinear algebraic equation (AE)
    Mass balance on A at steady state and constant T:
        0 = F_A0 - F_A - (k1 + k2)*C_A*V
    Substituting C_A and dividing by F_A0:
        f(X_A) = X_A - N_Da * (1 - X_A)/(1 + epsA*X_A) = 0
    with the Damkohler number  N_Da = (k1 + k2)*V/Q0  (dimensionless).
    N_Da compares the reaction rate with the flow rate: N_Da >> 1 means high
    conversion, N_Da << 1 low conversion.
    The volumetric expansion (epsA != 0) is what makes f NONLINEAR in X_A,
    even though the kinetics are first order.

WHAT THIS SCRIPT DOES
    1) Computes k1, k2, Q0 and N_Da at the operating point.
    2) Solves f(X_A) = 0 by scalar Newton-Raphson, printing every iterate to
       show QUADRATIC convergence.
    3) Checks the result against the ANALYTICAL solution (slide 12, Note):
       f = 0 is equivalent to the quadratic
           epsA*X_A^2 + (1 + N_Da)*X_A - N_Da = 0
    4) Checks robustness: a starting guess farther away converges to the
       same root.
    5) Computes the selectivity to styrene and plots f(X_A).

HOW TO RUN
    python example_2_1_cstr_isothermal.py      (requires numpy, matplotlib)

UNITS
    Rate constants in 1/s, volume in m^3, volumetric flow in m^3/s: then
    N_Da = (k1 + k2)*V/Q0 is dimensionless. Molar flows in kmol/s (the feed
    flows of Sec. 1.6 are given in kmol/h and must be divided by 3600).
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# HELPER FUNCTION (in MATLAB it is a local function at the end of the script;
# in Python a function must be defined BEFORE it is called)
# =============================================================================

def newton_scalar(f, df, x0, tol, maxiter):
    """Scalar Newton-Raphson: x_{n+1} = x_n - f(x_n)/f'(x_n).
    INPUT   f, df   - the function and its derivative
            x0      - initial guess
            tol     - tolerance on |f(x)| (stopping criterion)
            maxiter - maximum number of iterations
    OUTPUT  x       - final estimate of the root
            hist    - list of all iterates, hist[0] = x0"""
    x = x0
    hist = [x0]
    for it in range(maxiter):
        fx = f(x)
        if abs(fx) < tol:
            return x, hist                 # converged: residual below tolerance
        x = x - fx/df(x)                   # Newton update
        hist.append(x)                     # store the new iterate
    warnings.warn(f"newton_scalar: no convergence in {maxiter} iterations "
                  f"(|f| = {abs(f(x)):.2e})")
    return x, hist


# =============================================================================
# 1) PARAMETERS
# =============================================================================

# --- Kinetic parameters (Arrhenius, Sec. 1.5) ---
R    = 8.314        # gas constant                                 [kJ/(kmol K)]
k0_1 = 1.47e5       # pre-exponential factor, R1 (A -> ST)         [1/s]
Ea1  = 90.98e3      # activation energy, R1                        [kJ/kmol]
k0_2 = 9.02e10      # pre-exponential factor, R2 (A -> BZ)         [1/s]
Ea2  = 207.99e3     # activation energy, R2                        [kJ/kmol]

# --- Operating conditions (Sec. 1.6) ---
T      = 900.0         # reactor temperature, FIXED (isothermal)      [K]
P      = 1e5           # pressure (isobaric reactor)                  [Pa]
F_A0   = 1/3600        # inlet molar flow of A (EB),  1 kmol/h        [kmol/s]
F_W0   = 9/3600        # inlet molar flow of steam,   9 kmol/h        [kmol/s]
F_tot0 = F_A0 + F_W0   # total inlet molar flow                       [kmol/s]
V      = 0.250         # reactor volume                               [m^3]

# Expansion factor epsA = y_A0 * delta_nu.
# Each reaction turns 1 mole of A into 2 moles of products, so
# delta_nu = +1 per mole of A converted; with y_A0 = 0.10 -> epsA = 0.10.
# The steam is inert: it dilutes the feed and damps the expansion.
y_A0 = F_A0/F_tot0     # mole fraction of A in the feed               [-]
epsA = y_A0*1          # expansion factor                             [-]

# Inlet volumetric flow from the ideal-gas law, Q0 = F_tot0*R*T/P.
# The factor 1e3 converts kmol -> mol, because with R in J/(mol K) and
# P in Pa, R*T/P is the molar volume in m^3/mol.
Q0 = F_tot0*1e3*R*T/P  # inlet volumetric flow (0.208 m^3/s)          [m^3/s]

# --- Rate constants and Damkohler number at the fixed temperature ---
# Because T is constant, k1, k2 and N_Da are plain NUMBERS here
# (compare with Example 2.2, where they become functions of T).
k1   = k0_1*np.exp(-Ea1/(R*T))   # [1/s]  (0.7705 1/s at 900 K)
k2   = k0_2*np.exp(-Ea2/(R*T))   # [1/s]  (0.0764 1/s at 900 K)
N_Da = (k1 + k2)*V/Q0            # [-]    (1.0187)

print(f"Isothermal CSTR, T = {T:.0f} K, V = {V:.3f} m^3")
print(f"  k1 = {k1:.4f} 1/s,  k2 = {k2:.4f} 1/s")
print(f"  Q0 = {Q0:.5f} m^3/s,  epsA = {epsA:.2f},  N_Da = {N_Da:.4f}\n")

# --- Numerical settings ---
tol     = 1e-12        # tolerance on |f(X_A)| (stopping criterion)
maxiter = 50           # maximum number of Newton iterations
X_A0    = 0.5          # initial guess for the conversion

# =============================================================================
# 2) MODEL: the residual f(X_A) and its derivative
# =============================================================================
# f(X_A) = X_A - N_Da*(1 - X_A)/(1 + epsA*X_A)            (equation 2.1)
# At the solution f = 0. NumPy operations are element-wise, so f can also be
# evaluated on an array of X_A values (used for the plot).

def f(X_A):
    return X_A - N_Da*(1 - X_A)/(1 + epsA*X_A)


# Derivative f'(X_A), obtained by hand with the quotient rule:
#   d/dX [ (1 - X)/(1 + epsA*X) ] = -(1 + epsA)/(1 + epsA*X)^2
# hence
#   f'(X_A) = 1 + N_Da*(1 + epsA)/(1 + epsA*X_A)^2
# Note that f'(X_A) > 0 on the whole physical domain [0, 1): f is strictly
# increasing, so the root is UNIQUE and Newton is well conditioned.

def df(X_A):
    return 1 + N_Da*(1 + epsA)/(1 + epsA*X_A)**2


# =============================================================================
# 3) NUMERICAL SOLUTION: scalar Newton-Raphson
# =============================================================================
# Linearizing f around the current estimate X_n (first-order Taylor):
#   f(X_n + dX) ~ f(X_n) + f'(X_n)*dX = 0   ->   X_{n+1} = X_n - f(X_n)/f'(X_n)
# Near a simple root the convergence is QUADRATIC: the error at step n+1
# scales as the SQUARE of the error at step n, so the number of correct
# digits roughly doubles at every iteration (visible in the printed table).
X_A, hist = newton_scalar(f, df, X_A0, tol, maxiter)

print(f"Newton-Raphson from X_A0 = {X_A0:.2f}")
print(f"{'it':>4s} {'X_A':>12s} {'f(X_A)':>14s}")
for k, xk in enumerate(hist):
    print(f"{k:4d} {xk:12.6f} {f(xk):14.3e}")
print(f"Solution: X_A = {X_A:.4f} ({100*X_A:.2f}%)\n")

# =============================================================================
# 4) CHECK 1: analytical solution (slide 12, Note)
# =============================================================================
# Multiplying f(X_A) = 0 by (1 + epsA*X_A) gives the quadratic
#   epsA*X_A^2 + (1 + N_Da)*X_A - N_Da = 0
# whose physically meaningful root (0 <= X_A < 1) is the one with "+":
#   X_A = [ -(1 + N_Da) + sqrt((1 + N_Da)^2 + 4*epsA*N_Da) ] / (2*epsA)
# (the other root is negative, hence unphysical).
X_exact = (-(1 + N_Da) + np.sqrt((1 + N_Da)**2 + 4*epsA*N_Da))/(2*epsA)
print(f"Check 1 - analytical root of the quadratic: X_A = {X_exact:.12f}")
print(f"          |X_A(Newton) - X_A(analytical)| = {abs(X_A - X_exact):.1e}\n")

# =============================================================================
# 5) CHECK 2: robustness to the initial guess
# =============================================================================
# Since f is strictly increasing on [0, 1), Newton converges to the same
# unique root even from a guess three times farther away.
X_A_far, hist_far = newton_scalar(f, df, 0.75, tol, maxiter)
print(f"Check 2 - from X_A0 = 0.75: X_A = {X_A_far:.6f} in {len(hist_far) - 1} iterations\n")

print(f"Newton-Raphson from X_A0 = 0.75")
print(f"{'it':>4s} {'X_A':>12s} {'f(X_A)':>14s}")
for k, xk in enumerate(hist_far):
    print(f"{k:4d} {xk:12.6f} {f(xk):14.3e}")
print(f"Solution: X_A = {X_A:.4f} ({100*X_A:.2f}%)\n")

# =============================================================================
# 6) SELECTIVITY
# =============================================================================
# Both reactions have the same order and the same reagent: their rates scale
# identically with C_A, so their ratio is constant and the selectivity to
# styrene does not depend on X_A or V.
S_ST = k1/(k1 + k2)
print(f"Selectivity to styrene k1/(k1+k2) = {S_ST:.4f} ({100*S_ST:.1f}%)")
print(f"Outlet flows [kmol/h]: EB {3600*F_A0*(1 - X_A):.4f}, "
      f"ST {3600*S_ST*F_A0*X_A:.4f}, BZ {3600*(1 - S_ST)*F_A0*X_A:.4f}")

# =============================================================================
# 7) PLOT: the residual f(X_A) and the Newton iterates
# =============================================================================
Xplot = np.linspace(0, 0.99, 200)
hist_arr = np.array(hist)
plt.figure("Example 2.1 - Isothermal CSTR")
plt.plot(Xplot, f(Xplot), "-", lw=1.2, label="$f(X_A)$")
plt.plot(hist_arr, f(hist_arr), "o", ms=6, label="Newton iterates")
plt.plot([0, 1], [0, 0], "k:", label="Zero line")   # the root is where f crosses it
plt.xlabel("Conversion of EB, $X_A$ [-]")
plt.ylabel("Residual $f(X_A)$ [-]")
plt.title(f"Isothermal CSTR, T = {T:.0f} K, $N_{{Da}}$ = {N_Da:.4f}")
plt.legend(loc="upper left")
plt.grid(True)
plt.show()
