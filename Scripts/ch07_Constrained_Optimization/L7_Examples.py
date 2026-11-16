"""Lesson 7 - Examples 7.1-7.3 solved with SciPy (reference implementation)."""
import numpy as np
from scipy.optimize import linprog, minimize, differential_evolution, NonlinearConstraint

# ---------------- Example 7.1: bakery product mix (max -> min of -z)
c = np.array([3, 5, 6]); A = [[1, 2, 4], [2, 2, 1], [1, 3, 1]]; b = [20, 24, 40]
r = linprog(-c, A_ub=A, b_ub=b, method="highs")
print("Ex 7.1  x* =", r.x, " z* =", -r.fun, " shadow prices =", -r.ineqlin.marginals)

# ---------------- Example 7.2: least-cost burger formulation (100 kg batch)
# Two different quantities for each constraint (do not confuse them):
#   slack        = b_ub - A_ub x (r.ineqlin.residual): how much of the constraint is left unused
#   shadow price = Lagrange multiplier: how much the optimal cost decreases if the constraint is
#                  relaxed by one unit (row 0 of the final tableau; the lambda_i of the KKT conditions).
#                  SciPy returns marginals = d(cost)/d(b_ub), i.e. the shadow price with a minus sign.
# Complementary slackness: for each constraint, slack x shadow price = 0.
cost = [7.0, 2.5, 2.0]                         # beef, pork trim, hydrated soy [EUR/kg]
names = ["fat", "protein", "soy"]
for fat_max in (20, 15):
    A_ub = [[0.10, 0.35, 0.00],                # fat     <= fat_max kg
            [-0.20, -0.14, -0.18],             # protein >= 16 kg (sign changed: -protein <= -16)
            [0, 0, 1]]                         # soy     <= 10 kg
    r = linprog(cost, A_ub=A_ub, b_ub=[fat_max, -16, 10], A_eq=[[1, 1, 1]], b_eq=[100], method="highs")
    print(f"\nEx 7.2  fat <= {fat_max}%:  x* = {np.round(r.x, 2)} kg,  cost = {r.fun:g} EUR/100 kg")
    print(f"  {'constr.':<8} {'slack [kg]':>12} {'shadow price [EUR/kg]':>28}")
    for k in range(3):
        print(f"  {names[k]:<8} {r.ineqlin.residual[k]:12.2f} {-r.ineqlin.marginals[k] + 0.0:28.2f}")
    print(f"  mass balance: one more kg in the batch costs {r.eqlin.marginals[0]:.2f} EUR")

# ---------------- Example 7.3: thermal process design
TrF, zF, F0min = 121.1, 10.0, 3.0              # lethality
TrC, zC, Cmin = 100.0, 33.0, 35.0              # cook value
Trv, zv, Dv = 121.1, 25.0, 180.0               # thiamine
lb, ub = np.array([105.0, 1.0]), np.array([135.0, 120.0])
F0 = lambda T, t: t * 10 ** ((T - TrF) / zF)
C100 = lambda T, t: t * 10 ** ((T - TrC) / zC)
R = lambda T, t: 10 ** (-t * 10 ** ((T - Trv) / zv) / Dv)
f = lambda x: -R(*x)
x0 = np.array([121.1, 10.0])
cons = [{"type": "ineq", "fun": lambda x: F0(*x) - F0min}, {"type": "ineq", "fun": lambda x: C100(*x) - Cmin}]
s = minimize(f, x0, method="SLSQP", bounds=list(zip(lb, ub)), constraints=cons)
print(f"Ex 7.3  SLSQP:         T*={s.x[0]:.2f}  t*={s.x[1]:.2f}  R*={R(*s.x):.4f}  evals={s.nfev}")
s = minimize(f, x0, method="trust-constr", bounds=list(zip(lb, ub)), constraints=cons)
print(f"Ex 7.3  trust-constr:  T*={s.x[0]:.2f}  t*={s.x[1]:.2f}  R*={R(*s.x):.4f}  evals={s.nfev}")

def penalized(x, rho):
    T, t = x
    v = [max(0, F0min - F0(T, t)) / F0min, max(0, Cmin - C100(T, t)) / Cmin,   # scaled violations
         max(0, lb[0] - T), max(0, T - ub[0]), max(0, lb[1] - t), max(0, t - ub[1])]
    return f(x) + rho * sum(vi ** 2 for vi in v)

for rho in (1e1, 1e3, 1e5):
    s = minimize(penalized, x0, args=(rho,), method="Nelder-Mead",
                 options={"xatol": 1e-6, "fatol": 1e-10, "maxfev": 5000})
    print(f"Ex 7.3  NM rho={rho:.0e}: T*={s.x[0]:.2f}  t*={s.x[1]:.2f}  R*={R(*s.x):.4f}  "
          f"F0={F0(*s.x):.3f}  evals={s.nfev}")
s = differential_evolution(lambda x: penalized(x, 1e5), bounds=list(zip(lb, ub)), seed=1, tol=1e-10, polish=False)
print(f"Ex 7.3  DE + penalty:  T*={s.x[0]:.2f}  t*={s.x[1]:.2f}  R*={R(*s.x):.4f}  evals={s.nfev}")

# hidden LP in (T, u = ln t): min u + ln10/zv (T - Trv)
L = np.log(10)
r = linprog([L / zv, 1.0], A_ub=[[-L / zF, -1.0], [-L / zC, -1.0]],
            b_ub=[-(np.log(F0min) + TrF * L / zF), -(np.log(Cmin) + TrC * L / zC)],
            bounds=[(lb[0], ub[0]), (np.log(lb[1]), np.log(ub[1]))], method="highs")
print(f"Ex 7.3  hidden LP:     T*={r.x[0]:.2f}  t*={np.exp(r.x[1]):.2f}  R*={R(r.x[0], np.exp(r.x[1])):.4f}")
