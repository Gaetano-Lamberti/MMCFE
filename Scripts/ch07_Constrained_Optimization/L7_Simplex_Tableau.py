"""Lesson 7 - Tableau Simplex method (maximization, <= constraints, b >= 0).

Reference implementation: prints every tableau with exact fractions.
Problem:  max z = c'x   s.t.  A x <= b,  x >= 0
Notation as in the slides: row 0 = objective row (z - c'x = 0), RHS = right-hand side.
"""
from fractions import Fraction as Fr


def print_tableau(T, z, basis, names, k):
    print(f"\nTableau {k}")
    print("Basis " + "".join(f"{n:>8}" for n in names) + f"{'RHS':>8}")
    print("z     " + "".join(f"{str(v):>8}" for v in z))
    for i, row in enumerate(T):
        print(f"{names[basis[i]]:<6}" + "".join(f"{str(v):>8}" for v in row))


def simplex(c, A, b, names):
    m, n = len(A), len(c)
    # tableau: [A | I | b], objective row: [-c | 0 | 0]
    T = [[Fr(a) for a in A[i]] + [Fr(int(i == j)) for j in range(m)] + [Fr(b[i])] for i in range(m)]
    z = [Fr(-v) for v in c] + [Fr(0)] * m + [Fr(0)]
    basis = [n + i for i in range(m)]               # slack variables start in the basis
    k = 0
    print_tableau(T, z, basis, names, k)
    while True:
        col = min(range(n + m), key=lambda j: z[j])  # entering: most negative row-0 entry
        if z[col] >= 0:
            print("\nAll row-0 coefficients >= 0: optimum reached")
            break
        cand = [(T[i][-1] / T[i][col], i) for i in range(m) if T[i][col] > 0]
        if not cand:
            print("\nNo positive entry in the pivot column: LP is unbounded")
            return None
        ratio, row = min(cand)                       # leaving: minimum ratio test
        print(f"\nEntering {names[col]}, leaving {names[basis[row]]}, pivot = {T[row][col]}")
        p = T[row][col]
        T[row] = [v / p for v in T[row]]             # Gauss-Jordan pivot
        for i in range(m):
            if i != row:
                f = T[i][col]
                T[i] = [a - f * r for a, r in zip(T[i], T[row])]
        f = z[col]
        z = [a - f * r for a, r in zip(z, T[row])]
        basis[row] = col
        k += 1
        print_tableau(T, z, basis, names, k)
    x = [Fr(0)] * (n + m)
    for i, j in enumerate(basis):
        x[j] = T[i][-1]
    print("Solution: " + ", ".join(f"{names[j]} = {x[j]}" for j in range(n + m)) + f";  z* = {z[-1]}")
    print("Shadow prices (row 0 under the slacks): " + ", ".join(str(v) for v in z[n:n + m]))
    return x, z[-1]


def check_linprog(name, c, A, b, x_tab, z_tab):
    """Check with the built-in solver. linprog MINIMIZES c'x subject to A_ub x <= b_ub:
    pass -c for a maximization and change the sign of fun. The multipliers
    res.ineqlin.marginals come out with the opposite sign of the shadow prices."""
    import numpy as np
    from scipy.optimize import linprog
    res = linprog(-np.array(c, float), A_ub=A, b_ub=b, bounds=[(0, None)] * len(c), method="highs")
    if res.status != 0:
        print(f"{name}: linprog failed ({res.message})")
        return
    x = res.x
    n = len(c)
    print(f"\n{name}")
    print(f"  x*            linprog: {np.round(x, 4)}   tableau: {[str(v) for v in x_tab[:n]]}")
    print(f"  z*            linprog: {-res.fun:g}   tableau: {z_tab}")
    print(f"  slacks        linprog: {np.round(res.ineqlin.residual, 4)}")
    print(f"  shadow prices linprog: {np.round(-res.ineqlin.marginals, 4)}")
    print(f"  max |difference| on x*: {max(abs(x[j] - float(x_tab[j])) for j in range(n)):.1e}")


if __name__ == "__main__":
    cP, AP, bP = [3, 4], [[1, 1], [1, 2], [2, 1]], [8, 12, 14]                      # pasta plant
    cB, AB, bB = [3, 5, 6], [[1, 2, 4], [2, 2, 1], [1, 3, 1]], [20, 24, 40]         # bakery
    print("=== Pasta plant (Section 7.4) ===")
    xP, zP = simplex(cP, AP, bP, ["x1", "x2", "s1", "s2", "s3"])
    print("\n=== Example 7.1 - Bakery product mix ===")
    xB, zB = simplex(cB, AB, bB, ["x1", "x2", "x3", "s1", "s2", "s3"])
    print("\n=== Check with linprog ===")
    check_linprog("Pasta plant", cP, AP, bP, xP, zP)
    check_linprog("Example 7.1", cB, AB, bB, xB, zB)
