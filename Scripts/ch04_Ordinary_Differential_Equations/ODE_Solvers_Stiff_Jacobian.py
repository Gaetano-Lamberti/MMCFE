"""Lecture 4, Section 2.10.2 - Stiff solvers and analytical Jacobian (SciPy).

Robertson kinetics (Section 2.7.5), t in [0, 40], y(0) = (1, 0, 0).
"""
from scipy.integrate import solve_ivp

def rob(t, y):                                  # Robertson kinetics
    y1, y2, y3 = y
    return [-0.04*y1 + 1e4*y2*y3,
             0.04*y1 - 1e4*y2*y3 - 3e7*y2**2,
             3e7*y2**2]

def jac(t, y):                                  # analytical Jacobian df/dy
    y1, y2, y3 = y
    return [[-0.04,  1e4*y3,           1e4*y2],
            [ 0.04, -1e4*y3 - 6e7*y2, -1e4*y2],
            [ 0.0,   6e7*y2,           0.0  ]]

for m in ['RK45', 'BDF', 'Radau', 'LSODA']:
    kw = {} if m == 'RK45' else {'jac': jac}    # RK45: no Jacobian
    s = solve_ivp(rob, (0, 40), [1, 0, 0], method=m,
                  rtol=1e-6, atol=1e-10, **kw)
    print(f'{m:5s} steps={len(s.t)-1:6d} nfev={s.nfev:7d} '
          f'njev={s.njev:3d} y3(40)={s.y[2, -1]:.6f}')
