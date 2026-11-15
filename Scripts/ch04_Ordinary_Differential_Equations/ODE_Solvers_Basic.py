"""Lecture 4, Section 2.10.1 - Basic call and tolerances (SciPy solve_ivp).

Test problem of Section 2.2:  dy/dt = y - t^2 + 1,  y(0) = 0.5,  t in [0, 2]
Exact solution:               y(t) = (t + 1)^2 - 0.5 exp(t)
"""
import numpy as np
from scipy.integrate import solve_ivp

def f(t, y):                          # right-hand side: dy/dt = f(t, y)
    return y - t**2 + 1

sol = solve_ivp(f, (0.0, 2.0), [0.5],          # f, (t0, tf), y0
                method='RK45',                 # Dormand-Prince 5(4)
                rtol=1e-6, atol=1e-9,          # tolerances (2.8.4)
                t_eval=np.linspace(0, 2, 11))  # output times only

y_exact = (sol.t + 1)**2 - 0.5*np.exp(sol.t)
print('status:', sol.status, '| f-evals:', sol.nfev)
err = np.max(np.abs(sol.y[0] - y_exact))
print('max error at output times: %.2e' % err)
print('y(2) = %.8f (exact %.8f)' % (sol.y[0, -1], y_exact[-1]))
