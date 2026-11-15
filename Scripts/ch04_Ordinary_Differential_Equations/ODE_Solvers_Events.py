"""Lecture 4, Section 2.10.3 - Events and dense output (SciPy solve_ivp).

Test problem of Section 2.2: find the time t* at which y(t*) = 4, then stop.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

def f(t, y):
    return y - t**2 + 1

def reach4(t, y):              # event function: zero when y = 4
    return y[0] - 4.0
reach4.terminal = True         # stop the integration at the event
reach4.direction = 1           # upward crossings only

sol = solve_ivp(f, (0, 2), [0.5], rtol=1e-8, atol=1e-10,
                events=reach4, dense_output=True)
t_star = brentq(lambda t: (t + 1)**2 - 0.5*np.exp(t) - 4, 0, 2)
print('event at t = %.8f (exact %.8f)' % (sol.t_events[0][0], t_star))
print('status:', sol.status, '(1 = stopped by an event)')
print('dense output y(1.0) = %.8f (exact %.8f)'
      % (sol.sol(1.0)[0], 4 - 0.5*np.exp(1)))
