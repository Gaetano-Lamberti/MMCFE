"""Example 3.12.1 - Test case (c): transient conduction in a slab, same problem, four methods.
    u_t = alpha*u_xx on the HALF slab 0 <= x <= l (x = 0: centre, symmetry; x = l: wall, u = 0)
    u(x,0) = cos(pi*x/(2l))  ->  exact u = exp(-alpha*pi^2*t/(4 l^2)) * cos(pi*x/(2l))
Each method gives a linear ODE system  M du/dt = -K u ; it is integrated EXACTLY in time
(matrix exponential), so the error measured is the spatial error only."""
import numpy as np, json
from scipy.linalg import expm, solve
from scipy.special import roots_jacobi

alpha, l, tend = 1.0, 1.0, 0.2
k = np.pi/(2*l)
uex = lambda x, t: np.exp(-alpha*k**2*t)*np.cos(k*x)

def fd(N):
    """FD + MOL: N unknowns at x_i = i*h, i = 0..N-1 (centre included), wall node x_N = l (u = 0)."""
    h = l/N; x = np.arange(N)*h
    A = np.zeros((N, N))
    for i in range(N):
        A[i, i] = -2
        if i == 0: A[0, 1] = 2                       # ghost node u_{-1} = u_1 (symmetry)
        else:
            A[i, i-1] = 1
            if i < N-1: A[i, i+1] = 1                 # u_N = 0 at the wall
    return x, np.eye(N), -alpha*A/h**2

def fem(N, lumped=False):
    """Galerkin P1: N unknowns (nodes 0..N-1), wall node fixed. Symmetry at x=0 is a NATURAL BC: nothing to do."""
    h = l/N; x = np.arange(N)*h
    K = np.zeros((N, N)); M = np.zeros((N, N))
    for e in range(N):                                # element e joins nodes e, e+1 (node N = wall, dropped)
        Ke = alpha/h*np.array([[1, -1], [-1, 1]]); Me = h/6*np.array([[2, 1], [1, 2]])
        if lumped: Me = h/2*np.eye(2)
        for a in range(2):
            for b in range(2):
                i, j = e + a, e + b
                if i < N and j < N: K[i, j] += Ke[a, b]; M[i, j] += Me[a, b]
    return x, M, K

def fvm(N):
    """Cell-centred FVM: N cells of width h, centres x_i = (i+1/2) h. Centre face: zero flux. Wall face: u = 0 at distance h/2."""
    h = l/N; x = (np.arange(N) + 0.5)*h
    K = np.zeros((N, N))
    for i in range(N):
        if i > 0: K[i, i] += alpha/h; K[i, i-1] -= alpha/h          # west face (interior)
        if i < N-1: K[i, i] += alpha/h; K[i, i+1] -= alpha/h        # east face (interior)
        else: K[i, i] += alpha/(h/2)                                # wall face, u_wall = 0
    return x, h*np.eye(N), K

def oc(N):
    """Orthogonal collocation (Villadsen-Michelsen, symmetric, slab a = 1): N interior points + wall."""
    t_, _ = roots_jacobi(N, 1.0, -0.5); s = np.sort((1 + t_)/2); xp = np.append(np.sqrt(s), 1.0)
    i = np.arange(1, N + 2); Q = xp[:, None]**(2*i - 2)
    D = np.where(i >= 2, (2*i - 2)*(2*i - 3)*xp[:, None]**np.maximum(2*i - 4, 0), 0.0)
    B = D @ np.linalg.inv(Q)
    return xp[:N]*l, np.eye(N), -alpha*B[:N, :N]/l**2      # wall value u = 0 -> its column drops

def err(method, N):
    x, M, K = method(N)
    u0 = uex(x, 0.0)
    u = expm(-solve(M, K)*tend) @ u0
    return np.max(np.abs(u - uex(x, tend)))

res = {}
meths = {'FD + MOL': fd, 'FEM consistent': fem, 'FEM lumped': lambda N: fem(N, True), 'FVM': fvm, 'Orth. collocation': oc}
Ns = [2, 4, 8, 16, 32]
for name, m in meths.items():
    e = [float(err(m, N)) for N in (Ns if name != 'Orth. collocation' else [1, 2, 3, 4, 5, 6])]
    res[name] = e
    nn = Ns if name != 'Orth. collocation' else [1, 2, 3, 4, 5, 6]
    print(f'{name:18s}', '  '.join(f'N={n}:{v:.2e}' for n, v in zip(nn, e)))
    if name != 'Orth. collocation':
        print(' ' * 18, 'orders', [round(np.log2(e[j-1]/e[j]), 3) for j in range(1, len(e))])
# are FD and FEM-lumped identical?  (FEM lumped: symmetry node has half mass h/2 -> same as FD ghost row)
print('max |FD - FEM lumped| errors difference:', max(abs(a - b) for a, b in zip(res['FD + MOL'], res['FEM lumped'])))
res['Ns'] = Ns; res['Ns_oc'] = [1, 2, 3, 4, 5, 6]; res['tend'] = tend
json.dump(res, open('ex312a.json', 'w'), indent=1)

# ---------------------------------------------------------------- numbers quoted in the detailed slides 3.12.1
np.set_printoptions(precision=4, suppress=True)
h = l/4
print('\nN = 4 matrices (slides 3.12.1):')
print('FD:  K*h^2/alpha =\n', fd(4)[2]*h**2/alpha)
x, M, K = fem(4, True); print('FEM: K*h/alpha =\n', K*h/alpha, '\nM_lumped/h =\n', M/h, '\nM_L^-1 K * h^2/alpha =\n', solve(M, K)*h**2/alpha)
print('FEM consistent: M*6/h =\n', fem(4)[1]*6/h)
print('FVM: K*h/alpha =\n', fvm(4)[2]*h/alpha)
x, M, K = oc(1); print('OC N=1: x1 = %.4f, B11 = %.3f (exact decay rate pi^2/4 = %.3f)' % (x[0], -K[0, 0]*l**2/alpha, np.pi**2/4))
# Eq. (59o): the discrete cosine is an exact eigenvector of the FD matrix -> error prediction
print('\nFD error at the centre, predicted by (59o) vs measured:')
for N in Ns[1:]:
    h = l/N; lam_h = 4*alpha/h**2*np.sin(np.pi*h/(4*l))**2
    x, M, K = fd(N); v = np.cos(k*x)
    assert np.allclose(K @ v, lam_h*v)                       # exact eigenvector
    pred = alpha*np.pi**4*h**2/(192*l**4)*tend*np.exp(-alpha*k**2*tend)
    print(f'N={N:3d}  lambda_h={lam_h:.4f} (exact {k**2:.4f})  predicted {pred:.3e}  measured {err(fd, N):.3e}')
