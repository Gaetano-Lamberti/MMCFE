"""Module 1, Part 3 - PDE test case (b): 2D Laplace in a rectangle
Python port of PDE_CaseB_Laplace2D.m

Elliptic problem, single-mode boundary data (clean closed-form test case):
    Laplace(u) = 0   in (0,a) x (0,b)
    u(0,y) = u(a,y) = u(x,0) = 0,   u(x,b) = sin(pi*x/a)

Three solutions compared:
    1) Analytical (separation of variables, single mode)
    2) Fully-discretized finite differences: 5-point stencil at every interior node,
       assembled into a sparse linear system A*u = b, solved with spsolve
    3) In place of MATLAB's solvepde (PDE Toolbox): FEM with scikit-fem
       (pip install scikit-fem). As in MATLAB, this is FEM on a triangular mesh,
       not finite differences. If scikit-fem is missing, this part is skipped.
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

# --- Parameters ---
a = 1.0   # m, rectangle width
b = 0.6   # m, rectangle height

# --- 1) Analytical solution ---
def u_exact(x, y):
    return np.sin(np.pi*x/a)*np.sinh(np.pi*y/a)/np.sinh(np.pi*b/a)

print('=== Analytical solution check ===')
print(f'u(0,y)={u_exact(0, 0.3):.3e}  u(a,y)={u_exact(a, 0.3):.3e}  u(x,0)={u_exact(0.5, 0):.3e}  (should all be 0)')
print(f'u(x,b) at x=a/2: {u_exact(a/2, b):.6f}  (should equal sin(pi/2) = {np.sin(np.pi*0.5):.6f})\n')


# --- Local functions ---
def assemble_laplace_fd(Nx, Ny, dx, dy, x, y):
    """5-point stencil on the interior nodes i = 1..Nx-2, j = 1..Ny-2 (0-based).
    Known boundary values move to the right-hand side."""
    nx_i, ny_i = Nx - 2, Ny - 2
    ndof = nx_i*ny_i
    idx = lambda i, j: (j - 1)*nx_i + (i - 1)      # interior (i,j) -> 0..ndof-1, row by row
    rows, cols, vals = [], [], []
    rhs = np.zeros(ndof)
    for j in range(1, Ny - 1):
        for i in range(1, Nx - 1):
            k = idx(i, j)
            rows.append(k); cols.append(k); vals.append(-2/dx**2 - 2/dy**2)
            for (ii, jj, w) in ((i-1, j, 1/dx**2), (i+1, j, 1/dx**2), (i, j-1, 1/dy**2), (i, j+1, 1/dy**2)):
                if 1 <= ii <= Nx - 2 and 1 <= jj <= Ny - 2:
                    rows.append(k); cols.append(idx(ii, jj)); vals.append(w)
                else:
                    rhs[k] -= w*u_exact(x[ii], y[jj])      # neighbour on the boundary
    A = sp.csr_matrix((vals, (rows, cols)), shape=(ndof, ndof))
    return A, rhs, idx


def fd_solve(Nx, Ny):
    dx, dy = a/(Nx - 1), b/(Ny - 1)
    x, y = np.linspace(0, a, Nx), np.linspace(0, b, Ny)
    A, rhs, idx = assemble_laplace_fd(Nx, Ny, dx, dy, x, y)
    usol = spsolve(A.tocsc(), rhs)
    Xg, Yg = np.meshgrid(x, y)
    Ugrid = u_exact(Xg, Yg)                 # boundary values; the interior is overwritten
    Ugrid[1:-1, 1:-1] = usol.reshape(Ny - 2, Nx - 2)
    return x, y, Xg, Yg, Ugrid, A, dx, dy


# --- 2) Fully-discretized finite differences: assemble A*u = b ---
Nx, Ny = 41, 25            # grid points including boundary
x, y, Xg, Yg, Ugrid, A, dx, dy = fd_solve(Nx, Ny)
print('=== Finite-difference grid ===')
print(f'Nx={Nx}, Ny={Ny}, dx={dx:.5f} m, dy={dy:.5f} m\n')
print(f'Assembled sparse system: {A.shape[0]} unknowns, {A.nnz} nonzeros')
print(f'Sparsity: {100*A.nnz/A.shape[0]**2:.2f}% nonzero\n')

Uexact_grid = u_exact(Xg, Yg)
err_fd = np.abs(Ugrid - Uexact_grid)
print('=== FD vs analytical ===')
print(f'Max error: {err_fd.max():.4e}\n')

# --- Empirical convergence order (grid refinement) ---
print('=== Empirical convergence order (finite differences) ===')
grids = [(11, 7), (21, 13), (41, 25), (81, 49), (161, 97)]
errs, dxs = [], []
for (nx, ny) in grids:
    _, _, Xk, Yk, Uk, _, dxk, _ = fd_solve(nx, ny)
    errs.append(np.abs(Uk - u_exact(Xk, Yk)).max()); dxs.append(dxk)
errs, dxs = np.array(errs), np.array(dxs)
print(f'{"Nx":>6s} {"Ny":>6s} {"dx":>10s} {"max error":>14s}')
for (nx, ny), d, e in zip(grids, dxs, errs):
    print(f'{nx:6d} {ny:6d} {d:10.5f} {e:14.6e}')
print('\nEstimated order p = log2(err(dx)/err(dx/2)):')
for k in range(1, len(errs)):
    print(f'  dx={dxs[k-1]:.5f} -> dx={dxs[k]:.5f}: order = {np.log2(errs[k-1]/errs[k]):.3f}')
print('--> Order 2, as expected from the O(dx^2) 5-point stencil.\n')

# --- 3) In place of solvepde: FEM with scikit-fem (P1 triangles) ---
try:
    import skfem
    from skfem.models.poisson import laplace
    from scipy.interpolate import LinearNDInterpolator
    have_skfem = True
except ImportError:
    have_skfem = False

if have_skfem:
    print('=== FEM (scikit-fem, in place of solvepde) ===')
    # Triangular mesh with longest edge a/64 = 0.016 m <= 0.02, as generateMesh(model,'Hmax',0.02).
    # init_symmetric: the rectangle cut by its two diagonals into 4 triangles,
    # then refined 6 times (each triangle -> 4): 16 384 triangles in four
    # orientations. A mesh that does NOT reproduce the 5-point FD stencil.
    mesh = skfem.MeshTri.init_symmetric().scaled([a, b]).refined(6)
    basis = skfem.Basis(mesh, skfem.ElementTriP1())
    K = laplace.assemble(basis)                       # stiffness matrix: c = 1, f = 0
    u_fem = np.zeros(basis.N)
    xb, yb = mesh.p[0], mesh.p[1]
    D = mesh.boundary_nodes()                          # all boundary nodes: Dirichlet
    u_fem[D] = u_exact(xb[D], yb[D])                   # sin(pi x/a) on top, 0 elsewhere
    u_fem = skfem.solve(*skfem.condense(K, np.zeros(basis.N), x=u_fem, D=D))
    interp = LinearNDInterpolator(mesh.p.T, u_fem)     # as interpolateSolution
    u_fem_grid = interp(Xg, Yg)
    err_fem = np.abs(u_fem_grid - Uexact_grid)
    print(f'Mesh: {mesh.p.shape[1]} nodes, {mesh.t.shape[1]} triangles')
    print(f'Max error (FEM, scikit-fem) vs analytical: {np.nanmax(err_fem):.4e}')
    print('(FEM uses a triangular mesh internally - different discretization')
    print(' entirely from the hand-assembled FD grid above, same PDE, same answer.)\n')
else:
    print('=== FEM skipped: scikit-fem not installed (pip install scikit-fem) ===\n')
    u_fem_grid = np.full(Xg.shape, np.nan)

# ============================================================
#  PLOTS
# ============================================================
# --- Plot 1: analytical solution surface ---
fig = plt.figure('2D Laplace: analytical solution')
ax = fig.add_subplot(projection='3d')
ax.plot_surface(Xg, Yg, Uexact_grid, cmap='viridis', linewidth=0)
ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.set_zlabel('u')
ax.set_title('u(x,y) = sin(πx/a)·sinh(πy/a)/sinh(πb/a)'); ax.view_init(30, -135)

# --- Plot 2: FD solution and error ---
plt.figure('Finite-difference solution and error', figsize=(11, 4))
plt.subplot(1, 2, 1)
plt.contourf(Xg, Yg, Ugrid, 20); plt.colorbar()
plt.xlabel('x (m)'); plt.ylabel('y (m)'); plt.title('Finite-difference solution'); plt.axis('scaled')
plt.subplot(1, 2, 2)
plt.contourf(Xg, Yg, err_fd, 20); plt.colorbar()
plt.xlabel('x (m)'); plt.ylabel('y (m)'); plt.title(f'|FD - analytical|  (max={err_fd.max():.2e})'); plt.axis('scaled')
plt.tight_layout()

# --- Plot 3: empirical convergence order ---
plt.figure('Empirical convergence order - Laplace FD')
plt.loglog(dxs, errs, '-o', lw=1.5, label='Finite differences (measured)')
plt.loglog(dxs, errs[0]*(dxs/dxs[0])**2, 'k:', lw=1, label='order-2 reference')
plt.xlabel('Δx (m, log scale)'); plt.ylabel('max error (log scale)')
plt.title('Empirical convergence order: slope = 2'); plt.legend(loc='upper left')
plt.grid(True, which='both'); plt.gca().invert_xaxis()
plt.gca().xaxis.set_minor_formatter(plt.NullFormatter())

# --- Plot 4: FD vs FEM comparison (if scikit-fem available) ---
if have_skfem:
    plt.figure('FD vs FEM comparison', figsize=(11, 4))
    plt.subplot(1, 2, 1)
    plt.contourf(Xg, Yg, u_fem_grid, 20); plt.colorbar()
    plt.xlabel('x (m)'); plt.ylabel('y (m)'); plt.title('FEM solution (scikit-fem, triangular mesh)'); plt.axis('scaled')
    plt.subplot(1, 2, 2)
    plt.contourf(Xg, Yg, np.abs(u_fem_grid - Ugrid), 20); plt.colorbar()
    plt.xlabel('x (m)'); plt.ylabel('y (m)'); plt.title('|FEM - FD|  (two discretizations, same PDE)'); plt.axis('scaled')
    plt.tight_layout()

plt.show()
