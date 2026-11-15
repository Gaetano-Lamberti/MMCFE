%% Module 1, Part 3 - PDE test case (b): 2D Laplace in a rectangle
% Elliptic problem, single-mode boundary data (clean closed-form test
% case, same spirit as S17.7's ODE test problems - no Fourier series to
% truncate):
%   Laplace(u) = 0   in (0,a) x (0,b)
%   u(0,y)=u(a,y)=u(x,0)=0,   u(x,b)=sin(pi*x/a)
%
% Three solutions compared:
%   1) Analytical (separation of variables, single mode)
%   2) Fully-discretized finite differences: 5-point stencil at every
%      interior node, assembled into a sparse linear system A*u=b
%      (direct reuse of S16.4), solved with backslash
%   3) MATLAB built-in `solvepde` (Partial Differential Equation
%      Toolbox) - NOTE: this is FEM internally (triangular mesh), not
%      finite differences. Watching the discretization method change
%      underneath a "built-in solver" call is the point, not an
%      inconvenience - it's the live version of the FD/FEM/FVM
%      discussion. Requires the PDE Toolbox (not core MATLAB); if
%      unavailable, this section can be skipped - flagged explicitly.

clear; clc; close all;

%% --- Parameters ---
a = 1.0;   % m, rectangle width
b = 0.6;   % m, rectangle height

%% --- 1) Analytical solution ---
u_exact = @(x,y) sin(pi*x/a).*sinh(pi*y/a)/sinh(pi*b/a);

fprintf('=== Analytical solution check ===\n');
fprintf('u(0,y)=%.3e  u(a,y)=%.3e  u(x,0)=%.3e  (should all be 0)\n', ...
        u_exact(0,0.3), u_exact(a,0.3), u_exact(0.5,0));
fprintf('u(x,b) at x=a/2: %.6f  (should equal sin(pi/2)=1 scaled: sin(pi*0.5)=%.6f)\n\n', ...
        u_exact(a/2,b), sin(pi*0.5));

%% --- 2) Fully-discretized finite differences (S16.4: assemble A*u=b) ---
Nx = 41; Ny = 25;     % grid points including boundary
dx = a/(Nx-1); dy = b/(Ny-1);
x = linspace(0,a,Nx)'; y = linspace(0,b,Ny)';

fprintf('=== Finite-difference grid ===\n');
fprintf('Nx=%d, Ny=%d, dx=%.5f m, dy=%.5f m\n\n', Nx, Ny, dx, dy);

[A, rhs, idxmap] = assemble_laplace_fd(Nx, Ny, dx, dy, x, y, u_exact);
fprintf('Assembled sparse system: %d unknowns, %d nonzeros\n', size(A,1), nnz(A));
fprintf('Sparsity: %.2f%% nonzero\n\n', 100*nnz(A)/numel(A));

usol = A\rhs;   % direct sparse solve

% reassemble full grid (interior solution + known boundary values)
Ugrid = zeros(Ny,Nx);
for j = 2:Ny-1
    for i = 2:Nx-1
        Ugrid(j,i) = usol(idxmap(i,j));
    end
end
Ugrid(1,:)   = u_exact(x, 0);
Ugrid(Ny,:)  = u_exact(x, b);
Ugrid(:,1)   = u_exact(0, y);
Ugrid(:,Nx)  = u_exact(a, y);

[Xg,Yg] = meshgrid(x,y);
Uexact_grid = u_exact(Xg,Yg);
err_fd = abs(Ugrid - Uexact_grid);
fprintf('=== FD vs analytical ===\n');
fprintf('Max error: %.4e\n\n', max(err_fd(:)));

%% --- Empirical convergence order (grid refinement) ---
fprintf('=== Empirical convergence order (finite differences) ===\n');
grids = [11 7; 21 13; 41 25; 81 49; 161 97];
errs = zeros(size(grids,1),1);
dxs  = zeros(size(grids,1),1);
for k = 1:size(grids,1)
    [errs(k), dxs(k)] = fd_error_at(grids(k,1), grids(k,2), a, b, u_exact);
end
fprintf('%6s %6s %10s %14s\n','Nx','Ny','dx','max error');
for k = 1:size(grids,1)
    fprintf('%6d %6d %10.5f %14.6e\n', grids(k,1), grids(k,2), dxs(k), errs(k));
end
fprintf('\nEstimated order p = log2(err(dx)/err(dx/2)):\n');
for k = 2:length(errs)
    p = log2(errs(k-1)/errs(k));
    fprintf('  dx=%.5f -> dx=%.5f: order = %.3f\n', dxs(k-1), dxs(k), p);
end
fprintf('--> Order 2, as expected from the O(dx^2) 5-point stencil.\n\n');

%% --- 3) MATLAB built-in: solvepde (PDE Toolbox, FEM) ---
% Requires the Partial Differential Equation Toolbox. Wrapped in try/catch
% so the rest of the script (and its outputs/plots) still run without it.
have_pde_toolbox = license('test','PDE_Toolbox') && ~isempty(which('createpde'));

if have_pde_toolbox
    fprintf('=== solvepde (built-in, FEM) ===\n');
    model = createpde();

    % Standard decsg rectangle: corners (0,0)-(a,0)-(a,b)-(0,b), CCW.
    % Edge numbering from decsg for a simple rectangle in this corner
    % order is 1=bottom(y=0), 2=right(x=a), 3=top(y=b), 4=left(x=0) -
    % this is the standard result for this corner ordering, but ALWAYS
    % worth confirming visually with pdegplot(model,'EdgeLabels','on')
    % before trusting a BC assignment on unfamiliar geometry.
    R1 = [3,4, 0,a,a,0, 0,0,b,b]';
    g = decsg(R1);
    geometryFromEdges(model, g);

    specifyCoefficients(model,'m',0,'d',0,'c',1,'a',0,'f',0);   % Laplace(u)=0

    % Edge 3 (top, y=b) carries the sin(pi x/a) data; edges 1,2,4 are zero
    applyBoundaryCondition(model,'dirichlet','Edge',[1 2 4],'u',0);
    applyBoundaryCondition(model,'dirichlet','Edge',3,'u',@(loc,state) sin(pi*loc.x/a));

    generateMesh(model,'Hmax',0.02);
    results = solvepde(model);

    u_fem_at_grid = interpolateSolution(results, Xg(:), Yg(:));
    u_fem_grid = reshape(u_fem_at_grid, size(Xg));
    err_fem = abs(u_fem_grid - Uexact_grid);
    fprintf('Max error (FEM, solvepde) vs analytical: %.4e\n', max(err_fem(:), [], 'omitnan'));
    fprintf('(FEM uses a triangular mesh internally - different discretization\n');
    fprintf(' entirely from the hand-assembled FD grid above, same PDE, same answer.)\n\n');
else
    fprintf('=== solvepde skipped: PDE Toolbox not available on this machine ===\n\n');
    u_fem_grid = nan(size(Xg));
end

%% ============================================================
%  PLOTS
%  ============================================================

%% --- Plot 1: analytical solution surface ---
figure('Name','2D Laplace: analytical solution');
surf(Xg, Yg, Uexact_grid, 'EdgeColor','none');
xlabel('x (m)'); ylabel('y (m)'); zlabel('u');
title('Analytical solution: u(x,y) = sin(\pi x/a)\cdot sinh(\pi y/a)/sinh(\pi b/a)');
colorbar; view(45,30);

%% --- Plot 2: FD solution and error ---
figure('Name','Finite-difference solution and error');
subplot(1,2,1);
contourf(Xg, Yg, Ugrid, 20, 'LineStyle','none');
xlabel('x (m)'); ylabel('y (m)'); colorbar;
title('Finite-difference solution');
axis equal tight;

subplot(1,2,2);
contourf(Xg, Yg, err_fd, 20, 'LineStyle','none');
xlabel('x (m)'); ylabel('y (m)'); colorbar;
title(sprintf('|FD - analytical|  (max=%.2e)', max(err_fd(:))));
axis equal tight;

%% --- Plot 3: empirical convergence order ---
figure('Name','Empirical convergence order - Laplace FD');
loglog(dxs, errs, '-o', 'LineWidth',1.5,'MarkerFaceColor','b'); hold on
loglog(dxs, errs(1)*(dxs/dxs(1)).^2, 'k:', 'LineWidth',1);
xlabel('\Delta x (m, log scale)'); ylabel('max error (log scale)');
title('Empirical convergence order: slope = 2');
legend('Finite differences (measured)','order-2 reference','Location','northwest');
grid on; set(gca,'XDir','reverse');

%% --- Plot 4: FD vs FEM comparison (if PDE Toolbox available) ---
if have_pde_toolbox
    figure('Name','FD vs FEM (solvepde) comparison');
    subplot(1,2,1);
    contourf(Xg, Yg, u_fem_grid, 20, 'LineStyle','none');
    xlabel('x (m)'); ylabel('y (m)'); colorbar;
    title('FEM solution (solvepde, triangular mesh)');
    axis equal tight;

    subplot(1,2,2);
    contourf(Xg, Yg, abs(u_fem_grid-Ugrid), 20, 'LineStyle','none');
    xlabel('x (m)'); ylabel('y (m)'); colorbar;
    title('|FEM - FD|  (two different discretizations, same PDE)');
    axis equal tight;
end

%% ============================================================
%  LOCAL FUNCTIONS (must follow all executable code)
%  ============================================================
function [A, rhs, idxmap] = assemble_laplace_fd(Nx, Ny, dx, dy, x, y, u_exact)
    nx_i = Nx-2; ny_i = Ny-2;
    ndof = nx_i*ny_i;
    idxmap = @(i,j) (j-2)*nx_i + (i-1);   % maps interior (i,j), i,j=2..N-1, to 1..ndof

    A = spalloc(ndof, ndof, 5*ndof);
    rhs = zeros(ndof,1);

    for j = 2:Ny-1
        for i = 2:Nx-1
            k = idxmap(i,j);
            A(k,k) = -2/dx^2 - 2/dy^2;
            if i-1 >= 2
                A(k, idxmap(i-1,j)) = A(k, idxmap(i-1,j)) + 1/dx^2;
            else
                rhs(k) = rhs(k) - u_exact(x(1), y(j))/dx^2;
            end
            if i+1 <= Nx-1
                A(k, idxmap(i+1,j)) = A(k, idxmap(i+1,j)) + 1/dx^2;
            else
                rhs(k) = rhs(k) - u_exact(x(Nx), y(j))/dx^2;
            end
            if j-1 >= 2
                A(k, idxmap(i,j-1)) = A(k, idxmap(i,j-1)) + 1/dy^2;
            else
                rhs(k) = rhs(k) - u_exact(x(i), y(1))/dy^2;
            end
            if j+1 <= Ny-1
                A(k, idxmap(i,j+1)) = A(k, idxmap(i,j+1)) + 1/dy^2;
            else
                rhs(k) = rhs(k) - u_exact(x(i), y(Ny))/dy^2;
            end
        end
    end
end

function [err, dx] = fd_error_at(Nx, Ny, a, b, u_exact)
    dx = a/(Nx-1); dy = b/(Ny-1);
    x = linspace(0,a,Nx)'; y = linspace(0,b,Ny)';
    [A, rhs, idxmap] = assemble_laplace_fd(Nx, Ny, dx, dy, x, y, u_exact);
    usol = A\rhs;
    Ugrid = zeros(Ny,Nx);
    for j = 2:Ny-1
        for i = 2:Nx-1
            Ugrid(j,i) = usol(idxmap(i,j));
        end
    end
    Ugrid(1,:) = u_exact(x,0); Ugrid(Ny,:) = u_exact(x,b);
    Ugrid(:,1) = u_exact(0,y); Ugrid(:,Nx) = u_exact(a,y);
    [Xg,Yg] = meshgrid(x,y);
    err = max(max(abs(Ugrid - u_exact(Xg,Yg))));
end
