%% Lesson 5, Section 3.12.1 - Test case (c): one problem, four methods
% Transient conduction in a slab, solved on the HALF slab (symmetry):
%   du/dt = alpha * d2u/dx2,   0 <= x <= l
%   x = 0 : du/dx = 0   (centre, symmetry)
%   x = l : u = 0       (wall, Dirichlet)
%   IC    : u(x,0) = cos(pi*x/(2l))   -> single-mode exact solution
%   exact : u(x,t) = exp(-alpha*pi^2*t/(4 l^2)) * cos(pi*x/(2l))
%
% The same problem is discretized in space with four methods (Sections 3.3-3.9):
%   1) FD + method of lines  (ghost node at the centre)          -> du/dt = A u
%   2) FEM, Galerkin, linear elements (symmetry is a NATURAL BC)  -> M du/dt = -K u
%      with consistent and with lumped mass matrix
%   3) FVM, cell-centred (zero flux at the centre face)           -> h du/dt = -K u
%   4) Orthogonal collocation (Villadsen-Michelsen, symmetric)    -> du/dt = B u
% Every method gives a LINEAR ODE system. It is integrated EXACTLY in time with
% the matrix exponential (expm), so the error printed is the SPATIAL error only.
%
% HOW TO RUN: press F5. Helper functions are local functions at the end.

clear; clc; close all;

%% --- Parameters (dimensionless) ---
alpha = 1.0;   % diffusivity
l     = 1.0;   % half thickness
tend  = 0.2;   % final time
k     = pi/(2*l);
u_exact = @(x,t) exp(-alpha*k^2*t)*cos(k*x);

%% --- Error vs number of unknowns, for each method ---
Ns    = [2 4 8 16 32];          % unknowns for FD, FEM, FVM
Ns_oc = [1 2 3 4 5 6];          % interior collocation points
err = zeros(4, numel(Ns));      % rows: FD, FEM consistent, FEM lumped, FVM
for j = 1:numel(Ns)
    N = Ns(j);
    [x, M, K] = disc_fd(N, l, alpha);         err(1,j) = spatial_error(x, M, K, tend, u_exact);
    [x, M, K] = disc_fem(N, l, alpha, false); err(2,j) = spatial_error(x, M, K, tend, u_exact);
    [x, M, K] = disc_fem(N, l, alpha, true);  err(3,j) = spatial_error(x, M, K, tend, u_exact);
    [x, M, K] = disc_fvm(N, l, alpha);        err(4,j) = spatial_error(x, M, K, tend, u_exact);
end
err_oc = zeros(1, numel(Ns_oc));
for j = 1:numel(Ns_oc)
    [x, M, K] = disc_oc(Ns_oc(j), l, alpha);  err_oc(j) = spatial_error(x, M, K, tend, u_exact);
end

%% --- Tables ---
names = {'FD + MOL', 'FEM consistent', 'FEM lumped', 'FVM'};
fprintf('Max error at t = %.2f (spatial error only, exact time integration)\n', tend);
fprintf('%-16s', 'N'); fprintf('%11d', Ns); fprintf('\n');
for m = 1:4
    fprintf('%-16s', names{m}); fprintf('%11.2e', err(m,:)); fprintf('\n');
    fprintf('%-16s', '   order p'); fprintf('%11s', '-'); fprintf('%11.2f', log2(err(m,1:end-1)./err(m,2:end))); fprintf('\n');
end
fprintf('\n%-16s', 'Orth. coll. N'); fprintf('%11d', Ns_oc); fprintf('\n');
fprintf('%-16s', 'max error'); fprintf('%11.2e', err_oc); fprintf('\n\n');
fprintf('FD + MOL and FEM with lumped mass give IDENTICAL results: max difference %.1e\n', max(abs(err(1,:) - err(3,:))));

%% --- Plot ---
figure('Name', 'Test case (c): error vs number of unknowns');
loglog(Ns, err(1,:), 's-', 'LineWidth', 2); hold on
loglog(Ns, err(2,:), '^--', 'LineWidth', 1.5);
loglog(Ns, err(4,:), 'd:', 'LineWidth', 1.5);
loglog(Ns_oc(1:5), err_oc(1:5), 'o-', 'LineWidth', 2);
xlabel('Number of unknowns'); ylabel(sprintf('Max error at t = %.1f', tend));
legend('FD + MOL = FEM lumped', 'FEM consistent', 'FVM', 'Orthogonal collocation', 'Location', 'southwest');
grid on;

%% ========================================================================
%  LOCAL FUNCTIONS
%  ========================================================================
function e = spatial_error(x, M, K, tend, u_exact)
% SPATIAL_ERROR  Solve M du/dt = -K u exactly in time (matrix exponential) and
% return the max nodal error at tend.
    u0 = u_exact(x(:), 0);
    u  = expm(-(M\K)*tend) * u0;
    e  = max(abs(u - u_exact(x(:), tend)));
end

function [x, M, K] = disc_fd(N, l, alpha)
% DISC_FD  Central FD on nodes x_i = i h, i = 0..N-1; wall node x_N = l has u = 0.
% Centre: ghost node u_{-1} = u_1 (symmetry). Written as M du/dt = -K u with M = I.
    h = l/N; x = (0:N-1)'*h;
    A = diag(-2*ones(N,1)) + diag(ones(N-1,1), 1) + diag(ones(N-1,1), -1);
    A(1,2) = 2;                                   % ghost node at the centre
    M = eye(N); K = -alpha*A/h^2;
end

function [x, M, K] = disc_fem(N, l, alpha, lumped)
% DISC_FEM  Galerkin with linear (hat) elements; element e joins nodes e and e+1.
% The wall node (u = 0) is removed; the symmetry at x = 0 needs NO special treatment.
    h = l/N; x = (0:N-1)'*h;
    Ke = alpha/h*[1 -1; -1 1];
    if lumped, Me = h/2*eye(2); else, Me = h/6*[2 1; 1 2]; end
    K = zeros(N); M = zeros(N);
    for e = 1:N                                   % assembly
        nodes = [e, e+1];                         % 1-based: node N+1 is the wall
        for a = 1:2
            for b = 1:2
                i = nodes(a); j = nodes(b);
                if i <= N && j <= N
                    K(i,j) = K(i,j) + Ke(a,b);  M(i,j) = M(i,j) + Me(a,b);
                end
            end
        end
    end
end

function [x, M, K] = disc_fvm(N, l, alpha)
% DISC_FVM  Cell-centred finite volumes: N cells of width h, centres (i-1/2) h.
% Centre face: zero flux. Wall face: u = 0 at a distance h/2 from the last centre.
    h = l/N; x = ((1:N)' - 0.5)*h;
    K = zeros(N);
    for i = 1:N
        if i > 1, K(i,i) = K(i,i) + alpha/h; K(i,i-1) = K(i,i-1) - alpha/h; end
        if i < N, K(i,i) = K(i,i) + alpha/h; K(i,i+1) = K(i,i+1) - alpha/h;
        else,     K(i,i) = K(i,i) + alpha/(h/2);                       % wall face
        end
    end
    M = h*eye(N);
end

function [x, M, K] = disc_oc(N, l, alpha)
% DISC_OC  Symmetric orthogonal collocation (slab, a = 1): N interior points + wall.
% Points: x^2 = roots of the Jacobi polynomial with weight (1-s) s^(-1/2) on [0,1].
    t  = jacobi_roots(N, 1, -0.5);                % on [-1, 1]
    xp = [sqrt(sort((1 + t)/2)); 1];              % interior points + wall (x = 1)
    i  = 1:N+1;
    Q  = xp.^(2*i - 2);
    D  = zeros(N+1);
    for c = 2:N+1
        D(:,c) = (2*c - 2)*(2*c - 3)*xp.^(2*c - 4);   % d2/dx2 of x^(2c-2)
    end
    B = D / Q;                                    % Laplacian matrix, Eq. (45)
    x = xp(1:N)*l; M = eye(N); K = -alpha*B(1:N,1:N)/l^2;   % wall value u = 0
end

function t = jacobi_roots(N, a, b)
% JACOBI_ROOTS  Roots of the Jacobi polynomial P_N^(a,b) on [-1,1] (Golub-Welsch):
% eigenvalues of the symmetric tridiagonal matrix of the three-term recurrence.
    k  = (0:N-1)'; s = 2*k + a + b;
    al = (b^2 - a^2)./(s.*(s + 2));
    al(1) = (b - a)/(a + b + 2);
    kk = (1:N-1)'; s2 = 2*kk + a + b;
    be = 4*kk.*(kk + a).*(kk + b).*(kk + a + b)./(s2.^2.*(s2 + 1).*(s2 - 1));
    J  = diag(al) + diag(sqrt(be), 1) + diag(sqrt(be), -1);
    t  = sort(eig(J));
end
