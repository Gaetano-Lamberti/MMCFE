% Lecture 4, Section 2.11 - Boundary value problem: steady conduction in a wall
% with temperature-dependent conductivity and internal heat generation.
%
%   d/dxi[(1 + beta*theta) dtheta/dxi] + G = 0,  theta(0) = 1, theta(1) = 0
%   beta = 1, G = 4.  Exact solution via the Kirchhoff transform
%   u = theta + beta*theta^2/2  (u'' = -G).
clear; clc;
beta = 1; G = 4;
u0 = 1 + beta/2;
thetaExact = @(xi) (-1 + sqrt(1 + 2*beta*(-G*xi.^2/2 + (G/2 - u0)*xi + u0)))/beta;

% first-order system: y1 = theta, y2 = (1 + beta*theta) dtheta/dxi
rhs = @(xi, y) [y(2)/(1 + beta*y(1)); -G];

% ---------------------------------------------------------- 1) shooting
opts = odeset('RelTol', 1e-10, 'AbsTol', 1e-12);
F = @(s) shootResidual(s, rhs, beta, opts);   % theta(1; s) - 0
s0 = 0; s1 = 1;                               % two starting guesses (secant)
for k = 1:20
    s2 = s1 - F(s1)*(s1 - s0)/(F(s1) - F(s0));
    s0 = s1; s1 = s2;
    fprintf('secant iteration %2d: s = %.10f\n', k, s1);
    if abs(F(s1)) < 1e-12, break; end
end

% ---------------------------------------------------------- 2) finite differences + Newton
for N = [10 20 40 80]
    [xi, th] = fdSolve(N, beta, G);
    fprintf('FD, N = %3d: max error = %.3e\n', N, max(abs(th - thetaExact(xi))));
end

% ---------------------------------------------------------- 3) bvp4c (collocation)
bc = @(ya, yb) [ya(1) - 1; yb(1) - 0];        % boundary-condition residuals
solinit = bvpinit(linspace(0, 1, 5), [1; 0]); % initial mesh and guess
sol = bvp4c(rhs, bc, solinit, bvpset('RelTol', 1e-8));
xx = linspace(0, 1, 201);
yy = deval(sol, xx);
fprintf('bvp4c: mesh nodes = %d, max error = %.3e\n', numel(sol.x), ...
        max(abs(yy(1, :) - thetaExact(xx))));

% ------------------------------------------------------ local functions
function r = shootResidual(s, rhs, beta, opts)
    [~, y] = ode45(rhs, [0 1], [1; (1 + beta)*s], opts);
    r = y(end, 1) - 0;                        % miss at the right boundary
end

function [xi, th] = fdSolve(N, beta, G)
    % expanded form: (1 + beta*th) th'' + beta th'^2 + G = 0, central differences
    xi = linspace(0, 1, N + 1)'; d = 1/N;
    th = 1 - xi;                              % initial guess: linear profile
    for it = 1:30
        c   = 1 + beta*th(2:end-1);
        lap = (th(3:end) - 2*th(2:end-1) + th(1:end-2))/d^2;
        grd = (th(3:end) - th(1:end-2))/(2*d);
        r   = c.*lap + beta*grd.^2 + G;       % residuals at interior nodes
        up  = c/d^2 + beta*grd/d;             % dr_i/dth_(i+1)
        lo  = c/d^2 - beta*grd/d;             % dr_i/dth_(i-1)
        J   = diag(beta*lap - 2*c/d^2) + diag(up(1:end-1), 1) + diag(lo(2:end), -1);
        dth = -J\r;
        th(2:end-1) = th(2:end-1) + dth;
        if max(abs(dth)) < 1e-12, break; end
    end
end
