% Lecture 4, Section 2.12 - DAE: batch reactor with a fast equilibrium
%   A <=> B (fast, K = kf/kb),  B -> C (slow, k2)
% DAE (mass-matrix form), unknowns y = [A; B; C]:
%   d(A + B)/dt = -k2 B,   0 = B - K A,   dC/dt = k2 B
clear; clc;
K = 2; k2 = 1; tf = 5;
keff = k2*K/(1 + K);
exact = @(t) [exp(-keff*t)/(1 + K); K*exp(-keff*t)/(1 + K); 1 - exp(-keff*t)];

% ---------------------------------------------------------- DAE with ode15s
M = [1 1 0;                               % row 1: d(A + B)/dt
     0 0 0;                               % row 2: algebraic (no derivative)
     0 0 1];                              % row 3: dC/dt
f = @(t, y) [-k2*y(2); y(2) - K*y(1); k2*y(2)];
y0 = [1/(1 + K); K/(1 + K); 0];           % consistent initial values
opts = odeset('Mass', M, 'MStateDependence', 'none', ...   % constant, singular M
              'RelTol', 1e-6, 'AbsTol', 1e-8);
[t, y] = ode15s(f, [0 tf], y0, opts);
fprintf('DAE (ode15s, mass matrix): max error = %.2e\n', ...
        max(max(abs(y' - exact(t')))));

% ---------------------------------------------------------- same DAE, fully implicit form
F = @(t, y, yp) M*yp - f(t, y);           % F(t, y, y') = 0
yp0 = [-keff*y0(1); -keff*y0(2); k2*y0(2)];  % consistent initial slopes
[t1, y1] = ode15i(F, [0 tf], y0, yp0, odeset('RelTol', 1e-6, 'AbsTol', 1e-8));
fprintf('DAE (ode15i, implicit form): max error = %.2e\n', ...
        max(max(abs(y1' - exact(t1')))));

% ---------------------------------------------------------- stiff ODE (kb finite)
kb = 1e3; kf = K*kb;
g = @(t, y) [-kf*y(1) + kb*y(2); kf*y(1) - kb*y(2) - k2*y(2); k2*y(2)];
[t2, y2] = ode15s(g, [0 tf], [1; 0; 0], odeset('RelTol', 1e-6, 'AbsTol', 1e-8));
late = t2 >= 0.1;
fprintf('stiff ODE, kb = %g: max deviation from the DAE (t >= 0.1) = %.2e\n', ...
        kb, max(max(abs(y2(late, :)' - exact(t2(late)')))));
