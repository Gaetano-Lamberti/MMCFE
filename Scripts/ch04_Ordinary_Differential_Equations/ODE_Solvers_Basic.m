% Lecture 4, Section 2.10.1 - Basic call and tolerances (MATLAB ode45)
%
% Test problem of Section 2.2: dy/dt = y - t^2 + 1, y(0) = 0.5, t in [0, 2]
% Exact solution:               y(t) = (t + 1)^2 - 0.5 exp(t)
clear; clc;

f = @(t, y) y - t.^2 + 1;                 % right-hand side dy/dt = f(t, y)
tspan = linspace(0, 2, 11);              % output times only (not the steps)
y0 = 0.5;
opts = odeset('RelTol', 1e-6, 'AbsTol', 1e-9);   % tolerances (2.8.4)

[t, y] = ode45(f, tspan, y0, opts);      % Dormand-Prince 5(4)

yExact = (t + 1).^2 - 0.5*exp(t);
fprintf('max error at output times = %.2e\n', max(abs(y - yExact)));

sol = ode45(f, [0 2], y0, opts);         % structure: steps and statistics
fprintf('steps = %d, f-evals = %d\n', sol.stats.nsteps, sol.stats.nfevals);
