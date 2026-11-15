% Lecture 4, Example 4 - Stiffness: explicit vs implicit, built-in solvers
% dy/dt = A y, A = diag(-1, -1000), y(0) = [1; 1], t in [0, 1]
% Exact: y1 = exp(-t), y2 = exp(-1000 t); stiffness ratio 1000
clear; clc;
A = diag([-1 -1000]); I = eye(2);
y0 = [1; 1]; tf = 1;

% ------------------------------------------ explicit Euler, h = 0.1
h = 0.1; y = y0;
disp('explicit Euler, h = 0.1 (stability limit 2/1000 = 0.002):');
for n = 0:3
    fprintf('  t = %.1f   y1 = %.6f   y2 = %.6f\n', n*h, y(1), y(2));
    y = y + h*A*y;
end

% ------------------------------- implicit Euler, Crank-Nicolson, h = 0.1
N = round(tf/h); yIE = y0; yCN = y0;
for n = 1:N
    yIE = (I - h*A) \ yIE;                          % (I - hA) y_(n+1) = y_n
    yCN = (I - h/2*A) \ ((I + h/2*A)*yCN);          % Crank-Nicolson
end
fprintf('implicit Euler: error y1(1) = %.3e\n', abs(yIE(1) - exp(-1)));
fprintf('Crank-Nicolson: error y1(1) = %.3e\n', abs(yCN(1) - exp(-1)));

% ------------------------------------------ built-in solvers
opts = odeset('RelTol', 1e-6, 'AbsTol', 1e-9);
s45 = ode45(@(t, y) A*y, [0 tf], y0, opts);
s15 = ode15s(@(t, y) A*y, [0 tf], y0, opts);
fprintf('ode45 : steps = %d, f-evals = %d\n', ...
        s45.stats.nsteps, s45.stats.nfevals);
fprintf('ode15s: steps = %d, f-evals = %d\n', ...
        s15.stats.nsteps, s15.stats.nfevals);
