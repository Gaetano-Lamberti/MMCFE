%% Module 1, Part 1 - Theory: Implicit AE, Newton-Raphson and variants
% Generic example, independent of the reactor case study, illustrating
% step-by-step the base method and its variants on a scalar implicit
% equation. Includes convergence plots and "cobweb" visualization.
%
% BASE CASE (clean convergence):
%   f1(x) = x^3 - 2x - 5 = 0      (single real root, ~2.0946)
%
% CRITICAL CASE (why the robust variants are needed):
%   f2(x) = x^3 - 2x + 2 = 0      (single real root, ~-1.7693)
%   Pure Newton-Raphson from x0=0 enters an EXACT cycle: 0 -> 1 -> 0 -> ...
%   Classic textbook case, not an implementation bug.

clear; clc; close all;

%% =============================== 
%  BASE CASE: f1(x) = x^3 - 2x - 5
%  =============================== 
f1  = @(x) x.^3 - 2*x - 5;
f1p = @(x) 3*x.^2 - 2;

tol = 1e-10; maxiter = 50;

%% --- Newton-Raphson, x0 = 3 ---
fprintf('=== BASE CASE: Newton-Raphson, x0=3 ===\n');
fprintf('it        x                f(x)\n');
x = 3.0;
hist_NR.x = x; hist_NR.f = f1(x);
for it = 0:maxiter
    fx = f1(x);
    fprintf('%2d   %.10f   %.3e\n', it, x, fx);
    if abs(fx) < tol, break; end
    x = x - fx/f1p(x);
    hist_NR.x(end+1) = x; hist_NR.f(end+1) = f1(x); 
end
fprintf('Root: x = %.10f\n\n', x);

%% --- Secant method, x0=2, x1=3 (no analytical derivative) ---
fprintf('=== Secant method, x0=2, x1=3 ===\n');
fprintf('it        x                f(x)\n');
x0 = 2.0; x1 = 3.0;
f0 = f1(x0); fval1 = f1(x1);
hist_SEC.x = [x0 x1]; hist_SEC.f = [f0 fval1];
fprintf('%2d   %.10f   %.3e\n', 0, x0, f0);
fprintf('%2d   %.10f   %.3e\n', 1, x1, fval1);
for it = 2:maxiter
    x2 = x1 - fval1*(x1-x0)/(fval1-f0);
    f2 = f1(x2);
    fprintf('%2d   %.10f   %.3e\n', it, x2, f2);
    hist_SEC.x(end+1) = x2; hist_SEC.f(end+1) = f2;
    if abs(f2) < tol, break; end
    x0 = x1; f0 = fval1;
    x1 = x2; fval1 = f2;
end
fprintf(['Root: x = %.10f  (superlinear convergence, ~1.618: ' ...
    'slower than Newton)\n\n'], x2);

%% --- Modified Newton, x0=3 (Jacobian fixed, never recomputed) ---
fprintf('=== Modified Newton, x0=3 (f''(x0) fixed) ===\n');
fprintf('it        x                f(x)\n');
x = 2.0;
fp_fixed = f1p(x);   % computed only once
hist_MOD.x = x; hist_MOD.f = f1(x);
for it = 0:maxiter
    fx = f1(x);
    fprintf('%2d   %.10f   %.3e\n', it, x, fx);
    if abs(fx) < tol, break; end
    x = x - fx/fp_fixed;   % same denominator every iteration
    hist_MOD.x(end+1) = x; hist_MOD.f(end+1) = f1(x);
end
fprintf(['Root: x = %.10f  (linear convergence: %d iterations ' ...
    'instead of 4)\n\n'], x, it);

%% ============================================================
%  CRITICAL CASE: f2(x) = x^3 - 2x + 2  (real root ~ -1.7693)
%  ============================================================
f2fun = @(x) x.^3 - 2*x + 2;
f2p   = @(x) 3*x.^2 - 2;

%% --- Pure Newton-Raphson, x0=0: EXACT CYCLE ---
fprintf('=== CRITICAL CASE: pure Newton-Raphson, x0=0 ===\n');
fprintf('it        x                f(x)\n');
x = 0.0;
hist_CYC.x = x; hist_CYC.f = f2fun(x);
for it = 0:7
    fx = f2fun(x);
    fprintf('%2d   %.6f   %.6f\n', it, x, fx);
    x = x - fx/f2p(x);
    hist_CYC.x(end+1) = x; hist_CYC.f(end+1) = f2fun(x);
end
fprintf('--> Never converges: exact cycle 0 -> 1 -> 0 -> 1 -> ...\n');
fprintf('    (provable analytically, not an implementation error)\n\n');

%% --- "Naive" damped Newton (backtracking on |f| only) ---
% Breaks the cycle, but can get STUCK at a spurious stationary point
% (where f'(x)=0 but f(x)!=0): illustrates the LIMIT of naive damping,
% not just its benefit.
fprintf('=== "Naive" damped Newton (backtracking on |f| only), x0=0 ===\n');
fprintf('it        x                f(x)\n');
x = 0.0;
hist_DAMP.x = x; hist_DAMP.f = f2fun(x);
for it = 0:14
    fx = f2fun(x);
    fprintf('%2d   %.8f   %.6e\n', it, x, fx);
    if abs(fx) < 1e-10, break; end
    fpx = f2p(x);
    dx  = -fx/fpx;
    lam = 1.0;
    while abs(f2fun(x+lam*dx)) >= abs(fx) && lam > 1e-8
        lam = lam/2;
    end
    x = x + lam*dx;
    hist_DAMP.x(end+1) = x; hist_DAMP.f(end+1) = f2fun(x);
end
fprintf('--> Gets stuck near x=0.8165 (critical point, f''(x)=0), NOT a root\n');
fprintf('    (f is about 0.911 there, not zero): limit of naive damping.\n\n');

%% --- Hybrid Newton-bisection (safeguarded): converges correctly ---
% Maintains a bracket [a,b] with f(a)*f(b)<0. At each iteration it tries
% the Newton step; if the result falls outside the bracket, it uses a
% bisection step instead. ALWAYS guarantees progress toward a true root
% (bisection), with Newton's acceleration whenever possible.
fprintf('=== Hybrid Newton-bisection (safeguarded), initial bracket [-2, 0] ===\n');
fprintf('it        x                f(x)             bracket\n');
a = -2.0; b = 0.0;
fa = f2fun(a); fb = f2fun(b);
if fa*fb >= 0
    error('Invalid bracket: f(a) and f(b) must have opposite signs.');
end
x = 0.0;   % same "bad" guess as before
hist_HYB.x = x; hist_HYB.f = f2fun(x);
for it = 0:14
    fx = f2fun(x);
    fprintf('%2d   %.10f   %.6e   [%.6f, %.6f]\n', it, x, fx, a, b);
    if abs(fx) < 1e-12, break; end
    % update the bracket
    if fa*fx < 0
        b = x; fb = fx;
    else
        a = x; fa = fx;
    end
    % try the Newton step; accept it only if it stays inside the bracket
    fpx = f2p(x);
    accepted = false;
    if fpx ~= 0
        x_newton = x - fx/fpx;
        if x_newton > a && x_newton < b
            x = x_newton;
            accepted = true;
        end
    end
    if ~accepted
        x = 0.5*(a+b);   % fallback: bisection step
    end
    hist_HYB.x(end+1) = x; hist_HYB.f(end+1) = f2fun(x);
end
fprintf('--> Converges correctly to the true root: x = %.10f\n', x);
fprintf('    This is exactly the role of bisection as a "safety net" (S16.2):\n');
fprintf('    it does not replace Newton, it guarantees progress when Newton fails.\n\n');

%% ============================================================
%  PLOTS
%  ============================================================

%% --- Figure 1: base case - cobweb + convergence ---
figure('Name','Base case: f1(x) = x^3 - 2x - 5');

subplot(1,2,1);
xg = linspace(1.5, 3.2, 300);
plot(xg, f1(xg), 'b-', 'LineWidth', 1.8); hold on
plot(xg, zeros(size(xg)), 'k-');
plot(2.0945514815, 0, 'kp', 'MarkerSize', 14, 'MarkerFaceColor','y');
plot(hist_NR.x, hist_NR.f, '--ro', 'LineWidth', 1.2, 'MarkerFaceColor','r');
plot(hist_SEC.x, hist_SEC.f, '--gs', 'LineWidth', 1.2, 'MarkerFaceColor','g');
plot(hist_MOD.x, hist_MOD.f, '--m^', 'LineWidth', 1.0, 'MarkerFaceColor','m', 'MarkerSize',4);
xlabel('x'); ylabel('f_1(x)');
title('f_1(x)=x^3-2x-5: iteration path');
legend('f_1(x)','f_1(x)=0','root','Newton-Raphson','Secant','Modified Newton', ...
       'Location','southeast');
grid on;

subplot(1,2,2);
semilogy(0:length(hist_NR.f)-1, abs(hist_NR.f), '-ro', 'LineWidth',1.5,'MarkerFaceColor','r'); hold on
semilogy(0:length(hist_SEC.f)-1, abs(hist_SEC.f), '-gs', 'LineWidth',1.5,'MarkerFaceColor','g');
semilogy(0:length(hist_MOD.f)-1, abs(hist_MOD.f), '-m^', 'LineWidth',1.5,'MarkerFaceColor','m');
xlabel('Iteration'); ylabel('|f_1(x)|  (log scale)');
title('Convergence comparison');
legend(sprintf('Newton-Raphson (%d it.)', length(hist_NR.f)-1), ...
       sprintf('Secant (%d it.)', length(hist_SEC.f)-1), ...
       sprintf('Modified Newton (%d it.)', length(hist_MOD.f)-1), ...
       'Location','northeast');
grid on;

%% --- Figure 2: critical case - cycle, naive damping, hybrid ---
figure('Name','Critical case: f2(x) = x^3 - 2x + 2');

subplot(1,2,1);
xg2 = linspace(-2.2, 1.5, 300);
plot(xg2, f2fun(xg2), 'b-', 'LineWidth', 1.8); hold on
plot(xg2, zeros(size(xg2)), 'k-');
plot(-1.7692923542, 0, 'kp', 'MarkerSize', 14, 'MarkerFaceColor','y');
plot(hist_CYC.x, hist_CYC.f, '--rx', 'LineWidth', 1.5, 'MarkerSize', 10);
plot(hist_DAMP.x, hist_DAMP.f, '--c^', 'LineWidth', 1.2, 'MarkerFaceColor','c', 'MarkerSize',5);
plot(hist_HYB.x, hist_HYB.f, '--o', 'Color',[0 0.6 0], 'LineWidth', 1.2, ...
     'MarkerFaceColor',[0 0.6 0]);
xlabel('x'); ylabel('f_2(x)');
title('f_2(x)=x^3-2x+2: pure Newton cycles, "naive" gets stuck, hybrid converges');
legend('f_2(x)','f_2(x)=0','true root','pure NR (cycle)', ...
       'naive damping (stuck)','hybrid Newton-bisection (converges)', ...
       'Location','northwest');
grid on;

subplot(1,2,2);
plot(0:length(hist_CYC.x)-1, hist_CYC.x, '-rx', 'LineWidth',1.5,'MarkerSize',8); hold on
plot(0:length(hist_DAMP.x)-1, hist_DAMP.x, '-c^', 'LineWidth',1.5,'MarkerFaceColor','c');
plot(0:length(hist_HYB.x)-1, hist_HYB.x, '-o', 'Color',[0 0.6 0], 'LineWidth',1.5, ...
     'MarkerFaceColor',[0 0.6 0]);
yline_val = -1.7692923542;
plot(xlim, [yline_val yline_val], 'k:', 'LineWidth', 1);
xlabel('Iteration'); ylabel('x_n');
title('Trajectory of x_n: cycle vs stagnation vs convergence');
legend('pure NR (0/1 cycle)','naive damping (stalls ~0.82)', ...
       'hybrid (converges to -1.769)','true root','Location','east');
grid on;

%% --- Save figures ---
exportgraphics(figure(1), 'fig1_base_case_m.png', 'Resolution', 300);
exportgraphics(figure(2), 'fig2_critical_case_m.png', 'Resolution', 300);
