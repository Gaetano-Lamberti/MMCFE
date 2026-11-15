% Lecture 4, Example 1 - Convergence: Euler, Heun and RK4 written from scratch
% Test problem: dy/dt = y - t^2 + 1, y(0) = 0.5, t in [0, 2]
% Exact solution: y(t) = (t + 1)^2 - 0.5 exp(t)
clear; clc;
f = @(t, y) y - t^2 + 1;                 % right-hand side dy/dt = f(t, y)
yExact = @(t) (t + 1).^2 - 0.5*exp(t);
t0 = 0; tf = 2; y0 = 0.5;

hs = [0.2 0.1 0.05 0.025 0.0125];
err = zeros(numel(hs), 3);               % columns: Euler, Heun, RK4
for i = 1:numel(hs)
    h = hs(i); N = round((tf - t0)/h);   % number of steps
    err(i, 1) = abs(euler(f, t0, y0, h, N) - yExact(tf));
    err(i, 2) = abs(heun(f, t0, y0, h, N) - yExact(tf));
    err(i, 3) = abs(rk4(f, t0, y0, h, N) - yExact(tf));
end
disp('     h        Euler         Heun          RK4');
fprintf('%6.4f %12.3e %12.3e %12.3e\n', [hs' err]');
p = log2(err(1:end-1, :)./err(2:end, :));    % observed order
disp('observed order (Euler, Heun, RK4):'); disp(p);

figure; loglog(hs, err, '-o'); grid on
xlabel('step size h'); ylabel('global error at t = 2');
legend('Euler', 'Heun', 'RK4', 'Location', 'southeast');

% ------------------------------------------------------ local functions
function y = euler(f, t, y, h, N)
    for n = 1:N
        y = y + h*f(t, y);
        t = t + h;
    end
end

function y = heun(f, t, y, h, N)
    for n = 1:N
        k1 = f(t, y);
        k2 = f(t + h, y + h*k1);
        y = y + h/2*(k1 + k2);
        t = t + h;
    end
end

function y = rk4(f, t, y, h, N)
    for n = 1:N
        k1 = f(t, y);
        k2 = f(t + h/2, y + h/2*k1);
        k3 = f(t + h/2, y + h/2*k2);
        k4 = f(t + h, y + h*k3);
        y = y + h/6*(k1 + 2*k2 + 2*k3 + k4);
        t = t + h;
    end
end
