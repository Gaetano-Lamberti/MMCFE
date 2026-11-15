% Lecture 4, Example 3 - Non-stiff 2x2 system: Euler and RK4, vector form
% dy/dt = A y, A = [-2 1; 1 -2], y(0) = [2; 0], t in [0, 2]; eig: -1, -3
% Exact: y1 = exp(-t) + exp(-3t), y2 = exp(-t) - exp(-3t)
clear; clc;
A = [-2 1; 1 -2];
f = @(t, y) A*y;                          % vector right-hand side
y0 = [2; 0]; tf = 2;
yExact = @(t) [exp(-t) + exp(-3*t); exp(-t) - exp(-3*t)];
disp('eigenvalues of A:'); disp(eig(A)');

hs = [0.2 0.1 0.05 0.025 0.0125];
eE = zeros(size(hs)); eR = zeros(size(hs));
for i = 1:numel(hs)
    h = hs(i); N = round(tf/h);
    eE(i) = norm(eulerSys(f, y0, h, N) - yExact(tf));   % Euclidean norm
    eR(i) = norm(rk4Sys(f, y0, h, N) - yExact(tf));
end
fprintf('%6.4f %12.3e %12.3e\n', [hs; eE; eR]);
disp('observed order (Euler, RK4):');
disp([log2(eE(1:end-1)./eE(2:end)); log2(eR(1:end-1)./eR(2:end))]);

% ------------------------------------------------------ local functions
function y = eulerSys(f, y, h, N)
    t = 0;
    for n = 1:N
        y = y + h*f(t, y); t = t + h;
    end
end

function y = rk4Sys(f, y, h, N)
    t = 0;
    for n = 1:N
        k1 = f(t, y);                     % all components of k1 first ...
        k2 = f(t + h/2, y + h/2*k1);      % ... then k2, and so on
        k3 = f(t + h/2, y + h/2*k2);
        k4 = f(t + h, y + h*k3);
        y = y + h/6*(k1 + 2*k2 + 2*k3 + k4); t = t + h;
    end
end
