% Lecture 4, Example 3-bis - Undamped oscillator (imaginary eigenvalues)
% y'' + w^2 y = 0  ->  y1' = y2, y2' = -w^2 y1;  w = 1, y(0) = 1, y'(0) = 0
% Exact amplitude sqrt(y1^2 + y2^2) = 1 for all t
clear; clc;
w = 1; h = 0.1; tf = 20; N = round(tf/h);
A = [0 1; -w^2 0];
f = @(y) A*y;

Y = zeros(2, N + 1, 3); Y(:, 1, :) = repmat([1; 0], 1, 1, 3);
for n = 1:N
    y = Y(:, n, 1); Y(:, n + 1, 1) = y + h*f(y);                    % Euler
    y = Y(:, n, 2); k1 = f(y); k2 = f(y + h*k1);
    Y(:, n + 1, 2) = y + h/2*(k1 + k2);                             % Heun
    y = Y(:, n, 3); k1 = f(y); k2 = f(y + h/2*k1);
    k3 = f(y + h/2*k2); k4 = f(y + h*k3);
    Y(:, n + 1, 3) = y + h/6*(k1 + 2*k2 + 2*k3 + k4);               % RK4
end
names = {'Euler', 'Heun', 'RK4'};
for m = 1:3
    fprintf('%-5s: amplitude at t = %g: %.6f\n', names{m}, tf, ...
            norm(Y(:, end, m)));
end

t = linspace(0, tf, N + 1);
figure; plot(t, squeeze(sqrt(sum(Y.^2, 1)))); grid on
xlabel('t'); ylabel('amplitude'); legend(names, 'Location', 'northwest');
