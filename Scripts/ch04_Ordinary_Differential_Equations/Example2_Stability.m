% Lecture 4, Example 2 - Absolute stability of explicit Euler (and RK4)
% Test equation: dy/dt = lam*y, lam = -10, y(0) = 1, t in [0, 2]
% Euler stable for h < 2/|lam| = 0.2; RK4 for h < 2.785/|lam| = 0.2785
clear; clc;
lam = -10; y0 = 1; tf = 2;
hs = [0.05 0.15 0.2 0.25];

disp('    h    z = h*lam   R_Euler   |y_N| Euler    R_RK4     |y_N| RK4');
figure;
for i = 1:numel(hs)
    h = hs(i); N = round(tf/h); z = h*lam;
    RE = 1 + z;                              % Euler: y_(n+1) = (1+z) y_n
    RR = 1 + z + z^2/2 + z^3/6 + z^4/24;         % RK4 stability function
    yE = y0*RE.^(0:N);                           % Euler solution at t_n
    yR = y0*RR.^(0:N);                           % RK4 solution at t_n
    fprintf('%5.2f %9.2f %10.3f %12.3e %10.4f %12.3e\n', h, z, RE, ...
            abs(yE(end)), RR, abs(yR(end)));
    subplot(2, 2, i)
    t = linspace(0, tf, N + 1); tt = linspace(0, tf, 400);
    plot(tt, exp(lam*tt), '-', t, yE, 'o-'); grid on
    title(sprintf('h = %g, z = %g', h, z));
end
