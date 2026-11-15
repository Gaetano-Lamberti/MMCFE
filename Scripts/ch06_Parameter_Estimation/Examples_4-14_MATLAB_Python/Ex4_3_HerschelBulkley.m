% Ex4_3_HerschelBulkley.m
% Lecture 6 - Parameter estimation, Section 4.14.5-4.14.6, Example 4.3: Herschel-Bulkley fluid
%
% Flow curve of a ketchup-like sauce (rotational rheometer, 12 shear rates from 1 to 200 1/s,
% relative error 3 %), Herschel-Bulkley model (Eq. 53):
%     tau = tau0 + K * gdot^n        tau0: yield stress [Pa], K: consistency [Pa s^n], n: flow index
% Algorithms:
%   power law (tau0 = 0) by linear regression in log-log (a model error: the fluid has a yield stress);
%   Herschel-Bulkley by Levenberg-Marquardt and by separable least squares: for a fixed n the model is LINEAR in (tau0, K), so a 1-D search on n
%   (fminbnd) with weighted linear least squares inside.
% Weights: w_i = 1/(0.03 tau_i)^2 (relative error). Pseudo-data: tau0 = 15 Pa, K = 6 Pa s^n, n = 0.35;
% the noisy values are written below, identical to those used for the slides (Python, numpy seed 7).
% No toolbox required: LM is written out below; fminbnd is base MATLAB.
% Derivative-free and global methods on this problem (Nelder-Mead, differential evolution): Lecture 7.
% Evaluation counts depend on the implementation and differ from the scipy ones on the slide.

clear; close all; clc

%% data
gdot = logspace(0, log10(200), 12)';                          % 1/s
tau  = [21.0007749966 22.299841585 23.2133340033 24.2827663737 26.4110880922 28.0778513383 ...
        31.5551173607 35.9160436669 37.5508272312 41.5694278141 48.0777983238 53.8985688389]';   % Pa
sig  = 0.03 * tau;                                            % standard deviations (relative error 3 %)
N = numel(tau);
hb = @(g, p) p(1) + p(2) * g.^p(3);
resHB = @(p) (tau - hb(gdot, p)) ./ sig;

%% power law, linear regression in log-log
b = polyfit(log(gdot), log(tau), 1);
Kpl = exp(b(2));  npl = b(1);
fprintf('4.3 power law (log-log): K=%.2f n=%.3f\n', Kpl, npl);
r = (tau - Kpl * gdot.^npl) ./ sig;
fprintf('    power law  SSE_w=%.1f  s2=%.1f\n', r' * r, r' * r / (N - 2));

%% Herschel-Bulkley, two algorithms
p0 = [1; Kpl; npl];                                           % start from the power law
names = {'LM', 'separable (1-D on n)'};
P = zeros(3, 2);  nev = zeros(1, 2);
[P(:, 1), ~, nev(1)] = lm_fd(resHB, p0);
lin = @(n) linear_part(n, gdot, tau, sig);
[nOpt, ~, ~, out] = fminbnd(@(n) lin(n), 0.05, 1.5, optimset('TolX', 1e-10));
[~, q] = lin(nOpt);
P(:, 2) = [q; nOpt];  nev(2) = out.funcCount;
for j = 1:2
    r = resHB(P(:, j));
    fprintf('4.3 %-24s tau0=%.2f K=%.3f n=%.4f  SSE_w=%.2f  evals=%d\n', names{j}, P(:, j), r' * r, nev(j));
end

%% uncertainty at the LM solution: Eq. (45), finite-difference Jacobian
pH = P(:, 1);  r = resHB(pH);  J = zeros(N, 3);
for j = 1:3
    dp = pH;  h = 1e-7 * abs(pH(j));  dp(j) = dp(j) + h;
    J(:, j) = (resHB(dp) - r) / h;
end
s2 = r' * r / (N - 3);
C = s2 * inv(J' * J);                                         %#ok<MINV>
se = sqrt(diag(C));  Rm = C ./ (se * se');
fprintf('4.3 SE: tau0 %.2f K %.2f n %.3f  s2=%.2f ; rho(tau0,K)=%.2f rho(tau0,n)=%.2f rho(K,n)=%.2f\n', ...
        se, s2, Rm(1, 2), Rm(1, 3), Rm(2, 3));

%% figure: data and fits; profile of the SSE on n (minimum over tau0 and K)
fig = figure('Position', [100 100 960 360]);
subplot(1, 2, 1)
gg = logspace(0, log10(200), 200)';
errorbar(gdot, tau, sig, 'ko'); hold on; box on
plot(gg, Kpl * gg.^npl, '--', 'Color', [211 84 0] / 255, 'LineWidth', 2)
plot(gg, hb(gg, pH), '-', 'Color', [35 63 152] / 255, 'LineWidth', 2)
set(gca, 'XScale', 'log', 'YScale', 'log')
xlim([0.8 250]); ylim([15 65])
xlabel('shear rate [1/s]'); ylabel('shear stress \tau [Pa]')
legend('data', 'power law (log-log)', 'Herschel-Bulkley, LM', 'Location', 'northwest')
subplot(1, 2, 2)
nn = linspace(0.15, 0.7, 200);  prof = arrayfun(@(v) lin(v), nn);
semilogy(nn, prof, '-', 'Color', [35 63 152] / 255, 'LineWidth', 2); hold on; box on
semilogy(nOpt, lin(nOpt), 'o', 'Color', [211 84 0] / 255, 'MarkerSize', 8)
xlabel('flow index n (fixed)'); ylabel('min SSE_w over \tau_0, K')
print(fig, 'Ex4_3_herschel_bulkley.png', '-dpng', '-r150');


%% ======================================================================= local functions

function [sse, q] = linear_part(n, gdot, tau, sig)
% For a fixed n the model is linear in (tau0, K): weighted linear least squares (Section 4.8).
A = [ones(size(gdot)), gdot.^n] ./ sig;
q = A \ (tau ./ sig);
rr = tau ./ sig - A * q;
sse = rr' * rr;
end

function [p, ok, nev] = lm_fd(res, p0, tol)
% Levenberg-Marquardt with Marquardt diagonal scaling (Section 4.9.3) and forward-difference
% Jacobian (step 1e-6 relative). nev counts the residual evaluations (here: PDE solutions).
% tol: relative SSE decrease that stops the iterations (default 1e-10).
if nargin < 3, tol = 1e-10; end
lam = 1e-3;  xtol = 1e-8;  maxit = 300;  hRel = 1e-6;
p = p0(:);  r = res(p);  f = r' * r;  nev = 1;  ok = false;
for it = 1:maxit
    J = zeros(numel(r), numel(p));
    for j = 1:numel(p)
        dp = p;  h = hRel * max(abs(p(j)), 1e-12);  dp(j) = dp(j) + h;
        J(:, j) = (res(dp) - r) / h;  nev = nev + 1;
    end
    A = J' * J;  g = J' * r;
    while true
        d = (A + lam * diag(diag(A))) \ (-g);
        if max(abs(d) ./ max(abs(p), 1e-12)) < xtol
            ok = true;  return                      % the step has become negligible
        end
        rn = res(p + d);  nev = nev + 1;  fn = rn' * rn;
        if fn < f
            lam = lam / 10;  break                  % accept
        end
        lam = lam * 10;                             % reject
        if lam > 1e14
            return
        end
    end
    p = p + d;  r = rn;
    if (f - fn) < tol * fn || max(abs(d) ./ max(abs(p), 1e-12)) < xtol
        ok = true;  return
    end
    f = fn;
end
end
