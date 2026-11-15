% Lab7_GlobalFit_4Params.m
% Lecture 6 - Parameter estimation, Case history 4.15.4 (guiding example, Section 5.5)
%
% One-stage (global) estimation of the four Arrhenius parameters p = (k1ref, Ea1, k2ref, Ea2), centred at
% T0 = 900 K, directly from all the pressure and selectivity data of the three runs (N = 912), by
% Levenberg-Marquardt with the analytic Jacobian obtained by the chain rule. Start deliberately generic.
% No toolbox required (lsqnonlin, if present, is used only as a cross-check).

clear; close all; clc

c = lab_settings();
runs = generate_data(c);
p0 = [0.5; 100; 0.05; 150];                      % deliberately generic start
[p, sse, it, hist] = fit_global(runs, p0, c);
N = numel(runs) * (numel(c.tP) + numel(c.tGC));
fprintf('Lab 7 - global one-stage fit of the four Arrhenius parameters (centred at T0 = %.0f K)\n', c.T0);

%% Jacobian check
Ja = jacobian_global(p0, runs, c);  Jn = zeros(size(Ja));
for j = 1:4
    dp = p0;  h = 1e-7 * p0(j);  dp(j) = dp(j) + h;
    Jn(:, j) = (residuals_global(dp, runs, c) - residuals_global(p0, runs, c)) / h;
end
fprintf('  Jacobian check: max |analytic - finite difference| / max |J| = %.1e\n', ...
        max(abs(Ja(:) - Jn(:))) / max(abs(Ja(:))));

%% results
fprintf('  start: k1ref = %.3f, Ea1 = %.1f, k2ref = %.3f, Ea2 = %.1f\n', p0);
fprintf('  LM iterations: %d, N = %d, nu = %d, SSE_w = %.1f, s2 = %.3f\n', it, N, N - 4, sse, sse / (N - 4));
names = {'k1ref [1/s]', 'Ea1 [kJ/mol]', 'k2ref [1/s]', 'Ea2 [kJ/mol]'};
for j = 1:4
    fprintf('    %-13s %10.4f   (true %.4f)\n', names{j}, p(j), c.pTrue(j));
end
fprintf('  SSE history: [%s]\n', strjoin(arrayfun(@(v) sprintf('%.1e', v), hist, 'UniformOutput', false), ' '));

%% optional cross-check with a library LM (Optimization Toolbox)
if exist('lsqnonlin', 'file') == 2
    opts = optimoptions('lsqnonlin', 'Algorithm', 'levenberg-marquardt', 'SpecifyObjectiveGradient', true, ...
                        'StepTolerance', 1e-12, 'FunctionTolerance', 1e-12, 'Display', 'off');
    fun = @(q) deal(residuals_global(q, runs, c), jacobian_global(q, runs, c));
    x = lsqnonlin(fun, p0, [], [], opts);
    fprintf('  lsqnonlin: max relative difference %.1e\n', max(abs(x ./ p - 1)));
end

%% comparison with the two-stage approach (Labs 5-6)
fits = fit_all_runs(runs, c);
fprintf('  per-temperature k from the global fit vs two-stage (Lab 5):\n');
for i = 1:numel(fits)
    [k1, k2] = k_arrhenius(p, fits(i).T, c);
    fprintf('    T = %.0f K: k1 %.4f vs %.4f, k2 %.4f vs %.4f\n', fits(i).T, k1, fits(i).p(1), k2, fits(i).p(2));
end

%% figure
fig = figure('Position', [100 100 1000 380]);
subplot(1, 2, 1)
semilogy(0:numel(hist) - 1, hist, 'o-', 'Color', [35 63 152] / 255); box on
xlabel('LM iteration'); ylabel('weighted SSE')
subplot(1, 2, 2)
xx = linspace(1 / 940, 1 / 840, 100)';
[k1c, k2c] = k_arrhenius(p, 1 ./ xx, c);
h1 = semilogy(1000 * xx, k1c, '-', 'Color', [35 63 152] / 255); hold on; box on
h2 = semilogy(1000 * xx, k2c, '-', 'Color', [211 84 0] / 255);
for i = 1:numel(fits)
    semilogy(1000 / fits(i).T, fits(i).p(1), 'ko')
    semilogy(1000 / fits(i).T, fits(i).p(2), 'ks')
end
xlabel('1000/T [1/K]'); ylabel('k [1/s]'); legend([h1 h2], 'k1, global fit', 'k2, global fit')
title('symbols: per-temperature estimates (Lab 5)')
print(fig, 'Lab7_global_matlab.png', '-dpng', '-r150');

%% ======================================================================= local functions
% (MATLAB R2016b or later: local functions in a script must be at the end of the file)

function c = lab_settings()
% Settings shared by Labs 4-8 (guiding example, Sections 1.5 and 5.1-5.3).
c.R     = 8.314e-3;                    % kJ/(mol K)
c.T0    = 900;                         % K, reference temperature of the centred Arrhenius law
c.pTrue = [0.7705; 90.98; 0.0764; 207.99];   % k1ref [1/s], Ea1 [kJ/mol], k2ref [1/s], Ea2 [kJ/mol]
c.Truns = [850; 900; 923];             % K, the three isothermal runs
c.P0    = 100;                         % kPa, initial pressure (pure EB), all runs
c.tP    = linspace(0, 15, 300)';       % s, pressure sampling (300 points per run)
c.tGC   = [2; 5; 9; 13];               % s, GC sampling (4 per run)
c.sigP  = 0.20;                        % kPa, transducer noise (standard deviation)
c.sigS  = 0.01;                        % -,   GC selectivity noise (standard deviation)
c.seed  = 20261012;                    % seed of the portable generator
end

function [z, state] = prng_normal(n, state)
% n standard normal numbers: Park-Miller "minimal standard" generator
% (x <- 16807 x mod 2^31-1) + Box-Muller transform (cosine branch).
% All products are exact in double precision, so MATLAB and Python give the SAME sequence.
z = zeros(n, 1);
for i = 1:n
    state = mod(16807 * state, 2147483647);  u1 = state / 2147483647;
    state = mod(16807 * state, 2147483647);  u2 = state / 2147483647;
    z(i) = sqrt(-2 * log(u1)) * cos(2 * pi * u2);
end
end

function [k1, k2] = k_arrhenius(p, T, c)
% k1, k2 at temperature T from p = (k1ref, Ea1, k2ref, Ea2), Arrhenius law centred at T0:
%   k_i(T) = k_i,ref * exp(-Ea_i/R * (1/T - 1/T0))
tau = 1 ./ T - 1 / c.T0;
k1 = p(1) * exp(-p(2) / c.R * tau);
k2 = p(3) * exp(-p(4) / c.R * tau);
end

function runs = generate_data(c)
% Pseudo-data of the three runs: struct array with T, P (300x1), S (4x1) and the true values.
% Noise order: for each T, first the 300 P values, then the 4 S values (generator state carried over).
state = c.seed;
for i = 1:numel(c.Truns)
    T = c.Truns(i);
    [k1, k2] = k_arrhenius(c.pTrue, T, c);
    K = k1 + k2;
    Ptrue = c.P0 * (2 - exp(-K * c.tP));      % P(t) = P0 (2 - exp(-(k1+k2) t))
    Strue = k1 / K;                           % S = k1/(k1+k2)
    [e, state] = prng_normal(numel(c.tP), state);
    Pmeas = Ptrue + c.sigP * e;
    [e, state] = prng_normal(numel(c.tGC), state);
    Smeas = Strue + c.sigS * e;
    runs(i) = struct('T', T, 'k1', k1, 'k2', k2, 'P', Pmeas, 'S', Smeas, ...
                     'Ptrue', Ptrue, 'Strue', Strue); %#ok<AGROW>
end
end

function [p, sse, it, hist] = lm(res, jac, p0)
% Levenberg-Marquardt from scratch (Section 4.9.3), Marquardt diagonal scaling:
%   (J'J + lambda diag(J'J)) delta = -J' r
% res(p): weighted residual vector; jac(p): its Jacobian.
% lambda /10 after an accepted step, x10 after a rejected one; stop when the relative step
% or the relative SSE decrease falls below tol. Returns p, SSE, iterations, SSE history.
lam = 1e-3;  tol = 1e-10;  maxit = 200;
p = p0(:);
r = res(p);  sse = r' * r;  hist = sse;
for it = 1:maxit
    J = jac(p);  A = J' * J;  g = J' * r;
    while true
        delta = (A + lam * diag(diag(A))) \ (-g);
        rNew = res(p + delta);  sseNew = rNew' * rNew;
        if sseNew < sse                       % accept the step
            lam = max(lam / 10, 1e-15);
            break
        end
        lam = lam * 10;                       % reject: more damping
        if lam > 1e15
            return
        end
    end
    p = p + delta;  r = rNew;
    conv = max(abs(delta) ./ max(abs(p), 1e-12)) < tol || (sse - sseNew) < tol * sseNew;
    sse = sseNew;  hist(end + 1) = sse; %#ok<AGROW>
    if conv
        break
    end
end
end

% ------------------------------------------------------------------ model of one run, p = (k1, k2)
function r = residuals_run(p, run, c)
% Weighted residuals: 300 pressures (sigma_P) followed by 4 selectivities (sigma_S).
K = p(1) + p(2);
rP = (run.P - c.P0 * (2 - exp(-K * c.tP))) / c.sigP;
rS = (run.S - p(1) / K) / c.sigS;
r = [rP; rS];
end

function J = jacobian_run(p, run, c) %#ok<INUSL>
% Analytic Jacobian of the weighted residuals (guiding example, Section 5.3).
% The P block has two IDENTICAL columns: from P(t) alone only k1+k2 is identifiable.
K = p(1) + p(2);
dP = -c.P0 * c.tP .* exp(-K * c.tP) / c.sigP;
JP = [dP, dP];
JS = repmat([-p(2) / (c.sigS * K^2), p(1) / (c.sigS * K^2)], numel(c.tGC), 1);
J = [JP; JS];
end

function p0 = initial_guess(run, c)
% Non-arbitrary start: ln(2 - P/P0) = -(k1+k2) t is linear in t (through the origin);
% use the points with conversion below 50 %, then split K with the mean measured selectivity.
y = 2 - run.P / c.P0;
m = (c.tP > 0) & (y > 0.5);
K0 = -sum(c.tP(m) .* log(y(m))) / sum(c.tP(m).^2);
S0 = mean(run.S);
p0 = [S0 * K0; (1 - S0) * K0];
end

function fits = fit_all_runs(runs, c)
% Stage 1: k1, k2 at each temperature, with standard errors and correlation.
for i = 1:numel(runs)
    run = runs(i);
    p0 = initial_guess(run, c);
    [p, sse, it, hist] = lm(@(q) residuals_run(q, run, c), @(q) jacobian_run(q, run, c), p0);
    J = jacobian_run(p, run, c);
    nu = numel(c.tP) + numel(c.tGC) - 2;
    C = (sse / nu) * inv(J' * J);                           %#ok<MINV> covariance, Eq. (45)
    fits(i) = struct('T', run.T, 'p0', p0, 'p', p, 'sse', sse, 'it', it, 'hist', hist, ...
                     'se', sqrt(diag(C)), 'rho', C(1, 2) / sqrt(C(1, 1) * C(2, 2)), 'nu', nu); %#ok<AGROW>
end
end

% ------------------------------------------------------------------ global model, p = (k1ref, Ea1, k2ref, Ea2)
function r = residuals_global(p, runs, c)
% Weighted residuals of all three runs stacked (N = 3 x 304 = 912).
r = [];
for i = 1:numel(runs)
    [k1, k2] = k_arrhenius(p, runs(i).T, c);
    r = [r; residuals_run([k1; k2], runs(i), c)]; %#ok<AGROW>
end
end

function J = jacobian_global(p, runs, c)
% Chain rule: dr/dp = (dr/dk_i)(dk_i/dp), with dk_i/dk_i,ref = k_i/k_i,ref and
% dk_i/dEa_i = -k_i (1/T - 1/T0)/R. Every block depends on all four parameters through k1(T), k2(T).
J = [];
for i = 1:numel(runs)
    T = runs(i).T;
    [k1, k2] = k_arrhenius(p, T, c);
    Jk = jacobian_run([k1; k2], runs(i), c);        % columns: dr/dk1, dr/dk2
    tau = 1 / T - 1 / c.T0;
    J = [J; Jk(:, 1) * k1 / p(1), Jk(:, 1) * (-k1 * tau / c.R), ...
            Jk(:, 2) * k2 / p(3), Jk(:, 2) * (-k2 * tau / c.R)]; %#ok<AGROW>
end
end

function [p, sse, it, hist] = fit_global(runs, p0, c)
[p, sse, it, hist] = lm(@(q) residuals_global(q, runs, c), @(q) jacobian_global(q, runs, c), p0);
end
