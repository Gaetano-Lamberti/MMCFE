% Lab5_BatchFitting_LM_k1k2.m
% Lecture 6 - Parameter estimation, Case history 4.15.2 (guiding example, Sections 5.2-5.3)
%
% For each temperature, k1 and k2 are estimated from the pressure curve P(t) (300 points) and from the
% GC selectivity (4 points), with weighted residuals (Section 4.6.3) and Levenberg-Marquardt implemented
% from scratch with the analytic Jacobian (Section 4.9). The pseudo-data are regenerated exactly as in Lab 4.
% No toolbox required (lsqnonlin, if present, is used only as a cross-check).

clear; close all; clc

c = lab_settings();
runs = generate_data(c);
fits = fit_all_runs(runs, c);

%% Jacobian check: analytic vs forward differences, at the start of the 900 K run
run = runs(2);  p = initial_guess(run, c);  Ja = jacobian_run(p, run, c);  Jn = zeros(size(Ja));
for j = 1:2
    dp = p;  h = 1e-7 * p(j);  dp(j) = dp(j) + h;
    Jn(:, j) = (residuals_run(dp, run, c) - residuals_run(p, run, c)) / h;
end
fprintf('Lab 5 - k1, k2 at each temperature (weighted LM, analytic Jacobian)\n');
fprintf('  Jacobian check: max |analytic - finite difference| / max |J| = %.1e\n', ...
        max(abs(Ja(:) - Jn(:))) / max(abs(Ja(:))));

%% results
fprintf('   T [K]   start k1, k2          k1 (true)          k2 (true)          it   SSE_w   s2    rho(k1,k2)\n');
for i = 1:numel(fits)
    f = fits(i);
    fprintf('  %6.0f   %.4f, %.4f    %.4f (%.4f)    %.4f (%.4f)    %2d   %6.1f  %.3f   %+.3f\n', ...
            f.T, f.p0(1), f.p0(2), f.p(1), runs(i).k1, f.p(2), runs(i).k2, f.it, f.sse, f.sse / f.nu, f.rho);
end
fprintf('  standard errors:\n');
for i = 1:numel(fits)
    f = fits(i);
    fprintf('  %6.0f   SE(k1) = %.4f  SE(k2) = %.4f   relative %.2f %%, %.2f %%\n', ...
            f.T, f.se(1), f.se(2), 100 * f.se(1) / f.p(1), 100 * f.se(2) / f.p(2));
end

%% identifiability: the P block alone gives a singular J'J
J2 = jacobian_run(fits(2).p, runs(2), c);  JP = J2(1:numel(c.tP), :);
fprintf('  P block only: cond(J''J) = %.1e  (columns identical -> k1, k2 not separable)\n', cond(JP' * JP));

%% optional cross-check with a library LM (Optimization Toolbox)
if exist('lsqnonlin', 'file') == 2
    opts = optimoptions('lsqnonlin', 'Algorithm', 'levenberg-marquardt', 'SpecifyObjectiveGradient', true, ...
                        'StepTolerance', 1e-12, 'FunctionTolerance', 1e-12, 'Display', 'off');
    for i = 1:numel(fits)
        run = runs(i);
        fun = @(q) deal(residuals_run(q, run, c), jacobian_run(q, run, c));
        x = lsqnonlin(fun, fits(i).p0, [], [], opts);
        fprintf('  lsqnonlin, T=%.0f K: max relative difference %.1e\n', fits(i).T, max(abs(x ./ fits(i).p - 1)));
    end
end

%% figure
col = [35 63 152; 211 84 0; 46 125 50] / 255;
fig = figure('Position', [100 100 1300 380]);
for k = 1:3, subplot(1, 3, k); hold on; box on; end
for i = 1:numel(fits)
    f = fits(i);  K = sum(f.p);
    subplot(1, 3, 1)
    plot(c.tP, runs(i).P, '.', 'Color', col(i, :), 'MarkerSize', 4)
    plot(c.tP, c.P0 * (2 - exp(-K * c.tP)), '-', 'Color', 'k', 'LineWidth', 1)
    subplot(1, 3, 2)
    h(i) = plot(c.tGC, runs(i).S, 'o', 'Color', col(i, :)); %#ok<SAGROW>
    plot([0 15], f.p(1) / K * [1 1], '-', 'Color', col(i, :), 'LineWidth', 1)
    subplot(1, 3, 3)
    semilogy(0:numel(f.hist) - 1, f.hist, 'o-', 'Color', col(i, :))
end
subplot(1, 3, 1); xlim([0 15]); xlabel('t [s]'); ylabel('P [kPa]')
subplot(1, 3, 2); xlim([0 15]); xlabel('t [s]'); ylabel('S')
legend(h, arrayfun(@(T) sprintf('T = %.0f K', T), c.Truns, 'UniformOutput', false), 'Location', 'best')
subplot(1, 3, 3); set(gca, 'YScale', 'log'); xlabel('LM iteration'); ylabel('weighted SSE')
print(fig, 'Lab5_fits_matlab.png', '-dpng', '-r150');

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
