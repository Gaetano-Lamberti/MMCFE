% Lab6_ArrheniusFitting.m
% Lecture 6 - Parameter estimation, Case history 4.15.3 (guiding example, Section 5.4)
%
% Second stage of the two-stage approach: k1(T), k2(T) estimated at 850, 900, 923 K (Lab 5, recomputed here)
% are fitted with the Arrhenius law, in the classic form k0*exp(-Ea/RT) and in the form centred at T0 = 900 K.
% The real difficulty is not convergence, but the k0-Ea correlation, removed by the reparametrization.
% No toolbox required.

clear; close all; clc

c = lab_settings();
runs = generate_data(c);
fits = fit_all_runs(runs, c);                     % stage 1 = Lab 5
T  = [fits.T]';
P  = [fits.p];                                    % 2 x 3: row 1 = k1, row 2 = k2
k1 = P(1, :)';  k2 = P(2, :)';
fprintf('Lab 6 - Arrhenius from the per-temperature estimates (two-stage approach)\n');
fprintf('  stage 1 (Lab 5): k1 = [%.4f %.4f %.4f]  k2 = [%.4f %.4f %.4f]\n', k1, k2);
[pc1, pr1] = arrhenius_two_stage(T, k1, c.pTrue(2), 'reaction 1 (styrene)', c);
[pc2, pr2] = arrhenius_two_stage(T, k2, c.pTrue(4), 'reaction 2 (benzene)', c); %#ok<ASGLU>
fprintf('  note: 3 points, 2 parameters: nu = 1. The correlation depends only on the design (the T values).\n');

%% figure 1: Arrhenius plots
xx = linspace(1 / 940, 1 / 840, 100)';
fig1 = figure('Position', [100 100 1000 380]);
K = {k1, k2};  PR = {pr1, pr2};  PT = {c.pTrue(1:2), c.pTrue(3:4)};  nm = {'k1', 'k2'};
for j = 1:2
    subplot(1, 2, j)
    semilogy(1000 ./ T, K{j}, 'ko'); hold on; box on
    semilogy(1000 * xx, PR{j}(1) * exp(-PR{j}(2) / c.R * (xx - 1 / c.T0)), '-', 'Color', [35 63 152] / 255)
    semilogy(1000 * xx, PT{j}(1) * exp(-PT{j}(2) / c.R * (xx - 1 / c.T0)), 'k:')
    xlabel('1000/T [1/K]'); ylabel([nm{j} ' [1/s]'])
    legend('stage 1 estimates', 'Arrhenius fit', 'true', 'Location', 'best')
end
print(fig1, 'Lab6_arrhenius_matlab.png', '-dpng', '-r150');

%% figure 2: SSE landscape of the classic form vs the centred form (reaction 1): the 'banana' valley
fig2 = figure('Position', [100 100 1000 380]);
E   = linspace(pr1(2) - 25, pr1(2) + 25, 201);
lk0 = linspace(log(pc1(1)) - 3.5, log(pc1(1)) + 3.5, 201);
[EE, LL] = meshgrid(E, lk0);
S1 = zeros(size(EE));
for i = 1:3
    S1 = S1 + (k1(i) - exp(LL) .* exp(-EE / (c.R * T(i)))).^2;
end
subplot(1, 2, 1); contour(EE, LL, log10(S1), 25); xlabel('Ea [kJ/mol]'); ylabel('ln k0'); title('classic form')
kr = linspace(pr1(1) * 0.9, pr1(1) * 1.1, 201);
[EE, KK] = meshgrid(E, kr);
S2 = zeros(size(EE));
for i = 1:3
    S2 = S2 + (k1(i) - KK .* exp(-EE / c.R * (1 / T(i) - 1 / c.T0))).^2;
end
subplot(1, 2, 2); contour(EE, KK, log10(S2), 25); xlabel('Ea [kJ/mol]'); ylabel('kref [1/s]')
title('centred at T0 = 900 K')
print(fig2, 'Lab6_sse_valley_matlab.png', '-dpng', '-r150');

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

% ------------------------------------------------------------------ second stage (Arrhenius)
function rho = corr_from_J(J)
% Correlation coefficient of the two parameters from (J'J)^-1 (the factor s^2 cancels).
C = inv(J' * J);                                  %#ok<MINV>
rho = C(1, 2) / sqrt(C(1, 1) * C(2, 2));
end

function [pc, pr] = arrhenius_two_stage(T, k, EaTrue, name, c)
% Three ways of fitting k(T) from the per-temperature estimates (two-stage approach, Section 5.4).
x = 1 ./ T;
% (a) linear fit of ln k vs 1/T, raw regressor: ln k = ln k0 - (Ea/R) (1/T)
X = [ones(size(x)), -x / c.R];
q = (X' * X) \ (X' * log(k));                     % (ln k0, Ea)
rhoRaw = corr_from_J(X);
% (b) classic nonlinear form k = k0 exp(-Ea/(R T)), LM in k-space, started from the linear fit
resC = @(p) k - p(1) * exp(-p(2) ./ (c.R * T));
jacC = @(p) -[exp(-p(2) ./ (c.R * T)), -p(1) * exp(-p(2) ./ (c.R * T)) ./ (c.R * T)];
% J'J of this form is nearly singular (k0 ~ 1e5-1e11, columns almost parallel): that IS the point of the
% exercise, so the 'nearly singular matrix' warnings are switched off here and restored afterwards.
ws = warning('off', 'all');
[pc, ~, itc] = lm(resC, jacC, [exp(q(1)); q(2)]);
rhoClassic = corr_from_J(jacC(pc));
warning(ws);
% (c) linear fit with centred regressor: ln k = ln kref - (Ea/R)(1/T - 1/T0)
Xc = [ones(size(x)), -(x - 1 / c.T0) / c.R];
qc = (Xc' * Xc) \ (Xc' * log(k));                 % (ln kref, Ea)
rhoCent = corr_from_J(Xc);
% (d) reparametrized nonlinear form k = kref exp(-Ea/R (1/T - 1/T0)), LM in k-space, started from (c)
tau = x - 1 / c.T0;
resR = @(p) k - p(1) * exp(-p(2) / c.R * tau);
jacR = @(p) -[exp(-p(2) / c.R * tau), -p(1) * exp(-p(2) / c.R * tau) .* tau / c.R];
[pr, ~, itr] = lm(resR, jacR, [exp(qc(1)); qc(2)]);
rhoRep = corr_from_J(jacR(pr));
fprintf('  %s:\n', name);
fprintf('    (a) ln k vs 1/T, linear          k0 = %.4e 1/s   Ea = %7.2f kJ/mol   rho(ln k0, Ea) = %+.4f\n', ...
        exp(q(1)), q(2), rhoRaw);
fprintf('    (b) k0 exp(-Ea/RT), LM (%d it)    k0 = %.4e 1/s   Ea = %7.2f kJ/mol   rho(k0, Ea)    = %+.4f\n', ...
        itc, pc(1), pc(2), rhoClassic);
fprintf('    (c) centred at T0, linear        kref = %.4f 1/s     Ea = %7.2f kJ/mol   rho(ln kref, Ea) = %+.4f\n', ...
        exp(qc(1)), qc(2), rhoCent);
fprintf('    (d) centred at T0, LM (%d it)     kref = %.4f 1/s     Ea = %7.2f kJ/mol   rho(kref, Ea)    = %+.4f\n', ...
        itr, pr(1), pr(2), rhoRep);
fprintf('        true Ea = %.2f kJ/mol\n', EaTrue);
end
