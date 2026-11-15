% Ex4_1_Thermistor.m
% Lecture 6 - Parameter estimation, Section 4.14.1-4.14.2, Example 4.1: NTC thermistor
%
% Calibration of an NTC thermistor with the Steinhart-Hart equation (Eq. 50), R in kOhm, T in K:
%     1/T = A + B*ln(R) + C*(ln R)^3
% Two objectives:
%   (1) error on 1/T  -> linear in (A, B, C): closed form, linear least squares (Section 4.8);
%   (2) error on T    -> nonlinear: Levenberg-Marquardt (Section 4.9.3), from the closed form and from
%                        200 random starts. Derivative-free and global methods on this problem: Lecture 7.
% Five points, three parameters (nu = 2): a calibration, not a statistical estimation.
% The interest is in the algorithms: sensitivity to the initial guess and to parameter scaling.
%
% No toolbox required: LM is written out below. The random starts use a small portable generator
% (Park-Miller), so every run gives the same numbers. The slides were made with Python (numpy generator):
% the success rate from random starts is similar (about one in four), not identical.

clear; close all; clc

%% data
TC = [0.00; 10.00; 20.00; 30.16; 40.12];           % degC
T  = TC + 273.15;                                  % K
R  = [32.65; 19.90; 12.50; 8.00; 5.30];            % kOhm
L  = log(R);

%% (1) closed form: least squares on 1/T (linear in A, B, C)
X = [ones(size(L)), L, L.^3];
pLin = X \ (1 ./ T);                               % QR, Section 4.8
fprintf('4.1 closed form (1/T): A=%.4e B=%.4e C=%.4e  max |T err| = %.3f K\n', ...
        pLin, max(abs(1 ./ (X * pLin) - T)));
fprintf('    R=10 kOhm -> T = %.2f C\n', 1 / ([1, log(10), log(10)^3] * pLin) - 273.15);
cc = corrcoef(L, L.^3);
fprintf('    cond(X''X) = %.1e,  corr(lnR, ln^3R) = %.3f\n', cond(X' * X), cc(1, 2));

%% (2) error on T: nonlinear least squares
resT = @(p) res_T(p, T, L);

% LM from the closed form: the natural, non-arbitrary initial guess
[pRef, ok, nev] = lm_fd(resT, pLin);
fprintf('4.1 LM from the closed form: evals %d, A=%.4e B=%.4e C=%.4e\n', nev, pRef);
isRef = @(p) all(abs(p(:) - pRef(:)) <= 1e-8 + 1e-2 * abs(pRef(:)));   % as numpy.allclose(rtol=1e-2)

%% random starts: |p_j| log-uniform in [1e-5, 1], random signs
NST = 200;  state = 20261012;
succ = 0;  ev = zeros(NST, 1);
for k = 1:NST
    [u, state] = prng_uniform(6, state);
    p0 = sign(u(1:3) - 0.5) .* 10.^(-5 + 5 * u(4:6));
    [p, ~, ev(k)] = lm_fd(resT, p0);
    succ = succ + isRef(p);
end
fprintf('4.1 random starts (|p_j| 1e-5..1, random signs): LM success %3d/%d  median evals %d\n', ...
        succ, NST, round(median(ev)));

r = resT(pRef);
fprintf('4.1 LM in T from closed form: A=%.4e B=%.4e C=%.4e ; T residuals (mK): %s\n', pRef, ...
        sprintf('%.2f ', 1000 * r));

%% figure: residuals of the two objectives
fig = figure('Position', [100 100 900 360]);
subplot(1, 2, 1)
RR = logspace(log10(5), log10(33), 200)';  LL = log(RR);
plot(R, TC, 'ko', 'MarkerFaceColor', 'w'); hold on; box on
plot(RR, 1 ./ ([ones(size(LL)), LL, LL.^3] * pRef) - 273.15, '-', 'Color', [35 63 152] / 255, 'LineWidth', 1.5)
xlabel('R [k\Omega]'); ylabel('T [\circC]'); legend('data', 'Steinhart-Hart, fit in T')
subplot(1, 2, 2)
plot(TC, 1000 * (T - 1 ./ (X * pLin)), 's-', 'Color', [211 84 0] / 255); hold on; box on
plot(TC, 1000 * resT(pRef), 'o-', 'Color', [35 63 152] / 255)
plot([0 41], [0 0], 'k-')
xlabel('T [\circC]'); ylabel('T residual [mK]'); legend('fit in 1/T', 'fit in T')
print(fig, 'Ex4_1_thermistor.png', '-dpng', '-r150');


%% ======================================================================= local functions

function r = res_T(p, T, L)
% Residuals on T; where the denominator vanishes or the result is not finite, a large penalty.
r = T - 1 ./ (p(1) + p(2) * L + p(3) * L.^3);
r(~isfinite(r)) = 1e6;
end

function [p, ok, nev] = lm_fd(res, p0)
% Levenberg-Marquardt with Marquardt diagonal scaling (Section 4.9.3) and forward-difference
% Jacobian (step 1e-6 relative). nev counts the residual evaluations, Jacobian included.
lam = 1e-3;  tol = 1e-10;  maxit = 300;  hRel = 1e-6;
p = p0(:);  r = res(p);  f = r' * r;  nev = 1;  ok = false;
for it = 1:maxit
    J = zeros(numel(r), numel(p));
    for j = 1:numel(p)
        dp = p;  h = hRel * max(abs(p(j)), 1e-12);  dp(j) = dp(j) + h;
        J(:, j) = (res(dp) - r) / h;  nev = nev + 1;
    end
    A = J' * J;  g = J' * r;
    while true
        M = A + lam * diag(diag(A));
        ws = warning('off', 'all');                 % far from the solution M can be (nearly) singular
        d = M \ (-g);
        warning(ws);
        if ~all(isfinite(d))
            return                                  % singular system: give up (failure)
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
    if (f - fn) < tol * fn
        ok = true;  return
    end
    f = fn;
end
end

function [u, state] = prng_uniform(n, state)
% n uniform numbers in (0,1): Park-Miller minimal standard generator (exact in double precision).
u = zeros(n, 1);
for i = 1:n
    state = mod(16807 * state, 2147483647);
    u(i) = state / 2147483647;
end
end
