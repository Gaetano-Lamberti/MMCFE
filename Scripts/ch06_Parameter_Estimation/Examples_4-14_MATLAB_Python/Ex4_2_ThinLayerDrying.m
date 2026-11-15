% Ex4_2_ThinLayerDrying.m
% Lecture 6 - Parameter estimation, Section 4.14.3-4.14.4, Example 4.2: thin-layer drying of apple slices
%
% Slices 4 mm thick, dried from both faces (half thickness L = 2 mm). Moisture ratio MR every 15 min for 5 h.
% Two models:
%   Fick's second law (Eq. 51), one physical parameter D_eff:
%       dX/dt = D d2X/dx2,  dX/dx = 0 at x = 0 (symmetry),  X = 0 at x = L (surface at equilibrium),
%       MR(t) = (1/L) * integral_0^L X dx
%   Page (Eq. 52), empirical, two parameters:  MR = exp(-k t^n),  t in min
% Algorithms:
%   Page: linearized fit ln(-ln MR) = ln k + n ln t; Levenberg-Marquardt.
%   Fick: slope method (first term of the series solution at long times); LM on D with the PDE solved
%         numerically by the method of lines inside the fit (Section 4.10); 1-D bounded search (fminbnd);
%         LM with the series solution as a check.
% Pseudo-data: Fick with D = 3.0e-10 m^2/s, noise sigma = 0.01 on MR. The noisy values are written below,
% identical to those used for the slides (Python, numpy generator seed 52).
% No toolbox required: LM is written out below; ode15s and fminbnd are base MATLAB.

clear; close all; clc

%% settings and data
Lh    = 2.0e-3;                 % m, half thickness
Dtrue = 3.0e-10;                % m^2/s, used only to generate the pseudo-data
sigMR = 0.01;                   % noise on MR
tMin  = (0:15:300)';            % min
tS    = 60 * tMin;              % s
MR = [1 0.709263556518 0.584797210939 0.491868131 0.419813915045 0.359187965547 0.300307518469 ...
      0.254407063661 0.202491361696 0.186833890634 0.150021069907 0.139660385683 0.122984736022 ...
      0.0766730710269 0.0920292506389 0.0598427570314 0.0460508489257 0.0454459760594 ...
      0.0434328435582 0.0443257017916 0.0572081131026]';
N = numel(MR);

%% method of lines vs series solution
fprintf('4.2 MOL vs series at D_true: max diff = %.1e\n', max(abs(mr_mol(tS, Dtrue, Lh) - mr_series(tS, Dtrue, Lh))));

%% Page model
m = (tMin > 0) & (MR > 0.02) & (MR < 0.98);
b = polyfit(log(tMin(m)), log(-log(MR(m))), 1);           % linearized: ln(-ln MR) = ln k + n ln t
pgLin = [exp(b(2)); b(1)];
page  = @(t, p) exp(-p(1) * t.^p(2));
resPg = @(p) (MR - page(tMin, p)) / sigMR;
[pgLM, ~, nPg] = lm_fd(resPg, pgLin);
nm = {'Page linearized', 'Page LM'};  P = [pgLin, pgLM];
for j = 1:2
    r = resPg(P(:, j));
    fprintf('4.2 %-18s k=%.5f  n=%.4f   SSE_w=%.2f  s2=%.2f\n', nm{j}, P(:, j), r' * r, r' * r / (N - 2));
end
fprintf('    Page LM: %d model evaluations\n', nPg);

%% Fick: slope method, ln MR = ln(8/pi^2) - pi^2 D t /(4 L^2) at long times
late = (MR < 0.6) & (MR > 0.02);
b = polyfit(tS(late), log(MR(late)), 1);
Dslope = -b(1) * 4 * Lh^2 / pi^2;
fprintf('4.2 Fick slope method: D=%.3e (intercept %.3f vs ln(8/pi^2)=%.3f)\n', Dslope, b(2), log(8 / pi^2));

%% Fick: LM on the scaled parameter d = D/1e-10, PDE solved by the method of lines at every evaluation
% The SSE computed through the ODE solver carries the solver's error (RelTol = 1e-8): asking for a
% relative SSE decrease below ~1e-6 would only chase that noise (Section 4.10). Hence tol = 1e-6 here.
resF = @(q) (MR - mr_mol(tS, q * 1e-10, Lh)) / sigMR;
[dLM, ~, nPDE] = lm_fd(resF, Dslope / 1e-10, 1e-6);
r = resF(dLM);
J = (resF(dLM * (1 + 1e-6)) - r) / (dLM * 1e-6);              % 1 x 1 problem: J is a column
seD = sqrt((r' * r / (N - 1)) / (J' * J)) * 1e-10;            % Eq. (45)
fprintf('4.2 Fick LM (MOL): D=%.4e +- %.1e   SSE_w=%.2f  s2=%.2f  PDE solutions=%d\n', ...
        dLM * 1e-10, seD, r' * r, r' * r / (N - 1), nPDE);

%% Fick: 1-D bounded search (golden section + parabolic interpolation)
[dGS, ~, ~, out] = fminbnd(@(d) sum(resF(d).^2), 0.5, 10, optimset('TolX', 1e-6));
fprintf('4.2 Fick 1-D bounded search: D=%.4e   PDE solutions=%d\n', dGS * 1e-10, out.funcCount);

%% Fick: LM with the series solution (check of the numerical model)
dSer = lm_fd(@(q) (MR - mr_series(tS, q * 1e-10, Lh)) / sigMR, Dslope / 1e-10);
fprintf('4.2 Fick LM (series): D=%.4e\n', dSer * 1e-10);

%% figure
fig = figure('Position', [100 100 960 360]);
subplot(1, 2, 1)
tt = (0:300)';
plot(tMin, MR, 'ko', 'MarkerFaceColor', 'w'); hold on; box on
plot(tt, page(tt, pgLM), '--', 'Color', [211 84 0] / 255, 'LineWidth', 2)
plot(tt, mr_series(60 * tt, dLM * 1e-10, Lh), '-', 'Color', [35 63 152] / 255, 'LineWidth', 2)
xlabel('time t [min]'); ylabel('moisture ratio MR'); legend('data', 'Page, LM', 'Fick, LM (MOL)')
subplot(1, 2, 2)
plot([0 300], [0 0], 'k-'); hold on; box on
h1 = plot(tMin, resPg(pgLM), 's', 'Color', [211 84 0] / 255);
h2 = plot(tMin, resF(dLM), 'o', 'Color', [35 63 152] / 255);
ylim([-4.5 4]); xlabel('time t [min]'); ylabel('(y_i - y_{model})/\sigma'); legend([h1 h2], 'Page', 'Fick')
print(fig, 'Ex4_2_drying.png', '-dpng', '-r150');


%% ======================================================================= local functions

function MR = mr_series(t, D, Lh)
% Series solution of Fick's law for a slab, uniform initial moisture, surface at equilibrium (60 terms).
n = (0:59)';
MR = sum(8 ./ ((2 * n + 1).^2 * pi^2) .* exp(-(2 * n + 1).^2 * pi^2 * D * t(:)' / (4 * Lh^2)), 1)';
end

function MR = mr_mol(t, D, Lh)
% Method of lines (Lecture 5): 101 nodes on [0, Lh]; unknowns at nodes 1..100 (node 101 = surface, X = 0).
% Symmetry at x = 0 by a mirror node: d2X/dx2 = 2 (X2 - X1)/dx^2. Linear system dX/dt = D*A*X: ode15s
% with the exact (constant, tridiagonal) Jacobian. MR = average of X over the slab, trapezoidal rule.
NX = 101;  xg = linspace(0, Lh, NX)';  dx = xg(2) - xg(1);  n = NX - 1;
e = ones(n, 1);
A = spdiags([e, -2 * e, e], -1:1, n, n);  A(1, 2) = 2;        % mirror node at x = 0
A = D * A / dx^2;
% A small first step (0.01 s) helps the solver through the initial jump X = 1 -> X = 0 at the surface
opts = odeset('RelTol', 1e-8, 'AbsTol', 1e-10, 'Jacobian', A, 'InitialStep', 1e-2);
[~, U] = ode15s(@(~, u) A * u, t, ones(n, 1), opts);
U = [U, zeros(numel(t), 1)];                                  % add the surface node
MR = trapz(xg, U, 2) / Lh;
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
