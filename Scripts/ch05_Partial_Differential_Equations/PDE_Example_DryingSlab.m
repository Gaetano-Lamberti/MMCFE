%% Lesson 5, Section 3.12.3 - Drying of a food slab with moisture-dependent diffusivity
% Nonlinear parabolic PDE on the HALF slab (x = 0: centre, x = L: surface):
%   dX/dt = d/dx( D(X) dX/dx ),   D(X) = D_d * exp(b*X)
%   x = 0 : dX/dx = 0   (symmetry)
%   x = L : X = Xe      (negligible external resistance)
%   IC    : X = X0
% X = moisture content [kg water / kg dry solid]. D grows with moisture: a dry,
% low-diffusivity CRUST forms at the surface and slows the end of drying.
%
% Solved with:
%   1) FVM, cell-centred, face diffusivity from the face moisture (Section 3.9),
%      method of lines + ode15s with the Jacobian pattern (Section 3.4);
%   2) pdepe (Section 3.11), as an independent check (MATLAB only).
% Then compared with the CONSTANT-D model (Crank's series) with the best-fit
% effective diffusivity: a simplification quantified, not assumed.
%
% HOW TO RUN: press F5. Helper functions are local functions at the end.

clear; clc; close all;

%% --- Parameters ---
L  = 5e-3;                         % half thickness [m]
X0 = 4.0;  Xe = 0.1;               % initial and equilibrium moisture [kg/kg dry]
D_lo = 1e-10;  D_hi = 2e-9;        % D at Xe and at X0 [m^2/s]
b   = log(D_hi/D_lo)/(X0 - Xe);    % [-]
D_d = D_lo/exp(b*Xe);              % [m^2/s]
Dfun = @(X) D_d*exp(b*X);
t_end = 24*3600;  ts = linspace(0, t_end, 289)';    % output every 5 min

%% --- 1) FVM + method of lines ---
N = 80;  h = L/N;  xc = ((1:N)' - 0.5)*h;
% InitialStep: at t = 0 the surface jumps from X0 to Xe (huge initial flux); a small
% explicit first step lets the solver start reliably (needed by Octave's ode15s).
opts = odeset('RelTol', 1e-8, 'AbsTol', 1e-10, 'InitialStep', 1e-2, ...
              'JPattern', spdiags(ones(N,3), [-1 0 1], N, N));
[~, Y] = ode15s(@(t,X) fvm_rhs(X, h, Xe, Dfun), ts, X0*ones(N,1), opts);
Xm = mean(Y, 2);                                    % mean moisture (equal cells)
i1 = find(ts >= 3600, 1);
fprintf('FVM, %d cells. After 1 h: X = %.2f near the surface, %.2f at the centre;\n', N, Y(i1,end), Y(i1,1));
fprintf('  D(centre)/D(surface) = %.1f  -> a dry crust controls the drying rate\n', Dfun(Y(i1,1))/Dfun(Y(i1,end)));
fprintf('Mean moisture after 24 h: %.3f kg/kg\n', Xm(end));

%% --- 2) pdepe (independent check; not available in Octave) ---
if exist('pdepe', 'file') == 2
    xm  = linspace(0, L, 201);
    sol = pdepe(0, @(x,t,u,dudx) deal(1, Dfun(u)*dudx, 0), @(x) X0, ...
                @(xl,ul,xr,ur,t) deal(0, 1, ur - Xe, 0), xm, ts, ...
                odeset('RelTol', 1e-8, 'AbsTol', 1e-10));
    Xm_pdepe = trapz(xm, sol, 2)/L;
    fprintf('pdepe vs FVM: max |difference in mean moisture| = %.2e kg/kg\n', max(abs(Xm_pdepe - Xm)));
else
    fprintf('pdepe not available (e.g. Octave): check skipped\n');
end

%% --- 3) Constant-D model (Crank) with the best-fit effective diffusivity ---
lgD  = fminbnd(@(lg) sum((crank_mean(10^lg, ts, L, X0, Xe) - Xm).^2), -11, -8);
Deff = 10^lgD;  Xc = crank_mean(Deff, ts, L, X0, Xe);
[emax, imax] = max(abs(Xc - Xm));
fprintf('\nBest-fit constant D_eff = %.3e m^2/s\n', Deff);
fprintf('Max |Crank - nonlinear| = %.3f kg/kg at t = %.1f h\n', emax, ts(imax)/3600);
fprintf('After 24 h: nonlinear %.3f, constant D %.3f kg/kg\n', Xm(end), Xc(end));

%% --- Plots ---
figure('Name', 'Drying: moisture profiles');
hold on
for th = [0.5 1 2 4 8 16]
    [~, i] = min(abs(ts - th*3600));
    plot([xc; L]*1000, [Y(i,:)'; Xe], 'LineWidth', 1.8);
end
xlabel('x (mm), centre -> surface'); ylabel('X (kg/kg dry)'); grid on
legend('0.5 h', '1 h', '2 h', '4 h', '8 h', '16 h', 'Location', 'southwest');
figure('Name', 'Drying curves');
plot(ts/3600, Xm, 'LineWidth', 2); hold on
plot(ts/3600, Xc, '--', 'LineWidth', 2);
xlabel('t (h)'); ylabel('mean X (kg/kg dry)'); grid on
legend('D(X), FVM', 'constant D_{eff}, best fit');

%% ========================================================================
%  LOCAL FUNCTIONS
%  ========================================================================
function dX = fvm_rhs(X, h, Xe, Dfun)
% FVM_RHS  Cell balances: h dX_i/dt = J_{i-1/2} - J_{i+1/2} (Eq. 54).
% Interior faces: D from the face moisture; centre face: zero flux;
% surface face: X = Xe at a distance h/2.
    N  = numel(X);
    Xf = 0.5*(X(1:end-1) + X(2:end));
    J  = -Dfun(Xf).*(X(2:end) - X(1:end-1))/h;          % interior face fluxes
    Js = -Dfun(0.5*(X(end) + Xe))*(Xe - X(end))/(h/2);   % surface flux
    dX = zeros(N,1);
    dX(1:end-1) = dX(1:end-1) - J;
    dX(2:end)   = dX(2:end)   + J;
    dX(end)     = dX(end)     - Js;
    dX = dX/h;
end

function Xm = crank_mean(D, t, L, X0, Xe)
% CRANK_MEAN  Mean moisture of a slab (half thickness L) with constant D (Crank).
    n  = (0:199);
    Xm = Xe + (X0 - Xe)*sum(8./((2*n + 1).^2*pi^2).*exp(-D*((2*n + 1)*pi/(2*L)).^2.*t(:)), 2);
end
