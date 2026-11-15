%% Module 1, Part 3 - PDE test case (a): cylindrical transient conduction
% Radially-symmetric transient heat conduction in a solid cylinder, no
% generation:
%   dT/dt = alpha*(d2T/dr2 + (1/r)*dT/dr),   0 <= r <= R
%   BC: T(R,t) = 0 (Dirichlet),  dT/dr(0,t) = 0 (symmetry)
%   IC: single Bessel mode (clean closed-form test case, no series to
%       truncate - same spirit as the ODE test problems in S17.7)
%
% Three solutions compared:
%   1) Analytical (separation of variables -> Bessel equation of order 0)
%   2) Method of lines: discretize r only, integrate in t with RK4
%      (direct reuse of S17 - this IS an ODE system once r is discretized)
%   3) MATLAB built-in `pdepe`, using its native cylindrical-symmetry
%      support (m=1) - the natural built-in choice for this geometry

clear; clc; close all;

%% --- Parameters ---
R     = 0.05;     % m, cylinder radius
alpha = 1.2e-5;    % m^2/s, thermal diffusivity
T0    = 100;       % K, amplitude of the initial mode

% First zero of J0 (lambda_1 = j_{0,1}/R), found via fzero - no hardcoded
% "magic number", transparent about where it comes from.
j01 = fzero(@(x) besselj(0,x), 2.4);
lam1 = j01/R;
fprintf('j_{0,1} = %.6f   lambda_1 = %.4f 1/m\n', j01, lam1);

tau = 1/(alpha*lam1^2);   % characteristic decay time of this mode
fprintf('Characteristic time tau = 1/(alpha*lambda1^2) = %.4f s\n\n', tau);

%% --- 1) Analytical solution ---
T_exact = @(r,t) T0*besselj(0,lam1*r).*exp(-alpha*lam1^2*t);

fprintf('=== Analytical solution check ===\n');
fprintf('T(R,0) = %.3e  (should be 0, BC)\n', T_exact(R,0));
fprintf('T(r,0) near r=0: symmetric by construction (J0 is even)\n\n');

%% --- 2) Method of lines: discretize r, integrate in t with RK4 ---
N  = 41;              % number of radial nodes (r=0 ... r=R)
dr = R/(N-1);
r  = linspace(0,R,N)';

% Stability limit for EXPLICIT time-stepping on the diffusion operator
% (von Neumann / Fourier number, PDE intro S5): Fo = alpha*h/dr^2 <= 0.5
h_stable_limit = 0.5*dr^2/alpha;
fprintf('Spatial grid: N=%d, dr=%.5f m\n', N, dr);
fprintf('Explicit stability limit: h < %.5f s (Fourier number <= 0.5)\n\n', h_stable_limit);

%% --- Stability demonstration: stable vs unstable step size ---
h_bad = 2.5*h_stable_limit;   % deliberately unstable
fprintf('=== Instability demo: h=%.4f s (> stability limit) ===\n', h_bad);
T_bad = T0*besselj(0,lam1*r(1:end-1));
t_bad = 0; hist_bad = T_bad(1);
for step = 1:12
    T_bad = rk4_step(@(TT) mol_rhs(TT,r,dr,alpha,N), T_bad, h_bad);
    t_bad = t_bad + h_bad;
    hist_bad(end+1) = T_bad(1); %#ok<AGROW>
    if mod(step,3)==0
        fprintf('  step %2d: t=%.3f s   T(0,t) = %.4e\n', step, t_bad, T_bad(1));
    end
end
fprintf('--> Diverges within a handful of steps: numerical instability from\n');
fprintf('    violating the Fourier-number limit, not a truncation-error issue.\n\n');

%% --- Stable, accurate run ---
h    = 0.3*h_stable_limit;
tend = 0.5*tau;
nsteps = round(tend/h);
h = tend/nsteps;   % land exactly on tend

fprintf('=== Stable run: h=%.5f s (%.2f x limit), %d steps to t=%.3f s (%.2f tau) ===\n', ...
        h, h/h_stable_limit, nsteps, tend, tend/tau);

T_mol = T0*besselj(0,lam1*r(1:end-1));   % state = interior+center nodes (boundary fixed at 0)
t_hist = zeros(nsteps+1,1);
T_center_hist = zeros(nsteps+1,1);
T_center_hist(1) = T_mol(1);
t = 0;
for step = 1:nsteps
    T_mol = rk4_step(@(TT) mol_rhs(TT,r,dr,alpha,N), T_mol, h);
    t = t + h;
    t_hist(step+1) = t;
    T_center_hist(step+1) = T_mol(1);
end

T_mol_full = [T_mol; 0];   % re-attach the boundary node (fixed at 0)
T_ex_final = T_exact(r, tend);
err_mol = abs(T_mol_full - T_ex_final);
fprintf('Max error vs analytical at t=%.3f s: %.4e K\n\n', tend, max(err_mol));

%% --- 3) MATLAB built-in: pdepe (m=1, cylindrical symmetry) ---
m = 1;
tspan = linspace(0, tend, 50);
sol = pdepe(m, @(x,t,u,dudx) pdefun_cyl(x,t,u,dudx,alpha), ...
               @(x) icfun_cyl(x,T0,lam1), ...
               @(xl,ul,xr,ur,t) bcfun_cyl(xl,ul,xr,ur,t), ...
               r, tspan);
T_pdepe_final = sol(end,:,1)';

err_pdepe = abs(T_pdepe_final - T_ex_final);
fprintf('=== pdepe (built-in) vs analytical at t=%.3f s ===\n', tend);
fprintf('Max error: %.4e K\n\n', max(err_pdepe));

%% --- Empirical convergence order (spatial refinement, method of lines) ---
fprintf('=== Empirical convergence order (method of lines, spatial refinement) ===\n');
Ns = [11 21 41 81 161];
errs = zeros(size(Ns));
for k = 1:length(Ns)
    errs(k) = mol_error_at(Ns(k), R, alpha, T0, lam1, 0.3*tau, T_exact);
end
fprintf('%6s %10s %14s\n','N','dr','max error');
for k = 1:length(Ns)
    fprintf('%6d %10.6f %14.6e\n', Ns(k), R/(Ns(k)-1), errs(k));
end
fprintf('\nEstimated order p = log2(err(N)/err(2N-1)):\n');
for k = 2:length(Ns)
    p = log2(errs(k-1)/errs(k));
    fprintf('  N=%d -> N=%d: order = %.3f\n', Ns(k-1), Ns(k), p);
end
fprintf('--> Order 2, as expected from the O(dr^2) central-difference stencil.\n\n');

%% ============================================================
%  PLOTS
%  ============================================================

%% --- Plot 1: radial profiles at several times ---
figure('Name','Cylindrical transient conduction: radial profiles');
times_to_plot = [0, 0.1, 0.25, 0.5]*tau;
colors = lines(length(times_to_plot));
rfine = linspace(0,R,200);
hold on
for k = 1:length(times_to_plot)
    tk = times_to_plot(k);
    plot(rfine*1000, T_exact(rfine,tk), '-', 'Color', colors(k,:), 'LineWidth', 1.8);
end
% overlay method-of-lines markers at final analyzed time
plot(r*1000, T_mol_full, 'ko', 'MarkerFaceColor','k', 'MarkerSize', 4);
xlabel('r (mm)'); ylabel('T (K)');
title('Radial temperature profile: analytical (lines) vs method of lines (markers, t=0.5\tau)');
legendStrs = arrayfun(@(tk) sprintf('t=%.2f\\tau', tk/tau), times_to_plot, 'UniformOutput', false);
legend([legendStrs, {'Method of lines (RK4)'}], 'Location','northeast');
grid on;

%% --- Plot 2: instability demonstration ---
figure('Name','Explicit stability limit (Fourier number)');
subplot(1,2,1);
semilogy(0:length(hist_bad)-1, abs(hist_bad)+eps, '-ro', 'LineWidth',1.5,'MarkerFaceColor','r');
xlabel('Step'); ylabel('|T(0,t)|  (log scale)');
title(sprintf('h=%.3f s > limit %.3f s: blows up', h_bad, h_stable_limit));
grid on;

subplot(1,2,2);
plot(t_hist, T_center_hist, 'b-', 'LineWidth',1.8); hold on
plot(t_hist, T_exact(0,t_hist), 'k--', 'LineWidth',1.2);
xlabel('t (s)'); ylabel('T(0,t)  (K)');
title(sprintf('h=%.4f s < limit: stable and accurate', h));
legend('Method of lines','Analytical','Location','northeast');
grid on;

%% --- Plot 3: three-way comparison at final time ---
figure('Name','Three-way comparison at t=0.5 tau');
plot(r*1000, T_ex_final, 'k-', 'LineWidth',2); hold on
plot(r*1000, T_mol_full, 'ro', 'MarkerFaceColor','r', 'MarkerSize',5);
plot(r*1000, T_pdepe_final, 'bs', 'MarkerFaceColor','b', 'MarkerSize',4);
xlabel('r (mm)'); ylabel('T (K)');
title('Analytical vs method of lines (RK4, from scratch) vs pdepe (built-in)');
legend('Analytical','Method of lines','pdepe','Location','northeast');
grid on;

%% --- Plot 4: empirical convergence order ---
figure('Name','Empirical convergence order');
drs = R./(Ns-1);
loglog(drs, errs, '-o', 'LineWidth',1.5,'MarkerFaceColor','b'); hold on
loglog(drs, errs(1)*(drs/drs(1)).^2, 'k:', 'LineWidth',1);
xlabel('\Delta r (m, log scale)'); ylabel('max error (K, log scale)');
title('Empirical convergence order: slope = 2');
legend('Method of lines (measured)','order-2 reference','Location','northwest');
grid on; set(gca,'XDir','reverse');

%% ============================================================
%  LOCAL FUNCTIONS (must follow all executable code)
%  ============================================================
function dT = mol_rhs(T, r, dr, alpha, N)
    % T has N-1 entries: T(1)=center (r=0) ... T(N-1)=last interior node
    % (boundary node r=R is fixed at 0, not part of the state vector)
    dT = zeros(N-1,1);
    % center node (r=0): limiting form via Taylor/L'Hopital (2*T''(0)),
    % standard cylindrical-coordinate center-node formula
    dT(1) = 4*alpha/dr^2*(T(2)-T(1));
    % interior nodes
    for i = 2:N-2
        ri = r(i);
        dT(i) = alpha*((T(i+1)-2*T(i)+T(i-1))/dr^2 + (1/ri)*(T(i+1)-T(i-1))/(2*dr));
    end
    % last interior node: neighbor is the boundary, fixed at T=0
    i = N-1;
    ri = r(i);
    dT(i) = alpha*((0-2*T(i)+T(i-1))/dr^2 + (1/ri)*(0-T(i-1))/(2*dr));
end

function Tnew = rk4_step(f, T, h)
    k1 = f(T);
    k2 = f(T+h/2*k1);
    k3 = f(T+h/2*k2);
    k4 = f(T+h*k3);
    Tnew = T + h/6*(k1+2*k2+2*k3+k4);
end

function err = mol_error_at(N, R, alpha, T0, lam1, tend, T_exact)
    dr = R/(N-1);
    r = linspace(0,R,N)';
    h_lim = 0.5*dr^2/alpha;
    h = 0.3*h_lim;
    nsteps = round(tend/h);
    h = tend/nsteps;
    T = T0*besselj(0,lam1*r(1:end-1));
    for step = 1:nsteps
        T = rk4_step(@(TT) mol_rhs(TT,r,dr,alpha,N), T, h);
    end
    Tfull = [T; 0];
    err = max(abs(Tfull - T_exact(r,tend)));
end

function [c,f,s] = pdefun_cyl(~,~,~,dudx,alpha)
    c = 1;
    f = alpha*dudx;
    s = 0;
end

function u0 = icfun_cyl(x,T0,lam1)
    u0 = T0*besselj(0,lam1*x);
end

function [pl,ql,pr,qr] = bcfun_cyl(~,~,~,ur,~)
    pl = 0;    % symmetry (zero flux) at r=0
    ql = 1;
    pr = ur;   % Dirichlet T=0 at r=R
    qr = 0;
end
