%% Example 3.1 PFR isothermal
% =========================================================================
%  A guiding example: the styrene reactor
%  3. The ODE(s) problem(s) - 3.1 Steady-state PFR at constant temperature
%  (slides 16-18)
% =========================================================================
%
% NOTATION (as in the slides)
%   A = ethylbenzene (EB), B = styrene (St), C = benzene (Bz)
%       (1)  A -> B + H2          rate constant k1
%       (2)  A -> C + C2H4        rate constant k2
%   Both reactions are irreversible and first order in A.
%   X_A = (nA0 - nA)/nA0 is the conversion of A.
%
% PROBLEM (3.1): a single, nonlinear ODE (initial value problem)
%       dX_A/dV = (k1 + k2) * (1 - X_A) / ( Vdot0*(1 + epsA*X_A) )
%       X_A = 0  @  V = 0
%   It comes from the mass balance on A, dnA/dV = -(k1 + k2)*C_A (A),
%   with C_A = nA/Vdot(X_A), Vdot(X_A) = Vdot0*(1 + epsA*X_A), and
%   nA = nA0*(1 - X_A)  (so dnA = -nA0*dX_A).
%   The temperature is FIXED (T = 900 K), so k1 and k2 are constants and
%   the volumetric flow changes only with composition (no T/T0 factor).
%
% WHAT THIS SCRIPT DOES
%   1) Solves (3.1) numerically with the classical 4th-order Runge-Kutta
%      method (RK4), fixed step, from V = 0 to V = 0.30 m^3.
%   2) Checks it against the ANALYTICAL solution (slide 17, "Note"):
%          -epsA*X_A - (1 + epsA)*ln(1 - X_A) = (k1 + k2)/Vdot0 * V
%      which is implicit in X_A and is solved at each V by scalar
%      Newton-Raphson (the same machinery used for the CSTR in Sec. 2.1).
%   3) Recovers the styrene flow nB ALGEBRAICALLY (slide 18): from the
%      ratio (B)/(A), dnB/dnA = -k1/(k1 + k2) is constant, hence
%          nB = k1/(k1 + k2) * nA0 * X_A
%      so only ONE ODE needs to be integrated.
%   4) Plots the profiles.
%
% HOW TO RUN
%   Open the file in MATLAB and press F5 (or "Run"). Everything is in this
%   single file: the helper functions are LOCAL FUNCTIONS at the end of the
%   script (this requires MATLAB R2016b or later).
%
% UNITS
%   Rate constants in 1/s, volumes in m^3, molar flows in kmol/s,
%   volumetric flows in m^3/s. The feed flows of Sec. 1.6 are given in
%   kmol/h and must be divided by 3600.

clear; clc; close all;

%% ========================================================================
%  1) PARAMETERS
%  ========================================================================

%% --- Kinetic parameters (Arrhenius, Sec. 1.5) ---
R    = 8.314;          % gas constant                                [kJ/(kmol K)]
k0_1 = 1.47e5;         % pre-exponential factor, reaction 1 (A -> B) [1/s]
Ea1  = 90.98e3;        % activation energy, reaction 1               [kJ/kmol]
k0_2 = 9.02e10;        % pre-exponential factor, reaction 2 (A -> C) [1/s]
Ea2  = 207.99e3;       % activation energy, reaction 2               [kJ/kmol]

%% --- Operating conditions (Sec. 1.6) ---
T     = 900;           % reactor temperature, FIXED (isothermal)     [K]
p     = 1e5;           % pressure (isobaric reactor)                 [Pa]
nA0  = 1/3600;        % inlet molar flow of A (EB),  1 kmol/h       [kmol/s]
nW0  = 9/3600;        % inlet molar flow of steam,   9 kmol/h       [kmol/s]
n0 = nA0 + nW0;  % total inlet molar flow                      [kmol/s]

% Expansion factor epsA = y_A0 * delta_nu.
% Each reaction turns 1 mole of A into 2 moles of products, so
% delta_nu = +1 per mole of A converted; with y_A0 = 0.10 -> epsA = 0.10.
% The steam is inert: it dilutes the feed and damps the expansion.
y_A0 = nA0/n0;    % mole fraction of A in the feed              [-]
epsA = y_A0*1;         % expansion factor                            [-]

% Inlet volumetric flow from the ideal-gas law, Vdot0 = n0*R*T/p.
% The factor 1e3 converts kmol -> mol, because with R in J/(mol K) and
% p in Pa, R*T/p is the molar volume in m^3/mol.
Vdot0 = n0*1e3*R*T/p; % inlet volumetric flow                       [m^3/s]

%% --- Rate constants at the fixed temperature ---
% Because T is constant, k1 and k2 are plain NUMBERS here, not functions
% (compare with Example 3.2, where they become functions of T).
k1 = k0_1*exp(-Ea1/(R*T));   % [1/s]  (0.7705 1/s at 900 K, slide 18)
k2 = k0_2*exp(-Ea2/(R*T));   % [1/s]  (0.0764 1/s at 900 K, slide 18)

%% --- Numerical settings ---
V_end = 0.30;          % reactor volume to integrate over            [m^3]
h     = 0.01;          % RK4 fixed step in V (30 steps)              [m^3]
X_A0  = 0;             % initial condition: X_A = 0 @ V = 0          [-]

fprintf('Isothermal PFR, T = %.0f K\n', T);
fprintf('  k1 = %.4f 1/s,  k2 = %.4f 1/s,  k1+k2 = %.4f 1/s\n', k1, k2, k1+k2);
fprintf('  Vdot0 = %.5f m^3/s,  epsA = %.2f\n\n', Vdot0, epsA);

%% ========================================================================
%  2) MODEL: right-hand side of the ODE (3.1)
%  ========================================================================
% f(V, X_A) = (k1 + k2)*(1 - X_A)/(Vdot0*(1 + epsA*X_A))
% V does not appear explicitly (the ODE is AUTONOMOUS), but it is kept as
% the first argument, as is standard for ODE solvers (same signature as
% MATLAB's ode45: f(t, y)). Element-wise operators (./ .*) allow the
% function to be evaluated on vectors as well.
f = @(V, X_A) (k1 + k2)*(1 - X_A)./(Vdot0*(1 + epsA*X_A));

%% ========================================================================
%  3) NUMERICAL SOLUTION: classical RK4 with fixed step
%  ========================================================================
% rk4_fixed is a local function (end of file). It returns the grid of V
% values (column) and the solution, one ROW per grid point.
[Vgrid, X_rk4] = rk4_fixed(f, 0, V_end, X_A0, h);

fprintf('RK4 (h = %.3f m^3, %d steps): X_A at V = %.2f m^3 is %.4f (%.2f%%)\n', ...
        h, numel(Vgrid)-1, V_end, X_rk4(end), 100*X_rk4(end));

%% ========================================================================
%  4) CHECK AGAINST THE ANALYTICAL SOLUTION (slide 17, "Note")
%  ========================================================================
% The ODE is SEPARABLE. With the partial-fraction split
%       (1 + epsA*X_A)/(1 - X_A) = -epsA + (1 + epsA)/(1 - X_A)
% it integrates in closed form (X_A = 0 at V = 0) to
%       -epsA*X_A - (1 + epsA)*ln(1 - X_A) = (k1 + k2)/Vdot0 * V
% This relation is IMPLICIT in X_A: for each V we solve g(X_A) = 0 with
%       g(X_A)  = -epsA*X_A - (1 + epsA)*ln(1 - X_A) - (k1 + k2)/Vdot0*V
%       g'(X_A) = -epsA + (1 + epsA)/(1 - X_A)   (> 0 on [0,1): unique root)
% by scalar Newton-Raphson. This reference uses NO ODE solver, so it
% cannot share any blind spot with RK4.
g  = @(X_A, V) -epsA*X_A - (1 + epsA)*log(1 - X_A) - (k1 + k2)/Vdot0*V;
dg = @(X_A)    -epsA + (1 + epsA)./(1 - X_A);

X_exact = zeros(size(Vgrid));
for i = 1:numel(Vgrid)
    % Initial guess: the RK4 value (very close, Newton converges in 1-3 steps)
    X_exact(i) = newton_scalar(@(X_A) g(X_A, Vgrid(i)), dg, X_rk4(i), 1e-14, 50);
end

max_err = max(abs(X_rk4 - X_exact));
fprintf('Max |X_A(RK4) - X_A(analytical)| over the grid: %.1e\n', max_err);
fprintf('  (RK4 global error is O(h^4): halving h reduces it by about 16)\n\n');

%% ========================================================================
%  5) STYRENE AND BENZENE: obtained algebraically (slide 18)
%  ========================================================================
% Mass balance on B: dnB/dV = k1*C_A   (B)
% Ratio (B)/(A): dnB/dnA = k1*C_A / ( -(k1 + k2)*C_A ) = -k1/(k1 + k2)
% C_A cancels, the ratio is CONSTANT (T is fixed), and integrating from
% the inlet (nA = nA0, nB = 0):
%       nB = k1/(k1 + k2) * (nA0 - nA) = k1/(k1 + k2) * nA0 * X_A
% The same argument for C gives nC = k2/(k1 + k2) * nA0 * X_A.
S_B  = k1/(k1 + k2);              % selectivity to styrene (constant)  [-]
nA  = nA0*(1 - X_rk4);          % EB molar flow profile              [kmol/s]
nB  = S_B*nA0*X_rk4;            % St molar flow profile              [kmol/s]
nC  = (1 - S_B)*nA0*X_rk4;      % Bz molar flow profile              [kmol/s]

fprintf('Selectivity to styrene k1/(k1+k2) = %.4f (%.1f%%), constant along V\n', ...
        S_B, 100*S_B);
fprintf('Outlet flows [kmol/h]: EB %.4f, St %.4f, Bz %.4f\n', ...
        3600*nA(end), 3600*nB(end), 3600*nC(end));

%% ========================================================================
%  6) PLOTS
%  ========================================================================
figure('Name', 'Example 3.1 - Isothermal PFR');

subplot(1,2,1)
plot(Vgrid, X_rk4, 'o', 'MarkerSize', 4); hold on
plot(Vgrid, X_exact, '-', 'LineWidth', 1.2);
xlabel('Reactor volume V [m^3]'); ylabel('Conversion of EB, X_A [-]');
title('Conversion profile, T = 900 K');
legend('RK4', 'Analytical', 'Location', 'southeast');
grid on

subplot(1,2,2)
plot(Vgrid, 3600*nA, '-', 'LineWidth', 1.2); hold on
plot(Vgrid, 3600*nB, '-', 'LineWidth', 1.2);
plot(Vgrid, 3600*nC, '-', 'LineWidth', 1.2);
xlabel('Reactor volume V [m^3]'); ylabel('Molar flow [kmol/h]');
title('Molar flow profiles');
legend('EB (n_A)', 'ST (n_B)', 'BZ (n_C)', 'Location', 'east');
grid on

%% ========================================================================
%  LOCAL FUNCTIONS (must stay at the end of the script)
%  ========================================================================

function [t, Y] = rk4_fixed(f, t0, tend, y0, h)
% RK4_FIXED  Classical 4th-order Runge-Kutta with a fixed step.
%   Solves y' = f(t, y), y(t0) = y0, on [t0, tend]. Each step uses four
%   evaluations of the slope:
%     K1 = f(t,       y)
%     K2 = f(t + h/2, y + h/2*K1)
%     K3 = f(t + h/2, y + h/2*K2)
%     K4 = f(t + h,   y + h*K3)
%     y_new = y + h/6*(K1 + 2*K2 + 2*K3 + K4)
%   Global error O(h^4); only f itself is needed (no derivatives of f).
%   Works for a scalar ODE and for a system (y and f(t,y) COLUMN vectors).
%   OUTPUT: t (column of grid points), Y (one ROW per grid point).
    N = round((tend - t0)/h);          % number of steps (h must divide the interval)
    t = t0 + (0:N)'*h;                 % grid, column vector
    y = y0(:);                         % force a column state vector
    Y = zeros(N+1, numel(y));          % preallocate the solution
    Y(1,:) = y';                       % initial condition in the first row
    for n = 1:N
        K1 = f(t(n),       y);
        K2 = f(t(n) + h/2, y + h/2*K1);
        K3 = f(t(n) + h/2, y + h/2*K2);
        K4 = f(t(n) + h,   y + h*K3);
        y  = y + h/6*(K1 + 2*K2 + 2*K3 + K4);
        Y(n+1,:) = y';                 % store as a row
    end
end

function x = newton_scalar(g, dg, x0, tol, maxiter)
% NEWTON_SCALAR  Scalar Newton-Raphson: x_{n+1} = x_n - g(x_n)/g'(x_n).
%   g, dg   - handles to the function and to its derivative
%   x0      - initial guess; tol - tolerance on |g|; maxiter - max steps
    x = x0;
    for it = 1:maxiter
        gx = g(x);
        if abs(gx) < tol
            return                     % converged
        end
        x = x - gx/dg(x);              % Newton update
    end
    warning('newton_scalar:noConvergence', ...
            'No convergence in %d iterations (|g| = %.2e).', maxiter, abs(g(x)));
end
