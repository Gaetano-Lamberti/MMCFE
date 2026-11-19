%% Example 3.2 PFR non-isothermal
% =========================================================================
%  A guiding example: the styrene reactor
%  3. The ODE(s) problem(s) - 3.2 Steady-state PFR with heat exchange
%  (slides 19-20)
% =========================================================================
%
% NOTATION (as in the slides)
%   A = ethylbenzene (EB), B = styrene (St), C = benzene (Bz)
%       (1)  A -> B + H2          rate constant k1(T), endothermic
%       (2)  A -> C + C2H4        rate constant k2(T), endothermic
%   X_A = (nA0 - nA)/nA0 is the conversion of A.
%
% PROBLEM (3.2): a system of two nonlinear ODEs (initial value problem)
%   System's state:  x = [X_A; T]
%       dx/dV = f(x),   x = x0 = [0; T0]  @  V = 0
%   with
%   f1(X_A,T) = (k1(T) + k2(T)) * (1 - X_A) / ( Vdot0*(1 + epsA*X_A)*T/T0 )
%   f2(X_A,T) = { [(-dHr1)*k1(T) + (-dHr2)*k2(T)]*C_A(X_A,T)
%                 - UaV*(T - Tc) } / ( ndot(X_A)*Cp )
%   C_A(X_A,T) = nA0*(1 - X_A) / ( Vdot0*(1 + epsA*X_A)*T/T0 )
%   ndot(X_A) = n0*(1 + epsA*X_A)
%
%   Compared with Example 3.1:
%   - k1 and k2 are now FUNCTIONS of T (Arrhenius): the temperature couples
%     the two equations;
%   - the volumetric flow Vdot = Vdot0*(1 + epsA*X_A)*T/T0 contains the factor
%     T/T0 (ideal gas at constant pressure), which was identically 1 in
%     the isothermal case. T therefore enters the mass balance TWICE:
%     through the kinetics and through the gas expansion.
%
% WHAT THIS SCRIPT DOES
%   1) Solves (3.2) with the classical RK4 method, fixed step, for two
%      cases: adiabatic (UaV = 0) and with heat exchange (UaV > 0).
%   2) Recovers the styrene flow nB by integrating dnB/dV = k1(T)*C_A
%      along the computed profile. (Unlike Example 3.1, the selectivity is
%      NOT constant here, because k1/(k1+k2) depends on T.)
%   3) Stiffness check: eigenvalues of the Jacobian of f along the
%      adiabatic profile.
%   4) Step-size test: the same problem with a much larger step.
%   5) Plots the profiles, with the isothermal PFR as reference.
%
% HOW TO RUN
%   Open the file in MATLAB and press F5 (or "Run"). Everything is in this
%   single file: the helper functions are LOCAL FUNCTIONS at the end of the
%   script (this requires MATLAB R2016b or later).
%
% UNITS (important for the energy balance)
%   Rate constants in 1/s, volumes in m^3, molar flows in kmol/s,
%   enthalpies in kJ/kmol, Cp in kJ/(kmol K), UaV in kW/(m^3 K).
%   Then every term in the numerator of f2 is in kW/m^3, the denominator
%   ndot*Cp is in kW/K, and dT/dV is in K/m^3 as it should be.
%   The feed flows of Sec. 1.6 are given in kmol/h: divide by 3600.

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

% Rate constants as FUNCTIONS of T (anonymous functions).
% Element-wise operators (./) allow evaluation on vectors of T.
k1 = @(T) k0_1*exp(-Ea1./(R*T));   % [1/s]
k2 = @(T) k0_2*exp(-Ea2./(R*T));   % [1/s]
% Note: Ea2 >> Ea1, so k2 grows much faster with T than k1: higher T
% favours the side reaction (lower selectivity to styrene).

%% --- Operating conditions (Sec. 1.6) ---
T0    = 900;           % feed temperature (also the T at which Vdot0 is given) [K]
p     = 1e5;           % pressure (isobaric reactor)                 [Pa]
nA0  = 1/3600;        % inlet molar flow of A (EB),  1 kmol/h       [kmol/s]
nW0  = 9/3600;        % inlet molar flow of steam,   9 kmol/h       [kmol/s]
n0 = nA0 + nW0;  % total inlet molar flow                      [kmol/s]
y_A0  = nA0/n0;   % mole fraction of A in the feed (0.10)       [-]
epsA  = y_A0*1;        % expansion factor (delta_nu = +1 per mole of A) [-]
Vdot0    = n0*1e3*R*T0/p;  % inlet volumetric flow, ideal gas       [m^3/s]
                            % (1e3: kmol -> mol, with R in J/(mol K))

%% --- Thermal parameters ---
Cp   = 58;             % molar heat capacity of the (steam-rich) mixture [kJ/(kmol K)]
dHr1 = 124e3;          % reaction enthalpy, reaction 1 (> 0, endothermic) [kJ/kmol]
dHr2 = 105e3;          % reaction enthalpy, reaction 2 (> 0, endothermic) [kJ/kmol]
% Declared simplifications: Cp constant (feed composition), reaction
% enthalpies independent of T, pressure drop neglected.

%% --- Heat exchange: the two cases ---
% UaV is a VOLUMETRIC heat-transfer coefficient [kW/(m^3 K)]: a jacketed
% tube exchanges heat continuously along its length, so the exchange term
% is written per unit reactor volume.
UaV_adiab = 0;     Tc_adiab = T0;     % case 1: adiabatic (Tc irrelevant)
UaV_heat  = 2;     Tc_heat  = 1000;   % case 2: heated jacket, Tc > T0

%% --- Numerical settings ---
V_end = 0.30;          % reactor volume to integrate over            [m^3]
h     = 0.001;         % RK4 fixed step in V (300 steps)             [m^3]
x0    = [0; T0];       % initial state: X_A = 0 and T = T0 @ V = 0

%% ========================================================================
%  2) MODEL: right-hand side f(x) of the ODE system (3.2)
%  ========================================================================
% All quantities are written as functions of the state (X_A, T), exactly
% as in the slides. Anonymous functions "freeze" the parameter values at
% the moment they are defined: if a parameter is changed later, the
% functions must be re-defined.

% Local volumetric flow: expands with composition AND temperature
Vdot_fun    = @(X_A,T) Vdot0*(1 + epsA*X_A).*(T/T0);           % [m^3/s]

% Local concentration of A
CA_fun   = @(X_A,T) nA0*(1 - X_A)./Vdot_fun(X_A,T);        % [kmol/m^3]

% Total molar flow: increases with conversion (1 mole A -> 2 moles)
ndot_fun = @(X_A)   n0*(1 + epsA*X_A);               % [kmol/s]

% f1: mass balance on A written for X_A.
% From dnA/dV = -(k1 + k2)*C_A with dnA = -nA0*dX_A:
%   dX_A/dV = (k1 + k2)*C_A/nA0 = (k1 + k2)*(1 - X_A)/(Vdot0*(1+epsA*X_A)*T/T0)
f1 = @(X_A,T) (k1(T) + k2(T)).*CA_fun(X_A,T)/nA0;      % [1/m^3]

% f2: energy balance. Numerator: heat "generated" by the reactions per
% unit volume (NEGATIVE, both reactions are endothermic: they consume
% heat) plus heat received from the jacket (positive if Tc > T).
% Denominator: heat capacity flow of the gas mixture.
% UaV and Tc are ARGUMENTS, so the same f2 serves both cases.
f2 = @(X_A,T,UaV,Tc) ( ((-dHr1)*k1(T) + (-dHr2)*k2(T)).*CA_fun(X_A,T) ...
                       - UaV*(T - Tc) ) ./ (ndot_fun(X_A)*Cp);   % [K/m^3]

% f(x): packs f1 and f2 into the COLUMN vector expected by the solver,
% mapping the physical notation (X_A, T) onto the state x = [x(1); x(2)].
% V is kept as first argument (standard ODE-solver signature), although
% the system is autonomous.
f = @(V, x, UaV, Tc) [ f1(x(1), x(2));
                       f2(x(1), x(2), UaV, Tc) ];

%% ========================================================================
%  3) NUMERICAL SOLUTION: RK4 for the two cases
%  ========================================================================
% rk4_fixed expects f(V, x): the case-specific parameters are fixed by
% wrapping f in a new anonymous function of (V, x) only.
[Vgrid, Xad] = rk4_fixed(@(V,x) f(V, x, UaV_adiab, Tc_adiab), 0, V_end, x0, h);
[~,     Xht] = rk4_fixed(@(V,x) f(V, x, UaV_heat,  Tc_heat ), 0, V_end, x0, h);
% Each solution has one ROW per grid point: column 1 = X_A, column 2 = T.

%% --- Styrene flow nB along the reactor (post-processing) ---
% Mass balance on B: dnB/dV = k1(T)*C_A(X_A,T). The right-hand side is
% known at every grid point once X_A(V) and T(V) are known, so nB follows
% by numerical QUADRATURE (cumulative trapezoidal rule), with nB = 0 at
% V = 0. With h = 0.001 m^3 the quadrature error is negligible.
% (In Example 3.1 the ratio dnB/dnA = -k1/(k1+k2) was constant; here it
% depends on T, so nB cannot be obtained algebraically from X_A.)
FB_ad = cumtrapz(Vgrid, k1(Xad(:,2)).*CA_fun(Xad(:,1), Xad(:,2)));
FB_ht = cumtrapz(Vgrid, k1(Xht(:,2)).*CA_fun(Xht(:,1), Xht(:,2)));

% Overall selectivity to styrene at the outlet: moles of B formed per
% mole of A converted.
S_ad = FB_ad(end)/(nA0*Xad(end,1));
S_ht = FB_ht(end)/(nA0*Xht(end,1));

%% --- Isothermal reference (Example 3.1): analytical solution at V_end ---
% -epsA*X - (1 + epsA)*ln(1 - X) = (k1 + k2)/Vdot0*V, solved by fzero.
kiso  = k1(T0) + k2(T0);
X_iso = fzero(@(X) -epsA*X - (1 + epsA)*log(1 - X) - kiso/Vdot0*V_end, [0 0.99]);
S_iso = k1(T0)/kiso;

%% --- Results table ---
fprintf('Non-isothermal PFR, V = %.2f m^3, RK4 with h = %.3f m^3\n\n', V_end, h);
fprintf('%-40s %8s %12s %12s\n', 'Case', 'X_A', 'T out [K]', 'Selectivity');
fprintf('%-40s %8.3f %12.1f %12.3f\n', 'Isothermal (Example 3.1)', X_iso, T0, S_iso);
fprintf('%-40s %8.3f %12.1f %12.3f\n', 'Adiabatic (UaV = 0)', ...
        Xad(end,1), Xad(end,2), S_ad);
fprintf('%-40s %8.3f %12.1f %12.3f\n', ...
        sprintf('Heat exchange (UaV = %g, Tc = %g K)', UaV_heat, Tc_heat), ...
        Xht(end,1), Xht(end,2), S_ht);
fprintf('\n');
% Physical reading: the adiabatic reactor cools down (endothermic
% reactions) and self-limits: lower conversion, but higher selectivity.
% With heat exchange T rises: higher conversion, lower selectivity
% (Ea2 >> Ea1 favours the side reaction at high T).

%% ========================================================================
%  4) STIFFNESS CHECK: eigenvalues of the Jacobian along the adiabatic run
%  ========================================================================
% A system is "stiff" when its Jacobian has eigenvalues of very different
% magnitude AND the fast one forces an explicit method to take tiny steps.
% What matters for stability of an explicit method is h*|lambda_fast|
% relative to the method's stability region (for RK4, roughly
% h*|lambda| < 2.8 on the negative real axis), not the ratio alone.
fprintf('Stiffness check (adiabatic case)\n');
fprintf('%8s %10s %8s %12s %12s %8s\n', 'V', 'T [K]', 'X_A', 'lambda_1', 'lambda_2', 'ratio');
for Vcheck = [0 0.15 0.30]
    [~, i] = min(abs(Vgrid - Vcheck));                 % closest grid point
    xi = Xad(i,:)';
    J  = numjac_fd(@(x) f(0, x, UaV_adiab, Tc_adiab), xi);
    lam = eig(J);
    ratio = max(abs(lam))/min(abs(lam));
    fprintf('%8.2f %10.1f %8.3f %12.4f %12.4f %8.1f\n', ...
            Vgrid(i), xi(2), xi(1), lam(1), lam(2), ratio);
end
fprintf('\n');

%% ========================================================================
%  5) STEP-SIZE TEST: is an explicit method adequate?
%  ========================================================================
% Same adiabatic problem with a step 50 times larger (only 6 steps over the
% whole reactor). If the result stays close to the fine-step solution, RK4
% is stable and accurate here, and no implicit method is needed.
h_big = 0.05;
[~, Xbig] = rk4_fixed(@(V,x) f(V, x, UaV_adiab, Tc_adiab), 0, V_end, x0, h_big);
fprintf('Step-size test (adiabatic): h = %.3f -> X_A = %.5f, T = %.3f K\n', ...
        h, Xad(end,1), Xad(end,2));
fprintf('                            h = %.3f -> X_A = %.5f, T = %.3f K\n\n', ...
        h_big, Xbig(end,1), Xbig(end,2));

%% ========================================================================
%  6) PLOTS
%  ========================================================================
% Isothermal conversion profile for reference (analytical, via fzero)
X_iso_prof = zeros(size(Vgrid));
for i = 2:numel(Vgrid)
    X_iso_prof(i) = fzero(@(X) -epsA*X - (1 + epsA)*log(1 - X) ...
                               - kiso/Vdot0*Vgrid(i), [0 0.99]);
end

figure('Name', 'Example 3.2 - Non-isothermal PFR');

subplot(1,2,1)
plot(Vgrid, X_iso_prof, 'k--', 'LineWidth', 1.0); hold on
plot(Vgrid, Xad(:,1), '-', 'LineWidth', 1.2);
plot(Vgrid, Xht(:,1), '-', 'LineWidth', 1.2);
xlabel('Reactor volume V [m^3]'); ylabel('Conversion of EB, X_A [-]');
title('Conversion profiles');
legend('Isothermal', 'Adiabatic', 'Heat exchange', 'Location', 'southeast');
grid on

subplot(1,2,2)
plot(Vgrid, T0*ones(size(Vgrid)), 'k--', 'LineWidth', 1.0); hold on
plot(Vgrid, Xad(:,2), '-', 'LineWidth', 1.2);
plot(Vgrid, Xht(:,2), '-', 'LineWidth', 1.2);
xlabel('Reactor volume V [m^3]'); ylabel('Temperature T [K]');
title('Temperature profiles');
legend('Isothermal', 'Adiabatic', 'Heat exchange', 'Location', 'west');
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

function J = numjac_fd(F, x)
% NUMJAC_FD  Jacobian dF/dx by forward finite differences, column by column.
%   Column j contains the derivatives of all components of F with respect
%   to x(j). The step is RELATIVE to |x_j| (absolute 1e-6 if |x_j| < 1),
%   which matters here because X_A ~ 0.1-1 while T ~ 900 K.
    x  = x(:);
    F0 = F(x);
    n  = numel(x);
    J  = zeros(n, n);
    for j = 1:n
        hj = 1e-6*max(1, abs(x(j)));  % relative step
        xh = x;  xh(j) = xh(j) + hj;  % perturb only component j
        J(:, j) = (F(xh) - F0)/hj;    % column j = dF/dx_j
    end
end
