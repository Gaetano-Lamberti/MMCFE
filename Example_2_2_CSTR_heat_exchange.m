%% Example 2.2 CSTR with heat exchange
% =========================================================================
%  A guiding example: the styrene reactor
%  2. The Algebraic Equation(s) problem(s)
%  2.2 Steady-state CSTR with heat exchange (1. adiabatic / 2. heated)
%  (slides 13-15)
% =========================================================================
%
% NOTATION (as in the slides)
%   A = ethylbenzene (EB)
%       R1:  EB -> ST + H2        r1 = k1(T)*C_A     (endothermic)
%       R2:  EB -> BZ + C2H4      r2 = k2(T)*C_A     (endothermic)
%   X_A = (F_A0 - F_A)/F_A0 is the conversion of A.
%
% A NOTE ON TERMINOLOGY
%   A steady-state CSTR is ALWAYS isothermal: perfect mixing makes the
%   temperature uniform throughout the vessel and equal to the outlet
%   temperature T. What distinguishes this example from Example 2.1 is
%   that T is no longer FIXED at the feed value T0: it is an UNKNOWN,
%   set by the energy balance (heats of reaction plus heat exchange with
%   the surroundings, Ua = 0 being the adiabatic limit). Hence "CSTR with
%   heat exchange", not "non-isothermal CSTR".
%
% WHAT CHANGES WITH RESPECT TO EXAMPLE 2.1 (slide 13)
%   1. The volumetric flow changes with conversion AND temperature
%      (ideal gas at constant pressure):   Q(X_A,T) = Q0*(1 + epsA*X_A)*T/T0
%   2. Hence the concentration of A too:   C_A(X_A,T) = F_A0*(1 - X_A)/Q(X_A,T)
%   3. The kinetic constants depend on T:  k_i(T) = k_i0*exp(-Ea_i/(R*T))
%   4. Hence the Damkohler number too:     N_Da(T) = (k1(T) + k2(T))*V/Q0
%
% PROBLEM (2.2): a system of two nonlinear algebraic equations
%   System's state:  x = [X_A; T]   (column vector)
%   Statement:       F(x) = 0,  with x and F in R^2
%
%   F1(X_A,T) = X_A - N_Da(T)*(1 - X_A)/((1 + epsA*X_A)*T/T0) = 0   (2.2.1)
%       mass balance on A (dimensionless)
%
%   F2(X_A,T) = (1 - T/T0)
%             + { V*[(-dHr1)*k1(T) + (-dHr2)*k2(T)]*C_A(X_A,T)
%                 - Ua*(T - Ta) } / (F_tot0*cp*T0)             = 0   (2.2.2)
%       energy balance, made dimensionless by dividing the dimensional
%       balance F2_dim (a heat flow in kW) by the reference heat flow
%       F_tot0*cp*T0 (slide 14). Ua = 0: adiabatic; Ua ~= 0: heated.
%
%   The two equations are GENUINELY COUPLED: F1 depends on T (through
%   N_Da(T) and T/T0) and F2 depends on X_A (through C_A). They must be
%   solved SIMULTANEOUSLY, not one after the other.
%
% WHAT THIS SCRIPT DOES
%   1) Solves F(x) = 0 by multivariate Newton-Raphson with a numerical
%      Jacobian, for the adiabatic and the heated reactor.
%   2) Prints the iterates (quadratic convergence) and the heat duties.
%   3) Compares with the CSTR at fixed temperature T = T0 of Example 2.1.
%   4) Shows the effect of the T/T0 factor in Q (same problem without it).
%
% HOW TO RUN
%   Open the file in MATLAB and press F5 (or "Run"). Everything is in this
%   single file: the Newton solver and a print helper are LOCAL FUNCTIONS at
%   the end of the script (this requires MATLAB R2016b or later).
%
% UNITS (important for the energy balance)
%   Rate constants in 1/s, V in m^3, so V*k*C_A is in kmol/s: ALL molar
%   flows must be in kmol/s (not kmol/h). With enthalpies in kJ/kmol and
%   cp in kJ/(kmol K), every term of F2_dim is in kJ/s = kW, and Ua must be
%   in kW/K.

clear; clc; close all;

%% ========================================================================
%  1) PARAMETERS
%  ========================================================================

%% --- Kinetic parameters (Arrhenius, Sec. 1.5) ---
R    = 8.314;          % gas constant                                [kJ/(kmol K)]
k0_1 = 1.47e5;         % pre-exponential factor, R1 (A -> ST)        [1/s]
Ea1  = 90.98e3;        % activation energy, R1                       [kJ/kmol]
k0_2 = 9.02e10;        % pre-exponential factor, R2 (A -> BZ)        [1/s]
Ea2  = 207.99e3;       % activation energy, R2                       [kJ/kmol]

% Rate constants as FUNCTIONS of T (anonymous functions, slide 13, point 3).
% Element-wise operators (./) allow evaluation on vectors of T.
k1 = @(T) k0_1*exp(-Ea1./(R*T));   % [1/s]
k2 = @(T) k0_2*exp(-Ea2./(R*T));   % [1/s]

%% --- Operating conditions (Sec. 1.6) ---
T0    = 900;           % feed temperature (also the T at which Q0 is given) [K]
P     = 1e5;           % pressure (isobaric reactor)                 [Pa]
F_A0  = 1/3600;        % inlet molar flow of A (EB),  1 kmol/h       [kmol/s]
F_W0  = 9/3600;        % inlet molar flow of steam,   9 kmol/h       [kmol/s]
F_tot0 = F_A0 + F_W0;  % total inlet molar flow                      [kmol/s]
y_A0  = F_A0/F_tot0;   % mole fraction of A in the feed (0.10)       [-]
epsA  = y_A0*1;        % expansion factor (delta_nu = +1 per mole of A) [-]
Q0    = F_tot0*1e3*R*T0/P;  % inlet volumetric flow at T0, ideal gas [m^3/s]
                            % (1e3: kmol -> mol, with R in J/(mol K))
V     = 0.250;         % reactor volume (same as Example 2.1)        [m^3]

%% --- Thermal parameters ---
cp   = 58;             % molar heat capacity of the (steam-rich) feed [kJ/(kmol K)]
dHr1 = 124e3;          % reaction enthalpy, R1 (> 0, endothermic)    [kJ/kmol]
dHr2 = 105e3;          % reaction enthalpy, R2 (> 0, endothermic)    [kJ/kmol]
% Declared simplifications: cp constant (feed composition); the sensible
% heat term uses the inlet molar flow F_tot0 (the moles actually increase
% with X_A); reaction enthalpies independent of T.

%% --- Heat exchange: the two cases (slide title: 1. adiabatic / 2. heated) ---
% Ua = U*a: heat-exchange coefficient times heat-exchange surface area.
Ua_adiab = 0;      Ta_adiab = T0;     % case 1: adiabatic (Ta irrelevant)
Ua_heat  = 0.20;   Ta_heat  = 1000;   % case 2: heated, Ta > T0      [kW/K], [K]

%% --- Numerical settings ---
tol     = 1e-10;       % tolerance on max|F_i| (both residuals are O(1))
maxiter = 50;          % maximum number of Newton iterations
x0      = [0.49; 900]; % initial guess [X_A; T]: the solution at T = T0

%% ========================================================================
%  2) MODEL: the residual vector F(x), with x = [X_A; T]
%  ========================================================================
% All functions below are ANONYMOUS FUNCTIONS (@). MATLAB "freezes" the
% parameter values (V, Q0, T0, ...) at the moment each function is
% defined: if a parameter is changed afterwards, the functions must be
% re-defined, otherwise they keep the old values.

% Slide 13, point 1: volumetric flow, expanded by composition AND temperature.
% In Example 2.1 the factor T/T0 was identically 1; here T is an unknown.
Q_fun  = @(X_A,T) Q0*(1 + epsA*X_A).*(T/T0);                  % [m^3/s]

% Slide 13, point 2: concentration of A
CA_fun = @(X_A,T) F_A0*(1 - X_A)./Q_fun(X_A,T);               % [kmol/m^3]

% Slide 13, point 4: Damkohler number, referred to the INLET flow Q0.
% The expansion effects are NOT inside N_Da: they appear explicitly in F1.
NDa_fun = @(T) (k1(T) + k2(T))*V/Q0;                          % [-]

% (2.2.1) Mass balance on A, dimensionless residual, O(1).
% Derived from F_A0*X_A = (k1 + k2)*C_A*V, substituting C_A and dividing
% by F_A0. T enters TWICE: through N_Da(T) (kinetics) and through T/T0
% (gas expansion).
F1 = @(X_A,T) X_A - NDa_fun(T).*(1 - X_A)./((1 + epsA*X_A).*(T/T0));

% Energy balance, DIMENSIONAL form (slide 14), every term in kW:
%   F_tot0*cp*(T0 - T)        sensible heat released by the feed cooling
%                             from T0 to T (positive if T < T0)
%   V*[...]*C_A               heat "generated" by the reactions: NEGATIVE,
%                             since both are endothermic (they CONSUME heat)
%   -Ua*(T - Ta)              heat received from the heating medium
%                             (positive if Ta > T); Ua = 0 -> adiabatic
% Ua and Ta are ARGUMENTS, so the same function serves both cases.
F2_dim = @(X_A,T,Ua,Ta) F_tot0*cp*(T0 - T) ...
       + V*((-dHr1)*k1(T) + (-dHr2)*k2(T)).*CA_fun(X_A,T) ...
       - Ua*(T - Ta);                                          % [kW]

% (2.2.2) Energy balance, DIMENSIONLESS form (slide 14).
% F1 is O(1), while F2_dim is O(10) kW: without scaling, the stopping test
% max|F_i| < tol would be dominated by the energy balance, and tol would
% mean different things for the two equations. Dividing by the reference
% heat flow F_tot0*cp*T0 gives F2 = (1 - T/T0) + ..., i.e. the energy
% balance expressed as a fraction of T0: |F2| < tol means an error of about
% tol*T0 on T. Scaling changes neither the solution nor (in exact
% arithmetic) the Newton iterates; it changes the meaning of the stopping
% criterion.
Qref = F_tot0*cp*T0;                                           % [kW]
F2 = @(X_A,T,Ua,Ta) F2_dim(X_A,T,Ua,Ta)/Qref;                  % [-]

% (2.2) Residual vector F(x): packs F1 and F2 into the COLUMN vector
% expected by the solver (the ";" inside [ ] makes it a column) and maps
% the physical notation (X_A, T) onto the state x = [x(1); x(2)].
Fvec = @(x,Ua,Ta) [ F1(x(1), x(2));
                    F2(x(1), x(2), Ua, Ta) ];

%% ========================================================================
%  3) SOLUTION: multivariate Newton-Raphson, two cases
%  ========================================================================
% The scalar update of Example 2.1 generalizes to
%   J(x_n)*dx = -F(x_n),   x_{n+1} = x_n + dx
% where J is the 2x2 Jacobian matrix of partial derivatives dF_i/dx_j.
% Each iteration solves a 2x2 LINEAR system. J is evaluated by forward
% finite differences inside newton_system (differentiating F2 by hand
% through k1(T), k2(T) is tedious and error-prone, with no benefit).

%% --- Case 1: adiabatic reactor (Ua = 0) ---
[xA, histA] = newton_system(Fvec, x0, Ua_adiab, Ta_adiab, tol, maxiter);
print_history('Case 1 - adiabatic (Ua = 0)', histA, Fvec, Ua_adiab, Ta_adiab);

%% --- Case 2: heated reactor (Ua > 0, Ta > T0) ---
[xH, histH] = newton_system(Fvec, x0, Ua_heat, Ta_heat, tol, maxiter);
print_history(sprintf('Case 2 - heated (Ua = %.2f kW/K, Ta = %.0f K)', Ua_heat, Ta_heat), ...
              histH, Fvec, Ua_heat, Ta_heat);

%% --- Heat duties at the solution of Case 2 (diagnostics, kW) ---
% At the solution the three terms of F2_dim must sum to zero.
Xs = xH(1);  Ts = xH(2);
q_sens  = F_tot0*cp*(T0 - Ts);
q_react = V*((-dHr1)*k1(Ts) + (-dHr2)*k2(Ts))*CA_fun(Xs,Ts);
q_exch  = -Ua_heat*(Ts - Ta_heat);
fprintf('Case 2 heat duties [kW]: sensible %+.3f, reaction %+.3f, exchange %+.3f, sum %.1e\n\n', ...
        q_sens, q_react, q_exch, q_sens + q_react + q_exch);

%% --- Comparison with the CSTR at fixed temperature (Example 2.1) ---
% Reference at T = T0: analytical root of epsA*X^2 + (1 + N_Da)*X - N_Da = 0
NDa0  = NDa_fun(T0);
X_iso = (-(1 + NDa0) + sqrt((1 + NDa0)^2 + 4*epsA*NDa0))/(2*epsA);

fprintf('%-26s %10s %10s\n', 'Case', 'X_A [%]', 'T [K]');
fprintf('%-26s %10.2f %10.1f\n', 'T = T0 (Example 2.1)', 100*X_iso, T0);
fprintf('%-26s %10.2f %10.1f\n', 'Adiabatic',                100*xA(1), xA(2));
fprintf('%-26s %10.2f %10.1f\n\n', 'Heated',                 100*xH(1), xH(2));
% Physical reading: without an external heat supply the endothermic
% reactions cool the reactor down and self-limit (conversion drops from
% about 49% to about 29% at the SAME volume). With heat supplied
% (Ta > T0), temperature and conversion recover above the value at T = T0.

%% ========================================================================
%  4) WHY THE T/T0 FACTOR MATTERS: the same problem WITHOUT it
%  ========================================================================
% For comparison only: residuals re-written as if the volumetric flow did
% not change with temperature (Q = Q0*(1 + epsA*X_A)). Names end in "_noT"
% so that the correct model above is left untouched.
CA_noT = @(X_A) F_A0*(1 - X_A)./(Q0*(1 + epsA*X_A));
F1_noT = @(X_A,T) X_A - NDa_fun(T).*(1 - X_A)./(1 + epsA*X_A);
F2_noT = @(X_A,T,Ua,Ta) (F_tot0*cp*(T0 - T) ...
       + V*((-dHr1)*k1(T) + (-dHr2)*k2(T)).*CA_noT(X_A) - Ua*(T - Ta))/Qref;
Fvec_noT = @(x,Ua,Ta) [F1_noT(x(1), x(2)); F2_noT(x(1), x(2), Ua, Ta)];

xA_noT = newton_system(Fvec_noT, x0, Ua_adiab, Ta_adiab, tol, maxiter);
xH_noT = newton_system(Fvec_noT, x0, Ua_heat,  Ta_heat,  tol, maxiter);

fprintf('Effect of T/T0 in Q(X_A,T)\n');
fprintf('%-12s %16s %14s %16s %14s\n', 'Case', 'X_A no T/T0 [%]', 'X_A with [%]', ...
        'T no T/T0 [K]', 'T with [K]');
fprintf('%-12s %16.2f %14.2f %16.2f %14.2f\n', 'Adiabatic', ...
        100*xA_noT(1), 100*xA(1), xA_noT(2), xA(2));
fprintf('%-12s %16.2f %14.2f %16.2f %14.2f\n', 'Heated', ...
        100*xH_noT(1), 100*xH(1), xH_noT(2), xH(2));
% In the adiabatic case the correction RAISES the conversion: the cooled
% gas contracts, the volumetric flow falls, and both the residence time
% and C_A increase. In the heated case T > T0 and the effect reverses
% (smaller, because T departs from T0 by only about 7 K).

%% ========================================================================
%  LOCAL FUNCTIONS (must stay at the end of the script)
%  ========================================================================

function [x, hist, iters] = newton_system(Fvec, x0, Ua, Ta, tol, maxiter)
% NEWTON_SYSTEM  Multivariate Newton-Raphson with a numerical Jacobian
% (forward finite differences). At each iteration it solves the LINEAR
% system J*dx = -F to obtain the increment dx.
%   INPUT   Fvec    - handle to the residual vector F(x,Ua,Ta), a COLUMN
%           x0      - initial guess of the unknowns, e.g. [X_A; T]
%           Ua, Ta  - physical parameters passed through to Fvec
%           tol     - tolerance on max(abs(F)) (stopping criterion)
%           maxiter - maximum number of iterations
%   OUTPUT  x       - final estimate (column vector)
%           hist    - history of the iterates, one ROW per iteration (row 1 = x0)
%           iters   - number of Newton iterations actually performed

    x = x0(:);                 % force a column vector, whatever the shape of x0
    hist = x';                 % store the initial guess as the first row
    n = length(x);             % number of unknowns (size of J: n x n)

    for it = 1:maxiter
        F0 = Fvec(x, Ua, Ta);  % residuals at the current point (reused below)

        if max(abs(F0)) < tol  % converged: largest residual below tolerance
            iters = it-1;      % the point was produced by the previous iteration
            return
        end

        % Numerical Jacobian, one COLUMN at a time (forward differences):
        % column j = derivatives of all residuals with respect to x(j)
        J = zeros(n,n);
        for j = 1:n
            h = 1e-6*max(1, abs(x(j)));        % relative step (absolute if |x_j| < 1):
                                               % X_A ~ 0.5 and T ~ 900 K differ by
                                               % three orders of magnitude
            xh = x; xh(j) = xh(j) + h;         % perturb only component j
            J(:,j) = (Fvec(xh,Ua,Ta) - F0)/h;  % dF/dx_j
        end

        dx = J\(-F0);          % solve J*dx = -F0 (Gauss/LU via backslash;
                               % never compute inv(J))
        x = x + dx;            % full Newton step (no damping)
        hist = [hist; x'];     %#ok<AGROW> store the new iterate as a row
    end

    % Only reached if the loop ended WITHOUT meeting the stopping criterion
    iters = maxiter;
    warning('newton_system:noConvergence', ...
        'No convergence in %d iterations (max|F| = %.3e).', ...
        maxiter, max(abs(Fvec(x,Ua,Ta))));
end

function print_history(title_str, hist, Fvec, Ua, Ta)
% PRINT_HISTORY  Prints the Newton iterates and the residual norm max|F_i|.
    fprintf('%s\n', title_str);
    fprintf('%4s %12s %12s %12s\n', 'it', 'X_A', 'T [K]', 'max|F_i|');
    for k = 1:size(hist,1)
        Fk = Fvec(hist(k,:)', Ua, Ta);
        fprintf('%4d %12.6f %12.3f %12.2e\n', k-1, hist(k,1), hist(k,2), max(abs(Fk)));
    end
    fprintf('\n');
end
