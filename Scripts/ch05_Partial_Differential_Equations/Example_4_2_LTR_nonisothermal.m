%% Example 4.2 LTR non-isothermal
% =========================================================================
%  A guiding example: the styrene reactor
%  4. The PDE(s) problem(s)
%  4.2 Non-isothermal steady-state Laminar Tubular Reactor (LTR)  (slides 27-28)
% =========================================================================
%
% NOTATION (as in the slides)
%   A = ethylbenzene (EB), B = styrene (ST); also H2, BZ (benzene),
%   C2H4 (ethylene) and W (steam, inert).
%       R1:  EB -> ST + H2        rate constant k1(T), endothermic
%       R2:  EB -> BZ + C2H4      rate constant k2(T), endothermic
%   X_A(r,z) = 1 - n_A/n_A0   local CONVERSION of EB
%   Y_B(r,z) = n_B/n_A0       local YIELD of styrene
%   T(r,z)                    local temperature
%   (n_i = v_z*C_i: local axial molar flux of species i)
%   k   = thermal conductivity of the gas        [kW/(m K)]
%   alpha = thermal diffusivity                  [m^2/s]
%   h_w = wall heat-exchange coefficient         [kW/(m^2 K)]  (h_w = 0: adiabatic)
%   Tc = service (heating) fluid temperature    [K]
%
% PROBLEM (4.2): three coupled, nonlinear, PARABOLIC PDEs in (r, z)
%
%   v0(r) dX_A/dz = D*Lap(X_A) + (k1(T) + k2(T)) * c(X_A,T)
%   v0(r) dY_B/dz = D*Lap(Y_B) + k1(T) * c(X_A,T)
%   v0(r) dT/dz   = alpha*Lap(T) + (y_A0/Cp)*[(-dHr1)*k1(T) + (-dHr2)*k2(T)] * c(X_A,T)
%
%   with  v0(r) = 2*vbar0*(1 - (r/R)^2),   Lap = (1/r) d/dr (r d/dr),
%         c(X_A,T) = (1 - X_A)/((1 + epsA*X_A)*T/T0)
%
%                        X_A            Y_B            T
%   BC1 @ z = 0, all r : X_A = 0        Y_B = 0        T = T0
%   BC2 @ r = 0, all z : dX_A/dr = 0    dY_B/dr = 0    dT/dr = 0
%   BC3 @ r = R, all z : dX_A/dr = 0    dY_B/dr = 0    -k*dT/dr = h_w*(T - Tc)
%
%   BC3 for T is a ROBIN condition (a mix of the value and of the derivative).
%
% WHAT THIS SCRIPT DOES
%   The radius is discretized with N_r = 40 intervals and second-order
%   finite differences (METHOD OF LINES): each PDE becomes a set of ODEs in
%   z, one per radial node (3 variables per node). The system is marched in
%   z with three methods:
%     1) FTCS   - explicit Euler in z: simple, slow (tiny step imposed by
%                 STABILITY), shown for teaching purposes;
%     2) ode15s - MATLAB's stiff solver: the REFERENCE solution;
%     3) pdepe  - MATLAB's built-in solver for parabolic PDEs (black box,
%                 its own discretization). Not available in Octave: skipped.
%   Two cases are solved: adiabatic wall (h_w = 0) and heated wall.
%   Each is compared with its plug-flow limit (the PFR of Section 3).
%   Finally the concentrations of ALL species are computed along the reactor
%   (slide 28) and plotted.
%
% TREATMENT OF THE WALL (strategy "a")
%   At the wall v0(R) = 0: there is no marching equation there. The wall
%   values are obtained from the boundary conditions BC3 written with a
%   one-sided, second-order difference for d/dr:
%       du/dr|_R ~ (3*u_N - 4*u_{N-1} + u_{N-2}) / (2*dr)
%   which gives the wall value in closed form (see wall_values below):
%       X_A, Y_B :  u_N = (4*u_{N-1} - u_{N-2})/3
%       T        :  T_N = [ k*(4*T_{N-1} - T_{N-2})/(2*dr) + h_w*Tc ]
%                         / [ 3*k/(2*dr) + h_w ]
%   Only the nodes with v0 > 0 (axis ... last node before the wall) are
%   marched in z.
%
% HOW TO RUN
%   Open the file in MATLAB (or Octave) and press F5 / "Run". Everything is
%   in this single file: helper functions are LOCAL FUNCTIONS at the end
%   (MATLAB R2016b or later). Run time: a few seconds (FTCS is the slowest).
%
% UNITS
%   Lengths in m, velocities in m/s, D and alpha in m^2/s, rate constants in
%   1/s, enthalpies in kJ/kmol, Cp in kJ/(kmol K), k in kW/(m K), h_w in
%   kW/(m^2 K), molar flows in kmol/s, concentrations in kmol/m^3.

clear; clc; close all;

%% ========================================================================
%  1) PARAMETERS
%  ========================================================================

%% --- Kinetic parameters (Arrhenius, Sec. 1.5) ---
Rg   = 8.314;          % gas constant (NB: R is the tube radius)    [kJ/(kmol K)]
k0_1 = 1.47e5;   Ea1 = 90.98e3;    % R1: [1/s], [kJ/kmol]
k0_2 = 9.02e10;  Ea2 = 207.99e3;   % R2: [1/s], [kJ/kmol]
k1 = @(T) k0_1*exp(-Ea1./(Rg*T));  % rate constants as FUNCTIONS of T [1/s]
k2 = @(T) k0_2*exp(-Ea2./(Rg*T));

%% --- Thermal parameters (Sec. 2.2) ---
Cp   = 58;             % molar heat capacity of the feed mixture    [kJ/(kmol K)]
dHr1 = 124e3;          % reaction enthalpy, R1 (> 0: endothermic)   [kJ/kmol]
dHr2 = 105e3;          % reaction enthalpy, R2 (> 0: endothermic)   [kJ/kmol]

%% --- Operating conditions (Sec. 1.6), same as Sections 2-3 ---
T0     = 900;          % feed temperature                            [K]
p      = 1e5;          % pressure (isobaric)                         [Pa]
nA0   = 1/3600;       % inlet molar flow of EB,     1 kmol/h        [kmol/s]
nW0   = 9/3600;       % inlet molar flow of steam,  9 kmol/h        [kmol/s]
n0 = nA0 + nW0;  % total inlet molar flow                      [kmol/s]
y_A0   = nA0/n0;  % inlet mole fraction of EB (0.10)            [-]
epsA   = y_A0*1;       % expansion factor (delta_nu = +1)            [-]
Vdot0     = n0*1e3*Rg*T0/p;      % total inlet volumetric flow      [m^3/s]
C_tot0 = p/(1e3*Rg*T0);           % inlet total concentration p/RT0  [kmol/m^3]
C_A0   = y_A0*C_tot0;             % inlet concentration of EB        [kmol/m^3]
V_tot  = 0.30;                    % total volume (as the PFR, Sec. 3) [m^3]

%% --- Geometry of the multitubular reactor ---
R     = 0.015;                 % tube radius                         [m]
vbar0 = 1.0;                   % mean inlet velocity per tube        [m/s]
tau   = V_tot/Vdot0;              % mean residence time                 [s]
L     = vbar0*tau;             % tube length                         [m]
N_t   = Vdot0/(pi*R^2*vbar0);     % number of tubes                     [-]

%% --- Transport properties ---
D      = 1e-4;                 % mass diffusivity (all species)      [m^2/s]
alpha  = D;                    % thermal diffusivity: Lewis number = 1 (stated
                               % approximation, Sec. 4.4 of the document)
k_cond = alpha*C_tot0*Cp;      % thermal conductivity k = alpha*C_tot0*Cp [kW/(m K)]
                               % (~0.078 W/(m K), close to steam at 900 K)

%% --- The two cases: adiabatic and heated wall ---
% h_w is chosen so that 2*h_w/R = 2 kW/(m^3 K): the SAME volumetric
% coefficient UaV used for the PFR of Section 3.2.
cases(1).name = 'Adiabatic wall';  cases(1).h_w = 0;      cases(1).Tc = T0;
cases(2).name = 'Heated wall';     cases(2).h_w = 0.015;  cases(2).Tc = 1000;
%                                  [kW/(m^2 K)] = 15 W/(m^2 K);   [K]

%% --- Radial grid (common to all methods) ---
N_r = 40;                          % number of radial INTERVALS
r   = linspace(0, R, N_r+1)';      % nodes: r(1) = 0 (axis) ... r(N_r+1) = R (wall)
dr  = R/N_r;
v0  = 2*vbar0*(1 - (r/R).^2);      % Poiseuille profile at the nodes
v0(end) = 0;                       % exactly zero at the wall
n   = N_r;                         % number of MARCHED nodes per variable

% Collect everything the right-hand side needs in one structure
par = struct('D', D, 'alpha', alpha, 'k_cond', k_cond, 'epsA', epsA, ...
             'T0', T0, 'y_A0', y_A0, 'Cp', Cp, 'dHr1', dHr1, 'dHr2', dHr2, ...
             'k1', k1, 'k2', k2, 'r', r, 'dr', dr, 'v0', v0, 'n', n);

fprintf('Non-isothermal LTR: N_t = %.0f tubes, R = %.3f m, L = %.3f m, N_r = %d\n', ...
        N_t, R, L, N_r);
fprintf('  thermal conductivity k = %.3f W/(m K), Biot number h_w*R/k = %.2f (heated case)\n\n', ...
        1e3*k_cond, cases(2).h_w*R/k_cond);

%% ========================================================================
%  2) SOLUTION OF THE TWO CASES
%  ========================================================================
% State vector of the marched system (3n unknowns), stacked by variable:
%   U = [ X_A(r_1..r_n) ; Y_B(r_1..r_n) ; T(r_1..r_n) ]
z  = linspace(0, L, 301)';                         % output grid
U0 = [zeros(n,1); zeros(n,1); T0*ones(n,1)];       % BC1: feed conditions

% Sparsity pattern of the Jacobian: each unknown depends on the three
% variables at its own node and at the two neighbouring nodes
Jpat = kron(ones(3), spdiags(ones(n,3), [-1 0 1], n, n));

has_pdepe = exist('pdepe', 'file') == 2;
res = struct();

for ic = 1:2
    h_w = cases(ic).h_w;  Tc = cases(ic).Tc;
    f_ode = @(zz, U) rhs_ltr(U, par, h_w, Tc);    % dU/dz = f(U)

    %% --- 2a) ode15s: reference solution ---
    opts = odeset('RelTol', 1e-8, 'AbsTol', 1e-10, 'JPattern', Jpat, ...
                  'InitialStep', 1e-6*L);
    % (small first step: at z = 0 the wall temperature jumps to its BC3
    %  value while the neighbouring nodes are still at T0)
    tic;
    [z_steps, ~] = ode15s(f_ode, [0 L], U0, opts);        % to count the steps
    [~, Uz] = ode15s(f_ode, z, U0, opts);                  % on the output grid
    t_ode = toc;
    [Xs, Ys, Ts] = full_fields(Uz, par, h_w, Tc);         % add the wall nodes
    res(ic).X = Xs;  res(ic).Y = Ys;  res(ic).T = Ts;
    res(ic).nsteps_ode = numel(z_steps) - 1;  res(ic).t_ode = t_ode;

    %% --- 2b) FTCS: explicit Euler in z ---
    % Stability limit dz <= 2/|lambda|_max, with lambda the eigenvalues of
    % the Jacobian of f_ode (evaluated at the inlet). The largest |lambda|
    % comes from the nodes next to the wall, where v0 is very small.
    J   = numerical_jacobian(@(U) f_ode(0, U), U0);
    lam_max = max(abs(eig(J)));
    dz_crit = 2/lam_max;
    nz_ftcs = ceil(1.10*L/dz_crit);        % 10% margin: lambda changes along z
    tic;
    [U_ftcs, ok] = march_ftcs(f_ode, U0, L, nz_ftcs);
    t_ftcs = toc;
    res(ic).dz_crit = dz_crit;  res(ic).nz_ftcs = nz_ftcs;  res(ic).t_ftcs = t_ftcs;
    [Xf, Yf, Tf] = full_fields(U_ftcs(end,:), par, h_w, Tc);
    res(ic).ftcs = [bulk_average(Xf, v0, r), bulk_average(Tf, v0, r), ...
                    bulk_average(Yf, v0, r)/bulk_average(Xf, v0, r)];
    res(ic).ftcs_ok = ok;

    %% --- 2c) pdepe: black box ---
    % Form: c*du/dz = r^-1 d/dr( r*f ) + s,  m = 1 (cylindrical), u = [X_A; Y_B; T]
    %   c = [v0; v0; v0],  f = [D*dX/dr; D*dY/dr; alpha*dT/dr],  s = reaction terms
    % BC3 in the form p + q*f = 0:
    %   X_A, Y_B : p = 0, q = 1                       (zero flux)
    %   T        : f_T = alpha*dT/dr = -alpha*(h_w/k)*(T - Tc)
    %              -> p = alpha*(h_w/k)*(T - Tc), q = 1
    % At r = 0, with m = 1, pdepe imposes symmetry by itself (BC2).
    if has_pdepe
        tic;
        pde_fun = @(rr, zz, u, dudr) pde_ltr(rr, u, dudr, par, R, vbar0);
        ic_fun  = @(rr) [0; 0; T0];
        bc_fun  = @(rl, ul, rr, ur, zz) deal([0;0;0], [1;1;1], ...
                   [0; 0; alpha*(h_w/k_cond)*(ur(3) - Tc)], [1;1;1]);
        sol = pdepe(1, pde_fun, ic_fun, bc_fun, r', z', ...
                    odeset('RelTol', 1e-8, 'AbsTol', 1e-10));
        res(ic).t_pdepe = toc;
        Xp = sol(:,:,1);  Yp = sol(:,:,2);  Tp = sol(:,:,3);
        res(ic).pdepe = [bulk_average(Xp(end,:), v0, r), bulk_average(Tp(end,:), v0, r), ...
                         bulk_average(Yp(end,:), v0, r)/bulk_average(Xp(end,:), v0, r)];
    end

    %% --- 2d) plug-flow limit (radial diffusion -> infinity) ---
    % Averaging the three PDEs over the cross-section with a flat profile
    % gives the 1-D PFR model of Section 3.2, with UaV = 2*h_w/R:
    %   vbar0*dX/dz = (k1+k2)*c,   vbar0*dY/dz = k1*c,
    %   vbar0*dT/dz = (y_A0/Cp)*[..]*c + (2*h_w/R)*(Tc - T)/(C_tot0*Cp)
    f_pfr = @(zz, u) [ (k1(u(3)) + k2(u(3)))*cfac(u(1), u(3), epsA, T0);
                        k1(u(3))*cfac(u(1), u(3), epsA, T0);
                        (y_A0/Cp)*(-dHr1*k1(u(3)) - dHr2*k2(u(3)))*cfac(u(1), u(3), epsA, T0) ...
                        + (2*h_w/R)*(Tc - u(3))/(C_tot0*Cp) ] / vbar0;
    [~, up] = ode15s(f_pfr, [0 L], [0; 0; T0], ...
                     odeset('RelTol', 1e-9, 'AbsTol', 1e-11, 'InitialStep', 1e-6*L));
    res(ic).pfr = [up(end,1), up(end,3), up(end,2)/up(end,1)];
end

%% ========================================================================
%  3) RESULTS TABLE
%  ========================================================================
for ic = 1:2
    Xb = bulk_average(res(ic).X, v0, r);  Tb = bulk_average(res(ic).T, v0, r);
    Yb = bulk_average(res(ic).Y, v0, r);
    res(ic).Xb = Xb;  res(ic).Tb = Tb;  res(ic).Yb = Yb;
    fprintf('%s (h_w = %.0f W/(m^2 K), Tc = %.0f K)\n', cases(ic).name, 1e3*cases(ic).h_w, cases(ic).Tc);
    fprintf('  %-32s %9s %10s %12s %10s\n', 'Method', 'z-steps', 'X_bulk', 'T_bulk [K]', 'S');
    fprintf('  %-32s %9d %10.4f %12.2f %10.4f\n', 'ode15s (reference)', res(ic).nsteps_ode, ...
            Xb(end), Tb(end), Yb(end)/Xb(end));
    if res(ic).ftcs_ok
        fprintf('  %-32s %9d %10.4f %12.2f %10.4f\n', 'FTCS (explicit Euler)', res(ic).nz_ftcs, res(ic).ftcs);
    else
        fprintf('  %-32s %9d   DIVERGED\n', 'FTCS (explicit Euler)', res(ic).nz_ftcs);
    end
    if has_pdepe
        fprintf('  %-32s %9s %10.4f %12.2f %10.4f\n', 'pdepe', 'internal', res(ic).pdepe);
    else
        fprintf('  %-32s   (not available, e.g. in Octave: skipped)\n', 'pdepe');
    end
    fprintf('  %-32s %9s %10.4f %12.2f %10.4f\n', 'plug-flow limit (PFR, Sec. 3.2)', '-', res(ic).pfr);
    fprintf('  outlet T: centre %.1f K, wall %.1f K; FTCS step limit dz_crit = %.2e m\n\n', ...
            res(ic).T(end,1), res(ic).T(end,end), res(ic).dz_crit);
end

%% ========================================================================
%  4) CONCENTRATIONS OF ALL SPECIES ALONG THE REACTOR (slide 28, ode15s)
%  ========================================================================
% Local yields (stoichiometry, equal diffusivities):
%   EB: Y_A = 1 - X_A;  ST: Y_B;  H2: Y_H2 = Y_B;
%   BZ: Y_BZ = X_A - Y_B;  C2H4: Y_C2H4 = Y_BZ;  W: Y_W = nW0/nA0 = 9
% Local concentrations (slide 28):  C_i = C_A0*Y_i/((1 + epsA*X_A)*T/T0)
% CUP-MIXING concentration at position z (what a sample of the whole tube
% outlet at z would contain) = molar flow / volumetric flow:
%   C_i,cup = int( n_i r dr ) / int( v_z r dr )
%           = C_A0*int( v0*Y_i r dr ) / int( v0*(1 + epsA*X_A)*T/T0 r dr )
% since n_i = v0*C_A0*Y_i and v_z = v0*(1 + epsA*X_A)*T/T0 (slide 28).
names = {'EB', 'ST', 'H_2', 'BZ', 'C_2H_4', 'W'};
for ic = 1:2
    X = res(ic).X;  Y = res(ic).Y;  T = res(ic).T;
    Yi = cat(3, 1 - X, Y, Y, X - Y, X - Y, (nW0/nA0)*ones(size(X)));   % z x r x species
    w  = (v0.*r)';                                                        % weights v0*r
    den = trapz(r, w.*(1 + epsA*X).*(T/T0), 2);                           % int v_z r dr
    C_cup = zeros(numel(z), 6);
    for s = 1:6
        C_cup(:,s) = C_A0*trapz(r, w.*Yi(:,:,s), 2)./den;
    end
    res(ic).C_cup = C_cup;
    fprintf('%s: outlet flows [kmol/h] and cup-mixing concentrations [mol/m^3]\n', cases(ic).name);
    Yb_out = squeeze(trapz(r, w.*Yi(end,:,:), 2))/trapz(r, w);            % bulk yields
    for s = 1:6
        fprintf('  %-6s n = %.4f kmol/h   C = %8.3f mol/m^3\n', strrep(names{s}, '_', ''), ...
                3600*nA0*Yb_out(s), 1e3*C_cup(end,s));
    end
    fprintf('\n');
end

%% ========================================================================
%  5) PLOTS
%  ========================================================================
figure('Name', 'Example 4.2 - bulk profiles');
subplot(1,2,1); hold on
for ic = 1:2, plot(z, res(ic).Xb, 'LineWidth', 1.6); end
xlabel('Axial position z [m]'); ylabel('Bulk conversion X_{A,bulk} [-]');
title('Conversion'); legend({cases.name}, 'Location', 'southeast'); grid on
subplot(1,2,2); hold on
for ic = 1:2, plot(z, res(ic).Tb, 'LineWidth', 1.6); end
for ic = 1:2, plot(z, res(ic).T(:,1), ':', 'LineWidth', 1.2); end
for ic = 1:2, plot(z, res(ic).T(:,end), '--', 'LineWidth', 1.2); end
xlabel('Axial position z [m]'); ylabel('Temperature [K]');
title('Temperature: bulk (solid), centre (dotted), wall (dashed)'); grid on

figure('Name', 'Example 4.2 - radial profiles (heated wall)');
zplot = [0.1 0.25 0.5 1.0]*L;
subplot(1,2,1); hold on
for zz = zplot, [~, i] = min(abs(z - zz)); plot(r/R, res(2).T(i,:), 'LineWidth', 1.3); end
xlabel('r/R [-]'); ylabel('T [K]'); title('Heated wall: radial temperature profiles'); grid on
legend(arrayfun(@(zz) sprintf('z/L = %.2f', zz/L), zplot, 'UniformOutput', false), 'Location', 'northwest');
subplot(1,2,2); hold on
for zz = zplot, [~, i] = min(abs(z - zz)); plot(r/R, res(2).X(i,:), 'LineWidth', 1.3); end
xlabel('r/R [-]'); ylabel('X_A [-]'); title('Heated wall: radial conversion profiles'); grid on

figure('Name', 'Example 4.2 - species concentrations');
for ic = 1:2
    subplot(1,2,ic); hold on
    sty = {'-', '-', '--', '-', '--'};
    for s = 1:5, plot(z, 1e3*res(ic).C_cup(:,s), sty{s}, 'LineWidth', 1.4); end
    xlabel('Axial position z [m]'); ylabel('Cup-mixing concentration [mol/m^3]');
    title(cases(ic).name); legend(names(1:5), 'Location', 'east'); grid on
end
% NB: ST/H2 and BZ/C2H4 coincide (same reaction); steam is printed only.

%% ========================================================================
%  LOCAL FUNCTIONS (must stay at the end of the script)
%  ========================================================================

function c = cfac(X, T, epsA, T0)
% CFAC  Concentration factor c = C_A/C_A0 = (1 - X_A)/((1 + epsA*X_A)*T/T0)
    c = (1 - X)./((1 + epsA*X).*(T/T0));
end

function [Xw, Yw, Tw] = wall_values(X, Y, T, par, h_w, Tc)
% WALL_VALUES  Wall values from BC3 with a one-sided, second-order difference
%   du/dr|_R ~ (3*u_N - 4*u_{N-1} + u_{N-2})/(2*dr)
%   X_A, Y_B: du/dr = 0            ->  u_N = (4*u_{N-1} - u_{N-2})/3
%   T: -k*dT/dr = h_w*(T_N - Tc)  ->  T_N = [k*(4*T_{N-1} - T_{N-2})/(2*dr) + h_w*Tc]
%                                            / [3*k/(2*dr) + h_w]
%   X, Y, T: marched nodes (last two entries = the two nodes before the wall).
%   Works also on matrices (one row per axial position).
    a  = par.k_cond/(2*par.dr);
    Xw = (4*X(:,end) - X(:,end-1))/3;
    Yw = (4*Y(:,end) - Y(:,end-1))/3;
    Tw = (a*(4*T(:,end) - T(:,end-1)) + h_w*Tc)/(3*a + h_w);
end

function F = rhs_ltr(U, par, h_w, Tc)
% RHS_LTR  dU/dz for the marched nodes (method of lines):
%   v0_i dX_i/dz = D*(Lap X)_i     + (k1+k2)*c
%   v0_i dY_i/dz = D*(Lap Y)_i     + k1*c
%   v0_i dT_i/dz = alpha*(Lap T)_i + (y_A0/Cp)*[(-dHr1)k1 + (-dHr2)k2]*c
    n = par.n;
    X = U(1:n)';  Y = U(n+1:2*n)';  T = U(2*n+1:3*n)';      % row vectors
    [Xw, Yw, Tw] = wall_values(X, Y, T, par, h_w, Tc);    % BC3 (slaved wall)
    LX = lap_fd([X Xw], par);  LY = lap_fd([Y Yw], par);  LT = lap_fd([T Tw], par);
    c  = cfac(X, T, par.epsA, par.T0);
    a1 = par.k1(T);  a2 = par.k2(T);
    sX = (a1 + a2).*c;
    sY = a1.*c;
    sT = (par.y_A0/par.Cp)*(-par.dHr1*a1 - par.dHr2*a2).*c;
    v  = par.v0(1:n)';
    F  = [ (par.D*LX + sX)./v, (par.D*LY + sY)./v, (par.alpha*LT + sT)./v ]';
end

function Lu = lap_fd(u, par)
% LAP_FD  Radial Laplacian (1/r) d/dr (r du/dr) at the marched nodes 1..n,
%   given the values u at ALL nodes 1..n+1 (wall included).
%   Axis (BC2): (1/r) du/dr -> d2u/dr2 (L'Hopital), Laplacian = 2*d2u/dr2,
%   with the symmetric ghost node u(-dr) = u(+dr): 4*(u_2 - u_1)/dr^2.
    n = par.n;  dr = par.dr;  r = par.r';
    Lu = zeros(1, n);
    Lu(1) = 4*(u(2) - u(1))/dr^2;
    i = 2:n;
    Lu(i) = (u(i+1) - 2*u(i) + u(i-1))/dr^2 + (u(i+1) - u(i-1))./(2*r(i)*dr);
end

function [X, Y, T] = full_fields(U, par, h_w, Tc)
% FULL_FIELDS  From the marched unknowns (one ROW per z) to the full fields
% on all N_r+1 nodes, appending the wall values given by BC3.
    n = par.n;
    X = U(:, 1:n);  Y = U(:, n+1:2*n);  T = U(:, 2*n+1:3*n);
    [Xw, Yw, Tw] = wall_values(X, Y, T, par, h_w, Tc);
    X = [X Xw];  Y = [Y Yw];  T = [T Tw];
end

function [Uall, ok] = march_ftcs(f_ode, U0, L, nz)
% MARCH_FTCS  Explicit Euler in z:  U^{k+1} = U^k + dz*f(U^k).
%   Uall: one ROW per step. Stops (ok = false) if the solution blows up.
    dz = L/nz;  U = U0(:);  ok = true;
    Uall = zeros(nz+1, numel(U));  Uall(1,:) = U';
    for k = 1:nz
        U = U + dz*f_ode((k-1)*dz, U);
        if any(~isfinite(U)) || any(abs(U) > 1e4)
            ok = false;  Uall = Uall(1:k,:);  return
        end
        Uall(k+1,:) = U';
    end
end

function [c, f, s] = pde_ltr(rr, u, dudr, par, R, vbar0)
% PDE_LTR  Coefficients of problem (4.2) in pdepe's form
%   c .* du/dz = r^-1 d/dr( r*f ) + s,   u = [X_A; Y_B; T]
    v  = 2*vbar0*(1 - (rr/R)^2);
    X  = u(1);  T = u(3);
    cc = cfac(X, T, par.epsA, par.T0);
    a1 = par.k1(T);  a2 = par.k2(T);
    c  = [v; v; v];
    f  = [par.D*dudr(1); par.D*dudr(2); par.alpha*dudr(3)];
    s  = [(a1 + a2)*cc; a1*cc; (par.y_A0/par.Cp)*(-par.dHr1*a1 - par.dHr2*a2)*cc];
end

function J = numerical_jacobian(F, x)
% NUMERICAL_JACOBIAN  Forward finite differences, one column per variable
% (same scheme as in the Newton solver of Example 2.2).
    x = x(:);  F0 = F(x);  m = numel(x);  J = zeros(m);
    for j = 1:m
        h = 1e-7*max(1, abs(x(j)));
        xh = x;  xh(j) = xh(j) + h;
        J(:,j) = (F(xh) - F0)/h;
    end
end

function Xb = bulk_average(Xs, v0, r)
% BULK_AVERAGE  Flow-averaged ("cup-mixing") value over the cross-section,
% weighted by the local mass (and heat-capacity) flux v0*r:
%   Xb = integral(v0*X*r dr) / integral(v0*r dr)   (trapezoidal rule)
%   Xs: one ROW per axial position, one COLUMN per radial node.
    w  = (v0.*r)';
    Xb = trapz(r, Xs.*w, 2) / trapz(r, w);
end
