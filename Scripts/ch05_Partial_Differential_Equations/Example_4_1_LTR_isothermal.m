%% Example 4.1 LTR isothermal - comparison of solution methods
% =========================================================================
%  A guiding example: the styrene reactor
%  4. The PDE(s) problem(s)
%  4.1 Isothermal steady-state Laminar Tubular Reactor (LTR)  (slides 21-26)
% =========================================================================
%
% NOTATION (as in the slides)
%   A = ethylbenzene (EB), B = styrene (ST); also H2, BZ (benzene),
%   C2H4 (ethylene) and W (steam, inert).
%       R1:  EB -> ST + H2        rate constant k1
%       R2:  EB -> BZ + C2H4      rate constant k2
%   n_i = v_z*C_i : local axial molar flux of species i [kmol/(m^2 s)]
%   X_A(r,z) = 1 - n_A(r,z)/n_A0(r)   local CONVERSION of EB
%   Y_B(r,z) = n_B(r,z)/n_A0(r)       local YIELD of styrene
%
% PROBLEM (4.1): one nonlinear, PARABOLIC PDE in (r, z)
%
%   2*vbar0*(1 - (r/R)^2) * dX_A/dz =
%        = D*(1/r)*d/dr( r*dX_A/dr ) + (k1 + k2)*(1 - X_A)/(1 + epsA*X_A)
%
%   BC1  @ r = 0, for all z :  dX_A/dr = 0   (symmetry)
%   BC2  @ r = R, for all z :  dX_A/dr = 0   (impermeable wall)
%   BC3  @ z = 0, for all r :  X_A = 0       (feed)
%
%   z plays the role of time: the solution is MARCHED from z = 0 to z = L.
%
% ONE SPATIAL DISCRETIZATION, FOUR WAYS OF MARCHING IN z
%   The radius is discretized with N_r = 40 intervals and second-order
%   finite differences (METHOD OF LINES): the PDE becomes a system of ODEs
%   in z, one per radial node. The same discretized system is then marched
%   in z in four ways:
%     1) FTCS  - Forward-Time (explicit Euler in z), Centred-Space.
%                Simplest possible scheme; limited by STABILITY.
%     2) IMEX  - IMplicit-EXplicit Euler: radial diffusion (the stiff part)
%                implicit, reaction explicit. One tridiagonal LINEAR system
%                per step, with a constant matrix factorized once.
%                Limited by ACCURACY (first order in z).
%     3) ode15s - MATLAB's stiff solver (variable order, adaptive step):
%                 fewest steps for a given accuracy. Used as REFERENCE.
%     4) pdepe  - MATLAB's built-in solver for parabolic PDEs in one space
%                 variable: a "black box" doing internally a method of
%                 lines + ode15s, with its own spatial discretization.
%                 (Not available in Octave: skipped there.)
%   Finally, from the ode15s solution, the concentrations of ALL species
%   are computed along the reactor and plotted.
%
% HOW TO RUN
%   Open the file in MATLAB (or Octave) and press F5 / "Run". Everything is
%   in this single file: helper functions are LOCAL FUNCTIONS at the end
%   (MATLAB R2016b or later).
%
% UNITS
%   Lengths in m, velocities in m/s, D in m^2/s, rate constants in 1/s,
%   molar flows in kmol/s, concentrations in kmol/m^3.

clear; clc; close all;

%% ========================================================================
%  1) PARAMETERS
%  ========================================================================

%% --- Kinetic parameters (Arrhenius, Sec. 1.5) ---
Rg   = 8.314;          % gas constant (NB: R is the tube radius)    [kJ/(kmol K)]
k0_1 = 1.47e5;         % pre-exponential factor, R1                  [1/s]
Ea1  = 90.98e3;        % activation energy, R1                       [kJ/kmol]
k0_2 = 9.02e10;        % pre-exponential factor, R2                  [1/s]
Ea2  = 207.99e3;       % activation energy, R2                       [kJ/kmol]

%% --- Operating conditions (Sec. 1.6), same as Sections 2-3 ---
T0     = 900;          % temperature, FIXED (isothermal)             [K]
P      = 1e5;          % pressure (isobaric)                         [Pa]
F_A0   = 1/3600;       % inlet molar flow of EB,     1 kmol/h        [kmol/s]
F_W0   = 9/3600;       % inlet molar flow of steam,  9 kmol/h        [kmol/s]
F_tot0 = F_A0 + F_W0;  % total inlet molar flow                      [kmol/s]
y_A0   = F_A0/F_tot0;  % inlet mole fraction of EB (0.10)            [-]
epsA   = y_A0*1;       % expansion factor (delta_nu = +1)            [-]
Q0     = F_tot0*1e3*Rg*T0/P;   % total inlet volumetric flow (0.208) [m^3/s]
C_tot  = P/(1e3*Rg*T0);        % total molar concentration (P/RT)    [kmol/m^3]
C_A0   = y_A0*C_tot;           % inlet concentration of EB           [kmol/m^3]
V_tot  = 0.30;                 % total volume (as the PFR of Ex. 3.1) [m^3]

k1   = k0_1*exp(-Ea1/(Rg*T0)); % [1/s]
k2   = k0_2*exp(-Ea2/(Rg*T0)); % [1/s]
ksum = k1 + k2;                % [1/s]
S    = k1/ksum;                % selectivity to ST, constant at fixed T [-]

%% --- Geometry of the multitubular reactor ---
R     = 0.015;                 % tube radius                         [m]
vbar0 = 1.0;                   % mean inlet velocity per tube        [m/s]
tau   = V_tot/Q0;              % mean residence time                 [s]
L     = vbar0*tau;             % tube length                         [m]
N_t   = Q0/(pi*R^2*vbar0);     % number of tubes                     [-]
D     = 1e-4;                  % diffusivity (all species)           [m^2/s]

%% --- Radial grid (common to ALL methods) ---
N_r = 40;                          % number of radial INTERVALS
r   = linspace(0, R, N_r+1)';      % nodes: r(1) = 0 (axis) ... r(N_r+1) = R (wall)
dr  = R/N_r;
v0  = 2*vbar0*(1 - (r/R).^2);      % Poiseuille profile at the nodes
v0(end) = 0;                       % exactly zero at the wall (no slip)
n_in = N_r;                        % number of MARCHED nodes (all but the wall)

fprintf('Isothermal LTR: N_t = %.0f tubes, R = %.3f m, L = %.3f m, N_r = %d\n\n', ...
        N_t, R, L, N_r);

%% ========================================================================
%  2) THE DISCRETIZED SYSTEM (method of lines)
%  ========================================================================
% Finite-difference matrix of the radial Laplacian (1/r) d/dr (r d/dr),
% with BC1 (axis) and BC2 (wall) built in: Lap*X gives the Laplacian at
% all N_r+1 nodes (see build_laplacian at the end of the file).
Lap = build_laplacian(r);

% Reaction source term (same at every node, since T is fixed)
src = @(X) ksum*(1 - X)./(1 + epsA*X);

% At each marched node i (v0 > 0):   v0_i * dX_i/dz = D*(Lap*X)_i + src(X_i)
% At the wall (v0 = 0) the equation has NO dz-derivative: it is ALGEBRAIC,
%     0 = D*(Lap*X)_wall + src(X_wall)
% i.e. a quadratic in X_wall, solved in closed form by wall_value: the wall
% is "slaved" to its neighbour and is never marched.
%
% Right-hand side of the marched ODE system dXi/dz = f(Xi):
f_ode = @(z, Xi) rhs_interior(Xi, Lap, v0, D, ksum, epsA, dr);

%% ========================================================================
%  3) STABILITY OF EXPLICIT EULER: where the FTCS limit comes from
%  ========================================================================
% Explicit Euler applied to dX/dz = J*X (J = Jacobian of f_ode) multiplies,
% at each step, every eigen-component of the error by (1 + dz*lambda).
% Stability requires |1 + dz*lambda| <= 1 for ALL eigenvalues lambda of J.
% Here the eigenvalues are real and negative, so the condition becomes
%          dz <= 2/|lambda|_max
% The largest |lambda| comes from the node next to the wall: there the
% diffusion term is divided by a very small velocity v0(r_N). A quick
% (Gershgorin) upper bound of that row is
%          |lambda|_max  <~  4*D/(v0(r_N)*dr^2)
% and since v0(r_N) ~ 4*vbar0/N_r and dr = R/N_r:
%          |lambda|_max  ~  D*N_r^3/(vbar0*R^2)   ->   dz_crit ~ N_r^(-3)
% Here the EXACT value is computed from the eigenvalues of J at the inlet.
J  = numerical_jacobian(@(Xi) f_ode(0, Xi), zeros(n_in,1));
lam = eig(J);
lam_max = max(abs(lam));
dz_crit = 2/lam_max;                              % exact explicit-Euler limit
lam_gersh = 4*D/(v0(n_in)*dr^2);                  % quick upper bound
fprintf('Stability of explicit Euler (FTCS), N_r = %d\n', N_r);
fprintf('  |lambda|_max = %.3e 1/m (Gershgorin bound %.3e 1/m)\n', lam_max, lam_gersh);
fprintf('  critical step dz_crit = 2/|lambda|_max = %.3e m  ->  at least %d steps\n\n', ...
        dz_crit, ceil(L/dz_crit));

%% ========================================================================
%  4) METHOD 3 FIRST: ode15s (it provides the REFERENCE solution)
%  ========================================================================
opts = odeset('RelTol', 1e-8, 'AbsTol', 1e-10, ...
              'JPattern', spdiags(ones(n_in,3), [-1 0 1], n_in, n_in), ...
              'InitialStep', 1e-6*L);
% (JPattern: the Jacobian is tridiagonal. InitialStep: a small first step,
%  because at z = 0 the wall value jumps to X_wall > 0 while its neighbours
%  are still 0.)
tic;
[z_steps, ~] = ode15s(f_ode, [0 L], zeros(n_in,1), opts);   % its own steps
nsteps_ode = numel(z_steps) - 1;
z = linspace(0, L, 301)';                                    % output grid
[z, Xi] = ode15s(f_ode, z, zeros(n_in,1), opts);
t_ode = toc;
X_ode = [Xi, wall_value(Xi(:,end), D, dr, ksum, epsA)];      % append the wall
Xb_ode = bulk_average(X_ode, v0, r);                         % cup-mixing average
X_ref  = Xb_ode(end);

%% ========================================================================
%  5) METHOD 1: FTCS (explicit Euler in z)
%  ========================================================================
%   X_i^{n+1} = X_i^n + dz * f_i(X^n),   then the wall value is updated.
% (a) a step just BELOW the critical one: stable
nz_ftcs = ceil(1.05*L/dz_crit);                % 5% more steps than the minimum
tic;
[X_ftcs, ok] = march_ftcs(f_ode, n_in, L, nz_ftcs, D, dr, ksum, epsA);
t_ftcs = toc;
Xb_ftcs = bulk_average(X_ftcs, v0, r);          % bulk conversion at every step
z_ftcs  = linspace(0, L, nz_ftcs+1)';
% (b) a step just ABOVE the critical one: diverges
nz_bad = floor(0.95*L/dz_crit);                % 5% fewer steps than the minimum
[~, ok_bad, n_blow] = march_ftcs(f_ode, n_in, L, nz_bad, D, dr, ksum, epsA);

%% ========================================================================
%  6) METHOD 2: IMEX Euler (implicit diffusion, explicit reaction)
%  ========================================================================
%   v0_i*(X_i^{n+1} - X_i^n)/dz = D*(Lap*X^{n+1})_i + src(X_i^n)
% i.e. the LINEAR system  (Mdz - D*Lap) * X^{n+1} = Mdz*X^n + src(X^n),
% with Mdz = diag(v0)/dz. The wall row has v0 = 0: it becomes the
% algebraic wall equation (with the reaction evaluated at the previous
% step, an O(dz) approximation). The matrix is the same at every step:
% it is factorized ONCE (LU) and reused.
nz_imex_list = [200 2000];
Xb_imex = zeros(size(nz_imex_list));  t_imex = Xb_imex;
for k = 1:numel(nz_imex_list)
    tic;
    X_imex = march_imex(Lap, v0, D, src, L, nz_imex_list(k));
    t_imex(k) = toc;
    Xbz = bulk_average(X_imex, v0, r);          % bulk conversion at every step
    Xb_imex(k) = Xbz(end);
    if k == 1, Xbz_imex1 = Xbz; end             % keep the coarse history for the plot
end
z_imex1 = linspace(0, L, nz_imex_list(1)+1)';

%% ========================================================================
%  7) METHOD 4: pdepe (MATLAB built-in, black box)
%  ========================================================================
% pdepe solves  c*du/dt = x^-m * d/dx( x^m * f ) + s  in one space
% variable x and one marching variable t. Mapping for problem (4.1):
%   t -> z,  x -> r,  m = 1 (cylindrical),  u -> X_A,
%   c = v0(r),  f = D*dX_A/dr,  s = (k1+k2)*(1 - X_A)/(1 + epsA*X_A)
% c vanishes at the wall: allowed by pdepe, because r = R is a mesh point.
% Boundary conditions in the form p + q*f = 0:
%   r = 0 : with m = 1 pdepe imposes symmetry itself (pl, ql ignored)
%   r = R : p = 0, q = 1  ->  f = D*dX_A/dr = 0  (BC2)
has_pdepe = exist('pdepe', 'file') == 2;
if has_pdepe
    tic;
    pde_fun = @(rr, zz, X, dXdr) pde_ltr(rr, X, dXdr, R, vbar0, D, ksum, epsA);
    ic_fun  = @(rr) 0;                                   % BC3: X_A = 0 at z = 0
    bc_fun  = @(rl, Xl, rr, Xr, zz) deal(0, 1, 0, 1);    % pl, ql, pr, qr
    popts   = odeset('RelTol', 1e-8, 'AbsTol', 1e-10);   % tighten the defaults (1e-3)
    X_pdepe = pdepe(1, pde_fun, ic_fun, bc_fun, r', z', popts);   % rows: z, cols: r
    t_pdepe = toc;
    Xb_pdepe = bulk_average(X_pdepe, v0, r);
end

%% ========================================================================
%  8) COMPARISON TABLE
%  ========================================================================
fprintf('Comparison of marching methods (same radial grid, N_r = %d)\n', N_r);
fprintf('%-34s %10s %12s %12s %10s\n', 'Method', 'z-steps', 'X_bulk(L)', '|error|', 'CPU [s]');
fprintf('%-34s %10d %12.6f %12s %10.3f\n', 'ode15s (reference, rtol 1e-8)', nsteps_ode, X_ref, '-', t_ode);
fprintf('%-34s %10d %12.6f %12.1e %10.3f\n', 'FTCS, 5% below dz_crit', nz_ftcs, Xb_ftcs(end), ...
        abs(Xb_ftcs(end) - X_ref), t_ftcs);
if ~ok_bad
    fprintf('%-34s %10d %12s %12s %10s\n', 'FTCS, 5% above dz_crit', nz_bad, ...
            sprintf('DIVERGED@%d', n_blow), '-', '-');
end
for k = 1:numel(nz_imex_list)
    fprintf('%-34s %10d %12.6f %12.1e %10.3f\n', sprintf('IMEX Euler, %d steps', nz_imex_list(k)), ...
            nz_imex_list(k), Xb_imex(k), abs(Xb_imex(k) - X_ref), t_imex(k));
end
if has_pdepe
    fprintf('%-34s %10s %12.6f %12.1e %10.3f\n', 'pdepe (rtol 1e-8)', 'internal', ...
            Xb_pdepe(end), abs(Xb_pdepe(end) - X_ref), t_pdepe);
else
    fprintf('%-34s   (pdepe not available, e.g. in Octave: skipped)\n', 'pdepe');
end
fprintf(['Notes: FTCS pays for STABILITY (the step is fixed by the near-wall nodes);\n' ...
         '       IMEX pays for ACCURACY (first order: error ~ 1/steps);\n' ...
         '       ode15s adapts order and step; pdepe uses its own r-discretization,\n' ...
         '       so it differs from the others at the level of the grid error.\n\n']);

%% ========================================================================
%  9) CONCENTRATIONS OF ALL SPECIES ALONG THE REACTOR (from ode15s)
%  ========================================================================
% Local yields per mole of EB fed along the same streamline (slide 26:
% equal diffusivities -> stoichiometry holds at every point):
%   EB   : 1 - X_A
%   ST   : Y_B = S*X_A                     (R1, constant S at fixed T)
%   H2   : Y_H2   = Y_B                    (formed with ST by R1)
%   BZ   : Y_BZ   = (1 - S)*X_A            (R2)
%   C2H4 : Y_C2H4 = Y_BZ                   (formed with BZ by R2)
%   W    : F_W0/F_A0 = 9                   (inert)
% Total moles per mole of EB fed: 1/y_A0 + X_A  (each converted EB gives
% 2 moles). Hence the local mole fraction of i is
%   y_i = Y_i/(1/y_A0 + X_A) = y_A0*Y_i/(1 + epsA*X_A)
% and, isothermal and isobaric, the local concentration is
%   C_i = y_i*C_tot = C_A0*Y_i/(1 + epsA*X_A)
% (for EB this is the familiar C_A = C_A0*(1 - X_A)/(1 + epsA*X_A)).
names = {'EB', 'ST', 'H_2', 'BZ', 'C_2H_4', 'W'};
yieldfun = @(X) cat(3, 1 - X, S*X, S*X, (1 - S)*X, (1 - S)*X, (F_W0/F_A0)*ones(size(X)));
conc_of  = @(X) C_A0*yieldfun(X)./(1 + epsA*X);   % local C_i (3rd dim = species)

% (a) CUP-MIXING (flow-averaged) concentrations along z: what a sample of
%     the whole tube outlet at position z would contain. They follow from
%     the bulk conversion, since the flows add up linearly over the section:
%     F_i(z) = F_A0*Y_i,bulk(z),  C_i,cup(z) = F_i(z)/Q(z), Q = Q0*(1+epsA*X_bulk).
C_cup = squeeze(conc_of(Xb_ode));                 % rows: z, cols: species

% (b) LOCAL concentrations over the whole (r, z) domain
C_loc = conc_of(X_ode);                           % z x r x species

% Outlet flows of the whole reactor (all N_t tubes)
fprintf('Outlet flows of the whole reactor [kmol/h] and cup-mixing concentrations [mol/m^3]\n');
Yout = squeeze(yieldfun(Xb_ode(end)));
for s = 1:numel(names)
    fprintf('  %-7s  F = %.4f kmol/h   C = %8.3f mol/m^3\n', ...
            strrep(names{s}, '_', ''), 3600*F_A0*Yout(s), 1e3*C_cup(end,s));
end

%% ========================================================================
%  10) PLOTS
%  ========================================================================
% Figure 1: the four methods on the bulk conversion
figure('Name', 'Example 4.1 - methods comparison');
plot(z, Xb_ode, '-', 'LineWidth', 2.0); hold on
plot(z_ftcs, Xb_ftcs, '--', 'LineWidth', 1.2);
plot(z_imex1, Xbz_imex1, ':', 'LineWidth', 1.6);
if has_pdepe, plot(z, Xb_pdepe, '-.', 'LineWidth', 1.2); end
xlabel('Axial position z [m]'); ylabel('Bulk conversion X_{A,bulk} [-]');
title(sprintf('Isothermal LTR, N_r = %d (curves nearly coincide)', N_r));
leg = {'ode15s (reference)', 'FTCS', sprintf('IMEX, %d steps', nz_imex_list(1))};
if has_pdepe, leg{end+1} = 'pdepe'; end
legend(leg, 'Location', 'southeast'); grid on

% Figure 2: concentrations of all species (ode15s solution)
figure('Name', 'Example 4.1 - species concentrations');
subplot(1,3,1)                                   % reacting species, cup-mixing
plot(z, 1e3*C_cup(:,1:5), 'LineWidth', 1.4);
xlabel('Axial position z [m]'); ylabel('Cup-mixing concentration [mol/m^3]');
title('Reacting species'); legend(names(1:5), 'Location', 'east'); grid on
% NB: the ST and H2 curves coincide, and so do BZ and C2H4 (same reaction).

subplot(1,3,2)                                   % steam, cup-mixing
plot(z, 1e3*C_cup(:,6), 'LineWidth', 1.4);
xlabel('Axial position z [m]'); ylabel('Cup-mixing concentration [mol/m^3]');
title('Steam (inert): diluted by the mole increase'); grid on

subplot(1,3,3)                                   % radial profiles at the outlet
plot(r/R, 1e3*squeeze(C_loc(end,:,1)), 'LineWidth', 1.4); hold on
plot(r/R, 1e3*squeeze(C_loc(end,:,2)), 'LineWidth', 1.4);
plot(r/R, 1e3*squeeze(C_loc(end,:,4)), 'LineWidth', 1.4);
xlabel('Dimensionless radius r/R [-]'); ylabel('Local concentration [mol/m^3]');
title('Outlet radial profiles'); legend('EB', 'ST', 'BZ', 'Location', 'east'); grid on

%% ========================================================================
%  LOCAL FUNCTIONS (must stay at the end of the script)
%  ========================================================================

function Lap = build_laplacian(r)
% BUILD_LAPLACIAN  Sparse finite-difference matrix of (1/r) d/dr (r d/dr)
% on the nodes r(1) = 0 ... r(end) = R, with zero-gradient conditions at
% the axis (BC1) and at the wall (BC2), both imposed with a ghost node.
    n  = numel(r);  dr = r(2) - r(1);
    lo = zeros(n,1); di = zeros(n,1); up = zeros(n,1);
    % Axis: (1/r)dX/dr -> d2X/dr2 (L'Hopital), so Lap = 2*d2X/dr2, and the
    % symmetric ghost node X(-dr) = X(+dr) gives 4*(X2 - X1)/dr^2
    di(1) = -4/dr^2;  up(2) = 4/dr^2;
    % Interior: central differences for d2X/dr2 + (1/r)*dX/dr
    i = (2:n-1)';
    lo(i-1) = 1/dr^2 - 1./(2*r(i)*dr);
    di(i)   = -2/dr^2;
    up(i+1) = 1/dr^2 + 1./(2*r(i)*dr);
    % Wall: ghost node X(R+dr) = X(R-dr) -> 2*(X_{n-1} - X_n)/dr^2
    lo(n-1) = 2/dr^2;  di(n) = -2/dr^2;
    Lap = spdiags([lo di up], [-1 0 1], n, n);
end

function Xw = wall_value(XN, D, dr, ksum, epsA)
% WALL_VALUE  Wall conversion from its ALGEBRAIC equation (v0(R) = 0):
%     0 = a*(XN - Xw) + ksum*(1 - Xw)/(1 + epsA*Xw),   a = 2*D/dr^2
% (XN = neighbouring node). Multiplying by (1 + epsA*Xw) gives the quadratic
%     a*epsA*Xw^2 + b*Xw - c = 0,  b = a + ksum - a*epsA*XN,  c = a*XN + ksum
% whose positive root is written in the numerically stable form below.
    a  = 2*D/dr^2;
    b  = a + ksum - a*epsA*XN;
    c  = a*XN + ksum;
    Xw = 2*c ./ (b + sqrt(b.^2 + 4*a*epsA*c));
end

function f = rhs_interior(Xi, Lap, v0, D, ksum, epsA, dr)
% RHS_INTERIOR  dX_i/dz at the marched nodes (all but the wall):
%     f_i = [ D*(Lap*X)_i + (k1+k2)*(1 - X_i)/(1 + epsA*X_i) ] / v0_i
    n  = numel(Xi);
    X  = [Xi; wall_value(Xi(n), D, dr, ksum, epsA)];   % all nodes, wall slaved
    LX = Lap*X;
    f  = (D*LX(1:n) + ksum*(1 - Xi)./(1 + epsA*Xi)) ./ v0(1:n);
end

function [Xall, ok, n_blow] = march_ftcs(f_ode, n_in, L, nz, D, dr, ksum, epsA)
% MARCH_FTCS  Explicit Euler in z on the method-of-lines system:
%     X_i^{n+1} = X_i^n + dz*f_i(X^n)   (marched nodes), wall slaved.
%   Xall: one ROW per axial step (nz+1 rows), one COLUMN per node.
%   Stops (ok = false) as soon as the solution blows up.
    dz = L/nz;  Xi = zeros(n_in,1);  ok = true;  n_blow = NaN;
    Xall = zeros(nz+1, n_in+1);
    Xall(1,:) = [Xi; wall_value(Xi(end), D, dr, ksum, epsA)]';
    for n = 1:nz
        Xi = Xi + dz*f_ode((n-1)*dz, Xi);          % explicit Euler step
        if any(~isfinite(Xi)) || max(abs(Xi)) > 10
            ok = false;  n_blow = n;  break          % divergence detected
        end
        Xall(n+1,:) = [Xi; wall_value(Xi(end), D, dr, ksum, epsA)]';
    end
end

function X = march_imex(Lap, v0, D, src, L, nz)
% MARCH_IMEX  IMEX Euler: implicit radial diffusion, explicit reaction.
%   (diag(v0)/dz - D*Lap) * X^{n+1} = diag(v0)/dz * X^n + src(X^n)
%   The matrix does not change with n: LU-factorize once, then only
%   forward/back substitutions (cheap: the matrix is tridiagonal).
%   X: one ROW per axial step (nz+1 rows), one COLUMN per node.
    n  = numel(v0);  dz = L/nz;
    Mdz = spdiags(v0/dz, 0, n, n);
    [Lf, Uf, Pf, Qf] = lu(Mdz - D*Lap);            % factorize once: Pf*A*Qf = Lf*Uf
    Xk = zeros(n,1);
    X  = zeros(nz+1, n);
    for k = 1:nz
        b  = Mdz*Xk + src(Xk);                     % explicit part (reaction)
        Xk = Qf*(Uf \ (Lf \ (Pf*b)));              % implicit part (diffusion)
        X(k+1,:) = Xk';
    end
end

function [c, f, s] = pde_ltr(rr, X, dXdr, R, vbar0, D, ksum, epsA)
% PDE_LTR  Coefficients of problem (4.1) in pdepe's form
%   c*dX/dz = r^-1 * d/dr( r*f ) + s
    c = 2*vbar0*(1 - (rr/R)^2);                    % velocity profile
    f = D*dXdr;                                    % diffusive flux (per C_A0)
    s = ksum*(1 - X)/(1 + epsA*X);                 % reaction source
end

function J = numerical_jacobian(F, x)
% NUMERICAL_JACOBIAN  Forward finite differences, one column per variable
% (same scheme as in the Newton solver of Example 2.2).
    x = x(:);  F0 = F(x);  n = numel(x);  J = zeros(n);
    for j = 1:n
        h = 1e-7*max(1, abs(x(j)));
        xh = x;  xh(j) = xh(j) + h;
        J(:,j) = (F(xh) - F0)/h;
    end
end

function Xb = bulk_average(Xs, v0, r)
% BULK_AVERAGE  Flow-averaged ("cup-mixing") value over the cross-section:
%   Xb = integral(v0*X*r dr) / integral(v0*r dr)   (trapezoidal rule)
%   Xs: one ROW per axial position, one COLUMN per radial node.
    w  = (v0.*r)';
    Xb = trapz(r, Xs.*w, 2) / trapz(r, w);
end
