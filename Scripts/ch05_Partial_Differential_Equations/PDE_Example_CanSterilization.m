%% Lesson 5, Section 3.12.2 - Thermal sterilization of a canned food
% Conduction-heating food in a cylindrical can (radius R, height H). By symmetry
% only a quarter of the axial section is solved: 0 <= r <= R, 0 <= z <= Hh = H/2,
% with z measured from the MID-PLANE.
%
%   dT/dt = alpha * [ (1/r) d/dr( r dT/dr ) + d2T/dz2 ]
%   r = 0  : dT/dr = 0 (axis)        z = 0  : dT/dz = 0 (mid-plane)
%   r = R  : T = Tw(t)               z = Hh : T = Tw(t)   (lid)
%   Tw = TR (retort, condensing steam) for t < t_h, then TC (cooling water)
%   IC     : T = T0 everywhere
%
% TYPE: parabolic in t ONLY. In (r, z) the operator is ELLIPTIC (Laplacian):
% neither r nor z can be "marched". The method of lines therefore discretizes
% BOTH r and z (2D grid) and integrates in t: the ODE system has the 2D,
% block-tridiagonal matrix of Section 3.6 (cylindrical form, Eqs. 19-21).
%
% LETHALITY at the cold point (centre: r = 0, z = 0), added as one more ODE:
%   dF0/dt = 10^((T_c - Tref)/zD) / 60      [min of equivalent time at 121.1 C]
%
% VERIFICATION: exact double series (Bessel in r x cosine in z), with the
% cooling step added by superposition.
%
% HOW TO RUN: press F5. Helper functions are local functions at the end.

clear; clc; close all;

%% --- Parameters ---
R     = 0.038;   Hh = 0.050;      % can radius and half height [m] (76 x 100 mm can)
alpha = 1.5e-7;                   % thermal diffusivity of the food [m^2/s]
T0 = 40;  TR = 121.1;  TC = 20;   % initial, retort and cooling-water temperatures [C]
t_h   = 80*60;                    % end of heating [s]
t_end = 110*60;                   % end of cooling [s]
zD = 10;  Tref = 121.1;           % z-value and reference temperature [C]

%% --- Method of lines on a 2D (r,z) grid ---
Nr = 20;  Nz = 26;                % intervals: dr ~ dz ~ 1.9 mm
[A, b] = build_operator(Nr, Nz, R, Hh, alpha);
n = size(A, 1);
fprintf('MOL: %d x %d nodes -> %d ODEs (+1 for F0)\n', Nr, Nz, n);

% Stiffness of the semi-discrete system (Section 3.4.2)
ev = eig(full(A));
fprintf('Eigenvalues: slowest %.3e 1/s, fastest %.3e 1/s, stiffness ratio %.0f\n', ...
        -max(ev), -min(ev), min(ev)/max(ev));
fprintf('Explicit Euler would need dt < %.2f s\n\n', 2/(-min(ev)));

pattern = [A ~= 0, sparse(n,1); sparse(1,1,1,1,n), 0];    % Jacobian pattern (+ F0 row)
opts = odeset('RelTol', 1e-8, 'AbsTol', 1e-8, 'JPattern', pattern);
y0 = [T0*ones(n,1); 0];
% two phases: the wall temperature jumps at t_h
[t1, y1] = ode15s(@(t,y) rhs(y, A, b, TR, Tref, zD), linspace(0, t_h, 481), y0, opts);
[t2, y2] = ode15s(@(t,y) rhs(y, A, b, TC, Tref, zD), linspace(t_h, t_end, 181), y1(end,:)', opts);
t  = [t1; t2(2:end)];  Tc = [y1(:,1); y2(2:end,1)];  F0 = [y1(:,end); y2(2:end,end)];

%% --- Exact solution at the cold point (verification) ---
Mt = 80;
beta = bessel_zeros(Mt);                           % zeros of J0
cr   = 2./(beta.*besselj(1, beta));                % 1 = sum cr_m J0(beta_m r/R)
nn   = (0:Mt-1)';
gam  = (2*nn + 1)*pi/(2*Hh);                       % cos(gam_n z): zero at the lid
cz   = 4*(-1).^nn./((2*nn + 1)*pi);                % 1 = sum cz_n cos(gam_n z)
lam  = alpha*((beta/R).^2 + (gam').^2);            % decay rates (Mt x Mt)
Cmn  = cr*cz';
Tex = zeros(size(t));
for k = 1:numel(t)
    if t(k) <= t_h
        Tex(k) = TR + (T0 - TR)*sum(sum(Cmn.*exp(-lam*t(k))));
    else
        Tex(k) = TC + (T0 - TR)*sum(sum(Cmn.*exp(-lam*t(k)))) + (TR - TC)*sum(sum(Cmn.*exp(-lam*(t(k) - t_h))));
    end
end
F0ex = cumtrapz(t, 10.^((Tex - Tref)/zD))/60;

%% --- Results ---
[Tmax, imax] = max(Tc);
fprintf('Centre at end of heating: %.2f C (exact %.2f C)\n', Tc(numel(t1)), Tex(numel(t1)));
fprintf('Centre maximum: %.2f C at t = %.1f min, i.e. AFTER cooling has started\n', Tmax, t(imax)/60);
fprintf('F0 = %.3f min (exact %.3f), of which %.3f during heating\n', F0(end), F0ex(end), F0(numel(t1)));
fprintf('Max |T_centre - exact| = %.3f C (t > 0)\n', max(abs(Tc(2:end) - Tex(2:end))));

%% --- Plots ---
figure('Name', 'Can sterilization: cold point');
subplot(2,1,1)
plot(t/60, Tex, 'k-', 'LineWidth', 2); hold on
plot(t(1:12:end)/60, Tc(1:12:end), 'o', 'MarkerSize', 4);
plot([0 t_h t_h t_end]/60, [TR TR TC TC], '--', 'Color', [0.5 0.5 0.5]);
ylabel('T (C)'); legend('centre, exact', 'centre, MOL', 'wall', 'Location', 'south'); grid on
subplot(2,1,2)
plot(t/60, F0ex, 'k-', 'LineWidth', 2); hold on
plot(t(1:12:end)/60, F0(1:12:end), 'o', 'MarkerSize', 4);
xlabel('t (min)'); ylabel('F_0 (min)'); grid on

%% ========================================================================
%  LOCAL FUNCTIONS
%  ========================================================================
function [A, b] = build_operator(Nr, Nz, R, Hh, alpha)
% BUILD_OPERATOR  alpha * 2D cylindrical Laplacian on nodes r_i = i dr (i = 0..Nr-1),
% z_j = j dz (j = 0..Nz-1); the wall (i = Nr) and the lid (j = Nz) carry Tw.
% dT/dt = A*T + b*Tw ; unknown k = i*Nz + j + 1 (1-based).
    dr = R/Nr;  dz = Hh/Nz;  r = (0:Nr-1)'*dr;
    Ar = zeros(Nr);  br = zeros(Nr,1);
    Ar(1,1) = -4/dr^2;  Ar(1,2) = 4/dr^2;               % axis: L'Hopital + ghost, Eq. (21)
    for i = 2:Nr
        rm = r(i) - dr/2;  rp = r(i) + dr/2;            % conservative form, Eq. (19)
        Ar(i,i-1) = rm/(r(i)*dr^2);  Ar(i,i) = -(rm + rp)/(r(i)*dr^2);
        if i < Nr, Ar(i,i+1) = rp/(r(i)*dr^2); else, br(i) = rp/(r(i)*dr^2); end
    end
    Az = zeros(Nz);  bz = zeros(Nz,1);
    Az(1,1) = -2/dz^2;  Az(1,2) = 2/dz^2;               % mid-plane: ghost node
    for j = 2:Nz
        Az(j,j-1) = 1/dz^2;  Az(j,j) = -2/dz^2;
        if j < Nz, Az(j,j+1) = 1/dz^2; else, bz(j) = 1/dz^2; end
    end
    A = alpha*(kron(sparse(Ar), speye(Nz)) + kron(speye(Nr), sparse(Az)));
    b = alpha*(kron(br, ones(Nz,1)) + kron(ones(Nr,1), bz));
end

function dy = rhs(y, A, b, Tw, Tref, zD)
% RHS  Method-of-lines right-hand side + lethality rate at the cold point (y(1)).
    T  = y(1:end-1);
    dy = [A*T + b*Tw; 10^((T(1) - Tref)/zD)/60];
end

function z = bessel_zeros(M)
% BESSEL_ZEROS  First M positive zeros of J0 (McMahon initial guesses + fzero).
    z = zeros(M,1);
    for m = 1:M
        z(m) = fzero(@(x) besselj(0, x), (m - 0.25)*pi);
    end
end
