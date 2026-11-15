%% Lesson 5, Section 3.11.3 - The canned food of Section 3.12 solved with solvepde
% Same problem as PDE_Example_CanSterilization.m (quarter of the axial section,
% 0 <= r <= R, 0 <= z <= Hh, z from the MID-PLANE), now by FEM with the PDE Toolbox.
%
%   dT/dt = alpha * [ (1/r) d/dr( r dT/dr ) + d2T/dz2 ]
%
% The PDE Toolbox works in CARTESIAN coordinates (x, y). An axisymmetric problem
% is brought to its coefficient form by multiplying the PDE by r:
%
%   r dT/dt = d/dr( alpha r dT/dr ) + d/dz( alpha r dT/dz ) = div( alpha r grad T )
%
% i.e. Eq. (58) with x = r, y = z and   m = 0, d = r, c = alpha*r, a = 0, f = 0.
%
% BCs:  wall (r = R) and lid (z = Hh): T = Tw(t), Dirichlet
%       axis (r = 0) and mid-plane (z = 0): symmetry, zero flux = NO condition
%       (an edge without a condition gets the zero generalized Neumann BC)
%
% Requires the Partial Differential Equation Toolbox.
% Expected (exact series, 3.12): centre 116.6 C at the end of heating,
% maximum 117.4 C at about 85.3 min, F0 = 8.89 min.

clear; clc; close all;

%% --- Parameters (as in PDE_Example_CanSterilization.m) ---
R     = 0.038;   Hh = 0.050;      % can radius and half height [m]
alpha = 1.5e-7;                   % thermal diffusivity [m^2/s]
T0 = 40;  TR = 121.1;  TC = 20;   % initial, retort and cooling-water temperatures [C]
t_h   = 80*60;                    % end of heating [s]
t_end = 110*60;                   % end of cooling [s]
zD = 10;  Tref = 121.1;           % z-value and reference temperature [C]

%% --- 1) Model and geometry ---
model = createpde();                              % one scalar PDE
gd = [3; 4; 0; R; R; 0; 0; 0; Hh; Hh];            % rectangle: 4 corners, x = r, y = z
geometryFromEdges(model, decsg(gd));
figure; pdegplot(model, 'EdgeLabels', 'on'); axis equal
title('Check: 1 mid-plane, 2 wall, 3 lid, 4 axis');

%% --- 2) Coefficients of Eq. (58): the PDE multiplied by r ---
specifyCoefficients(model, 'm', 0, ...
                           'd', @(location, state) location.x, ...          % r
                           'c', @(location, state) alpha*location.x, ...    % alpha*r
                           'a', 0, 'f', 0);

%% --- 3) Boundary and initial conditions ---
applyBoundaryCondition(model, 'dirichlet', 'Edge', [2 3], 'u', TR);    % wall and lid
% edges 1 (mid-plane) and 4 (axis): nothing -> zero flux (symmetry)
setInitialConditions(model, T0);

%% --- 4) Mesh ---
generateMesh(model, 'Hmax', 0.002);               % 2 mm triangles, quadratic (default)
figure; pdemesh(model); axis equal
fprintf('Mesh: %d nodes, %d triangles\n', size(model.Mesh.Nodes, 2), size(model.Mesh.Elements, 2));

%% --- 5) Solve: heating, then cooling ---
model.SolverOptions.RelativeTolerance = 1e-6;
model.SolverOptions.AbsoluteTolerance = 1e-6;
tlist1 = linspace(0, t_h, 481);
res1 = solvepde(model, tlist1);                   % heating, Tw = TR

applyBoundaryCondition(model, 'dirichlet', 'Edge', [2 3], 'u', TC);    % the last assignment wins
setInitialConditions(model, res1, numel(tlist1)); % start from the last heating state
tlist2 = linspace(t_h, t_end, 181);
res2 = solvepde(model, tlist2);                   % cooling, Tw = TC

%% --- 6) Cold point and lethality ---
Tc1 = interpolateSolution(res1, 0, 0, 1:numel(tlist1));   % r = 0, z = 0, all times
Tc2 = interpolateSolution(res2, 0, 0, 1:numel(tlist2));
t  = [tlist1(:); tlist2(2:end).'];
Tc = [Tc1(:); reshape(Tc2(2:end), [], 1)];
F0 = cumtrapz(t, 10.^((Tc - Tref)/zD))/60;

[Tmax, imax] = max(Tc);
fprintf('Centre at end of heating: %.2f C   (exact 116.56 C)\n', Tc(numel(tlist1)));
fprintf('Centre maximum: %.2f C at t = %.1f min   (exact 117.42 C at 85.3 min)\n', Tmax, t(imax)/60);
fprintf('F0 = %.3f min   (exact 8.89 min)\n', F0(end));

%% --- Plots ---
figure;
subplot(1,2,1);
pdeplot(model, 'XYData', res1.NodalSolution(:, end), 'Contour', 'on'); axis equal
xlabel('r (m)'); ylabel('z (m)'); title('T at the end of heating (C)');
subplot(1,2,2);
yyaxis left;  plot(t/60, Tc, 'LineWidth', 1.5); ylabel('T at the cold point (C)');
yyaxis right; plot(t/60, F0, 'LineWidth', 1.5); ylabel('F_0 (min)');
xlabel('t (min)'); grid on; title('solvepde (FEM)');
