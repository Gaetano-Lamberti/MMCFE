% Lecture 4, Section 2.10.2 - Stiff solvers and analytical Jacobian
%
% Robertson kinetics (Section 2.7.5), t in [0, 40], y(0) = (1, 0, 0).
clear; clc;

opts = odeset('RelTol', 1e-6, 'AbsTol', 1e-10, ...
              'Jacobian', @robJac);      % analytical Jacobian df/dy

solvers = {@ode45, @ode15s, @ode23s, @ode23tb};
for k = 1:numel(solvers)
    sol = solvers{k}(@rob, [0 40], [1; 0; 0], opts); % ode45: no Jacobian
    fprintf('%-8s steps = %6d  f-evals = %7d  y3(40) = %.6f\n', ...
            func2str(solvers{k}), sol.stats.nsteps, sol.stats.nfevals, ...
            sol.y(3, end));
end

% ------------------------------------------------------ local functions
function dydt = rob(~, y)                % Robertson kinetics
    dydt = [-0.04*y(1) + 1e4*y(2)*y(3);
             0.04*y(1) - 1e4*y(2)*y(3) - 3e7*y(2)^2;
             3e7*y(2)^2];
end

function J = robJac(~, y)                % analytical Jacobian df/dy
    J = [-0.04,  1e4*y(3),             1e4*y(2);
          0.04, -1e4*y(3) - 6e7*y(2), -1e4*y(2);
          0,     6e7*y(2),             0];
end
