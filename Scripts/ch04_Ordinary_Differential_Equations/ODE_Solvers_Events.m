% Lecture 4, Section 2.10.3 - Events and dense output (MATLAB ode45)
%
% Test problem of Section 2.2: find the time t* at which y(t*) = 4.
clear; clc;

f = @(t, y) y - t.^2 + 1;
opts = odeset('RelTol', 1e-8, 'AbsTol', 1e-10, 'Events', @reach4);

sol = ode45(f, [0 2], 0.5, opts);        % structure output
fprintf('event at t = %.8f\n', sol.xe);  % time of the event
y1 = deval(sol, 1.0);                    % dense output: y at any t
fprintf('dense output y(1.0) = %.8f\n', y1);

% ------------------------------------------------------ local functions
function [value, isterminal, direction] = reach4(~, y)
    value = y(1) - 4;     % zero when y = 4
    isterminal = 1;       % stop the integration at the event
    direction = 1;        % upward crossings only
end
