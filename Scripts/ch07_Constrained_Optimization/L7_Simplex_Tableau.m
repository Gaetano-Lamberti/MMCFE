% L7_Simplex_Tableau.m - Lesson 7, tableau Simplex method
% (maximization, <= constraints, b >= 0). Prints every tableau.
%   max z = c'x   s.t.  A x <= b,  x >= 0
% Row 0 = objective row (z - c'x = 0), last column = RHS (as in the slides).
% Values are shown as fractions with format rat.
clear; clc; format rat

% Problem data
c_P = [3 4];    A_P = [1 1; 1 2; 2 1];       b_P = [8; 12; 14];      % pasta plant
c_B = [3 5 6];  A_B = [1 2 4; 2 2 1; 1 3 1]; b_B = [20; 24; 40];     % bakery

fprintf('=== Pasta plant (Section 7.4) ===\n');
[x_P, z_P] = simplex_tableau(c_P, A_P, b_P, {'x1','x2','s1','s2','s3'});

fprintf('\n=== Example 7.1 - Bakery product mix ===\n');
[x_B, z_B] = simplex_tableau(c_B, A_B, b_B, {'x1','x2','x3','s1','s2','s3'});

format short

% ------------------------------------------------------------ check with the built-in solver
% linprog (Optimization Toolbox) MINIMIZES f'x subject to A x <= b, Aeq x = beq, lb <= x <= ub:
%     [x, fval, exitflag, output, lambda] = linprog(f, A, b, Aeq, beq, lb, ub)
% For a maximization pass f = -c and change the sign of fval.
% [] = argument not used (here: no equality constraints, no upper bounds).
% lambda.ineqlin = Lagrange multipliers of A x <= b = shadow prices
%                  (row 0 of the final tableau under the slack variables).
fprintf('\n=== Check with linprog ===\n');
check_linprog('Pasta plant', c_P, A_P, b_P, x_P, z_P);
check_linprog('Example 7.1', c_B, A_B, b_B, x_B, z_B);

% ------------------------------------------------------------ local functions
function [x, zopt] = simplex_tableau(c, A, b, names)
    [m, n] = size(A);
    T = [A, eye(m), b(:)];                  % constraint rows [A | I | b]
    z = [-c(:)', zeros(1, m), 0];           % row 0: [-c | 0 | 0]
    basis = n + (1:m);                      % slack variables start in the basis
    k = 0;
    print_tableau(T, z, basis, names, k);
    while true
        [zmin, col] = min(z(1:n+m));        % entering: most negative row-0 entry
        if zmin >= 0
            fprintf('All row-0 coefficients >= 0: optimum reached\n'); break
        end
        pos = find(T(:, col) > 0);
        if isempty(pos)
            fprintf('No positive entry in the pivot column: LP is unbounded\n');
            x = []; zopt = Inf; return
        end
        [~, j] = min(T(pos, end) ./ T(pos, col));   % leaving: minimum ratio test
        row = pos(j);
        fprintf('Entering %s, leaving %s, pivot = %s\n', names{col}, ...
                names{basis(row)}, strtrim(rats(T(row, col))));
        T(row, :) = T(row, :) / T(row, col);        % Gauss-Jordan pivot
        for i = [1:row-1, row+1:m]
            T(i, :) = T(i, :) - T(i, col) * T(row, :);
        end
        z = z - z(col) * T(row, :);
        basis(row) = col;
        k = k + 1;
        print_tableau(T, z, basis, names, k);
    end
    x = zeros(1, n + m); x(basis) = T(:, end);
    zopt = z(end);
    for j = 1:n+m, fprintf('%s = %s  ', names{j}, strtrim(rats(x(j)))); end
    fprintf('\nz* = %s\nShadow prices (row 0 under the slacks): %s\n', ...
            strtrim(rats(zopt)), rats(z(n+1:n+m)));
end

function print_tableau(T, z, basis, names, k)
    fprintf('\nTableau %d\n%-6s', k, 'Basis');
    fprintf('%10s', names{:}, 'RHS'); fprintf('\n%-6s', 'z');
    s = arrayfun(@(v) strtrim(rats(v)), z, 'UniformOutput', false);
    fprintf('%10s', s{:});
    fprintf('\n');
    for i = 1:size(T, 1)
        fprintf('%-6s', names{basis(i)});
        s = arrayfun(@(v) strtrim(rats(v)), T(i, :), 'UniformOutput', false);
        fprintf('%10s', s{:}); fprintf('\n');
    end
end

function check_linprog(name, c, A, b, x_tab, z_tab)
    n = numel(c);
    lb = zeros(n, 1);                                   % non-negativity x >= 0
    [x, fval, exitflag, ~, lambda] = linprog(-c(:), A, b, [], [], lb, []);
    if exitflag ~= 1
        fprintf('%s: linprog did not converge (exitflag = %d)\n', name, exitflag); return
    end
    z = -fval;                                          % back to the maximum
    s = b - A*x;                                        % slack variables
    fprintf('\n%s\n', name);
    fprintf('  x*            linprog: %s   tableau: %s\n', mat2str(x', 4), mat2str(x_tab(1:n), 4));
    fprintf('  z*            linprog: %g   tableau: %g\n', z, z_tab);
    fprintf('  slacks        linprog: %s\n', mat2str(s', 4));
    fprintf('  shadow prices linprog: %s\n', mat2str(lambda.ineqlin', 4));
    fprintf('  max |difference| on x*: %.1e\n', max(abs(x' - x_tab(1:n))));
end
