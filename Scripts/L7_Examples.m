% L7_Examples.m - Lesson 7, Examples 7.1-7.3 (reference implementation)
% Requires the Optimization Toolbox (linprog, fmincon); fminsearch is in base MATLAB.
clear; clc

%% Example 7.1 - bakery product mix (linprog minimizes: change the sign of c)
c = [3 5 6];  A = [1 2 4; 2 2 1; 1 3 1];  b = [20; 24; 40];
[x, fval, ~, ~, lam] = linprog(-c, A, b, [], [], zeros(3,1));
fprintf('Ex 7.1  x* = [%g %g %g], z* = %g, shadow prices = [%g %g %g]\n', x, -fval, lam.ineqlin);

%% Example 7.2 - least-cost burger formulation (100 kg batch)
% Two different quantities for each constraint (do not confuse them):
%   slack        = b_ub - A_ub*x : how much of the constraint is left unused
%                  (value of s_i, read in the RHS column of the final tableau)
%   shadow price = Lagrange multiplier lam.ineqlin : how much the optimal cost
%                  decreases if the constraint is relaxed by one unit
%                  (row 0 of the final tableau, under s_i; the same lambda_i of the KKT conditions)
% Complementary slackness: for each constraint, slack x shadow price = 0.
cost  = [7.0 2.5 2.0];                  % beef, pork trim, hydrated soy [EUR/kg]
names = {'fat', 'protein', 'soy'};
for fat_max = [20 15]
    A_ub = [ 0.10  0.35  0.00;         % fat     <= fat_max kg
            -0.20 -0.14 -0.18;         % protein >= 16 kg (sign changed: -protein <= -16)
             0     0     1   ];        % soy     <= 10 kg
    b_ub = [fat_max; -16; 10];
    [x, fval, ~, ~, lam] = linprog(cost, A_ub, b_ub, [1 1 1], 100, zeros(3,1));
    slack = b_ub - A_ub*x;             % unused part of each constraint [kg]
    fprintf('\nEx 7.2  fat <= %d%%:  x* = [%g %g %g] kg,  cost = %g EUR/100 kg\n', fat_max, x, fval);
    fprintf('  %-8s %12s %28s\n', 'constr.', 'slack [kg]', 'shadow price [EUR/kg]');
    for k = 1:3
        fprintf('  %-8s %12.2f %28.2f\n', names{k}, slack(k), lam.ineqlin(k));
    end
end

%% Example 7.3 - thermal process design, x = [T, t]
TrF = 121.1; zF = 10; F0min = 3;       % lethality
TrC = 100;   zC = 33; Cmin  = 35;      % cook value
Trv = 121.1; zv = 25; Dv    = 180;     % thiamine
lb = [105 1];  ub = [135 120];  x0 = [121.1 10];
F0   = @(x) x(2) * 10^((x(1) - TrF) / zF);
C100 = @(x) x(2) * 10^((x(1) - TrC) / zC);
R    = @(x) 10^(-x(2) * 10^((x(1) - Trv) / zv) / Dv);
f    = @(x) -R(x);
nonlcon = @(x) deal([F0min - F0(x); Cmin - C100(x)], []);   % c(x) <= 0

% Default limit of fmincon: MaxFunctionEvaluations = 100*(number of variables) = 200.
% R is very flat near the optimum and T, t have different scales: with the default
% limit 'sqp' may stop before converging (exitflag = 0). Always check exitflag:
%   1 = converged,  0 = stopped by the iteration/evaluation limit,  < 0 = failure.
for alg = {'sqp', 'interior-point'}
    opt = optimoptions('fmincon', 'Algorithm', alg{1}, 'Display', 'off', ...
                       'MaxFunctionEvaluations', 5000, 'MaxIterations', 1000);
    [x, ~, exitflag, out] = fmincon(f, x0, [], [], [], [], lb, ub, nonlcon, opt);
    fprintf('Ex 7.3  fmincon %-15s T* = %.2f  t* = %.2f  R* = %.4f  evals = %d  exitflag = %d\n', ...
            alg{1}, x, R(x), out.funcCount, exitflag);
end

% Nelder-Mead on the penalty function (7.8), scaled violations
pen = @(x, rho) f(x) + rho * ( max(0, (F0min - F0(x)) / F0min)^2 + ...
                               max(0, (Cmin - C100(x)) / Cmin)^2 + ...
                               sum(max(0, lb - x).^2) + sum(max(0, x - ub).^2) );
opt = optimset('TolX', 1e-6, 'TolFun', 1e-10, 'MaxFunEvals', 5000);
for rho = [1e1 1e3 1e5]
    [x, ~, ~, out] = fminsearch(@(x) pen(x, rho), x0, opt);
    fprintf('Ex 7.3  NM rho = %.0e: T* = %.2f  t* = %.2f  R* = %.4f  F0 = %.3f  evals = %d\n', ...
            rho, x, R(x), F0(x), out.funcCount);
end

% Hidden LP in (T, u = ln t):  min u + ln10/zv (T - Trv)
L = log(10);
cLP  = [L/zv, 1];
A_LP = [-L/zF, -1; -L/zC, -1];
b_LP = [-(log(F0min) + TrF*L/zF); -(log(Cmin) + TrC*L/zC)];
y = linprog(cLP, A_LP, b_LP, [], [], [lb(1) log(lb(2))], [ub(1) log(ub(2))]);
fprintf('Ex 7.3  hidden LP:      T* = %.2f  t* = %.2f  R* = %.4f\n', y(1), exp(y(2)), R([y(1) exp(y(2))]));
