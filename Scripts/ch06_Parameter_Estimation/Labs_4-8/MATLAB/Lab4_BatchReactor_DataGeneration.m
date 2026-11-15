% Lab4_BatchReactor_DataGeneration.m
% Lecture 6 - Parameter estimation, Case history 4.15.1 (guiding example, Sections 5.1-5.3)
%
% Pseudo-experimental data for the isothermal, constant-volume batch reactor charged with pure ethylbenzene:
%     EB -> St + H2      r1 = k1*C_EB
%     EB -> Bz + C2H4    r2 = k2*C_EB
% Analytical solution and measured quantities:
%     C_EB(t) = C_EB0*exp(-(k1+k2)*t)
%     P(t)    = P0*(2 - exp(-(k1+k2)*t))            (pressure transducer, dense sampling)
%     S       = k1/(k1+k2)  (constant in time)      (GC selectivity, a few samples)
% True kinetic parameters: Arrhenius form centred at T0 = 900 K (guiding example, Section 1.5):
%     k_i(T) = k_i,ref*exp(-Ea_i/R*(1/T - 1/T0))
%
% A small portable random-number generator (Park-Miller + Box-Muller) is used instead of randn,
% so that the MATLAB and Python versions of Labs 4-8 produce EXACTLY the same pseudo-data.
% No toolbox required.

clear; close all; clc

c = lab_settings();
runs = generate_data(c);

fprintf('Lab 4 - pseudo-experimental batch data (P0 = %.0f kPa, sigma_P = %.2f kPa, sigma_S = %.2f)\n', ...
        c.P0, c.sigP, c.sigS);
fprintf('   T [K]   k1 [1/s]   k2 [1/s]   tau=1/(k1+k2) [s]   S=k1/(k1+k2)   mean S_meas\n');
for i = 1:numel(runs)
    r = runs(i);
    fprintf('  %6.0f   %8.4f   %8.4f   %10.2f          %8.4f       %8.4f\n', ...
            r.T, r.k1, r.k2, 1 / (r.k1 + r.k2), r.Strue, mean(r.S));
end

% C_EB from P (linear inversion): C_EB/C_EB0 = 2 - P/P0; error propagation: sigma = sigma_P/P0 (Section 4.7)
fprintf('  noise on C_EB/C_EB0 (delta method) = sigma_P/P0 = %.4f (constant)\n', c.sigP / c.P0);

%% save the data actually 'measured' (true values are NOT saved: they are unknown in a real experiment)
fid = fopen('BatchReactor_PseudoData.csv', 'w');
fprintf(fid, 'T_K,type,t_s,value\n');
for i = 1:numel(runs)
    for j = 1:numel(c.tP)
        fprintf(fid, '%.0f,P_kPa,%.6f,%.6f\n', runs(i).T, c.tP(j), runs(i).P(j));
    end
    for j = 1:numel(c.tGC)
        fprintf(fid, '%.0f,S,%.6f,%.6f\n', runs(i).T, c.tGC(j), runs(i).S(j));
    end
end
fclose(fid);
fprintf('  data written to BatchReactor_PseudoData.csv\n');

%% figure
col = [35 63 152; 211 84 0; 46 125 50] / 255;
fig = figure('Position', [100 100 1000 380]);
subplot(1, 2, 1); hold on; box on
subplot(1, 2, 2); hold on; box on
for i = 1:numel(runs)
    subplot(1, 2, 1)
    plot(c.tP, runs(i).P, '.', 'Color', col(i, :), 'MarkerSize', 4)
    plot(c.tP, runs(i).Ptrue, '-', 'Color', 'k', 'LineWidth', 0.8)
    subplot(1, 2, 2)
    h(i) = plot(c.tGC, runs(i).S, 'o', 'Color', col(i, :)); %#ok<SAGROW>
    plot([0 15], runs(i).Strue * [1 1], '--', 'Color', col(i, :), 'LineWidth', 0.8)
end
subplot(1, 2, 1); xlim([0 15]); xlabel('t [s]'); ylabel('P [kPa]'); title('pressure (dots: measured)')
subplot(1, 2, 2); xlim([0 15]); xlabel('t [s]'); ylabel('S = C_{St}/(C_{St}+C_{Bz})'); title('GC selectivity')
legend(h, arrayfun(@(T) sprintf('T = %.0f K', T), c.Truns, 'UniformOutput', false), 'Location', 'best')
print(fig, 'Lab4_data_matlab.png', '-dpng', '-r150');

%% ======================================================================= local functions
% (MATLAB R2016b or later: local functions in a script must be at the end of the file)

function c = lab_settings()
% Settings shared by Labs 4-8 (guiding example, Sections 1.5 and 5.1-5.3).
c.R     = 8.314e-3;                    % kJ/(mol K)
c.T0    = 900;                         % K, reference temperature of the centred Arrhenius law
c.pTrue = [0.7705; 90.98; 0.0764; 207.99];   % k1ref [1/s], Ea1 [kJ/mol], k2ref [1/s], Ea2 [kJ/mol]
c.Truns = [850; 900; 923];             % K, the three isothermal runs
c.P0    = 100;                         % kPa, initial pressure (pure EB), all runs
c.tP    = linspace(0, 15, 300)';       % s, pressure sampling (300 points per run)
c.tGC   = [2; 5; 9; 13];               % s, GC sampling (4 per run)
c.sigP  = 0.20;                        % kPa, transducer noise (standard deviation)
c.sigS  = 0.01;                        % -,   GC selectivity noise (standard deviation)
c.seed  = 20261012;                    % seed of the portable generator
end

function [z, state] = prng_normal(n, state)
% n standard normal numbers: Park-Miller "minimal standard" generator
% (x <- 16807 x mod 2^31-1) + Box-Muller transform (cosine branch).
% All products are exact in double precision, so MATLAB and Python give the SAME sequence.
z = zeros(n, 1);
for i = 1:n
    state = mod(16807 * state, 2147483647);  u1 = state / 2147483647;
    state = mod(16807 * state, 2147483647);  u2 = state / 2147483647;
    z(i) = sqrt(-2 * log(u1)) * cos(2 * pi * u2);
end
end

function [k1, k2] = k_arrhenius(p, T, c)
% k1, k2 at temperature T from p = (k1ref, Ea1, k2ref, Ea2), Arrhenius law centred at T0:
%   k_i(T) = k_i,ref * exp(-Ea_i/R * (1/T - 1/T0))
tau = 1 ./ T - 1 / c.T0;
k1 = p(1) * exp(-p(2) / c.R * tau);
k2 = p(3) * exp(-p(4) / c.R * tau);
end

function runs = generate_data(c)
% Pseudo-data of the three runs: struct array with T, P (300x1), S (4x1) and the true values.
% Noise order: for each T, first the 300 P values, then the 4 S values (generator state carried over).
state = c.seed;
for i = 1:numel(c.Truns)
    T = c.Truns(i);
    [k1, k2] = k_arrhenius(c.pTrue, T, c);
    K = k1 + k2;
    Ptrue = c.P0 * (2 - exp(-K * c.tP));      % P(t) = P0 (2 - exp(-(k1+k2) t))
    Strue = k1 / K;                           % S = k1/(k1+k2)
    [e, state] = prng_normal(numel(c.tP), state);
    Pmeas = Ptrue + c.sigP * e;
    [e, state] = prng_normal(numel(c.tGC), state);
    Smeas = Strue + c.sigS * e;
    runs(i) = struct('T', T, 'k1', k1, 'k2', k2, 'P', Pmeas, 'S', Smeas, ...
                     'Ptrue', Ptrue, 'Strue', Strue); %#ok<AGROW>
end
end
