"""
Lab4_BatchReactor_DataGeneration.py
Lecture 6 - Parameter estimation, Case history 4.15.1 (guiding example, Sections 5.1-5.3)

Pseudo-experimental data for the isothermal, constant-volume batch reactor charged with pure ethylbenzene:
    EB -> St + H2      r1 = k1*C_EB
    EB -> Bz + C2H4    r2 = k2*C_EB
Analytical solution and measured quantities:
    C_EB(t) = C_EB0*exp(-(k1+k2)*t)
    P(t)    = P0*(2 - exp(-(k1+k2)*t))            (pressure transducer, dense sampling)
    S       = k1/(k1+k2)  (constant in time)      (GC selectivity, a few samples)
True kinetic parameters: Arrhenius form centred at T0 = 900 K (guiding example, Section 1.5):
    k_i(T) = k_i,ref*exp(-Ea_i/R*(1/T - 1/T0))

A small portable random-number generator (Park-Miller + Box-Muller) is used instead of numpy.random,
so that the MATLAB and Python versions of Labs 4-8 produce EXACTLY the same pseudo-data.
"""
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- settings (same in Labs 4-8)
R_GAS = 8.314e-3                       # kJ/(mol K)
T0 = 900.0                             # K, reference temperature
P_TRUE = np.array([0.7705, 90.98, 0.0764, 207.99])   # k1ref [1/s], Ea1 [kJ/mol], k2ref [1/s], Ea2 [kJ/mol]
T_RUNS = np.array([850.0, 900.0, 923.0])             # K
P0 = 100.0                             # kPa, initial pressure (pure EB), all runs
T_P = np.linspace(0.0, 15.0, 300)      # s, pressure sampling (300 points per run)
T_GC = np.array([2.0, 5.0, 9.0, 13.0]) # s, GC sampling (4 per run)
SIG_P = 0.20                           # kPa, transducer noise
SIG_S = 0.01                           # -, GC selectivity noise
SEED = 20261012


def prng_normal(n, state):
    """n standard normal numbers. Park-Miller minimal standard generator + Box-Muller (cosine branch).
    Returns (z, new_state). Identical sequence in MATLAB and Python."""
    z = np.empty(n)
    for i in range(n):
        state = (16807.0 * state) % 2147483647.0
        u1 = state / 2147483647.0
        state = (16807.0 * state) % 2147483647.0
        u2 = state / 2147483647.0
        z[i] = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
    return z, state


def k_arrhenius(p, T):
    """k1, k2 at temperature T from p = (k1ref, Ea1, k2ref, Ea2), centred at T0."""
    tau = 1.0 / T - 1.0 / T0
    return p[0] * np.exp(-p[1] / R_GAS * tau), p[2] * np.exp(-p[3] / R_GAS * tau)


def generate_data():
    """Pseudo-data for the three runs: list of dicts with T, P (300), S (4) and the true values."""
    state = float(SEED)
    runs = []
    for T in T_RUNS:
        k1, k2 = k_arrhenius(P_TRUE, T)
        K = k1 + k2
        P_true = P0 * (2.0 - np.exp(-K * T_P))
        S_true = k1 / K
        e, state = prng_normal(T_P.size, state)
        P_meas = P_true + SIG_P * e
        e, state = prng_normal(T_GC.size, state)
        S_meas = S_true + SIG_S * e
        runs.append(dict(T=T, k1=k1, k2=k2, P=P_meas, S=S_meas, P_true=P_true, S_true=S_true))
    return runs


if __name__ == "__main__":
    runs = generate_data()
    print("Lab 4 - pseudo-experimental batch data (P0 = %.0f kPa, sigma_P = %.2f kPa, sigma_S = %.2f)" % (P0, SIG_P, SIG_S))
    print("   T [K]   k1 [1/s]   k2 [1/s]   tau=1/(k1+k2) [s]   S=k1/(k1+k2)   mean S_meas")
    for r in runs:
        print("  %6.0f   %8.4f   %8.4f   %10.2f          %8.4f       %8.4f"
              % (r["T"], r["k1"], r["k2"], 1 / (r["k1"] + r["k2"]), r["S_true"], r["S"].mean()))

    # C_EB from P (linear inversion): C_EB/C_EB0 = 2 - P/P0; error propagation: sigma = sigma_P/P0 (Section 4.7)
    for r in runs:
        r["CEB"] = 2.0 - r["P"] / P0
    print("  noise on C_EB/C_EB0 (delta method) = sigma_P/P0 = %.4f (constant)" % (SIG_P / P0))

    # save the data actually 'measured' (true values are NOT saved: they are unknown in a real experiment)
    with open("BatchReactor_PseudoData.csv", "w") as f:
        f.write("T_K,type,t_s,value\n")
        for r in runs:
            for t, v in zip(T_P, r["P"]):
                f.write("%.0f,P_kPa,%.6f,%.6f\n" % (r["T"], t, v))
            for t, v in zip(T_GC, r["S"]):
                f.write("%.0f,S,%.6f,%.6f\n" % (r["T"], t, v))
    print("  data written to BatchReactor_PseudoData.csv")

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    for r, c in zip(runs, ["#233F98", "#D35400", "#2E7D32"]):
        axs[0].plot(T_P, r["P"], ".", color=c, ms=2)
        axs[0].plot(T_P, r["P_true"], "-", color="black", lw=0.8)
        axs[1].plot(T_GC, r["S"], "o", color=c, label="T = %.0f K" % r["T"])
        axs[1].axhline(r["S_true"], color=c, lw=0.8, ls="--")
    axs[0].set_xlabel("t [s]"); axs[0].set_ylabel("P [kPa]"); axs[0].set_title("pressure (dots: measured)")
    axs[1].set_xlabel("t [s]"); axs[1].set_ylabel("S = C_St/(C_St+C_Bz)"); axs[1].set_title("GC selectivity")
    axs[1].legend()
    fig.tight_layout(); fig.savefig("Lab4_data.png", dpi=150)
    plt.show()
