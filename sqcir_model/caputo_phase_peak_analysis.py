"""
Caputo fractional-order SQCIR phase-plane and peak-shift analysis.

This script examines how the fractional order alpha affects:
1. The S-I phase portrait.
2. The infection curve I(t).
3. The peak infection time and peak infection magnitude.

Output:
    fig_caputo_analysis.pdf
"""

import numpy as np
import matplotlib.pyplot as plt
from math import gamma as gamma_function
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------

plt.rcParams.update({
    "figure.dpi": 200,
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "axes.linewidth": 1.1,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ---------------------------------------------------------------------
# Model settings
# ---------------------------------------------------------------------

ALPHA_COLORS = {
    1.00: "#1B4F72",
    0.90: "#2E86C1",
    0.80: "#27AE60",
    0.70: "#E74C3C",
    0.60: "#8E44AD",
}

PARAMS = {
    "Pi": 0.5,
    "beta1": 0.14,
    "beta2": 0.05,
    "sigma": 0.5,
    "xi": 0.04,
    "rho": 0.09,
    "gamma": 0.8,
    "mu": 0.2,
    "omega": 0.01,
}

Q0 = 1e-4
C0 = 1e-4
I0 = 1e-6

INITIAL_CONDITIONS = np.array([
    1 - (Q0 + C0 + I0),
    Q0,
    C0,
    I0,
    0.0,
])

ALPHAS = [1.00, 0.90, 0.80, 0.70, 0.60]
T_END = 100
N_STEPS = 500


# ---------------------------------------------------------------------
# SQCIR model
# ---------------------------------------------------------------------

def sqcir_rhs(y, params):
    """Right-hand side of the SQCIR model."""
    S, Q, C, I, R = y

    dSdt = (
        params["Pi"]
        - params["beta1"] * S * Q
        - params["beta2"] * S * I
        - params["sigma"] * S * C
        - params["xi"] * S
        - params["rho"] * S
    )

    dQdt = (
        params["beta1"] * S * Q
        + params["beta2"] * S * I
        - params["sigma"] * Q * C
        - params["rho"] * Q
    )

    dCdt = (
        params["sigma"] * S * C
        + params["sigma"] * Q * C
        - params["gamma"] * C * I
        - params["mu"] * C
        - params["rho"] * C
    )

    dIdt = (
        params["gamma"] * C * I
        - params["omega"] * I
        - params["rho"] * I
    )

    dRdt = (
        params["mu"] * C
        + params["omega"] * I
        + params["xi"] * S
        - params["rho"] * R
    )

    return np.array([dSdt, dQdt, dCdt, dIdt, dRdt])


# ---------------------------------------------------------------------
# Classical and fractional solvers
# ---------------------------------------------------------------------

def solve_classical_sqcir(total_time, n_steps, y0, params):
    """Solve the classical SQCIR model for alpha = 1."""
    time_grid = np.linspace(0, total_time, n_steps + 1)

    solution = solve_ivp(
        lambda t, y: sqcir_rhs(y, params),
        (0, total_time),
        y0,
        t_eval=time_grid,
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
    )

    return time_grid, solution.y


def solve_caputo_l1_sqcir(total_time, n_steps, alpha, y0, params):
    """
    Solve the Caputo fractional-order SQCIR model using an L1-type scheme.
    """
    step_size = total_time / n_steps
    time_grid = np.linspace(0, total_time, n_steps + 1)

    solution = np.zeros((5, n_steps + 1))
    solution[:, 0] = y0

    weights = np.array([
        (j + 1) ** (1 - alpha) - j ** (1 - alpha)
        for j in range(n_steps)
    ])

    coefficient = step_size ** alpha * gamma_function(2 - alpha)

    for n in range(1, n_steps + 1):
        history_sum = sum(
            weights[j] * (solution[:, n - j] - solution[:, n - j - 1])
            for j in range(1, n)
        )

        predictor = np.clip(
            solution[:, n - 1]
            + coefficient * sqcir_rhs(solution[:, n - 1], params)
            - history_sum,
            0,
            None,
        )

        solution[:, n] = np.clip(
            solution[:, n - 1]
            + 0.5
            * coefficient
            * (
                sqcir_rhs(solution[:, n - 1], params)
                + sqcir_rhs(predictor, params)
            )
            - history_sum,
            0,
            None,
        )

    return time_grid, solution


def run_alpha_sweep():
    """Run classical and fractional simulations for all alpha values."""
    results = {}

    for alpha in ALPHAS:
        if alpha == 1.0:
            results[alpha] = solve_classical_sqcir(
                T_END,
                N_STEPS,
                INITIAL_CONDITIONS,
                PARAMS,
            )
        else:
            results[alpha] = solve_caputo_l1_sqcir(
                T_END,
                N_STEPS,
                alpha,
                INITIAL_CONDITIONS,
                PARAMS,
            )

    return results


# ---------------------------------------------------------------------
# Plotting and peak analysis
# ---------------------------------------------------------------------

def plot_phase_peak_analysis(output_path="fig_caputo_analysis.pdf"):
    """Generate phase-plane and peak-shift analysis figure."""
    results = run_alpha_sweep()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    fig.suptitle(
        r"Caputo Fractional SQCIR — Effect of Memory Order $\alpha$",
        fontsize=13,
        fontweight="bold",
    )

    # -------------------------------------------------------------
    # (a) S-I phase portrait
    # -------------------------------------------------------------
    ax = axes[0]

    for alpha in ALPHAS:
        time_grid, solution = results[alpha]

        line_style = "-" if alpha == 1.0 else "--"
        line_width = 2.2 if alpha == 1.0 else 1.6

        ax.plot(
            solution[0],
            solution[3],
            color=ALPHA_COLORS[alpha],
            ls=line_style,
            lw=line_width,
            label=rf"$\alpha={alpha}$",
        )

    ax.set_xlabel("Susceptible $S(t)$")
    ax.set_ylabel("Infected $I(t)$")
    ax.set_title("(a) Phase portrait $S$–$I$")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.2, linestyle="--")

    # -------------------------------------------------------------
    # (b) Infection curves and peak markers
    # -------------------------------------------------------------
    ax = axes[1]
    peak_data = {}

    for alpha in ALPHAS:
        time_grid, solution = results[alpha]

        infected_curve = solution[3]

        line_style = "-" if alpha == 1.0 else "--"
        line_width = 2.2 if alpha == 1.0 else 1.6

        ax.plot(
            time_grid,
            infected_curve,
            color=ALPHA_COLORS[alpha],
            ls=line_style,
            lw=line_width,
            label=rf"$\alpha={alpha}$",
        )

        peak_index = np.argmax(infected_curve)

        ax.scatter(
            time_grid[peak_index],
            infected_curve[peak_index],
            color=ALPHA_COLORS[alpha],
            s=55,
            marker="^",
            zorder=5,
            edgecolors="white",
            linewidths=0.7,
        )

        peak_data[alpha] = (
            time_grid[peak_index],
            infected_curve[peak_index],
        )

    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Infected fraction $I(t)$")
    ax.set_title("(b) Infection dynamics and peak shift")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.2, linestyle="--")

    # -------------------------------------------------------------
    # (c) Peak time and peak magnitude bar chart
    # -------------------------------------------------------------
    ax = axes[2]

    alpha_values = list(peak_data.keys())
    peak_times = [peak_data[alpha][0] for alpha in alpha_values]
    peak_values = [peak_data[alpha][1] for alpha in alpha_values]

    x_positions = np.arange(len(alpha_values))
    bar_width = 0.38

    bars_peak_time = ax.bar(
        x_positions - bar_width / 2,
        peak_times,
        width=bar_width,
        color="#2166AC",
        alpha=0.85,
        label="Peak time (days)",
        zorder=3,
    )

    ax.set_ylabel("Peak time (days)", color="#2166AC")
    ax.tick_params(axis="y", labelcolor="#2166AC")

    ax_secondary = ax.twinx()

    bars_peak_value = ax_secondary.bar(
        x_positions + bar_width / 2,
        peak_values,
        width=bar_width,
        color="#8B0000",
        alpha=0.85,
        label="Peak $I$ value",
        zorder=3,
    )

    ax_secondary.set_ylabel("Peak infected fraction", color="#8B0000")
    ax_secondary.tick_params(axis="y", labelcolor="#8B0000")

    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [rf"$\alpha={alpha}$" for alpha in alpha_values],
        fontsize=8,
    )

    ax.set_title(r"(c) Peak metrics vs $\alpha$")
    ax.grid(axis="y", alpha=0.2, linestyle="--", zorder=0)

    bars_all = [bars_peak_time, bars_peak_value]
    labels_all = [bar.get_label() for bar in bars_all]

    ax.legend(
        bars_all,
        labels_all,
        loc="upper left",
        fontsize=8,
        framealpha=0.9,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(output_path, bbox_inches="tight")
    plt.show()

    print("\n   Alpha     Peak time        Peak I")
    print("--------------------------------------")

    for alpha, (peak_time, peak_value) in peak_data.items():
        print(f"   {alpha:>5.2f}      {peak_time:>8.2f}      {peak_value:>10.5f}")


if __name__ == "__main__":
    plot_phase_peak_analysis()
