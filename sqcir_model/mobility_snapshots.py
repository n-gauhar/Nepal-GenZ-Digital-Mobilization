"""
SQCIR mobility snapshot visualization.

This script:
1. Simulates the SQCIR model over time.
2. Normalizes the compartment populations.
3. Generates agent-based visual snapshots at selected time points.
4. Plots each snapshot beside the corresponding compartment dynamics.

Output:
    fig_mobility_snapshots.pdf
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------

plt.rcParams.update({
    "figure.dpi": 200,
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.linewidth": 0.9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
})


# ---------------------------------------------------------------------
# Colors and line styles
# ---------------------------------------------------------------------

COLORS = {
    "S": "#2166AC",
    "Q": "#F4A582",
    "C": "#4DAC26",
    "I": "#8B0000",
    "R": "#808080",
}

LINE_STYLES = {
    "S": "-",
    "Q": "--",
    "C": "-.",
    "I": ":",
    "R": "-",
}


# ---------------------------------------------------------------------
# SQCIR model
# ---------------------------------------------------------------------

def sqcir_rhs(t, y, params):
    """Right-hand side of the SQCIR model."""
    S, Q, C, I, R = y

    Pi, beta1, beta2, sigma, xi, rho, gamma, mu, omega = params

    dSdt = Pi - beta1 * S * Q - beta2 * S * I - sigma * S * C - xi * S - rho * S
    dQdt = beta1 * S * Q + beta2 * S * I - sigma * Q * C - rho * Q
    dCdt = sigma * S * C + sigma * Q * C - gamma * C * I - mu * C - rho * C
    dIdt = gamma * C * I - omega * I - rho * I
    dRdt = mu * C + omega * I + xi * S - rho * R

    return [dSdt, dQdt, dCdt, dIdt, dRdt]


# ---------------------------------------------------------------------
# Simulation settings
# ---------------------------------------------------------------------

PARAMS = (
    0.5,   # Pi
    0.14,  # beta1
    0.05,  # beta2
    0.5,   # sigma
    0.04,  # xi
    0.09,  # rho
    0.8,   # gamma
    0.2,   # mu
    0.01,  # omega
)

INITIAL_CONDITIONS = [
    1 - 3e-4 - 1e-6,  # S0
    1e-4,             # Q0
    1e-4,             # C0
    1e-6,             # I0
    0.0,              # R0
]

SNAPSHOT_TIMES = [0, 24, 48, 99]
TIME_EVAL = np.linspace(0, 99, 600)


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def simulate_model():
    """Solve the SQCIR system and return normalized compartment trajectories."""
    solution = solve_ivp(
        lambda t, y: sqcir_rhs(t, y, PARAMS),
        (TIME_EVAL[0], TIME_EVAL[-1]),
        INITIAL_CONDITIONS,
        t_eval=TIME_EVAL,
        method="RK45",
        rtol=1e-7,
        atol=1e-9,
    )

    Y = solution.y

    total_population = np.sum(Y, axis=0)
    total_population[total_population < 1e-12] = 1.0

    Y_normalized = Y / total_population

    return Y_normalized


def state_at_time(Y_normalized, time_mark):
    """Return compartment values closest to a selected time."""
    index = np.argmin(np.abs(TIME_EVAL - time_mark))
    return Y_normalized[:, index]


def generate_snapshot_agents(state_vector, n_agents=900, seed=0):
    """
    Generate random agent positions and assign compartment labels.

    The number of agents assigned to each compartment follows the normalized
    compartment proportions at the selected time point.
    """
    rng = np.random.default_rng(seed)

    probabilities = np.clip(state_vector, 0, None)
    probabilities = probabilities / probabilities.sum()

    counts = rng.multinomial(n_agents, probabilities)

    labels = np.array(
        ["S"] * counts[0]
        + ["Q"] * counts[1]
        + ["C"] * counts[2]
        + ["I"] * counts[3]
        + ["R"] * counts[4]
    )

    x_positions = rng.random(n_agents)
    y_positions = rng.random(n_agents)

    return x_positions, y_positions, labels


def plot_snapshot(ax, x_positions, y_positions, labels, time_mark):
    """Plot a spatial snapshot of agents colored by compartment."""
    for label in ["S", "Q", "C", "I", "R"]:
        mask = labels == label

        if mask.any():
            ax.scatter(
                x_positions[mask],
                y_positions[mask],
                s=7,
                c=COLORS[label],
                alpha=0.85,
                edgecolors="none",
                rasterized=True,
            )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_aspect("equal", "box")
    ax.set_title(f"$t = {time_mark}$", fontsize=9.5, pad=4)

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_edgecolor("0.35")


def plot_dynamics(ax, Y_normalized, time_mark):
    """Plot normalized compartment dynamics with snapshot time marker."""
    for index, label in enumerate(["S", "Q", "C", "I", "R"]):
        ax.plot(
            TIME_EVAL,
            Y_normalized[index],
            lw=1.5,
            color=COLORS[label],
            ls=LINE_STYLES[label],
            label=label,
        )

    ax.axvline(time_mark, ls="--", lw=1.1, color="black", alpha=0.55)

    ax.set_xlabel("Time (days)", fontsize=8)
    ax.set_ylabel("Fraction", fontsize=8)

    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(TIME_EVAL[0], TIME_EVAL[-1])

    ax.grid(True, alpha=0.20, linestyle="--")

    ax.legend(
        loc="upper right",
        fontsize=6.5,
        ncol=2,
        handlelength=1.5,
        framealpha=0.85,
    )


# ---------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------

def plot_mobility_snapshots(output_path="fig_mobility_snapshots.pdf"):
    """Generate the full mobility snapshot figure."""
    Y_normalized = simulate_model()

    snapshot_width = 3.6
    dynamics_width = 2.4
    gap_width = 0.5

    fig = plt.figure(
        figsize=(
            2 * snapshot_width + 2 * dynamics_width + gap_width + 0.3,
            2 * snapshot_width + 1.0,
        )
    )

    grid = gridspec.GridSpec(
        2,
        5,
        width_ratios=[
            snapshot_width,
            dynamics_width,
            gap_width,
            snapshot_width,
            dynamics_width,
        ],
        height_ratios=[
            snapshot_width,
            snapshot_width,
        ],
        left=0.03,
        right=0.99,
        top=0.92,
        bottom=0.09,
        hspace=0.35,
        wspace=0.18,
    )

    snapshot_columns = [0, 3]
    dynamics_columns = [1, 4]
    panel_labels = list("abcd")

    for index, (time_mark, panel_label) in enumerate(
        zip(SNAPSHOT_TIMES, panel_labels)
    ):
        row = index // 2
        column_pair = index % 2

        ax_snapshot = fig.add_subplot(grid[row, snapshot_columns[column_pair]])
        ax_dynamics = fig.add_subplot(grid[row, dynamics_columns[column_pair]])

        state_vector = state_at_time(Y_normalized, time_mark)

        x_positions, y_positions, labels = generate_snapshot_agents(
            state_vector,
            seed=123 + int(time_mark),
        )

        plot_snapshot(
            ax_snapshot,
            x_positions,
            y_positions,
            labels,
            time_mark,
        )

        plot_dynamics(ax_dynamics, Y_normalized, time_mark)

        ax_snapshot.text(
            0.0,
            -0.09,
            f"({panel_label})",
            transform=ax_snapshot.transAxes,
            fontsize=11,
            fontweight="bold",
            va="top",
        )

    legend_patches = [
        mpatches.Patch(color=COLORS["S"], label="S - Susceptible"),
        mpatches.Patch(color=COLORS["Q"], label="Q - Quarantined"),
        mpatches.Patch(color=COLORS["C"], label="C - Contacted"),
        mpatches.Patch(color=COLORS["I"], label="I - Infected"),
        mpatches.Patch(color=COLORS["R"], label="R - Recovered"),
    ]

    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.5, 0.005),
        fontsize=9,
        framealpha=0.9,
        edgecolor="0.6",
    )

    plt.savefig(output_path, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    plot_mobility_snapshots()
