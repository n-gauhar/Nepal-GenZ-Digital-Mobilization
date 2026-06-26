"""
Sensitivity analysis of the basic reproduction number R0 for the SQCIR model.

This script:
1. Defines baseline SQCIR parameters.
2. Computes R0.
3. Computes normalized sensitivity indices of R0.
4. Generates a publication-style bar chart.

Output:
    fig_sensitivity.pdf
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch


# ---------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------

plt.rcParams.update({
    "figure.dpi": 200,
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "axes.linewidth": 1.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ---------------------------------------------------------------------
# Baseline parameters
# ---------------------------------------------------------------------

theta_hat = {
    "Pi": 0.5,
    "sigma": 0.03,
    "xi": 0.04,
    "rho": 0.09,
    "mu": 0.10,
}


# ---------------------------------------------------------------------
# R0 and sensitivity functions
# ---------------------------------------------------------------------

def R0_from_params(params):
    """
    Compute the basic reproduction number R0.

    R0 = (Pi * sigma) / ((xi + rho) * (mu + rho))
    """
    denominator = (params["xi"] + params["rho"]) * (params["mu"] + params["rho"])

    if denominator <= 0:
        return np.nan

    return (params["Pi"] * params["sigma"]) / denominator


def sensitivity_indices(theta_dict, rel_step=1e-5):
    """
    Compute normalized sensitivity indices of R0.

    The normalized sensitivity index is:

        S_theta^R0 = (theta / R0) * (dR0 / dtheta)

    A positive value means the parameter increases R0.
    A negative value means the parameter decreases R0.
    """
    parameter_names = ["Pi", "sigma", "xi", "rho", "mu"]

    base_params = theta_dict.copy()
    R0_base = R0_from_params(base_params)

    sensitivity = {}

    for name in parameter_names:
        p0 = base_params[name]

        # Small perturbation for numerical derivative
        h = rel_step * (abs(p0) + 1e-8)

        params_plus = {**base_params, name: p0 + h}
        params_minus = {**base_params, name: max(p0 - h, 1e-12)}

        derivative = (
            R0_from_params(params_plus) - R0_from_params(params_minus)
        ) / (params_plus[name] - params_minus[name])

        sensitivity[name] = (p0 / R0_base) * derivative

    return R0_base, sensitivity


# ---------------------------------------------------------------------
# Plotting function
# ---------------------------------------------------------------------

def plot_sensitivity_bar(sensitivity_dict, output_path="fig_sensitivity.pdf"):
    """Create and save a bar chart of normalized sensitivity indices."""

    symbol_map = {
        "Pi": r"$\Pi$",
        "sigma": r"$\sigma$",
        "xi": r"$\xi$",
        "rho": r"$\rho$",
        "mu": r"$\mu$",
    }

    names = list(sensitivity_dict.keys())
    values = np.array(list(sensitivity_dict.values()))
    labels = [symbol_map[name] for name in names]

    blue = "#2166AC"
    red = "#D6604D"
    colors = [blue if value > 0 else red for value in values]

    fig, ax = plt.subplots(figsize=(7, 4.2))

    bars = ax.bar(
        labels,
        values,
        color=colors,
        width=0.55,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )

    # Add value labels to bars
    for bar, value in zip(bars, values):
        vertical_alignment = "bottom" if value >= 0 else "top"
        y_offset = 0.01 if value >= 0 else -0.01

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + y_offset,
            f"{value:+.3f}",
            ha="center",
            va=vertical_alignment,
            fontsize=10,
            fontweight="bold",
        )

    ax.axhline(0, color="black", linewidth=0.9, zorder=4)

    ax.set_xlabel("Model Parameters", labelpad=6)
    ax.set_ylabel(
        r"Normalized Sensitivity Index $\mathcal{S}^{R_0}_{\theta_i}$",
        labelpad=6,
    )

    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())

    ax.grid(axis="y", which="major", alpha=0.25, linestyle="--", zorder=0)
    ax.grid(axis="y", which="minor", alpha=0.10, linestyle=":", zorder=0)

    legend_elements = [
        Patch(facecolor=blue, label="Positive influence"),
        Patch(facecolor=red, label="Negative influence"),
    ]

    ax.legend(handles=legend_elements, loc="upper right", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------

if __name__ == "__main__":
    R0_value, sensitivity_dict = sensitivity_indices(theta_hat)

    print(f"R0 = {R0_value:.4f}")

    for parameter, value in sensitivity_dict.items():
        print(f"  {parameter}: {value:.4f}")

    plot_sensitivity_bar(sensitivity_dict)
