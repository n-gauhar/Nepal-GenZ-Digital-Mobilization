"""
Caputo fractional-order SQCIR model examples.

This script:
1. Defines the classical SQCIR model.
2. Implements a numerical L1-type approximation for the Caputo fractional model.
3. Compares the classical model alpha = 1 with fractional cases alpha = 0.85
   and alpha = 0.70.
4. Generates compartment-wise comparison plots.

Output:
    fig_caputo_examples.pdf
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
    "legend.fontsize": 8,
})


# ---------------------------------------------------------------------
# Model settings
# ---------------------------------------------------------------------

COLORS = {
    "S": "#2166AC",
    "Q": "#F4A582",
    "C": "#4DAC26",
    "I": "#8B0000",
    "R": "#808080",
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
    1 - (Q0 + C0 + I0),  # S0
    Q0,                  # Q0
    C0,                  # C0
    I0,                  # I0
    0.0,                 # R0
])

COMPARTMENT_NAMES = ["S", "Q", "C", "I", "R"]


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
    """Solve the classical integer-order SQCIR model."""
    t_eval = np.linspace(0, total_time, n_steps + 1)

    solution = solve_ivp(
        lambda t, y: sqcir_rhs(y, params),
        (0, total_time),
        y0,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
    )

    return t_eval, solution.y


def solve_caputo_l1_sqcir(total_time, n_steps, alpha, y0, params):
    """
    Solve the Caputo fractional-order SQCIR model using an L1-type scheme.

    alpha controls the memory effect:
        alpha close to 1.0 -> weaker memory, closer to classical model
        lower alpha        -> stronger memory effect
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


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def plot_caputo_examples(output_path="fig_caputo_examples.pdf"):
    """Generate comparison plots for alpha = 1, 0.85, and 0.70."""
    total_time = 100
    n_steps = 500

    t_classical, y_classical = solve_classical_sqcir(
        total_time,
        n_steps,
        INITIAL_CONDITIONS,
        PARAMS,
    )

    t_alpha_085, y_alpha_085 = solve_caputo_l1_sqcir(
        total_time,
        n_steps,
        alpha=0.85,
        y0=INITIAL_CONDITIONS,
        params=PARAMS,
    )

    t_alpha_070, y_alpha_070 = solve_caputo_l1_sqcir(
        total_time,
        n_steps,
        alpha=0.70,
        y0=INITIAL_CONDITIONS,
        params=PARAMS,
    )

    line_styles = {
        1.00: "-",
        0.85: "--",
        0.70: ":",
    }

    example_configs = [
        (0.85, t_alpha_085, y_alpha_085, "Example 1 (alpha=0.85, mild memory)"),
        (0.70, t_alpha_070, y_alpha_070, "Example 2 (alpha=0.70, strong memory)"),
    ]

    fig, axes = plt.subplots(2, 5, figsize=(16, 7), sharey="row")

    fig.suptitle(
        "Caputo Fractional SQCIR Model\n"
        r"Effect of fractional order $\alpha$ on epidemic compartments",
        fontsize=13,
        fontweight="bold",
    )

    for row, (alpha_value, t_fractional, y_fractional, example_label) in enumerate(
        example_configs
    ):
        for column, compartment_name in enumerate(COMPARTMENT_NAMES):
            ax = axes[row, column]

            ax.plot(
                t_classical,
                y_classical[column],
                color=COLORS[compartment_name],
                lw=2.2,
                ls=line_styles[1.00],
                label=r"$\alpha=1$ (classical)",
            )

            ax.plot(
                t_fractional,
                y_fractional[column],
                color=COLORS[compartment_name],
                lw=1.8,
                ls=line_styles[alpha_value],
                label=rf"$\alpha={alpha_value}$",
            )

            ax.set_title(f"{compartment_name} compartment", fontsize=9)

            if column == 0:
                ax.set_ylabel(f"{example_label}\nFraction", fontsize=8)

            if row == 1:
                ax.set_xlabel("Time (days)", fontsize=8)

            ax.grid(True, alpha=0.2, linestyle="--")
            ax.legend(loc="best", fontsize=7, handlelength=2)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(output_path, bbox_inches="tight")
    plt.show()

    print("Caputo examples plot done.")


if __name__ == "__main__":
    plot_caputo_examples()
