"""
SQCIR model fitting and data-fit visualization.

This script:
1. Defines the SQCIR model equations.
2. Generates a synthetic cumulative engagement curve for demonstration.
3. Fits the SQCIR model to the normalized cumulative curve.
4. Computes goodness-of-fit metrics.
5. Generates a publication-style data-fit figure.

Note:
    The synthetic data section can be replaced with the real processed
    YouTube engagement data when the full dataset is available.

Output:
    fig_datafit.pdf
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.integrate import solve_ivp
from scipy.optimize import minimize


# ---------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------

plt.rcParams.update({
    "figure.dpi": 200,
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "axes.linewidth": 1.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.7",
})


# ---------------------------------------------------------------------
# SQCIR model
# ---------------------------------------------------------------------

def sqcir_rhs(t, y, params):
    """Right-hand side of the SQCIR model."""
    S, Q, C, I, R = y

    Pi, beta1, beta2, sigma, xi, rho, gamma, mu, omega = params

    dSdt = Pi - beta1 * S * Q - beta2 * S * I - sigma * S * C - xi * S - rho * S
    dQdt = beta1 * S * Q + beta2 * S * I - sigma * Q * C - rho * Q
    dCdt = sigma * S * C - sigma * Q * C - gamma * C * I - mu * C - rho * C
    dIdt = gamma * C * I - omega * I - rho * I
    dRdt = mu * C + omega * I + xi * S - rho * R

    return [dSdt, dQdt, dCdt, dIdt, dRdt]


def simulate_sqcir(t_eval, y0, params):
    """Simulate the SQCIR model over the time points in t_eval."""
    try:
        solution = solve_ivp(
            lambda t, y: sqcir_rhs(t, y, params),
            (t_eval[0], t_eval[-1]),
            y0,
            t_eval=t_eval,
            method="RK45",
            rtol=1e-6,
            atol=1e-8,
            max_step=0.5,
        )

        if solution.success:
            return solution.y

        return None

    except Exception:
        return None


def cumulative_trapezoid(x, t):
    """Compute cumulative integral using the trapezoidal rule."""
    output = np.zeros_like(x)

    for k in range(1, len(x)):
        output[k] = output[k - 1] + 0.5 * (x[k] + x[k - 1]) * (t[k] - t[k - 1])

    return output


# ---------------------------------------------------------------------
# Demonstration data
# ---------------------------------------------------------------------

def generate_synthetic_engagement_data(seed=42):
    """
    Generate synthetic S-shaped cumulative engagement data.

    This section is included so that the script can run without the full
    research dataset. Replace this section with real CSV loading when needed.
    """
    np.random.seed(seed)

    n_points = 200
    t_days = np.linspace(0, 30, n_points)

    sigmoid_curve = 1 / (1 + np.exp(-0.35 * (t_days - 10)))
    sigmoid_curve = (sigmoid_curve - sigmoid_curve[0]) / (
        sigmoid_curve[-1] - sigmoid_curve[0]
    )

    noise = 0.006 * np.random.randn(n_points)

    raw_increment = np.diff(
        np.clip(sigmoid_curve + noise, 0, None),
        prepend=0
    ).clip(0)

    cumulative_likes = (
        np.cumsum(raw_increment)
        / np.cumsum(raw_increment).max()
        * 5e6
    )

    plot_dates = pd.date_range(
        "2025-09-08",
        periods=n_points,
        freq="216min"
    )

    return t_days, cumulative_likes, plot_dates


# ---------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------

def fit_sqcir_model(t_days, cumulative_data, y0):
    """Fit SQCIR parameters to normalized cumulative engagement data."""
    cumulative_normalized = cumulative_data / cumulative_data.max()

    def objective(log_params):
        params = tuple(np.exp(log_params))

        Y = simulate_sqcir(t_days, y0, params)

        if Y is None:
            return 1e6

        cumulative_I = cumulative_trapezoid(Y[3], t_days)
        max_cumulative_I = cumulative_I.max()

        if max_cumulative_I < 1e-15:
            return 1e6

        model_normalized = cumulative_I / max_cumulative_I

        return np.sqrt(
            np.mean((model_normalized - cumulative_normalized) ** 2)
        )

    initial_guess = np.log([
        0.5,   # Pi
        0.14,  # beta1
        0.05,  # beta2
        0.03,  # sigma
        0.04,  # xi
        0.09,  # rho
        0.10,  # gamma
        0.10,  # mu
        0.01,  # omega
    ])

    result = minimize(
        objective,
        initial_guess,
        method="Nelder-Mead",
        options={
            "maxiter": 5000,
            "xatol": 1e-5,
            "fatol": 1e-6,
        },
    )

    fitted_params = tuple(np.exp(result.x))

    return fitted_params, result.fun


def evaluate_fit(t_days, cumulative_data, y0, fitted_params):
    """Evaluate fitted SQCIR model and compute goodness-of-fit metrics."""
    Y_hat = simulate_sqcir(t_days, y0, fitted_params)

    cumulative_I_hat = cumulative_trapezoid(Y_hat[3], t_days)

    scale_factor = np.dot(cumulative_data, cumulative_I_hat) / (
        np.dot(cumulative_I_hat, cumulative_I_hat) + 1e-15
    )

    model_counts = scale_factor * cumulative_I_hat

    residual_error = model_counts - cumulative_data

    rmse = np.sqrt(np.mean(residual_error ** 2))
    correlation = np.corrcoef(cumulative_data, model_counts)[0, 1]
    relative_error = np.linalg.norm(residual_error) / (
        np.linalg.norm(cumulative_data) + 1e-12
    )

    normalized_data = cumulative_data / cumulative_data.max()
    normalized_model = model_counts / (model_counts.max() + 1e-15)

    normalized_rmse = np.sqrt(
        np.mean((normalized_model - normalized_data) ** 2)
    )

    metrics = {
        "RMSE": rmse,
        "correlation": correlation,
        "relative_error": relative_error,
        "normalized_RMSE": normalized_rmse,
    }

    return model_counts, normalized_data, normalized_model, metrics


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def plot_data_fit(plot_dates, normalized_data, normalized_model, metrics):
    """Generate normalized cumulative data-fit plot."""
    data_style = {
        "marker": "o",
        "markersize": 4,
        "markerfacecolor": "red",
        "markeredgecolor": "darkred",
        "markeredgewidth": 0.5,
        "linestyle": "none",
        "alpha": 0.7,
        "label": "Gen Z YouTube Data",
    }

    model_style = {
        "lw": 2,
        "color": "#2166AC",
        "label": "SQCIR fit",
    }

    text_box_style = {
        "boxstyle": "round,pad=0.4",
        "facecolor": "lightyellow",
        "edgecolor": "0.6",
        "alpha": 0.9,
    }

    fig, ax = plt.subplots(1, 1, figsize=(8, 4.8))

    ax.plot(plot_dates, normalized_data, **data_style)
    ax.plot(plot_dates, normalized_model, **model_style)

    ax.set_ylabel("Normalized cumulative value")
    ax.set_xlabel("Date (Gen Z Mob 2025)")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))

    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    ax.text(
        0.97,
        0.06,
        f"RMSE(norm) = {metrics['normalized_RMSE']:.4f}\n"
        f"r = {metrics['correlation']:.4f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=text_box_style,
    )

    ax.legend(loc="upper left")
    ax.set_axisbelow(True)
    ax.grid(
        True,
        color="gray",
        linestyle="--",
        linewidth=0.5,
        alpha=0.4,
    )

    plt.tight_layout()
    plt.savefig("fig_datafit.pdf", bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------

if __name__ == "__main__":
    t_days, cumulative_likes, plot_dates = generate_synthetic_engagement_data()

    Q0 = 1e-4
    C0 = 1e-4
    I0 = 1e-6
    R0_initial = 0.0

    y0 = [
        1 - (Q0 + C0 + I0 + R0_initial),
        Q0,
        C0,
        I0,
        R0_initial,
    ]

    params_hat, fit_rmse = fit_sqcir_model(t_days, cumulative_likes, y0)

    print(f"RMSE(norm) after fit = {fit_rmse:.4f}")

    model_counts, norm_data, norm_model, fit_metrics = evaluate_fit(
        t_days,
        cumulative_likes,
        y0,
        params_hat,
    )

    print(
        f"Corr={fit_metrics['correlation']:.4f} | "
        f"E_rel={fit_metrics['relative_error']:.4f} | "
        f"RMSE_n={fit_metrics['normalized_RMSE']:.4f}"
    )

    plot_data_fit(plot_dates, norm_data, norm_model, fit_metrics)
