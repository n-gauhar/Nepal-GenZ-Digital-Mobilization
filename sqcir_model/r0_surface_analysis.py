"""
3D parameter-surface visualization of the basic reproduction number R0.

This script:
1. Defines baseline SQCIR parameters.
2. Varies selected parameter pairs by +/- 50%.
3. Computes R0 over each parameter grid.
4. Generates 3D surface plots showing how R0 changes with parameter pairs.

Output:
    fig_R0_surfaces.pdf
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ---------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------

plt.rcParams.update({
    "figure.dpi": 200,
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
})


# ---------------------------------------------------------------------
# Baseline parameters
# ---------------------------------------------------------------------

PI_0 = 0.5
SIGMA_0 = 0.03
XI_0 = 0.04
RHO_0 = 0.09
MU_0 = 0.10


# ---------------------------------------------------------------------
# R0 and helper functions
# ---------------------------------------------------------------------

def compute_R0(Pi, sigma, xi, rho, mu):
    """
    Compute the basic reproduction number R0.

    R0 = (Pi * sigma) / ((xi + rho) * (mu + rho))
    """
    denominator = (xi + rho) * (mu + rho)

    return (Pi * sigma) / np.maximum(denominator, 1e-12)


def vary_parameter(x0, pct=0.5, n=70, floor=1e-8):
    """
    Create a parameter range around a baseline value.

    By default, the parameter varies from 50% below to 50% above
    its baseline value.
    """
    lower_bound = max(floor, x0 * (1 - pct))
    upper_bound = x0 * (1 + pct)

    return np.linspace(lower_bound, upper_bound, n)


# ---------------------------------------------------------------------
# Parameter grids
# ---------------------------------------------------------------------

Pi_values = vary_parameter(PI_0)
sigma_values = vary_parameter(SIGMA_0)
xi_values = vary_parameter(XI_0)
rho_values = vary_parameter(RHO_0)
mu_values = vary_parameter(MU_0)


# ---------------------------------------------------------------------
# Surface plot settings
# ---------------------------------------------------------------------

COLOR_MAP = "RdYlBu_r"
SURFACE_ALPHA = 0.85

panels = [
    (
        r"$R_0$ vs $\Pi$ and $\sigma$",
        Pi_values,
        sigma_values,
        r"$\Pi$",
        r"$\sigma$",
        lambda X, Y: compute_R0(X, Y, XI_0, RHO_0, MU_0),
    ),
    (
        r"$R_0$ vs $\xi$ and $\rho$",
        xi_values,
        rho_values,
        r"$\xi$",
        r"$\rho$",
        lambda X, Y: compute_R0(PI_0, SIGMA_0, X, Y, MU_0),
    ),
    (
        r"$R_0$ vs $\sigma$ and $\rho$",
        sigma_values,
        rho_values,
        r"$\sigma$",
        r"$\rho$",
        lambda X, Y: compute_R0(PI_0, X, XI_0, Y, MU_0),
    ),
    (
        r"$R_0$ vs $\rho$ and $\mu$",
        rho_values,
        mu_values,
        r"$\rho$",
        r"$\mu$",
        lambda X, Y: compute_R0(PI_0, SIGMA_0, XI_0, X, Y),
    ),
]


# ---------------------------------------------------------------------
# Generate figure
# ---------------------------------------------------------------------

fig = plt.figure(figsize=(13, 9))

for panel_index, (title, x_values, y_values, x_label, y_label, r0_function) in enumerate(panels, 1):
    ax = fig.add_subplot(2, 2, panel_index, projection="3d")

    X, Y = np.meshgrid(x_values, y_values)
    Z = r0_function(X, Y)

    surface = ax.plot_surface(
        X,
        Y,
        Z,
        cmap=COLOR_MAP,
        alpha=SURFACE_ALPHA,
        linewidth=0,
        antialiased=True,
        rcount=60,
        ccount=60,
    )

    # Add a contour projection on the bottom plane
    ax.contourf(
        X,
        Y,
        Z,
        zdir="z",
        offset=Z.min() * 0.95,
        cmap=COLOR_MAP,
        alpha=0.3,
        levels=12,
    )

    colorbar = fig.colorbar(
        surface,
        ax=ax,
        shrink=0.52,
        pad=0.10,
        aspect=14,
    )

    colorbar.set_label(r"$R_0$", fontsize=9)

    ax.set_title(title, pad=6)
    ax.set_xlabel(x_label, labelpad=4)
    ax.set_ylabel(y_label, labelpad=4)
    ax.set_zlabel(r"$R_0$", labelpad=4)

    ax.tick_params(labelsize=8)

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.grid(True, linestyle=":", alpha=0.4)


plt.tight_layout()
plt.savefig("fig_R0_surfaces.pdf", bbox_inches="tight")
plt.show()
