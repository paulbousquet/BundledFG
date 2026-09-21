"""Appendix B figure `gradient.png`: identifying power of the factor targets, by parameter.

Reads the baseline weighted derivative matrices `D_S` (scalar path, 40 x 9) and `D_F` (path
factors, 80 x 9) from `data/irfmatching/profile_md_jacobians.mat`, the file that
`code/irfmatching/matlab/profile_md.m` writes. Column j of each matrix is the derivative of the
weighted targets with respect to free parameter j, evaluated at the true parameter vector.

For each free parameter the script forms the concentrated curvature of the impulse-response
matching loss under both designs and plots the ratio (factors over scalar). A ratio above one
means the factor design leaves more curvature in that parameter once the other eight are
re-optimized, i.e. it identifies the parameter more sharply.

Run from the repository root:

    python code/irfmatching/make_gradient_figure.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.io import loadmat  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
JACOBIAN_FILE = REPO_ROOT / "data" / "irfmatching" / "profile_md_jacobians.mat"
FIGURE_DIR = REPO_ROOT / "output" / "figures"
STATS_DIR = REPO_ROOT / "output" / "stats"

STEM_COLOR = "#B34816"
REFERENCE_COLOR = "#999999"

# Free-parameter names as they appear in the MATLAB `names` cell, mapped to their paper labels.
LATEX_LABELS = {
    "h": r"$h$",
    "xi_p": r"$\xi_p$",
    "iota_p": r"$\iota_p$",
    "xi_w": r"$\xi_w$",
    "iota_w": r"$\iota_w$",
    "kappa": r"$\kappa$",
    "m_d": r"$m_d$",
    "m_f": r"$m_f$",
    "varphi": r"$\varphi$",
}

# Ratios reported in the paper, in `free9` order. The script refuses to write a figure that
# disagrees with these by more than TOLERANCE.
EXPECTED_RATIOS = np.array([2.38, 2.43, 1.96, 7.92, 9.08, 8.93, 12.44, 6.05, 8.62])
TOLERANCE = 0.05

RC_PARAMS = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "mathtext.fontset": "dejavuserif",
    "pdf.fonttype": 42,
}


def concentrated_curvature(design_matrix: np.ndarray) -> np.ndarray:
    """Curvature of the loss in each parameter after the other parameters are re-optimized.

    For the Gauss-Newton loss q(theta) = 0.5 * ||D (theta - theta_0)||^2, profiling out the other
    parameters leaves parameter j with curvature equal to the squared norm of column j of D after
    the remaining columns are projected out. By the partitioned-inverse (Frisch-Waugh) identity
    that residual squared norm is exactly 1 / [(D'D)^{-1}]_jj, so no explicit regression is needed.
    """
    return 1.0 / np.diag(np.linalg.inv(design_matrix.T @ design_matrix))


def load_jacobians() -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return the nine free-parameter names and the scalar and factor derivative matrices."""
    data = loadmat(JACOBIAN_FILE, simplify_cells=True)
    names = np.asarray(data["names"])
    free9 = np.asarray(data["free9"], dtype=int) - 1  # stored 1-based by MATLAB
    return [str(name) for name in names[free9]], data["D_S"], data["D_F"]


def make_figure(labels: list[str], ratios: np.ndarray) -> plt.Figure:
    """Stem plot of the factor-to-scalar curvature ratio, one stem per free parameter."""
    positions = np.arange(len(ratios))
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    ax.axhline(1.0, color=REFERENCE_COLOR, linewidth=0.7, zorder=1)
    ax.vlines(positions, 1.0, ratios, color=STEM_COLOR, linewidth=1.4, zorder=2)
    ax.plot(positions, ratios, linestyle="none", marker="o", markersize=7.5,
            color=STEM_COLOR, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(1, 150)
    ax.set_yticks([1, 10, 100])
    ax.set_yticklabels(["1", "10", "100"])
    ax.set_yticks([], minor=True)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.tick_params(axis="x", length=0)
    # "Gradient" is right even though the plotted quantity is a curvature. The loss is quadratic
    # near the truth, so at any displacement its gradient is curvature times displacement; the
    # displacement cancels in the ratio, making the ratio of loss gradients equal the ratio of
    # curvatures. The label matches the published figure -- do not "correct" it to curvature.
    ax.set_ylabel("Loss gradient, factors / scalar")
    fig.tight_layout()
    return fig


def main() -> None:
    names, d_scalar, d_factors = load_jacobians()
    curvature_scalar = concentrated_curvature(d_scalar)
    curvature_factors = concentrated_curvature(d_factors)
    ratios = curvature_factors / curvature_scalar

    print("parameter  curvature_scalar  curvature_factors     ratio  expected")
    for name, c_s, c_f, ratio, expected in zip(names, curvature_scalar, curvature_factors,
                                               ratios, EXPECTED_RATIOS):
        print(f"{name:>9}  {c_s:16.6e}  {c_f:17.6e}  {ratio:8.2f}  {expected:8.2f}")

    deviation = np.abs(ratios - EXPECTED_RATIOS)
    if deviation.max() > TOLERANCE:
        offenders = [f"{names[j]}: got {ratios[j]:.4f}, expected {EXPECTED_RATIOS[j]:.2f}"
                     for j in np.flatnonzero(deviation > TOLERANCE)]
        raise SystemExit("Curvature ratios do not reproduce the paper:\n  "
                         + "\n  ".join(offenders))
    print(f"\nAll nine ratios match the paper within {TOLERANCE} "
          f"(largest deviation {deviation.max():.4f}).")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    STATS_DIR.mkdir(parents=True, exist_ok=True)

    with plt.rc_context(RC_PARAMS):
        fig = make_figure([LATEX_LABELS[name] for name in names], ratios)
        fig.savefig(FIGURE_DIR / "gradient.pdf")
        fig.savefig(FIGURE_DIR / "gradient.png", dpi=200)
        plt.close(fig)

    stats_path = STATS_DIR / "loss_gradient_ratio.csv"
    with stats_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("parameter,curvature_scalar,curvature_factors,ratio\n")
        for name, c_s, c_f, ratio in zip(names, curvature_scalar, curvature_factors, ratios):
            handle.write(f"{name},{c_s:.10e},{c_f:.10e},{ratio:.10f}\n")

    print(f"wrote {FIGURE_DIR / 'gradient.pdf'}")
    print(f"wrote {FIGURE_DIR / 'gradient.png'}")
    print(f"wrote {stats_path}")


if __name__ == "__main__":
    main()
