"""Appendix B figure `losscurve.png`: loss profile in m_d after the other parameters are refit.

Reads `data/irfmatching/profile_md.csv`, written by `code/irfmatching/matlab/profile_md.m`. Each
row holds one perturbation `t` of the household cognitive-discounting parameter m_d away from its
true value 0.65, together with the impulse-response matching loss that survives after the other
eight free parameters are re-optimized: `q_S` against the scalar path target, `q_F` against the
two path-factor targets.

The figure plots both profiles over the inner part of the grid, with the perturbation expressed
as a percent of the true m_d. The factor profile rising faster is the identification claim: the
factor targets punish a wrong m_d that the scalar target lets the other parameters absorb.

Run from the repository root:

    python code/irfmatching/make_losscurve_figure.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FixedLocator  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILE_FILE = REPO_ROOT / "data" / "irfmatching" / "profile_md.csv"
FIGURE_DIR = REPO_ROOT / "output" / "figures"
STATS_DIR = REPO_ROOT / "output" / "stats"

M_D_TRUE = 0.65
T_WINDOW = 0.04

SCALAR_COLOR = "#1F77B4"
FACTOR_COLOR = "#D2601A"
ZERO_LINE_COLOR = "#999999"

PERCENT_TICKS = [-6, -4, -2, 0, 2, 4, 6]

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


def load_profile() -> pd.DataFrame:
    """Rows of the m_d profile inside the plotted window, with the perturbation in percent."""
    frame = pd.read_csv(PROFILE_FILE)
    frame = frame.loc[frame["t"].abs() <= T_WINDOW + 1e-9, ["t", "q_S", "q_F"]].copy()
    frame = frame.sort_values("t").reset_index(drop=True)
    frame["percent"] = 100.0 * frame["t"] / M_D_TRUE
    return frame


def percent_label(value: float) -> str:
    """Tick label in the paper's style: a signed percent, with no sign on zero."""
    if value == 0:
        return "0%"
    return f"{value:+.0f}%"


def make_figure(frame: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    ax.axvline(0.0, color=ZERO_LINE_COLOR, linewidth=0.7, zorder=1)
    ax.plot(frame["percent"], frame["q_S"], marker="o", markersize=4, linewidth=1.5,
            color=SCALAR_COLOR, label="Scalar path", zorder=2)
    ax.plot(frame["percent"], frame["q_F"], marker="o", markersize=4, linewidth=1.5,
            color=FACTOR_COLOR, label="Path factors", zorder=3)
    # Headroom above the peak so the tick locator lands on whole multiples of 1e-3 and the
    # legend clears the factor curve.
    ax.set_ylim(0.0, 1.5 * float(frame["q_F"].max()))
    # Losses are order 1e-3; pull the exponent out into an offset label so the ticks read 0..4.
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3))
    ax.xaxis.set_major_locator(FixedLocator(PERCENT_TICKS))
    ax.set_xticklabels([percent_label(tick) for tick in PERCENT_TICKS])
    ax.set_xlabel(r"Perturbation in $m_d$")
    ax.set_ylabel("Distance from targets after refit")
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    return fig


def main() -> None:
    frame = load_profile()
    # Both losses are zero at the truth, so the ratio is undefined there. The shipped grid skips
    # t = 0, but a regenerated one need not, and a bare division would emit inf rather than say so.
    positive = frame["q_S"] > 0
    frame["ratio_factors_scalar"] = (frame["q_F"] / frame["q_S"].where(positive)).where(positive)

    print("     t   percent            q_S            q_F   q_F/q_S")
    for row in frame.itertuples(index=False):
        print(f"{row.t:+6.3f}  {row.percent:+7.2f}  {row.q_S:13.6e}  {row.q_F:13.6e}  "
              f"{row.ratio_factors_scalar:8.2f}")

    ends = frame.loc[np.isclose(frame["t"].abs(), T_WINDOW)]
    for row in ends.itertuples(index=False):
        print(f"\nAt t = {row.t:+.2f} ({row.percent:+.1f}% of m_d = {M_D_TRUE}): "
              f"q_S = {row.q_S:.3e}, q_F = {row.q_F:.3e}, "
              f"factor/scalar = {row.ratio_factors_scalar:.2f}x")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    STATS_DIR.mkdir(parents=True, exist_ok=True)

    with plt.rc_context(RC_PARAMS):
        fig = make_figure(frame)
        fig.savefig(FIGURE_DIR / "losscurve.pdf")
        fig.savefig(FIGURE_DIR / "losscurve.png", dpi=200)
        plt.close(fig)

    stats_path = STATS_DIR / "md_loss_profile.csv"
    frame.rename(columns={"q_S": "q_scalar", "q_F": "q_factors"}).to_csv(
        stats_path, index=False, float_format="%.10g")

    print(f"\nwrote {FIGURE_DIR / 'losscurve.pdf'}")
    print(f"wrote {FIGURE_DIR / 'losscurve.png'}")
    print(f"wrote {stats_path}")


if __name__ == "__main__":
    main()
