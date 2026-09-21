"""Part A figures: PC loadings, the rolling GSS rotation angle, and event-window outliers.

Draws `fig:load`, `fig:gss` (= appendix `fig:angle`), and appendix `fig:window1`. Every
number comes from `common.py`; this module only picks samples, writes files, and plots.
Also emits the pgfplots `.dat` files the paper's TeX version of the rotation figure reads.

Run from the repository root:

    python code/highfreq/run_figures.py [--variant full_untransformed]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    ORIGINAL,
    OUTPUT_DIR,
    RICHER,
    ROLLING_WINDOW,
    SPLIT_DATE,
    VARIANTS,
    insert_gaps,
    pca,
    quadratic_reconstruction,
    rolling_rotation,
    sample,
    variant_frame,
)

# --------------------------------------------------------------------------- #
# Presentation constants
# --------------------------------------------------------------------------- #

PENN_RED = "#95001A"  # RGB (149, 0, 26)
BLUE = "#1414B4"
ZLB_SHADE = "#E7E7E7"
GRID = "#E6E6E6"

# Each published figure was built under its own convention, so there is no single default.
# gssdrift and its .dat come from the dashboard-adjusted strip; the loadings figure does not;
# and window1 must be untransformed, since the two contracts it flags as mismeasured (ED6 on
# 2001-05-15, ED7 on 2004-03-16) are exactly the ones the dashboard corrects. Running with no
# --variant reproduces all three as published; passing --variant builds them on one convention.
PUBLISHED_VARIANT = {
    "loadings": "full_untransformed",
    "gssdrift": "full_adjusted",
    "window1": "full_untransformed",
}

FIGURE_DIR = OUTPUT_DIR / "figures"
DAT_DIR = FIGURE_DIR / "dat"

# Decimal-year edges of the shaded zero-lower-bound bands in the rotation figure.
ZLB_BANDS = [(2008.83, 2015.75), (2020.17, 2022.00)]

ANGLE_XLIM = (2000.0, 2026.5)
ANGLE_YLIM = (25.0, 100.0)

# The two announcements the appendix uses to show a single mismeasured contract.
WINDOW_EVENTS = [
    ("2001-05-15", "(a) May 15, 2001", PENN_RED),
    ("2004-03-16", "(b) March 16, 2004", BLUE),
]

PP_TO_BP = 100.0  # source surprises are in percentage points

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.linewidth": 0.8,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.frameon": True,
        "legend.framealpha": 1.0,
        "legend.edgecolor": "black",
        "lines.solid_capstyle": "round",
    }
)


# --------------------------------------------------------------------------- #
# File helpers
# --------------------------------------------------------------------------- #

def _stem(stem: str, variant: str) -> str:
    """Unsuffixed names are reserved for the variant each figure was published under."""
    figure = "gssdrift" if stem.startswith("section2_rotation40") else stem
    published = PUBLISHED_VARIANT.get(figure)
    return stem if variant == published else f"{stem}_{variant}"


def write_window_spans(rows: list[dict], stem: str, variant: str) -> Path:
    """Record what calendar range each plotted rolling window actually covers.

    The .dat files carry only the window's end date, which is what the published figure plots.
    Because zero-lower-bound meetings are removed before the 40-meeting windows are formed, a
    point drawn at, say, 2016 can be estimated on meetings reaching back before the 2008 block.
    This file makes that visible without altering the .dat the figure consumes.
    """
    frame = pd.DataFrame(rows)[["t", "start", "end", "angle"]].copy()
    frame["span_years"] = frame["end"] - frame["start"]
    path = DAT_DIR / f"{_stem(stem, variant)}_windows.csv"
    frame.to_csv(path, index=False, float_format="%.6f", lineterminator='\n')
    return path


def save_figure(fig: plt.Figure, stem: str, variant: str) -> list[Path]:
    name = _stem(stem, variant)
    paths = []
    for suffix, kwargs in (("pdf", {}), ("png", {"dpi": 150})):
        path = FIGURE_DIR / f"{name}.{suffix}"
        fig.savefig(path, bbox_inches="tight", **kwargs)
        paths.append(path)
    plt.close(fig)
    return paths


def write_dat(frame: pd.DataFrame, stem: str, variant: str) -> Path:
    """Two-column pgfplots table; `nan` rows are the line breaks across the ZLB gaps."""
    path = DAT_DIR / f"{_stem(stem, variant)}.dat"
    lines = ["t angle"]
    lines += [f"{t:.18e} {angle:.18e}" for t, angle in zip(frame["t"], frame["angle"])]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Figure 1 -- unrotated PC1 and PC2 loadings, full sample vs 2009 onward
# --------------------------------------------------------------------------- #

def figure_loadings(df: pd.DataFrame, variant: str) -> tuple[list[Path], pd.DataFrame]:
    full = sample(
        df,
        RICHER,
        require_scheduled=False,
        exclude_2009_2014=VARIANTS[variant]["exclude_2009_2014"],
    )
    recent = full.loc[full["date"].ge(SPLIT_DATE)]
    v_full, _, _ = pca(full[RICHER].to_numpy(float))
    v_recent, _, _ = pca(recent[RICHER].to_numpy(float))

    x = np.arange(len(RICHER), dtype=float)
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)

    for k, color, name in ((0, PENN_RED, "PC1"), (1, BLUE, "PC2")):
        a, b = v_full[:, k], v_recent[:, k]
        ax.vlines(x, np.minimum(a, b), np.maximum(a, b), color=color, linewidth=0.9,
                  alpha=0.55, zorder=2)
        ax.plot(x, a, linestyle="none", marker="o", markersize=5.5, color=color,
                label=f"{name}, full", zorder=4)
        # Hollow markers are drawn larger so they stay visible when the two samples agree.
        ax.plot(x, b, linestyle="none", marker="o", markersize=8.0, markerfacecolor="none",
                markeredgecolor=color, markeredgewidth=1.3, label=f"{name}, 2009+", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(RICHER)
    ax.set_xlim(-0.6, len(RICHER) - 0.4)
    ax.set_xlabel("contract")
    ax.set_ylabel("loading")
    ax.legend(loc="upper right", borderpad=0.5, labelspacing=0.4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    paths = save_figure(fig, "loadings", variant)
    table = pd.DataFrame(
        {
            "contract": RICHER,
            "pc1_full": v_full[:, 0],
            "pc1_2009plus": v_recent[:, 0],
            "pc2_full": v_full[:, 1],
            "pc2_2009plus": v_recent[:, 1],
        }
    )
    return paths, table


# --------------------------------------------------------------------------- #
# Figure 2 -- 40-meeting rolling GSS rotation angle
# --------------------------------------------------------------------------- #

def figure_gssdrift(df: pd.DataFrame, variant: str) -> tuple[list[Path], list[Path], dict]:
    exclude_2009_2014 = VARIANTS[variant]["exclude_2009_2014"]
    specs = [
        ("original", ORIGINAL, True, PENN_RED, "-", "Original GSS"),
        ("richer", RICHER, False, BLUE, "--", "Expanded Set"),
    ]

    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    for lo, hi in ZLB_BANDS:
        ax.axvspan(lo, hi, color=ZLB_SHADE, linewidth=0, zorder=0)

    dat_paths, summary = [], {}
    for stem, cols, require_scheduled, color, style, label in specs:
        rows = rolling_rotation(
            variant_frame(variant, cols),
            cols,
            require_scheduled=require_scheduled,
            exclude_2009_2014=exclude_2009_2014,
            window=ROLLING_WINDOW,
        )
        plotted = insert_gaps(rows)
        ax.plot(plotted["t"], plotted["angle"], linestyle=style, color=color, linewidth=1.7,
                label=label, zorder=3)
        dat_paths.append(write_dat(plotted, f"section2_rotation40_{stem}", variant))
        dat_paths.append(write_window_spans(rows, f"section2_rotation40_{stem}", variant))
        angles = np.array([r["angle"] for r in rows], float)
        spans = np.array([r["end"] - r["start"] for r in rows], float)
        # A window holds 40 retained meetings, but ZLB meetings are dropped before windowing, so
        # a window can reach back across a whole excluded block while only its end date is
        # plotted. Report how often that happens; the companion spans file has the detail.
        straddling = int((spans > 2 * ROLLING_WINDOW / 8).sum())
        summary[stem] = {
            "windows": len(rows),
            "first_t": rows[0]["t"],
            "last_t": rows[-1]["t"],
            "min_angle": float(angles.min()),
            "max_angle": float(angles.max()),
            "median_span_years": float(np.median(spans)),
            "max_span_years": float(spans.max()),
            "windows_spanning_over_10y": int((spans > 10).sum()),
        }

    ax.set_xlim(*ANGLE_XLIM)
    ax.set_ylim(*ANGLE_YLIM)
    ax.set_xlabel("window end date")
    ax.set_ylabel("rotation angle (degrees)")
    ax.legend(loc="upper left", borderpad=0.5)
    paths = save_figure(fig, "gssdrift", variant)
    return paths, dat_paths, summary


# --------------------------------------------------------------------------- #
# Figure 3 -- event-window measurement outliers
# --------------------------------------------------------------------------- #

def figure_window1(df: pd.DataFrame, variant: str) -> tuple[list[Path], list[dict]]:
    block = sample(
        df,
        RICHER,
        require_scheduled=False,
        exclude_2009_2014=VARIANTS[variant]["exclude_2009_2014"],
    )
    x = np.arange(len(RICHER), dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.6), sharey=True)
    diagnostics: list[dict] = []
    for ax, (date, title, color) in zip(axes, WINDOW_EVENTS):
        ax.set_title(title, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(RICHER)
        ax.set_xlim(-0.35, len(RICHER) - 0.65)
        ax.axhline(0.0, color="black", linewidth=0.8, zorder=1)

        row = block.loc[block["date"].eq(date), RICHER]
        if row.empty:
            diagnostics.append({"date": date, "present": False})
            continue

        raw = row.to_numpy(float)[:1]
        observed = raw[0] * PP_TO_BP
        fitted = quadratic_reconstruction(raw)[0] * PP_TO_BP
        residual = observed - fitted

        ax.plot(x, observed, linestyle="-", marker="o", markersize=5.0, color=color,
                linewidth=2.0, zorder=3)
        ax.plot(x, fitted, linestyle=":", color="black", linewidth=1.8, zorder=4)

        order = np.argsort(np.abs(residual))[::-1]
        diagnostics.append(
            {
                "date": date,
                "present": True,
                "observed_bp": observed,
                "fitted_bp": fitted,
                "residual_bp": residual,
                "worst": RICHER[order[0]],
                "worst_bp": float(residual[order[0]]),
                "runner_up": RICHER[order[1]],
                "runner_up_bp": float(residual[order[1]]),
            }
        )

    axes[0].set_ylabel("realized revision (bp)")
    for ax in axes:
        ax.margins(y=0.12)
    fig.subplots_adjust(wspace=0.08)
    paths = save_figure(fig, "window1", variant)
    return paths, diagnostics


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Part A high-frequency figures.")
    parser.add_argument("--variant", default=None, choices=sorted(VARIANTS),
                        help="build every figure on this convention instead of the one each "
                             "was published under")
    args = parser.parse_args()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    DAT_DIR.mkdir(parents=True, exist_ok=True)

    def chosen(figure: str) -> str:
        return args.variant or PUBLISHED_VARIANT[figure]

    if args.variant:
        print(f"variant: {args.variant} (overriding each figure's published convention)")
    else:
        print("variant: per figure -- " + ", ".join(f"{k} {v}" for k, v in PUBLISHED_VARIANT.items()))

    variant = chosen("loadings")
    paths, loadings = figure_loadings(variant_frame(variant, RICHER), variant)
    print("\n[fig:load] unrotated loadings, expanded set")
    print(loadings.to_string(index=False, float_format=lambda v: f"{v: .3f}"))
    for path in paths:
        print(f"  wrote {path}")

    variant = chosen("gssdrift")
    paths, dat_paths, angles = figure_gssdrift(variant_frame(variant, RICHER), variant)
    print("\n[fig:gss] 40-meeting rolling rotation angle")
    for stem, info in angles.items():
        print(f"  {stem:9s} windows={info['windows']:3d} "
              f"t in [{info['first_t']:.3f}, {info['last_t']:.3f}] "
              f"angle in [{info['min_angle']:.2f}, {info['max_angle']:.2f}] degrees")
        print(f"            window span: median {info['median_span_years']:.1f} yr, "
              f"max {info['max_span_years']:.1f} yr; "
              f"{info['windows_spanning_over_10y']} of {info['windows']} exceed 10 yr "
              f"(they reach back across a dropped zero-lower-bound block)")
    for path in paths + dat_paths:
        print(f"  wrote {path}")

    variant = chosen("window1")
    paths, events = figure_window1(variant_frame(variant, RICHER), variant)
    print("\n[fig:window1] event-window residuals vs the event's own quadratic in horizon")
    for event in events:
        if not event["present"]:
            print(f"  {event['date']}: NOT IN SAMPLE under this variant")
            continue
        print(f"  {event['date']}: largest residual {event['worst']} "
              f"{event['worst_bp']:+.2f} bp, next largest {event['runner_up']} "
              f"{event['runner_up_bp']:+.2f} bp")
    for path in paths:
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
