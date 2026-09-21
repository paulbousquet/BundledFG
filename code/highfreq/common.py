"""Shared data layer for the high-frequency factor-model exercises.

Every Part A script imports from here. The sample filters, PCA normalization, and
GSS rotation defined below are the locked contract: change them and every table and
figure in the paper moves.

Ported from mindcraft1997/RelDim `stability_figures/sim/section2_fresh_artifacts.py`,
which produced the .dat and .tex artifacts the paper currently inputs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Contract sets
# --------------------------------------------------------------------------- #

# Gurkaynak, Sack & Swanson (2005): current- and next-meeting fed funds target
# surprises plus the 2nd-4th quarterly Eurodollar futures.
ORIGINAL = ["MP1", "MP2", "ED2", "ED3", "ED4"]

# The expanded set used in the main analysis: the unscaled 2nd fed funds futures
# contract plus the Eurodollar strip out to 7 quarters ahead.
RICHER = ["FF2", "ED2", "ED3", "ED4", "ED5", "ED6", "ED7", "ED8"]

# The Eurodollar strip alone, used for the event-by-event quadratic comparison.
ED_STRIP = ["ED2", "ED3", "ED4", "ED5", "ED6", "ED7", "ED8"]

# Zero-lower-bound months, excluded from the rolling-window rotation estimates.
ZLB_WINDOWS = [("2008-11-01", "2015-10-31"), ("2020-03-01", "2022-01-31")]

# The window the "exclude_2009_2014" variant drops.
EXCLUDE_WINDOW = ("2009-01-01", "2014-12-31")

SAMPLE_START = "1994-01-01"
ED2_THRESHOLD = 0.01  # drop meetings whose ED2 surprise is under one basis point
SPLIT_DATE = "2009-01-01"
ROLLING_WINDOW = 40  # meetings

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "output"


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def year_fraction(ts: pd.Timestamp) -> float:
    """Decimal year used as the x axis of the rolling-window figures."""
    return float(ts.year + (ts.month - 1) / 12)


def is_zlb(date: pd.Series) -> pd.Series:
    out = pd.Series(False, index=date.index)
    for start, end in ZLB_WINDOWS:
        out |= date.between(pd.Timestamp(start), pd.Timestamp(end))
    return out


def load_source(path: Path | str | None = None) -> pd.DataFrame:
    """Read the announcement-window futures surprises.

    Columns of interest are in percentage points. `main` flags scheduled meetings and
    `Unscheduled` flags intermeeting moves.
    """
    path = Path(path) if path is not None else DATA_DIR / "jk_source_old.csv"
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="coerce")
    for col in set(ORIGINAL + RICHER + ["main", "Unscheduled", "SP500"]):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["zlb"] = is_zlb(df["date"])
    return df


def apply_adjustments(df: pd.DataFrame, fits_path: Path | str | None = None) -> pd.DataFrame:
    """Overwrite strip observations that the termprop dashboard flags as mismeasured.

    The dashboard stores a `Modified` fitted value on a 12.5 scale; the raw-unit
    replacement is `Modified / 12.5 * SourceED2`. Only rows where `Modified` differs
    from `Base` are substituted.
    """
    fits_path = Path(fits_path) if fits_path is not None else DATA_DIR / "ed_quadratic_fits.csv"
    out = df.copy()
    fits = pd.read_csv(fits_path)
    for col in ["Modified", "Base", "SourceED2"]:
        fits[col] = pd.to_numeric(fits[col], errors="coerce")
    mods = fits[(fits["Modified"] - fits["Base"]).abs() > 1e-9].copy()
    mods["date"] = pd.to_datetime(mods["Date"], errors="coerce")
    mods["raw_mod"] = mods["Modified"] * mods["SourceED2"] / 12.5
    for _, row in mods.iterrows():
        if row["Horizon"] in out.columns:
            out.loc[out["date"].eq(row["date"]), row["Horizon"]] = row["raw_mod"]
    return out


def sample(
    df: pd.DataFrame,
    cols: list[str],
    *,
    exclude_zlb: bool = False,
    require_scheduled: bool = True,
    exclude_2009_2014: bool = False,
) -> pd.DataFrame:
    """Apply the paper's sample filters and return the rows sorted by date.

    Base filters: on or after 1994, all `cols` observed, |ED2| above one basis point.
    """
    keep = (
        df["date"].ge(SAMPLE_START)
        & df[cols].notna().all(axis=1)
        & df["ED2"].abs().gt(ED2_THRESHOLD)
    )
    if require_scheduled:
        keep &= df["main"].eq(1) & df["Unscheduled"].fillna(0).eq(0)
    if exclude_zlb:
        keep &= ~df["zlb"]
    if exclude_2009_2014:
        start, end = EXCLUDE_WINDOW
        keep &= ~df["date"].between(pd.Timestamp(start), pd.Timestamp(end))
    return df.loc[keep].sort_values("date").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Factor model
# --------------------------------------------------------------------------- #

def pca(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardized principal components.

    Returns (V, eig, evr): unit-norm loading vectors in columns, eigenvalues, and
    explained-variance ratios. Signs are fixed so PC1 sums positive (a tightening
    is positive) and PC2 loads positively on the front contract.
    """
    X = np.asarray(X, float)
    Xs = (X - X.mean(0)) / X.std(0, ddof=0)
    _, S, Vt = np.linalg.svd(Xs, full_matrices=False)
    V = Vt.T.copy()
    if V[:, 0].sum() < 0:
        V[:, 0] *= -1
    if V[0, 1] < 0:
        V[:, 1] *= -1
    eig = S**2 / (len(Xs) - 1)
    evr = S**2 / (S**2).sum()
    return V, eig, evr


def rotation(block: pd.DataFrame, cols: list[str]) -> dict[str, object]:
    """GSS target/path rotation of the first two components.

    The target factor is the direction aligned with the front contract's position in
    the first two scaled loadings; the path factor is orthogonal to it, so it has
    zero loading on the front contract. `angle` is that alignment in degrees.
    """
    V, eig, evr = pca(block[cols].to_numpy(float))
    loadings = V[:, :2] * np.sqrt(eig[:2])
    anchor = loadings[0]
    angle = float(np.degrees(np.arctan2(anchor[1], anchor[0])))
    u0 = anchor / np.linalg.norm(anchor)
    u1 = np.array([-u0[1], u0[0]])
    rotated = loadings @ np.column_stack([u0, u1])
    if rotated[-1, 1] < 0:
        rotated[:, 1] *= -1
    return {
        "angle": angle,
        "target": rotated[:, 0],
        "path": rotated[:, 1],
        "pc1": V[:, 0] / np.linalg.norm(V[:, 0]),
        "pc2": V[:, 1] / np.linalg.norm(V[:, 1]),
        "evr": evr,
    }


def rolling_rotation(
    data: pd.DataFrame,
    cols: list[str],
    *,
    require_scheduled: bool,
    exclude_2009_2014: bool = False,
    window: int = ROLLING_WINDOW,
) -> list[dict[str, float]]:
    """Rotation angle on consecutive `window`-meeting blocks, ZLB meetings dropped."""
    data = sample(
        data,
        cols,
        exclude_zlb=True,
        require_scheduled=require_scheduled,
        exclude_2009_2014=exclude_2009_2014,
    )
    rows = []
    for end in range(window, len(data) + 1):
        block = data.iloc[end - window : end]
        rot = rotation(block, cols)
        rows.append(
            {
                "t": year_fraction(block["date"].iloc[-1]),
                "angle": rot["angle"],
                "start": year_fraction(block["date"].iloc[0]),
                "end": year_fraction(block["date"].iloc[-1]),
            }
        )
    return rows


def insert_gaps(rows: list[dict[str, float]], max_gap: float = 1.0) -> pd.DataFrame:
    """Break the plotted line where the rolling windows skip more than `max_gap` years."""
    plot_rows: list[dict[str, float]] = []
    for i, row in enumerate(rows):
        if i and row["t"] - rows[i - 1]["t"] > max_gap:
            plot_rows.append({"t": (row["t"] + rows[i - 1]["t"]) / 2, "angle": np.nan})
        plot_rows.append({"t": row["t"], "angle": row["angle"]})
    return pd.DataFrame(plot_rows)


# --------------------------------------------------------------------------- #
# Reconstruction fit
# --------------------------------------------------------------------------- #

def pooled_r2(actual: np.ndarray, fitted: np.ndarray) -> float:
    """Share of pooled standardized variation captured, across all contracts at once."""
    y = np.asarray(actual, float).reshape(-1)
    yhat = np.asarray(fitted, float).reshape(-1)
    ssr = float(np.sum((y - yhat) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    return 1 - ssr / sst


def pc_subset_reconstruction(X_all: np.ndarray, fit_indices: list[int], k: int) -> np.ndarray:
    """Rank-k PC reconstruction fit on the full contract set, read off `fit_indices`."""
    Xs = (X_all - X_all.mean(0)) / X_all.std(0, ddof=0)
    _, _, Vt = np.linalg.svd(Xs, full_matrices=False)
    V = Vt[:k].T
    return (Xs @ V @ V.T)[:, fit_indices]


def quadratic_reconstruction(X: np.ndarray) -> np.ndarray:
    """Event-by-event quadratic in horizon: three free parameters per announcement."""
    horizons = np.arange(X.shape[1], dtype=float)
    design = np.column_stack([np.ones_like(horizons), horizons, horizons**2])
    hat = design @ np.linalg.inv(design.T @ design) @ design.T
    return np.asarray(X, float) @ hat.T


# --------------------------------------------------------------------------- #
# Variants
# --------------------------------------------------------------------------- #

# The paper's exhibits exist under three sample/data conventions. See README.
VARIANTS = {
    "full_adjusted": dict(adjusted=True, exclude_2009_2014=False),
    "untransformed_exclude_2009_2014": dict(adjusted=False, exclude_2009_2014=True),
    "full_untransformed": dict(adjusted=False, exclude_2009_2014=False),
}


def variant_frame(
    variant: str,
    cols: list[str] | None = None,
    data_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Load the source data with the adjustment convention this variant calls for.

    The dashboard corrections are applied to the expanded set only. The original GSS set is
    always read raw: the dashboard fits the FF2+ED2-ED8 strip, and the published tables treat
    the original set as untouched by it. Pass `cols` to get that scoping; passing ORIGINAL
    returns the raw frame under every variant.
    """
    if variant not in VARIANTS:
        raise KeyError(f"unknown variant {variant!r}; expected one of {sorted(VARIANTS)}")
    data_dir = Path(data_dir) if data_dir is not None else DATA_DIR
    df = load_source(data_dir / "jk_source_old.csv")
    if VARIANTS[variant]["adjusted"] and list(cols or RICHER) != ORIGINAL:
        df = apply_adjustments(df, data_dir / "ed_quadratic_fits.csv")
    return df


def fmt(x: float) -> str:
    return f"{x:.2f}"


def pct(x: float) -> str:
    return f"{100 * x:.1f}\\%"
