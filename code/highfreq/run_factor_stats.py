"""Factor-model statistics for the high-frequency appendix.

Writes `output/stats/factor_stats.json` with the rotation, loading, and fit numbers
behind Appendix Tables 1-3, for every sample/data variant in `common.VARIANTS`, and
writes the Appendix Table 3 variance-share table for each variant.

    python code/highfreq/run_factor_stats.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from common import (
    ED_STRIP,
    ORIGINAL,
    OUTPUT_DIR,
    RICHER,
    ROLLING_WINDOW,
    SPLIT_DATE,
    VARIANTS,
    pc_subset_reconstruction,
    pca,
    pct,
    pooled_r2,
    quadratic_reconstruction,
    rolling_rotation,
    rotation,
    sample,
    variant_frame,
)

# Each contract set carries its own sample filter (the original GSS set is scheduled
# meetings only) and its own strip columns for the reconstruction-fit rows.
SETS: dict[str, dict[str, object]] = {
    "original": dict(
        cols=ORIGINAL,
        require_scheduled=True,
        label="Original GSS",
        fit_cols=["ED2", "ED3", "ED4"],
    ),
    "richer": dict(
        cols=RICHER,
        require_scheduled=False,
        label="Expanded Set",
        fit_cols=list(ED_STRIP),
    ),
}

SET_ORDER = ["original", "richer"]


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #

def split_blocks(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Full sample plus the pre-2009 / 2009-onward halves used in Tables 1 and 2."""
    return {
        "full": data,
        "pre_2009": data.loc[data["date"] < pd.Timestamp(SPLIT_DATE)],
        "post_2009": data.loc[data["date"] >= pd.Timestamp(SPLIT_DATE)],
    }


def loadings(block: pd.DataFrame, cols: list[str]) -> dict[str, dict[str, float]]:
    """Unit-norm PC1 and PC2 loadings, keyed by contract name."""
    V, _, _ = pca(block[cols].to_numpy(float))
    return {
        "pc1": {c: float(V[i, 0]) for i, c in enumerate(cols)},
        "pc2": {c: float(V[i, 1]) for i, c in enumerate(cols)},
    }


def strip_fit(data: pd.DataFrame, cols: list[str], fit_cols: list[str]) -> dict[str, object]:
    """Variance shares and reconstruction fit over the Eurodollar strip.

    Two different quantities live here and they are not interchangeable. The
    `*_variance_share` entries are the full-contract-set explained-variance ratios,
    which is what Appendix Table 3 prints. The `*_pooled_r2` entries fit the PCA on
    the full contract set, read the rank-k reconstruction off the strip columns only,
    and compare it to the standardized strip. The quadratic is fit on the raw strip
    and standardized by the strip's own mean and sd before the comparison.
    """
    X_all = data[cols].to_numpy(float)
    idx = [cols.index(c) for c in fit_cols]
    _, _, evr = pca(X_all)

    standardized_all = (X_all - X_all.mean(0)) / X_all.std(0, ddof=0)
    actual = standardized_all[:, idx]

    strip = data[fit_cols].to_numpy(float)
    strip_mean, strip_sd = strip.mean(0), strip.std(0, ddof=0)
    strip_standardized = (strip - strip_mean) / strip_sd
    # Three free parameters per event, so a three-contract strip is exactly identified
    # and its fit is a tautological 1.0. Record nothing rather than a fit number.
    quad_r2 = None
    if len(fit_cols) > 3:
        quad = (quadratic_reconstruction(strip) - strip_mean) / strip_sd
        quad_r2 = float(pooled_r2(strip_standardized, quad))

    return {
        "columns": list(fit_cols),
        "pc1_variance_share": float(evr[0]),
        "pc1_pc2_variance_share": float(evr[:2].sum()),
        "pc1_pooled_r2": float(pooled_r2(actual, pc_subset_reconstruction(X_all, idx, 1))),
        "pc1_pc2_pooled_r2": float(pooled_r2(actual, pc_subset_reconstruction(X_all, idx, 2))),
        "quadratic_pooled_r2": quad_r2,
        "quadratic_exactly_identified": len(fit_cols) <= 3,
    }


def set_record(df: pd.DataFrame, spec: dict[str, object], *, exclude: bool) -> dict[str, object]:
    """Every statistic for one contract set under one variant."""
    cols = list(spec["cols"])
    scheduled = bool(spec["require_scheduled"])
    data = sample(df, cols, require_scheduled=scheduled, exclude_2009_2014=exclude)
    blocks = split_blocks(data)

    far = cols[-1]
    rot: dict[str, dict[str, object]] = {}
    for period, key in [("pre", "pre_2009"), ("post", "post_2009")]:
        r = rotation(blocks[key], cols)
        rot[period] = {
            "angle": float(r["angle"]),
            "far_contract": far,
            "far_loading": float(r["target"][-1]),
        }

    rolling = rolling_rotation(df, cols, require_scheduled=scheduled, exclude_2009_2014=exclude)
    angles = [row["angle"] for row in rolling]
    _, _, evr = pca(data[cols].to_numpy(float))

    return {
        "label": spec["label"],
        "contracts": cols,
        "require_scheduled": scheduled,
        "n": int(len(data)),
        "n_pre_2009": int(len(blocks["pre_2009"])),
        "n_post_2009": int(len(blocks["post_2009"])),
        "evr_pc1": float(evr[0]),
        "evr_pc1_pc2": float(evr[:2].sum()),
        "loadings": {name: loadings(block, cols) for name, block in blocks.items()},
        "rotation": rot,
        "rolling_rotation": {
            "window": ROLLING_WINDOW,
            "n_windows": int(len(rolling)),
            "angle_min": float(min(angles)) if angles else None,
            "angle_max": float(max(angles)) if angles else None,
        },
        "fit": strip_fit(data, cols, list(spec["fit_cols"])),
    }


def variant_record(variant: str) -> dict[str, object]:
    exclude = bool(VARIANTS[variant]["exclude_2009_2014"])
    return {
        "adjusted": bool(VARIANTS[variant]["adjusted"]),
        "exclude_2009_2014": exclude,
        "contract_sets": {
            name: set_record(
                variant_frame(variant, SETS[name]["cols"]), SETS[name], exclude=exclude
            )
            for name in SET_ORDER
        },
    }


# --------------------------------------------------------------------------- #
# Appendix Table 3
# --------------------------------------------------------------------------- #

def edstrip_note(variant: str) -> str:
    adjusted = VARIANTS[variant]["adjusted"]
    excluded = VARIANTS[variant]["exclude_2009_2014"]
    if excluded:
        lead = "Full sample outside 2009--2014, using untransformed source data."
    elif adjusted:
        lead = "Full sample."
    else:
        lead = "Full sample, using untransformed source data."
    return (
        f"{lead} PC rows are full-set standardized variance shares. Quadratic is fit "
        "separately by event over the Eurodollar strip; the Original GSS quadratic is "
        "exactly identified because ED2--ED4 gives three contracts for three parameters."
    )


def edstrip_table(variant: str, record: dict[str, object]) -> str:
    """Appendix Table 3: share of variance captured by the components and a quadratic."""
    sets = record["contract_sets"]
    fits = [sets[name]["fit"] for name in SET_ORDER]
    headers = " & ".join(str(sets[name]["label"]) for name in SET_ORDER)
    quad = [
        "---" if f["quadratic_pooled_r2"] is None else pct(f["quadratic_pooled_r2"]) for f in fits
    ]
    lines = [
        "\\begin{tabular}{lcc}",
        "\\toprule",
        f"Representation & {headers} \\\\",
        "\\midrule",
        "PC1 & " + " & ".join(pct(f["pc1_variance_share"]) for f in fits) + " \\\\",
        "PC1+PC2 & " + " & ".join(pct(f["pc1_pc2_variance_share"]) for f in fits) + " \\\\",
        "Quadratic & " + " & ".join(quad) + " \\\\",
        "\\midrule",
        "\\multicolumn{3}{p{0.72\\linewidth}}{\\footnotesize " + edstrip_note(variant) + "}\\\\",
        "\\bottomrule",
        "\\end{tabular}",
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def write(path: Path, text: str) -> None:
    """Write with LF endings on every platform, so the .tex artifacts stay diffable."""
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def summarize(variant: str, record: dict[str, object]) -> None:
    print(f"\n{variant}")
    for name in SET_ORDER:
        rec = record["contract_sets"][name]
        roll = rec["rolling_rotation"]
        fit = rec["fit"]
        print(
            f"  {rec['label']:<13} n={rec['n']} "
            f"(pre {rec['n_pre_2009']} / post {rec['n_post_2009']})  "
            f"full-set variance share: PC1 {100 * rec['evr_pc1']:.1f}%  "
            f"PC1+PC2 {100 * rec['evr_pc1_pc2']:.1f}%"
        )
        pre, post = rec["rotation"]["pre"], rec["rotation"]["post"]
        print(
            f"    rotation angle {pre['angle']:.1f} -> {post['angle']:.1f} deg; "
            f"{pre['far_contract']} target loading {pre['far_loading']:.2f} -> "
            f"{post['far_loading']:.2f}"
        )
        print(
            f"    rolling angle over {roll['n_windows']} windows of {roll['window']} "
            f"meetings: {roll['angle_min']:.1f} to {roll['angle_max']:.1f} deg"
        )
        quad = (
            "exactly identified"
            if fit["quadratic_pooled_r2"] is None
            else f"{100 * fit['quadratic_pooled_r2']:.1f}%"
        )
        print(
            f"    pooled R^2 over {'+'.join(fit['columns'])}: "
            f"PC1 {100 * fit['pc1_pooled_r2']:.1f}%  "
            f"PC1+PC2 {100 * fit['pc1_pc2_pooled_r2']:.1f}%  quadratic {quad}"
        )


def main() -> None:
    stats_dir = OUTPUT_DIR / "stats"
    tables_dir = OUTPUT_DIR / "tables"
    stats_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    records = {variant: variant_record(variant) for variant in VARIANTS}
    for variant, record in records.items():
        path = tables_dir / f"tab_section2_edstrip_fit_{variant}.tex"
        write(path, edstrip_table(variant, record))
        summarize(variant, record)
        print(f"    wrote {path.relative_to(OUTPUT_DIR.parent)}")

    out = stats_dir / "factor_stats.json"
    write(out, json.dumps({"variants": records}, indent=2) + "\n")
    print(f"\nwrote {out.relative_to(OUTPUT_DIR.parent)}")


if __name__ == "__main__":
    main()
