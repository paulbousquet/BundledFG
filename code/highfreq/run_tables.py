"""LaTeX tables for the high-frequency appendix.

Writes Appendix Table 1 (rotation instability) and Appendix Table 2 (raw PC loadings)
for every sample/data variant in `common.VARIANTS`, plus Appendix Table 4, the S&P 500
return regressions on the principal components.

    python code/highfreq/run_tables.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from common import (
    ORIGINAL,
    OUTPUT_DIR,
    RICHER,
    SPLIT_DATE,
    VARIANTS,
    fmt,
    pca,
    rotation,
    sample,
    variant_frame,
)

# Table 4 is reported for one convention only: the expanded contract set on
# untransformed data with 2009-2014 dropped.
SP500_VARIANT = "untransformed_exclude_2009_2014"
SP500_SCALE_CONTRACT = "ED4"
SP500_TRIM = 2  # observations trimmed from each tail of the S&P 500 return

SETS = {
    "original": dict(cols=ORIGINAL, require_scheduled=True, label="Original GSS"),
    "richer": dict(cols=RICHER, require_scheduled=False, label="Expanded Set"),
}
SET_ORDER = ["original", "richer"]


# --------------------------------------------------------------------------- #
# Variant-dependent note text
# --------------------------------------------------------------------------- #

def data_word(variant: str) -> str:
    return "adjusted" if VARIANTS[variant]["adjusted"] else "untransformed"


def convention_note(variant: str) -> str:
    """Sentence naming the data convention and, if it applies, the dropped window."""
    if VARIANTS[variant]["adjusted"]:
        return ""
    if VARIANTS[variant]["exclude_2009_2014"]:
        return (
            " Untransformed source data are used, and all observations dated "
            "2009--2014 are excluded."
        )
    return " Untransformed source data are used."


def exclusion_note(variant: str) -> str:
    """Just the dropped-window sentence, for notes that name the data convention inline."""
    if VARIANTS[variant]["exclude_2009_2014"]:
        return " All observations dated 2009--2014 are excluded."
    return ""


# --------------------------------------------------------------------------- #
# Appendix Table 1: rotation instability
# --------------------------------------------------------------------------- #

def rotation_rows(variant: str) -> list[str]:
    exclude = bool(VARIANTS[variant]["exclude_2009_2014"])
    rows: list[str] = []
    for i, name in enumerate(SET_ORDER):
        spec = SETS[name]
        cols = list(spec["cols"])
        data = sample(
            variant_frame(variant, cols), cols,
            require_scheduled=bool(spec["require_scheduled"]),
            exclude_2009_2014=exclude,
        )
        halves = {
            "pre": data.loc[data["date"] < pd.Timestamp(SPLIT_DATE)],
            "post": data.loc[data["date"] >= pd.Timestamp(SPLIT_DATE)],
        }
        if i:
            rows.append("\\midrule")
        for period, block in halves.items():
            rot = rotation(block, cols)
            rows.append(
                f"{spec['label']} & {period} & {rot['angle']:.1f}$^\\circ$ & "
                f"{fmt(rot['target'][-1])} \\\\"
            )
    return rows


def rotation_table(variant: str) -> str:
    far = f"{ORIGINAL[-1]}/{RICHER[-1]}"
    note = (
        f"Scheduled FOMC meetings for Original GSS; {data_word(variant)} FF2+ED2--ED8 "
        f"event sample for Expanded Set. Far loading is {far}.{convention_note(variant)}"
    )
    lines = [
        "\\begin{tabular}{llcc}",
        "\\toprule",
        "Set & Period & Rotation angle & Target far loading \\\\",
        "\\midrule",
        *rotation_rows(variant),
        "\\midrule",
        "\\multicolumn{4}{p{0.62\\linewidth}}{\\footnotesize " + note + "}\\\\",
        "\\bottomrule",
        "\\end{tabular}",
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Appendix Table 2: raw PC loadings before and after 2009
# --------------------------------------------------------------------------- #

def loading_panel(variant: str, name: str, panel: str, title: str) -> str:
    spec = SETS[name]
    cols = list(spec["cols"])
    data = sample(
        variant_frame(variant, cols), cols,
        require_scheduled=bool(spec["require_scheduled"]),
        exclude_2009_2014=bool(VARIANTS[variant]["exclude_2009_2014"]),
    )
    blocks = {
        "Full": data,
        "pre-2009": data.loc[data["date"] < pd.Timestamp(SPLIT_DATE)],
        "2009+": data.loc[data["date"] >= pd.Timestamp(SPLIT_DATE)],
    }
    V = {label: pca(block[cols].to_numpy(float))[0] for label, block in blocks.items()}

    body: list[str] = []
    for k, pc in enumerate(["PC1", "PC2"]):
        if k:
            body.append("\\addlinespace")
        for label in blocks:
            cells = " & ".join(fmt(v) for v in V[label][:, k])
            body.append(f"{pc} & {label} & {cells} \\\\")

    ncols = len(cols) + 2
    lines = [
        "\\begin{tabular}{@{}ll" + "r" * len(cols) + "@{}}",
        "\\toprule",
        f"\\multicolumn{{{ncols}}}{{l}}{{Panel {panel}: {title}}} \\\\",
        "\\midrule",
        "Component & Sample & " + " & ".join(cols) + " \\\\",
        "\\midrule",
        *body,
        "\\bottomrule",
        "\\end{tabular}",
    ]
    return "\n".join(lines)


def minipage(note: str) -> str:
    return (
        "\\smallskip\n\\begin{minipage}{0.82\\linewidth}\\footnotesize "
        + note
        + "\\end{minipage}"
    )


def stability_table(variant: str) -> str:
    note_a = (
        "Loadings from standardized PCA of MP1, MP2, and ED2--ED4. The sample is "
        "scheduled FOMC meetings since 1994 with $|\\Delta ED2|>1$bp."
        f"{convention_note(variant)}"
    )
    note_b = (
        f"Loadings from standardized PCA of the {data_word(variant)} FF2+ED2--ED8 event "
        "sample. PC1 is signed so positive denotes tightening; PC2 is signed so the "
        f"near/current contract loads positively.{exclusion_note(variant)}"
    )
    parts = [
        loading_panel(variant, "original", "A", "Original GSS set"),
        "",
        minipage(note_a),
        "",
        "\\medskip",
        "",
        loading_panel(variant, "richer", "B", "Expanded set"),
        "",
        minipage(note_b),
    ]
    return "\n".join(parts) + "\n"


# --------------------------------------------------------------------------- #
# Appendix Table 4: S&P 500 returns on the principal components
# --------------------------------------------------------------------------- #

def sp500_scores(data: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, float]:
    """PC1 and PC2 scores rescaled so one unit of PC1 is one basis point of ED4.

    `pca` standardizes each contract, so a one-unit move in the raw PC1 score moves
    ED4 by `V[ED4, 0] * sd(ED4)` in percentage points, i.e. 100 times that in basis
    points. Call that constant `bp_per_unit`. Multiplying the raw scores by it puts
    PC1 in basis points of ED4; PC2 is multiplied by the same constant, so the two
    columns stay on a common scale rather than each being normalized separately.
    """
    X = data[cols].to_numpy(float)
    V, _, _ = pca(X)
    sd = X.std(0, ddof=0)
    raw = ((X - X.mean(0)) / sd) @ V[:, :2]
    j = cols.index(SP500_SCALE_CONTRACT)
    bp_per_unit = float(V[j, 0] * sd[j] * 100)
    return raw * bp_per_unit, bp_per_unit


def trim_tails(y: np.ndarray, k: int) -> np.ndarray:
    """Boolean mask dropping the `k` largest and `k` smallest values of `y`."""
    order = np.argsort(y, kind="stable")
    keep = np.ones(len(y), bool)
    keep[order[:k]] = False
    keep[order[len(y) - k:]] = False
    return keep


def stars(pvalue: float) -> str:
    for cut, mark in [(0.01, "***"), (0.05, "**"), (0.10, "*")]:
        if pvalue < cut:
            return mark
    return ""


def sp500_regressions() -> tuple[str, dict[str, object]]:
    """Fit the two S&P 500 regressions and render Appendix Table 4."""
    df = variant_frame(SP500_VARIANT)
    cols = list(RICHER)
    data = sample(
        df, cols, require_scheduled=False,
        exclude_2009_2014=bool(VARIANTS[SP500_VARIANT]["exclude_2009_2014"]),
    )
    data = data.loc[data["SP500"].notna()].reset_index(drop=True)

    # The components are estimated on the full event sample; the trim applies only to
    # the regression sample, as the paper's note describes it.
    scores, bp_per_unit = sp500_scores(data, cols)
    y = data["SP500"].to_numpy(float)
    keep = trim_tails(y, SP500_TRIM)

    fits = []
    for k in (1, 2):
        X = sm.add_constant(scores[keep, :k], has_constant="add")
        fits.append(sm.OLS(y[keep], X).fit(cov_type="HC1"))

    def cell(fit, j: int) -> tuple[str, str]:
        b, se, p = fit.params[j], fit.bse[j], fit.pvalues[j]
        return f"${b:.3f}^{{{stars(p)}}}$" if stars(p) else f"${b:.3f}$", f"$({se:.3f})$"

    pc1 = [cell(fit, 1) for fit in fits]
    pc2 = cell(fits[1], 2)
    note = (
        "\\textit{Notes:} PC1 is scaled so that a unit increase corresponds to a one "
        "basis-point increase in ED4. PC2 is scaled by the same factor. The sample "
        "excludes observations dated 2009--2014 and trims the two largest and two "
        "smallest S\\&P 500 returns. All regressions include a constant. HC1 standard "
        "errors are in parentheses. $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$."
    )
    lines = [
        "\\begin{tabular}{@{}lcc@{}}",
        "\\toprule",
        "& (1) & (2) \\\\",
        "\\midrule",
        f"PC1  & {pc1[0][0]} & {pc1[1][0]} \\\\",
        f"& {pc1[0][1]} & {pc1[1][1]} \\\\",
        "\\addlinespace",
        f"PC2  & & {pc2[0]} \\\\",
        f"& & {pc2[1]} \\\\",
        "\\midrule",
        "Observations & " + " & ".join(f"{int(fit.nobs)}" for fit in fits) + " \\\\",
        "$R^2$ & " + " & ".join(f"{fit.rsquared:.3f}" for fit in fits) + " \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "",
        "\\smallskip",
        "\\begin{minipage}{0.72\\linewidth}\\footnotesize",
        note,
        "\\end{minipage}",
    ]
    summary = {
        "n_events": int(len(data)),
        "n_regression": int(keep.sum()),
        "bp_per_unit": bp_per_unit,
        "fits": fits,
    }
    return "\n".join(lines) + "\n", summary


# --------------------------------------------------------------------------- #

def write(path: Path, text: str) -> None:
    """Write with LF endings on every platform, so the .tex artifacts stay diffable."""
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def main() -> None:
    tables_dir = OUTPUT_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    for variant in VARIANTS:
        for stem, text in [
            (f"tab_section2_rotation_instability_{variant}", rotation_table(variant)),
            (f"tab_section2_raw_pc_stability_{variant}", stability_table(variant)),
        ]:
            path = tables_dir / f"{stem}.tex"
            write(path, text)
            print(f"wrote {path.relative_to(OUTPUT_DIR.parent)}")

    table4, summary = sp500_regressions()
    path = tables_dir / "tab_sp500_pc_regressions.tex"
    write(path, table4)
    print(f"wrote {path.relative_to(OUTPUT_DIR.parent)}")

    print(
        f"\nS&P 500 regressions: {summary['n_events']} events before trimming, "
        f"{summary['n_regression']} after trimming {SP500_TRIM} from each tail"
    )
    print(
        f"  scaling constant: one unit of the raw PC1 score = "
        f"{summary['bp_per_unit']:.4f} bp of {SP500_SCALE_CONTRACT}; "
        "scores multiplied by it"
    )
    for k, fit in enumerate(summary["fits"], start=1):
        terms = " ".join(
            f"PC{j}={fit.params[j]:.4f} (se {fit.bse[j]:.4f})" for j in range(1, k + 1)
        )
        print(f"  column ({k}): {terms}  R^2={fit.rsquared:.4f}  n={int(fit.nobs)}")


if __name__ == "__main__":
    main()
