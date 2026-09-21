"""Appendix B Table B.1: the nine estimated parameters and the values used to generate the data.

The values are read from `param0` in `data/irfmatching/profile_md_jacobians.mat` rather than
retyped, so the table cannot drift away from what the estimation actually ran at. `free9` in that
file lists the nine estimated parameters as 1-based indices into the ten names; the tenth,
`psi_u`, is held fixed and is reported in the note rather than the body.

Run from the repository root:

    python code/irfmatching/make_params_table.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat

REPO_ROOT = Path(__file__).resolve().parents[2]
JACOBIAN_FILE = REPO_ROOT / "data" / "irfmatching" / "profile_md_jacobians.mat"
TABLES_DIR = REPO_ROOT / "output" / "tables"

# Symbol and description for each name stored in the .mat file.
PARAMETERS: dict[str, tuple[str, str]] = {
    "h": (r"$h$", "habit in consumption"),
    "xi_p": (r"$\xi_p$", "Calvo price parameter"),
    "iota_p": (r"$\iota_p$", "price indexation parameter"),
    "xi_w": (r"$\xi_w$", "Calvo wage parameter"),
    "iota_w": (r"$\iota_w$", "wage indexation parameter"),
    "kappa": (r"$\kappa$", "investment adjustment cost"),
    "psi_u": (r"$\psi_u$", "capacity utilization cost"),
    "m_d": (r"$m_d$", "cognitive discounting, households"),
    "m_f": (r"$m_f$", "cognitive discounting, price and wage setters"),
    "varphi": (r"$\varphi$", "Frisch elasticity"),
}

# The order the paper prints, which puts the Frisch elasticity before the two discount factors.
PRINT_ORDER = ["h", "xi_p", "iota_p", "xi_w", "iota_w", "kappa", "varphi", "m_d", "m_f"]


def trim(value: float) -> str:
    """Drop trailing zeros so 0.750 prints as 0.75 and 5.500 as 5.5."""
    return f"{value:.3f}".rstrip("0").rstrip(".")


def main() -> None:
    saved = loadmat(JACOBIAN_FILE, simplify_cells=True)
    names = [str(n) for n in np.atleast_1d(saved["names"])]
    values = dict(zip(names, np.asarray(saved["param0"], float).ravel()))
    free = [names[int(j) - 1] for j in np.atleast_1d(saved["free9"])]

    missing = [n for n in PRINT_ORDER if n not in free]
    if missing:
        raise SystemExit(f"expected these to be estimated but they are not free: {missing}")
    extra = [n for n in free if n not in PRINT_ORDER]
    if extra:
        raise SystemExit(f"estimated parameters absent from PRINT_ORDER: {extra}")

    fixed = [n for n in names if n not in free]
    fixed_note = ", ".join(f"{PARAMETERS[n][0]} is held at {trim(values[n])}" for n in fixed)

    lines = [
        r"\begin{tabular}{@{}llc@{}}",
        r"\toprule",
        r"Parameter & Description & Value \\",
        r"\midrule",
    ]
    for name in PRINT_ORDER:
        symbol, description = PARAMETERS[name]
        lines.append(f"{symbol} & {description} & {trim(values[name])} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        "",
        r"\smallskip",
        r"\begin{minipage}{0.72\linewidth}\footnotesize",
        r"\textit{Notes:} Values used to generate the data, read from the saved estimation "
        r"inputs. " + fixed_note + ". Remaining parameters and the solution method are taken "
        r"from the replication codes of Caravello, McKay and Wolf.",
        r"\end{minipage}",
    ]

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLES_DIR / "tab_params.tex"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(REPO_ROOT)}")
    for name in PRINT_ORDER:
        print(f"  {name:8s} {trim(values[name])}")
    for name in fixed:
        print(f"  {name:8s} {trim(values[name])}  (fixed, not estimated)")


if __name__ == "__main__":
    main()
