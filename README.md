# BundledFG

Replication package for "[Estimating Impulse Responses to Bundled Forward Guidance](https://pbousquet.com/assets/BundledFG.pdf)". See the Online Appendix at `BundledOnline.pdf`. 

## Requirements

Python 3.10 or later.

```
pip install -r requirements.txt
```

MATLAB is needed only to re-run the model estimation behind the Online Appendix results
(see the last section). Its saved output is included, so every figure and table builds
without it.

## Run

A runner command for the entire suite is 

```
python code/highfreq/run_factor_stats.py
python code/highfreq/run_tables.py
python code/highfreq/run_figures.py
python code/irfmatching/make_params_table.py
python code/irfmatching/make_gradient_figure.py
python code/irfmatching/make_losscurve_figure.py
```

Each finishes in seconds. Output goes to `output/figures`, `output/tables`, and
`output/stats`.

## Exhibits

The termprop folder generates a dashboard to visualize each individual announcement. The remainder of the repo is for replicating the main figures in the paper. 

| In the paper | Script | Output |
|---|---|---|
| Section 3, variance shares quoted in the text (85.5%, 96.3%) | `run_factor_stats.py` | `output/stats/factor_stats.json` |
| Section 3 figure, PC loadings panel | `run_figures.py` | `output/figures/loadings.pdf` |
| Section 3 figure, rotation-angle panel | `run_figures.py` | `output/figures/gssdrift.pdf` |
| Online Appendix Table 1, rotation angle and target loading | `run_tables.py` | `output/tables/tab_section2_rotation_instability_*.tex` |
| Online Appendix Table 2, raw PC1 and PC2 loadings | `run_tables.py` | `output/tables/tab_section2_raw_pc_stability_*.tex` |
| Online Appendix Table 3, variance shares vs. quadratic | `run_factor_stats.py` | `output/tables/tab_section2_edstrip_fit_*.tex` |
| Online Appendix Table 4, S&P 500 returns on components | `run_tables.py` | `output/tables/tab_sp500_pc_regressions.tex` |
| Online Appendix figure, two announcements with an off-curve contract | `run_figures.py` | `output/figures/window1.pdf` |
| Online Appendix table of estimated parameters | `make_params_table.py` | `output/tables/tab_params.tex` |
| Online Appendix figure, loss curvature by parameter | `make_gradient_figure.py` | `output/figures/gradient.pdf` |
| Online Appendix figure, loss after refit as `m_d` moves | `make_losscurve_figure.py` | `output/figures/losscurve.pdf` |

Tables 1 to 3 are produced under three sample and data conventions; the file suffix names
the convention, and the `untransformed_exclude_2009_2014` file is the one in the paper.
`run_figures.py --variant <name>` rebuilds the figures under one convention.

## Data

Two input files, described in [`data/README.md`](data/README.md): announcement-window
futures surprises and the per-contract corrections applied to FF2 and the Eurodollar strip. The
corrections were made with the dashboard in [`termprop/`](termprop/README.md).

## Re-running the estimation (MATLAB)

The model is the behavioral New Keynesian model of Caravello, McKay and Wolf. Their
replication package is third-party and not included.

```
git clone https://github.com/tcaravello/mp_modelcnfctls.git
cd mp_modelcnfctls
git apply ../BundledFG/code/irfmatching/mp_modelcnfctls_pcpath.patch
```

Then in MATLAB (R2024a), with `MP_MODELCNFCTLS` set to that checkout:

```
profile_md([-0.10:0.01:-0.01, -0.005, 0.005, 0.01:0.01:0.10])
```

That grid is the one behind the vendored `data/irfmatching/profile_md.csv` and
`profile_md_jacobians.mat`. The drivers are in `code/irfmatching/matlab/`.
