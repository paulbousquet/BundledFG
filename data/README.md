# Data

Two inputs. Both are vendored snapshots. The upstream sources are live and move, so the
numbers in the paper are reproducible only against a pinned vintage.

## `jk_source_old.csv`

Announcement-window changes in interest-rate futures and asset prices around FOMC
announcements. One row per announcement, 385 rows, first observation February 1988.

- **Upstream:** <https://raw.githubusercontent.com/paulbousquet/GBMPSurprise/main/jk_source_old.csv>
- **Snapshot taken:** 2026-09-19
- **Units:** percentage points. A value of `0.25` in `ED4` is a 25 basis point move.

Columns this repository uses:

| Column | Meaning |
|---|---|
| `Date` | announcement date, `M/D/YYYY` |
| `main` | 1 for a scheduled FOMC meeting |
| `Unscheduled` | 1 for an intermeeting move |
| `MP1`, `MP2` | current- and next-meeting fed funds target surprises (scaled) |
| `FF2` | second fed funds futures contract, unscaled |
| `ED2`–`ED8` | Eurodollar futures, 1 through 7 quarters ahead |
| `SP500` | announcement-window S&P 500 return |

**Vintage matters.** The file grows as new meetings are added. The appendix tables built on
the *adjusted* data were produced from a June 2026 vintage and sit about 0.1 percentage points
away from what the September 2026 vintage gives (86.3% vs 86.2% for the PC1 variance share).
The *untransformed* tables were produced in September 2026 and reproduce exactly.

## `ed_quadratic_fits.csv`

Per-event, per-contract fitted values from the term-structure dashboard in `termprop/`, used
to overwrite strip observations flagged as mismeasured.

- **Snapshot taken:** 2026-09-19
- Columns used: `Date`, `Horizon`, `Base`, `Modified`, `SourceED2`
- Only rows where `Modified` differs from `Base` are substituted. The raw-unit replacement is
  `Modified / 12.5 * SourceED2`; the dashboard stores `Modified` on a 12.5 scale.

## Sample definition

Every exercise in this repository uses the same base filters, in `code/highfreq/common.py`:

- date on or after 1994-01-01
- all contracts in the set observed
- `|ED2| > 0.01` (one basis point), which drops announcements with no meaningful policy news

The original GSS set additionally requires a scheduled meeting (`main == 1` and
`Unscheduled != 1`); the expanded set does not.
