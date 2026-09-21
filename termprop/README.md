# termprop

Offline dashboard for the Eurodollar strip (ED2 to ED8) and the second fed funds futures
contract (FF2) at each FOMC announcement. Each profile is normalized to ED2 = 12.5 bp and
fitted with a quadratic in the horizon index; FF2 is shown and fitted only when switched on.
Coefficients are summarized as histograms and a joint scatter over the selected range.

## Run

Open `termprop/index.html` in a browser, by double-clicking it or with File > Open. No
server, packages, or network connection is needed.

If your browser blocks storage for local files, serve the folder instead and open the
printed address:

```
cd termprop
python -m http.server 8000
```

## Data

The page embeds `data/ed_quadratic_fits.csv` with `Modified` set equal to `Base`, so it
carries the source values for FF2 and ED2 to ED8 and none of the corrections used by the
paper's *adjusted* variant. 178 announcements from 1994-02-04 through the 2026-09-19 snapshot; sample filter in
`data/README.md`. After refreshing the snapshot, re-embed with:

```
python termprop/build_index.py
```

## Controls

| Control | Effect |
|---|---|
| Range / Date | Overlay all announcements in the year range, or show one with its fit. |
| Year range | Free text such as `1993 onward` or `2004-2007`; sets the years shown in Range mode. |
| Meeting date, `<` `>` | Step through announcements in Date mode. |
| Reset date to Base | Drop your adjustments for the selected announcement. |
| Reset all dates to Base | Drop every adjustment saved in this browser. |
| Actual realized units | Raw, un-normalized changes, read-only. |
| Include FF2 | Add FF2 as horizon 0 and refit on eight points. |
| Overlay GSS PC fit, PC1 only | Principal-component fit on the active sample, two components or one. |
| Joint: midpoint slope | Joint-scatter y-axis becomes the midpoint slope instead of `b`. |
| Exclude ZLB | Drop 2009-01-01 to 2015-12-31 and 2020-01-01 to 2021-12-31 from the chart, date list, histograms, and PC fit. |

The ZLB windows are `ZLB_WINDOWS` in `index.html`. The paper's tables cut 2009 to 2014.

## Adjustments

In Date mode, drag a Modified point; the fit updates and the values save to this browser's
`localStorage` (key `termprop-static-adjustments-v1`). They never leave the browser and are
keyed by origin, so `file://` copies share them. If storage is blocked, edits show for the
session but saving reports an error.

## Files

- `index.html`: the dashboard, single file.
- `build_index.py`: re-embeds `../data/ed_quadratic_fits.csv` with `Modified` reset to `Base`.
