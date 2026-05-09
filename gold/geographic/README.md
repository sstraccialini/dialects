# Geographic gold matrix

A simple great-circle (Haversine) distance matrix between the centroids
of each variety.  Captures pure areal proximity — no genealogy, no
contact, no typology.

## Files

```
varieties.py                hand-coded (lat, lon) per variety
build_haversine.py          builds the matrix from varieties.LATLON
matrices/
    geographic_haversine.npz   the gold matrix (13×13, [0, 1] normalised)
```

## How to rebuild

```bash
python -m gold.geographic.build_haversine \
    --out-dir gold/geographic/matrices
```

When new varieties are added, edit `gold/geographic/varieties.py` to
include their coordinates and rerun.

## Caveats

* Centroids approximate the speech area for dialects with broad spread
  (Lombard, Sardinian).  Document this in the paper.
* Geographic distance only; for areal *contact* effects (e.g. German
  borrowings into Friulian via the Slovene–German border zone) use
  the historical-influence gold under `gold/historical_influence/`.
