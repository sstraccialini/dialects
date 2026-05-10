"""
Geographic-centroid coordinates per variety, used by the great-circle gold.

Coordinates are taken from Glottolog 5.x languoid records (or the capital
city for standard languages with broad geographic spread).  Update this
file when adding new varieties.
"""
from __future__ import annotations

from typing import Dict, Tuple


# (latitude, longitude) in decimal degrees.
LATLON: Dict[str, Tuple[float, float]] = {
    # Italian dialects — Glottolog languoid centroids
    "fur": (46.07, 13.23),    # Friulian — Udine area
    "lij": (44.40,  8.95),    # Ligurian — Genoa
    "lmo": (45.45,  9.18),    # Lombard — Milan (Western Lombard centroid)
    "sc":  (40.30,  9.15),    # Sardinian (Logudorese) — central-northern Sardinia
    "scn": (37.50, 14.00),    # Sicilian — Sicily centroid
    "vec": (45.45, 12.33),    # Venetan — Venice
    # Standards — capital city of the speech community
    "ita": (41.90, 12.49),    # Italian        — Rome
    "fra": (48.85,  2.35),    # French         — Paris
    "spa": (40.42, -3.70),    # Spanish        — Madrid
    "cat": (41.39,  2.17),    # Catalan        — Barcelona
    "por": (38.72, -9.14),    # Portuguese     — Lisbon
    "oci": (43.60,  1.44),    # Occitan        — Toulouse
    "deu": (52.52, 13.40),    # German         — Berlin
    "eng": (51.51, -0.13),    # English        — London
    "slv": (46.06, 14.51),    # Slovenian      — Ljubljana
    "hrv": (45.81, 15.98),    # Croatian       — Zagreb
    "hun": (47.50, 19.04),    # Hungarian      — Budapest
}
