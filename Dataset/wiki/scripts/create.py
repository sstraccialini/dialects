"""
Download Wikipedia dumps for the 6 italo-romance varieties (intersection
OLDI ∩ FLORES) and extract them with wikiextractor.

Caching: dumps and wikiextractor outputs are kept under
    Dataset/wiki/_cache/
so that subsequent runs skip the download and extraction steps. Delete
that folder manually to force a fresh re-download.

Pipeline:
    1. download <lang>wiki-2026-04-01-pXpY.xml.bz2 (skip if cached)
    2. wikiextractor → <lang>_texts/AA/wiki_*.json (skip if cached)
    3. invoke generation.py (writes to wiki/ subfolders)
"""

import os
import subprocess
import sys
from pathlib import Path

import requests


# (url, output_dir name)
# Group A — italo-romance varieties in OLDI ∩ FLORES (the 6 we use as
# downstream training/eval set).
# Group B — other italo-romance varieties on Wikipedia (in ITDI 2022 but
# not in OLDI/FLORES); useful as a comparison set.
dumps = [
    # --- Group A: OLDI ∩ FLORES ---
    ('https://dumps.wikimedia.org/other/mediawiki_content_current/furwiki/2026-04-01/xml/bzip2/furwiki-2026-04-01-p1p14311.xml.bz2',     'fur_texts'),
    ('https://dumps.wikimedia.org/other/mediawiki_content_current/lijwiki/2026-04-01/xml/bzip2/lijwiki-2026-04-01-p2p32679.xml.bz2',     'lij_texts'),
    ('https://dumps.wikimedia.org/other/mediawiki_content_current/lmowiki/2026-04-01/xml/bzip2/lmowiki-2026-04-01-p1p279182.xml.bz2',    'lmo_texts'),
    ('https://dumps.wikimedia.org/other/mediawiki_content_current/scwiki/2026-04-01/xml/bzip2/scwiki-2026-04-01-p2p23475.xml.bz2',       'sc_texts'),
    ('https://dumps.wikimedia.org/other/mediawiki_content_current/scnwiki/2026-04-01/xml/bzip2/scnwiki-2026-04-01-p1p67086.xml.bz2',     'scn_texts'),
    ('https://dumps.wikimedia.org/other/mediawiki_content_current/vecwiki/2026-04-01/xml/bzip2/vecwiki-2026-04-01-p3p154496.xml.bz2',    'vec_texts'),
    # --- Group B: other italo-romance varieties on Wikipedia ---
    ('https://dumps.wikimedia.org/other/mediawiki_content_current/lldwiki/2026-04-01/xml/bzip2/lldwiki-2026-04-01-p1p189680.xml.bz2',    'lld_texts'),       # Ladino (in FLORES but not OLDI)
    ('https://dumps.wikimedia.org/other/mediawiki_content_current/napwiki/2026-04-01/xml/bzip2/napwiki-2026-04-01-p1p66122.xml.bz2',     'nap_texts'),       # Napoletano
    ('https://dumps.wikimedia.org/other/mediawiki_content_current/pmswiki/2026-04-01/xml/bzip2/pmswiki-2026-04-01-p1p111428.xml.bz2',    'pms_texts'),       # Piemontese
    ('https://dumps.wikimedia.org/other/mediawiki_content_current/roa_tarawiki/2026-04-01/xml/bzip2/roa_tarawiki-2026-04-01-p2p21745.xml.bz2', 'roa_tara_texts'),  # Tarantino
]

SCRIPT_DIR = Path(__file__).resolve().parent  # Dataset/wiki/scripts/
BASE_DIR = SCRIPT_DIR.parent                  # Dataset/wiki/
CACHE_DIR = BASE_DIR / "_cache"  # cached dumps + extractions

BASE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


for url, output_dir in dumps:
    filename = url.split('/')[-1]
    dump_path = CACHE_DIR / filename
    output_path = CACHE_DIR / output_dir

    # 1) download if not cached
    if dump_path.exists():
        print(f'[cache] dump already present: {filename}')
    else:
        print(f'Downloading {filename}')
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with dump_path.open('wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

    # 2) wikiextractor if not cached
    if output_path.is_dir() and any(output_path.iterdir()):
        print(f'[cache] extraction already present: {output_dir}')
    else:
        print(f'Extracting {filename} to {output_dir}')
        subprocess.run([
            sys.executable, '-m', 'wikiextractor.WikiExtractor',
            str(dump_path), '-o', str(output_path), '--json',
        ], check=True)


# 3) run generation.py from the cache directory so it picks up the
# *_texts/ folders. generation.py decides where to write its outputs
# (default: ../ subfolders dialects_in_both_OLDI_and_Flores/ and
# others_dialects/, can be overridden inside generation.py).
print('Running generation.py')
subprocess.run([sys.executable, str(SCRIPT_DIR / 'generation.py')],
               check=True, cwd=str(CACHE_DIR))

print('Done. Cache kept at', CACHE_DIR)
