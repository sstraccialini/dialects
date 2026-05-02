import requests
import subprocess
import os
import glob
import shutil
import sys
from pathlib import Path

# List of (url, output_dir)
# Already downloaded (kept commented for reference):
#   napwiki  -> nap_texts   p1p66122
#   vecwiki  -> vec_texts   p3p154496
#   lmowiki  -> lmo_texts   p1p279182
#   scnwiki  -> scn_texts   p1p67086
#   scwiki   -> sc_texts    p2p23475
#   itwiki, eswiki, frwiki, cawiki, dewiki, elwiki, arwiki, slwiki, enwiki
dumps = [
    ('https://dumps.wikimedia.org/other/mediawiki_content_current/furwiki/2026-04-01/xml/bzip2/furwiki-2026-04-01-p1p14311.xml.bz2', 'fur_texts'),
    ('https://dumps.wikimedia.org/other/mediawiki_content_current/lijwiki/2026-04-01/xml/bzip2/lijwiki-2026-04-01-p2p32679.xml.bz2', 'lij_texts'),
]

SCRIPT_DIR = Path(__file__).resolve().parent  # Dataset/wiki/scripts/
BASE_DIR = SCRIPT_DIR.parent                  # Dataset/wiki/

# create wiki dataset directory if it doesn't exist
BASE_DIR.mkdir(parents=True, exist_ok=True)

for url, output_dir in dumps:
    filename = url.split('/')[-1]
    dump_path = BASE_DIR / filename
    output_path = BASE_DIR / output_dir

    print(f'Downloading {filename}')
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with dump_path.open('wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    
    print(f'Extracting {filename} to {output_dir}')
    # Use the current interpreter to avoid picking a different global Python install.
    subprocess.run([sys.executable, '-m', 'wikiextractor.WikiExtractor', str(dump_path), '-o', str(output_path), '--json'], check=True)

print('Running generation.py')
subprocess.run([sys.executable, str(SCRIPT_DIR / 'generation.py')], check=True, cwd=str(BASE_DIR))

print('Cleaning up')
# Only remove what THIS run produced/downloaded — never touch dumps from
# previous runs that the team has already kept under Dataset/wiki/.
for url, output_dir in dumps:
    filename = url.split('/')[-1]
    dump_path = BASE_DIR / filename
    texts_path = BASE_DIR / output_dir
    if texts_path.is_dir():
        shutil.rmtree(texts_path)
    if dump_path.is_file():
        os.remove(dump_path)