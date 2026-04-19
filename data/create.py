import requests
import subprocess
import os
import glob
import shutil
import sys
from pathlib import Path

# List of (url, output_dir)
dumps = [
    ('https://dumps.wikimedia.org/other/mediawiki_content_current/napwiki/2026-04-01/xml/bzip2/napwiki-2026-04-01-p1p66122.xml.bz2', 'nap_texts'),
    ('https://dumps.wikimedia.org/other/mediawiki_content_current/vecwiki/2026-04-01/xml/bzip2/vecwiki-2026-04-01-p3p154496.xml.bz2', 'vec_texts'),
    ('https://dumps.wikimedia.org/other/mediawiki_content_current/lmowiki/2026-04-01/xml/bzip2/lmowiki-2026-04-01-p1p279182.xml.bz2', 'lmo_texts'),
    ('https://dumps.wikimedia.org/other/mediawiki_content_current/scnwiki/2026-04-01/xml/bzip2/scnwiki-2026-04-01-p1p67086.xml.bz2', 'scn_texts'),
]

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
BASE_DIR = PROJECT_ROOT / 'datasets'

# create datasets directory if it doesn't exist
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
for pattern in ['*_texts', '*.xml.bz2']:
    for path in glob.glob(str(BASE_DIR / pattern)):
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)