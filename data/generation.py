import os
import json
import pandas as pd
import re
from tqdm import tqdm
import numpy as np
import urllib.request
import urllib.parse
from pathlib import Path
import time

print("Generating the dataset...")

# list of dialects in the training set
dialects = [d for d in os.listdir(".") if not os.path.isfile(d) and d != '.ipynb_checkpoints' and d != '.cache' and not d.startswith('.')]

# define dictionaries for string->int and int->label conversion
fold_label = {
    'eml_texts' : 0,
    'nap_texts' : 1,
    'pms_texts' : 2,
    'fur_texts' : 3,
    'lld_texts' : 4,
    'lij_texts' : 5,
    'lmo_texts' : 6,
    'roa_tara_texts' : 7,
    'scn_texts' : 8,
    'vec_texts' : 9,
    'sc_texts' : 10,
    'it_texts' : 11
}
dial_label = {
    0 : 'EML',
    1 : 'NAP',
    2 : 'PMS',
    3 : 'FUR',
    4 : 'LLD',
    5 : 'LIJ',
    6 : 'LMO',
    7 : 'ROA_TARA',
    8 : 'SCN',
    9 : 'VEC',
    10 : 'SC',
    11 : 'ITA'
}

# folder -> Wikidata site id (for wbgetentities sites=...)
folder_to_wiki = {
    'eml_texts': 'emlwiki',
    'nap_texts': 'napwiki',
    'pms_texts': 'pmswiki',
    'fur_texts': 'furwiki',
    'lld_texts': 'lldwiki',
    'lij_texts': 'lijwiki',
    'lmo_texts': 'lmowiki',
    'roa_tara_texts': 'roa_tarawiki',
    'scn_texts': 'scnwiki',
    'vec_texts': 'vecwiki',
    'sc_texts': 'scwiki',
    'it_texts': 'itwiki',
}

# folder -> hostname for MediaWiki category API
folder_to_host = {
    'eml_texts': 'eml.wikipedia.org',
    'nap_texts': 'nap.wikipedia.org',
    'pms_texts': 'pms.wikipedia.org',
    'fur_texts': 'fur.wikipedia.org',
    'lld_texts': 'lld.wikipedia.org',
    'lij_texts': 'lij.wikipedia.org',
    'lmo_texts': 'lmo.wikipedia.org',
    'roa_tara_texts': 'roa-tara.wikipedia.org',
    'scn_texts': 'scn.wikipedia.org',
    'vec_texts': 'vec.wikipedia.org',
    'sc_texts': 'sc.wikipedia.org',
    'it_texts': 'it.wikipedia.org',
}

# P31 (instance of) QID -> macro topic bucket
P31_TO_TOPIC = {
    # persone
    'Q5': 'persona',
    # luoghi
    'Q515': 'luogo',            # citta
    'Q747074': 'luogo',         # comune in Italia
    'Q1549591': 'luogo',        # big city
    'Q3957': 'luogo',           # town
    'Q532': 'luogo',            # village
    'Q15284': 'luogo',          # municipality
    'Q6256': 'luogo',           # country
    'Q10742': 'luogo',          # frazione
    'Q486972': 'luogo',         # human settlement
    'Q22865': 'luogo',          # capital
    'Q23397': 'luogo',          # lake
    'Q8502': 'luogo',           # mountain
    'Q4022': 'luogo',           # river
    # opere
    'Q11424': 'opera',          # film
    'Q7725634': 'opera',        # opera letteraria
    'Q571': 'opera',            # libro
    'Q2188189': 'opera',        # composizione musicale
    'Q7366': 'opera',           # canzone
    'Q482994': 'opera',         # album
    'Q5398426': 'opera',        # serie TV
    # eventi
    'Q1190554': 'evento',
    'Q13418847': 'evento',      # evento storico
    'Q198': 'evento',           # guerra
    'Q178561': 'evento',        # battaglia
    # organizzazioni
    'Q43229': 'organizzazione',
    'Q4830453': 'organizzazione',   # business
    'Q783794': 'organizzazione',    # company
    'Q327333': 'organizzazione',    # government agency
    # specie / tassonomia
    'Q16521': 'specie',
    # meta (da droppare)
    'Q4167410': 'meta',         # disambigua
    'Q13406463': 'meta',        # lista
    'Q11266439': 'meta',        # template
    'Q4167836': 'meta',         # category
}

CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)

USER_AGENT = "italian-dialects-research/1.0 (educational project)"

# Toggle: se False, salta le chiamate API a Wikidata e MediaWiki.
# I campi qid/topic/title_en/categories restano stringhe vuote nel meta CSV.
# Utile quando non si vuole dipendere dalla rete o si sta debuggando la pipeline.
ENRICH_VIA_API = False

# create training dataset
data = []

for d in tqdm(dialects):
    aa_dir = os.path.join(d, "AA")
    if not os.path.isdir(aa_dir):
        continue
    for name in os.listdir(aa_dir):
        f = open(os.path.join(aa_dir, name), "r")
        lines = f.readlines()
        for l in lines:
            jline = json.loads(l)
            if not jline['text']:
                continue
            data.append([int(jline['id']), jline['url'], jline['title'], jline['text'], fold_label[d]])


columns = ['id', 'url', 'title', 'text', 'label']
df = pd.DataFrame(data, columns=columns)

# (was: df = df.drop(columns=["id", "url", "title"]))
# keep id, title, url for the meta file

# clean text
def clean(text):
    text = re.sub(r'==.*?==+', '', text)

    text = text.replace("\n", " ")

    text = text.replace('"', " ")

    regex = re.compile('&[^;]+;')
    text = re.sub(regex, '', text)


    regex = re.compile('(graph.*/graph|\(.*\)|\[.*\]|parentid>.*/parentid>|BR[^>]+>|bR[^>]+>|Br[^>]+>|br[^>]+>|ns>.*/ns>|timestamp>.*/timestamp>|revision>.*/revision>|contributor>.*/contributor>|model>.*/model>|format>.*/format>|comment>.*/comment>)')
    text = re.sub(regex, '', text)
    regex = re.compile('(parentid.*/parentid|ns.*/ns|timestamp.*/timestamp|revision.*/revision|contributor.*/contributor|model.*/model|format.*/format|comment.*/comment)')
    text = re.sub(regex, '', text)

    text = text.replace("revision>", "")
    text = text.replace("br>", "")
    text = text.replace("Br>", "")
    text = text.replace("bR>", "")
    text = text.replace("BR>", "")
    text = text.replace("/br>", "")
    text = text.replace("/Br>", "")
    text = text.replace("/bR>", "")
    text = text.replace("/BR>", "")

    text = text.replace("&quot;","")

    text = text.replace("br clear=all>", "")

    if(len(text) < 50):
        text = np.nan

    return text

# print("Saving uncleaned dataset...")
# df.to_csv("uncleaned.csv", index=None)

print("Cleaning text...")

df['text'] = df['text'].apply(clean)

# drop rows with nan values
df.dropna(inplace=True)

# drop duplicate entries in the samples
df.drop_duplicates(subset='text', keep=False, inplace=True)

# create sentences
print("Splitting sentences...")

import spacy

nlp = spacy.load("it_core_news_sm", disable=['ner', 'lemmatizer', "textcat", "custom", "tagger"])

from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True, verbose=0)

df['text'] = df['text'].parallel_apply(nlp)

X = df["text"].to_numpy()
y = df["label"].to_numpy()
ids = df["id"].to_numpy()
titles = df["title"].to_numpy()
urls = df["url"].to_numpy()

print("Creating new data...")
X_train, y_train, id_train, title_train, url_train = [], [], [], [], []
for i, article in tqdm(enumerate(X), total=X.shape[0]):
    for sentence in article.sents:
        X_train.append(sentence)
        y_train.append(y[i])
        id_train.append(ids[i])
        title_train.append(titles[i])
        url_train.append(urls[i])

X_train = np.array(X_train, dtype=object)
y_train = np.array(y_train, dtype=object)
id_train = np.array(id_train, dtype=object)
title_train = np.array(title_train, dtype=object)
url_train = np.array(url_train, dtype=object)

print("Cleaning sentences...")
df = pd.DataFrame({
    'text': X_train,
    'label': y_train,
    'article_id': id_train,
    'title': title_train,
    'url': url_train,
}, index=None)

df["text"] = df['text'].apply(lambda x: ''.join(x.text))

# pms documents have a lot of these
df["text"] = df['text'].apply(lambda x: x.replace("http://www.sil.org/iso639-3/documentation.asp?id=", ""))
# other minor corrections
df['text'] = df['text'].apply(lambda x: x.replace("&lt;br clear=all&gt;", ""))
df['text'] = df['text'].apply(lambda x: x.replace("Evulusiù demogràfica.", ""))
df['text'] = df['text'].apply(lambda x: x.replace("&lt;br&gt;&lt;br&gt;", ""))
df['text'] = df['text'].apply(lambda x: x.replace("ł", "l"))
df['text'] = df['text'].apply(lambda x: x.replace("Ł", "l"))

df["text"] = df['text'].apply(lambda x: np.nan if len(x)<=20 else x)
df.dropna(inplace=True)

df.loc[df['label'] == 2, 'text'] = df.loc[df['label'] == 2, 'text'].apply(lambda x: np.nan if ("grup ëd popolassion." in x or "A confin-a con " in x or "a l’é na comun-a ëd" in x or "con na densità" in x or "A së stend" in x or "As dëstend për" in x or "a l'é na comun" in x or "La lenga" in x or "Në schema" in x or "Ël sìndich a l'é" in x or "a l'é un comun" in x) else x)
df.loc[df['label'] == 6, 'text'] = df.loc[df['label'] == 6, 'text'].apply(lambda x: np.nan if ("La Stazzion de" in x or "El cumün" in x or "a l'è una cità" in x or "El Passaport" in x or "la se tróa a 'na" in x or "a l'è 'na ferrovia" in x or "L'è taccada a stazione di" in x or "La a l'è 'na strada" in x or "L'andament del numer de abitant" in x or "L'andament del nömer dei abitàncc" in x or "l'è menziunaa la prima volta" in x or "l'è 'na stazion de la" in x or "L'andamènt del nömer dei abitàncc" in x or "La Stazion de" in x or "El Distret" in x or "El cümü" in x or "km²" in x or "Al gh’ha pressapoch abitant" in x or "l'è un cumün" in x or "El cumün de" in x or "El cunfìna coi cümü" in x or "l'è un cümü" in x or "l'è 'n cümü" in x or "e 'na densità de" in x) else x)
df.loc[df['label'] == 9, 'text'] = df.loc[df['label'] == 9, 'text'].apply(lambda x: np.nan if ("el xe on comun de" in x or "el xe un comun" in x or "gregorian" in x) else x)

df.dropna(inplace=True)
df.drop_duplicates(subset='text', keep=False, inplace=True)


# ====== Wikidata + MediaWiki enrichment ======

def _load_cache(path):
    if path.exists():
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_cache(cache, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False)


def _chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]


def _http_get_json(url, params, timeout=30):
    full_url = url + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(full_url, headers={'User-Agent': USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode('utf-8'))


def _with_retry(fn, *args, max_attempts=5, base_wait=2.0, label="batch"):
    """Call fn(*args); on HTTP 429 back off exponentially. Returns None on final failure."""
    for attempt in range(max_attempts):
        try:
            return fn(*args)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = base_wait * (2 ** attempt)
                print(f"    {label}: 429 rate-limited, sleeping {wait:.1f}s (attempt {attempt+1}/{max_attempts})")
                time.sleep(wait)
                continue
            print(f"    {label}: HTTP error {e.code}: {e}")
            return None
        except Exception as e:
            print(f"    {label}: error {type(e).__name__}: {e}")
            if attempt < max_attempts - 1:
                time.sleep(base_wait)
                continue
            return None
    return None


def fetch_wikidata_batch(batch_titles, wiki_site):
    """Given a batch of up to 50 page titles, return {title: {qid, topic, title_en}}."""
    if not batch_titles:
        return {}
    # note: 'normalize' only works for single-title calls on wbgetentities,
    # so we do our own loose matching on underscores/spaces via _norm() below.
    params = {
        'action': 'wbgetentities',
        'sites': wiki_site,
        'titles': "|".join(batch_titles),
        'languages': 'en',
        'props': 'labels|claims|sitelinks',
        'format': 'json',
    }
    data = _http_get_json("https://www.wikidata.org/w/api.php", params)
    out = {}
    entities = data.get('entities', {})
    for qid, entity in entities.items():
        if 'missing' in entity or qid.startswith('-'):
            continue
        sitelinks = entity.get('sitelinks', {})
        if wiki_site not in sitelinks:
            continue
        wiki_title = sitelinks[wiki_site].get('title', '')
        # topic via P31
        topic = 'altro'
        for claim in entity.get('claims', {}).get('P31', []):
            try:
                p31_qid = claim['mainsnak']['datavalue']['value']['id']
            except (KeyError, TypeError):
                continue
            if p31_qid in P31_TO_TOPIC:
                topic = P31_TO_TOPIC[p31_qid]
                break
        # title_en
        title_en = entity.get('labels', {}).get('en', {}).get('value', '')
        out[wiki_title] = {'qid': qid, 'topic': topic, 'title_en': title_en}
    return out


def fetch_categories_batch(batch_titles, host):
    """Given a batch of up to 50 page titles, return {title: [category, ...]}."""
    if not batch_titles:
        return {}
    params = {
        'action': 'query',
        'prop': 'categories',
        'titles': "|".join(batch_titles),
        'cllimit': 'max',
        'clshow': '!hidden',
        'format': 'json',
    }
    data = _http_get_json(f"https://{host}/w/api.php", params)
    out = {}
    pages = data.get('query', {}).get('pages', {})
    for page in pages.values():
        title = page.get('title', '')
        cats = []
        for c in page.get('categories', []):
            name = c.get('title', '')
            if ':' in name:
                name = name.split(':', 1)[1]
            cats.append(name)
        out[title] = cats
    return out


def _norm(t):
    """Normalize a title for loose matching (underscores vs spaces)."""
    return t.replace('_', ' ').strip()


if ENRICH_VIA_API:
    print("Enriching via Wikidata + MediaWiki (cached on disk)...")
else:
    print("Skipping API enrichment (ENRICH_VIA_API=False). "
          "qid/topic/title_en/categories resteranno vuoti.")

# Build meta DataFrame: one row per unique (label, article_id)
meta_df = df[['label', 'article_id', 'title', 'url']].drop_duplicates(
    subset=['label', 'article_id']
).reset_index(drop=True).copy()
meta_df['qid'] = ''
meta_df['topic'] = ''
meta_df['title_en'] = ''
meta_df['categories'] = ''

for folder, lbl in (fold_label.items() if ENRICH_VIA_API else []):
    if folder not in dialects:
        continue
    sub_mask = meta_df['label'] == lbl
    if not sub_mask.any():
        continue
    wiki_site = folder_to_wiki[folder]
    host = folder_to_host[folder]

    unique_titles = meta_df.loc[sub_mask, 'title'].unique().tolist()

    # Wikidata
    wd_path = CACHE_DIR / f"wikidata_{folder}.json"
    wd_cache = _load_cache(wd_path)
    missing = [t for t in unique_titles if t not in wd_cache]
    if missing:
        print(f"  {folder}: {len(unique_titles)} titoli unici, {len(missing)} da recuperare da Wikidata")
        for batch in tqdm(list(_chunks(missing, 50)), desc=f"Wikidata {folder}"):
            result = _with_retry(fetch_wikidata_batch, batch, wiki_site, label=f"Wikidata {folder}")
            if result is None:
                # batch failed after retries: do NOT cache, so next run retries
                time.sleep(1.0)
                continue
            norm_result = {_norm(k): v for k, v in result.items()}
            for t in batch:
                match = result.get(t) or norm_result.get(_norm(t))
                wd_cache[t] = match if match is not None else {'qid': '', 'topic': '', 'title_en': ''}
            _save_cache(wd_cache, wd_path)
            time.sleep(0.3)

    # MediaWiki categories
    cat_path = CACHE_DIR / f"categories_{folder}.json"
    cat_cache = _load_cache(cat_path)
    missing = [t for t in unique_titles if t not in cat_cache]
    if missing:
        print(f"  {folder}: {len(missing)} da recuperare da MediaWiki (categorie)")
        for batch in tqdm(list(_chunks(missing, 50)), desc=f"Categorie {folder}"):
            result = _with_retry(fetch_categories_batch, batch, host, label=f"Categorie {folder}")
            if result is None:
                # batch failed after retries: do NOT cache, so next run retries
                time.sleep(1.0)
                continue
            norm_result = {_norm(k): v for k, v in result.items()}
            for t in batch:
                match = result.get(t)
                if match is None:
                    match = norm_result.get(_norm(t), [])
                cat_cache[t] = match
            _save_cache(cat_cache, cat_path)
            time.sleep(0.3)

    # Apply to meta_df rows of this dialect
    def _lookup_wd(t, field):
        v = wd_cache.get(t, {})
        if isinstance(v, dict):
            return v.get(field, '')
        return ''

    def _lookup_cats(t):
        v = cat_cache.get(t, [])
        if isinstance(v, list):
            return '|'.join(v)
        return ''

    titles_sub = meta_df.loc[sub_mask, 'title']
    meta_df.loc[sub_mask, 'qid'] = titles_sub.map(lambda t: _lookup_wd(t, 'qid'))
    meta_df.loc[sub_mask, 'topic'] = titles_sub.map(lambda t: _lookup_wd(t, 'topic'))
    meta_df.loc[sub_mask, 'title_en'] = titles_sub.map(lambda t: _lookup_wd(t, 'title_en'))
    meta_df.loc[sub_mask, 'categories'] = titles_sub.map(_lookup_cats)


# n_sentences per (label, article_id) from the main df
counts = df.groupby(['label', 'article_id']).size().reset_index(name='n_sentences')
meta_df = meta_df.merge(counts, on=['label', 'article_id'], how='left')
meta_df['n_sentences'] = meta_df['n_sentences'].fillna(0).astype(int)

# drop meta/disambigua/lista articles from BOTH files
meta_drop = meta_df[meta_df['topic'] == 'meta'][['label', 'article_id']]
if len(meta_drop) > 0:
    drop_keys = set(zip(meta_drop['label'].tolist(), meta_drop['article_id'].tolist()))
    keys_series = list(zip(df['label'].tolist(), df['article_id'].tolist()))
    keep_mask = [k not in drop_keys for k in keys_series]
    df = df[keep_mask].reset_index(drop=True)
    meta_df = meta_df[meta_df['topic'] != 'meta'].reset_index(drop=True)

# ====== Save ======

print("Salvataggio file per dialetto...")

for lbl, sub in df.groupby("label"):
    fname = dial_label[int(lbl)].lower() + ".csv"
    sub[['text', 'label', 'article_id']].to_csv(fname, index=None)
    print(f"  -> {fname} ({len(sub)} frasi)")

for lbl, sub in meta_df.groupby("label"):
    fname = dial_label[int(lbl)].lower() + "_meta.csv"
    cols = ['article_id', 'title', 'url', 'qid', 'topic', 'title_en', 'categories', 'n_sentences']
    sub[cols].to_csv(fname, index=None)
    print(f"  -> {fname} ({len(sub)} articoli)")

print("Dataset created.")
