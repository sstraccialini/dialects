## 1. What we did in one paragraph

We took the 14 language varieties (5 Italo-Romance dialects + Italian +
3 other Romance languages + German, English, Greek, Arabic, Slovenian),
concatenated ~16,000 Wikipedia sentences per variety into one big
"super-document" each, and turned those 14 documents into numeric
vectors using **TF-IDF**. TF-IDF counts how often each feature appears
in a document and down-weights features that appear in almost every
document (common, uninformative features). The result is one vector
per variety: a numerical "fingerprint" of that language.

We did this **twice**, in parallel, with two different definitions of
"feature":

- **Word pipeline** — features are full words and pairs of consecutive
  words (unigrams and bigrams).
- **Char pipeline** — features are sequences of 3 to 5 characters
  inside words (for example `sch`, `chio`, `cchio`).

Once each variety is a vector, we measure how similar two varieties
are with **cosine distance**. Cosine distance is a number between 0
and 1:

- 0 = identical vectors (same content mix),
- 1 = completely orthogonal (nothing in common).

Doing this for every pair gives us a **14 × 14 distance matrix**,
which is the raw output everything else is built on.

---

## 2. What the different outputs mean

For each pipeline (`word/` and `char/`) you will find these files:

### `distances.csv` — the raw numbers
A 14 × 14 table with the cosine distance between every pair of
varieties. This is the ground truth; all the pictures are just
different ways of looking at this table.

### `nearest_neighbors.csv` — the top-3 closest neighbors
For each variety, the three closest varieties and their distances.
Think of it as "given this language, which 3 languages look most like
it according to TF-IDF". Fast sanity check: if Neapolitan's closest
neighbor is Italian, the method is behaving reasonably.

### `top_features.csv` — which features define each variety
For each variety, the 30 TF-IDF features that score highest — i.e. the
words or character sequences that are most typical of that variety.
Useful to see *why* two languages are considered similar.

### `dendrogram.png` — a family tree built from the distances
A hierarchical clustering of the 14 varieties. Read the **vertical
axis**: it is the cosine distance at which two branches get merged.
Low merge = the two branches are very similar; high merge = they are
very different. By drawing an imaginary horizontal line you cut the
tree into groups.

### `projection_mds.png` — a 2D map of the 14 varieties
MDS (multidimensional scaling) takes the 14 × 14 distance matrix and
places the 14 points on a flat plane **trying to preserve distances**.
On an MDS plot, two dots that look close on screen really are close
according to TF-IDF, and two dots that look far really are far.
Distances are globally meaningful.

### `projection_tsne.png` — a 2D map that emphasises clusters
t-SNE also gives a 2D layout, but with a different trade-off: it
preserves **local neighborhoods**, not global distances. Clusters are
easier to see, but the distance *between* clusters on the plot is not
meaningful (you cannot say "cluster A is twice as far as cluster B").

### `silhouette_report.txt` (in `results/shared/`) — a single number
The silhouette score rates how well the family labels (Italo-Romance,
Germanic, etc.) match the structure of the distances. It ranges from
-1 to +1:

- ~ 0 → no structure,
- > 0.2 → decent,
- > 0.5 → excellent.

We report two versions:
- **family**: against the 7 fine-grained labels,
- **romance vs rest**: simpler binary label (Romance = Italo-Romance +
  Italian + other Romance, everything else = not Romance).

---

## 3. Results of the **word** pipeline

Silhouette scores: **family = -0.005, romance-vs-rest = +0.016** — both
essentially zero.

Look at `results/word/dendrogram.png`: all 14 varieties merge in a
very narrow band between ~0.91 and ~1.00. The tree is almost flat.
That means the word-level vectors say *"every language is almost
equally distant from every other language"*.

Why? Because at the word level the dialects share almost nothing.
Neapolitan `'o` is not the same string as Italian `il`, Sicilian
`chi` is not Italian `che`, Venetian `xe` is not Italian `è`. Every
dialect essentially has its own vocabulary from the raw-string point
of view, so the word pipeline cannot see that they are related.

The nearest-neighbor table still gets *some* things right (Neapolitan
→ Italian, Catalan → Spanish, Venetian → Italian), but distances are
all around 0.9, so the "top 3 closest" is only marginally closer than
the other 10.

**Take-away for the team:** plain words are not a useful signal for
comparing dialects. This is an expected, classical result in
dialectology and is the main reason we also run the char pipeline.

---

## 4. Results of the **char** pipeline

Silhouette scores: **family = -0.050, romance-vs-rest = +0.114** —
much better on the romance-vs-rest split, clearly measuring something.

This pipeline looks at 3-to-5-character sequences, which captures
spelling habits, frequent endings, typical clusters of letters. Two
languages that share the same family will naturally share many such
sub-strings even if they use different full words.

### Dendrogram (`results/char/dendrogram.png`)
The tree is no longer flat. Reading from the lowest merges upward:

- **Neapolitan + Italian** merge at ~0.54 — the closest pair in the
  whole dataset.
- **Sicilian** joins them at ~0.59.
- **Venetian + Lombard** form their own small pair at ~0.60.
- **Sardinian** attaches to the Sicilian/Neapolitan/Italian group.
- **Spanish + Catalan** form a tight pair at ~0.63.
- **French + English** merge at ~0.65 (a spurious pair driven by the
  international vocabulary shared on Wikipedia; not a real linguistic
  signal).
- **German** joins the Latin-alphabet cluster at ~0.83.
- **Slovenian** at ~0.79.
- **Arabic** and **Greek** only merge at ~0.98–0.99, confirming they
  are effectively isolated (different scripts → almost no shared
  character n-grams).

### MDS map (`results/char/projection_mds.png`)
The left half is dominated by Italo-Romance (Sardinian, Lombard,
Venetian, Sicilian, Neapolitan) with Italian sitting right in the
middle. The top-right shows the other Romance languages (Catalan,
Spanish, French) together with German. English sits between the
Romance block and Slovenian. Arabic and Greek are pushed to the bottom
corners — they are the outliers.

### t-SNE map (`results/char/projection_tsne.png`)
Same information, reorganised to emphasise cluster membership. The
Italo-Romance block sits together on one side, the Romance trio
(Spanish, French, Catalan) on another, the two Germanic languages
close to each other, and Italian floats between the Italo-Romance
block and the other Romance languages — which is exactly where a
standard language should be.

### Nearest neighbors (`results/char/nearest_neighbors.csv`)
Every Italo-Romance dialect has Italian or another Italo-Romance
language in its top-3. Catalan → Spanish, French → English/Italian,
Italian → Neapolitan/Venetian. The signal is clean.

---

## 5. Summary for the team

- **TF-IDF + cosine distance** is a very simple method: count features,
  down-weight common ones, measure angles between vectors.
- **Word n-grams do not work for dialects** because each dialect
  writes the same concepts with different strings (flat dendrogram,
  near-zero silhouette).
- **Character n-grams do work** as a first approximation: they
  recover the Italo-Romance cluster, pair Spanish with Catalan,
  isolate Arabic and Greek, and place Italian correctly between
  dialects and the other Romance languages.
- The char silhouette on the romance-vs-rest label is still modest
  (+0.11), so TF-IDF finds the families but does not cleanly separate
  them. This is the **baseline**: the more powerful models in the
  project (Word2Vec, SBERT, mBERT/XLM-R, FastText) are expected to go
  further and, in particular, capture signals TF-IDF cannot see:
  historical contact across different scripts, deeper structural
  similarity, and cross-lingual semantic alignment.
