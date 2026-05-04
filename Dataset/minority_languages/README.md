# Minority languages of Italy — non-Wikipedia corpora

This folder collects sentence-level corpora for **three minority languages
of Italy** that are NOT covered by FLORES+ / OLDI / Wikipedia in our main
pipeline:

- **Griko** (`grk`, Salentino-Calabrian Greek isolate)
- **Molise Slavic / Na-našu** (`svm`, South Slavic enclave in Molise)
- **South Tyrolean German** (`de-st`, Bavarian variety of Alto Adige) plus
  bonus **Bernese Swiss German** (`gsw`) and **Neapolitan** (`nap`) from
  the same XSID/SID4LR shared tasks.

These corpora were compiled by independent academic projects, are openly
released, and complement our main 10-dialect Wikipedia training set
(see `Dataset/wiki/PIPELINE.md`).

## Folder layout

```
Dataset/minority_languages/
├── README.md                      this file
├── griko/
│   ├── uoi/                       Boito et al. 2018, SLTU — 330 parallel sentences
│   │   ├── README.md
│   │   ├── all/                   {transcriptions, translations, italian_gloss, ...}
│   │   ├── train/                 {gr/, it/, train.ids}
│   │   └── dev/                   {gr/, it/, dev.ids}
│   └── zrc/                       Boito ZRC reference, phonological transcriptions
│       ├── README.md
│       ├── SIL_version/
│       └── eval/
├── molise_slavic/                 Breu (2017) / EuroSlav 2010, Pangloss CNRS
│   ├── acquaviva_collecroce/      27 texts, ~890 sentences, parallel svm/ita/deu/eng
│   ├── montemitro/                22 texts, ~592 sentences, parallel svm/ita/deu
│   └── san_felice_del_molise/     14 texts, ~628 sentences, parallel svm/ita/deu
└── south_tyrolean/
    ├── xsid/                      van der Goot et al. 2021, NAACL — de-st intent + slot tags
    │   ├── LICENSE                CC BY-SA 4.0
    │   ├── README.md
    │   ├── de-st.test.conll       500 sentences
    │   └── de-st.valid.conll      300 sentences
    └── sid4lr/                    Aepli et al. 2023, VarDial
        ├── de-st.{test,valid}.conll
        ├── gsw.{test,valid}.conll  bonus: Bernese Swiss German
        └── nap.{test,valid}.conll  bonus: Neapolitan
```

## Coverage summary

| Variety | Sentences | Parallel langs | Source | License |
|---|---:|---|---|---|
| **grk** Griko UOI | 330 | grk + ita + speech audio (skipped) | Boito et al. 2018 | citation requested |
| **grk** Griko ZRC | (phonological reference, ~minutes audio) | grk transcriptions | Boito ZRC | (research) |
| **svm** Molise Slavic Acquaviva | ~890 | svm + ita + deu + eng | Breu 2017 / Pangloss CNRS | CC BY-NC 2.5 |
| **svm** Molise Slavic Montemitro | ~592 | svm + ita + deu | Breu 2017 | CC BY-NC 2.5 |
| **svm** Molise Slavic San Felice | ~628 | svm + ita + deu | Breu 2017 | CC BY-NC 2.5 |
| **de-st** South Tyrolean (xSID v0.7) | 800 | de-st (intent + slot tags, English-aligned) | van der Goot et al. 2021 | CC BY-SA 4.0 |
| **de-st** South Tyrolean (SID4LR snapshot) | 800 | same as xSID v0.4 | Aepli et al. 2023 | CC BY-SA 4.0 |
| **gsw** Bernese Swiss German (SID4LR) | 800 | gsw | Aepli et al. 2023 | CC BY-SA 4.0 |
| **nap** Neapolitan (SID4LR) | 800 | nap | Aepli et al. 2023 | CC BY-SA 4.0 |

## File formats

### Griko UOI (`griko/uoi/`)
Per-sentence files, numeric IDs `1.words` … `330.words`:
- `all/transcriptions/<id>.words` — Griko text, segmented (orthographic, Spitta)
- `all/transcriptions/<id>.words.unseg` — same, unsegmented
- `all/translations/<id>.it` — Italian translation
- `all/italian_gloss/<id>.gloss` — word-level Italian gloss
- `all/alignment.gr-it.txt` — sentence-level alignment table
- `all/id_list.txt` — list of valid sentence IDs
- `all/wav2gr/`, `all/wav2it/` — alignment between speech and text (kept,
  but the audio WAV files themselves are NOT mirrored here to save space —
  re-clone `https://github.com/antonisa/griko-italian-parallel-corpus` for
  audio)
- `train/`, `dev/` — pre-split versions, with `gr/` and `it/` per-sentence
  text files.

### Griko ZRC (`griko/zrc/`)
Two-format text files (with/without silence markers) for Zero Resource
Challenge speech baseline. We mirror the text portions; for audio files
re-clone `https://github.com/mzboito/ZRC_corpora`.

### Molise Slavic (`molise_slavic/<village>/crdo-SVM_*.xml`)
Pangloss XML format. Each `.xml` is one narrative, with structure:

```xml
<TEXT xml:lang="svm" id="crdo-SVM_TRACTEUR">
  <S id="s1">
    <AUDIO start="0.4030" end="3.7176"/>
    <FORM>Ja mahu kupi trator nonda, one dana.</FORM>           <!-- Slavic -->
    <TRANSL xml:lang="it">Io dovevo comprare un trattore...</TRANSL>
    <TRANSL xml:lang="de">Ich musste einen Traktor...</TRANSL>
    <W>                                                          <!-- words -->
      <M><FORM>ja</FORM><TRANSL>I.NOM</TRANSL></M>              <!-- morphemes -->
      ...
    </W>
  </S>
</TEXT>
```

Audio files (`.wav`/`.mp3`) are NOT mirrored here — fetch from the
per-text DOIs at `https://doi.org/10.24397/pangloss-XXXXXXX` if needed.
Acquaviva texts also include `<TRANSL xml:lang="en">` (English).

### XSID / SID4LR (`south_tyrolean/{xsid,sid4lr}/<lang>.{train,valid,test}.conll`)
SID-specific CoNLL flavour, tab-separated columns:

```
# id: 1
# text: Bring di Cocacola in dr Schubladn
# intent: PlayMusic
1   Bring     O
2   di        O
3   Cocacola  B-music_item
4   in        O
5   dr        B-playlist
6   Schubladn I-playlist
```

Comment lines (`# text:`, `# intent:`) carry the gold sentence and
intent label; token rows have IOB slot tags. South Tyrolean files use
language code `de-st` consistently.

## Citation

When using these corpora in publications, cite the original papers:

- **Griko UOI**:
  Boito, M. Z., Anastasopoulos, A., Lekakou, M., Villavicencio, A., &
  Besacier, L. (2018). *A Small Griko-Italian Speech Translation Corpus*.
  SLTU 2018. arXiv:1807.10740.

- **Molise Slavic / Na-našu**:
  Breu, W. (2017). *Slavische Mikrosprachen im Absoluten Sprachkontakt.
  Teil I: Moliseslavische Texte aus Acquaviva Collecroce, Montemitro und
  San Felice del Molise*. Slavistische Beiträge 505. Harrassowitz.
  ISBN 978-3-447-10865-2. DOI: 10.2307/j.ctv11sn5zw.
  EuroSlav 2010 deposit at LACITO/Pangloss CNRS.
  Per-text DOIs: `https://doi.org/10.24397/pangloss-XXXXXXX`.

- **xSID / South Tyrolean**:
  van der Goot, R., Sharaf, I., Imankulova, A., Üstün, A., Stepanović, M.,
  Ramponi, A., Khairunnisa, S. O., Komachi, M., & Plank, B. (2021).
  *From Masked Language Modeling to Translation: Non-English Auxiliary
  Tasks Improve Zero-shot Spoken Language Understanding*. NAACL 2021.

- **SID4LR (gsw, nap)**:
  Aepli, N., Çöltekin, Ç., van der Goot, R., et al. (2023). *Findings of
  the VarDial Evaluation Campaign 2023*. VarDial 2023.

## Tier 3 — items NOT mirrored here (require contact / login)

These three datasets were excluded because they require email contact or
academic CLARIN authentication. Documented here for the record:

| Dataset | Why not mirrored | How to obtain |
|---|---|---|
| **GRIKONARRATIVE** (Anastasopoulos et al. 2018, COLING) | no public mirror; ~10,100 Griko sentences across 114 narratives, ~943 with gold POS | email `antonis@gmu.edu` (George Mason) describing intended use |
| **DiDi** (Eurac, South Tyrolean Facebook CMC) | CLARIN academic login + NDA for private messages tier | request CLARIN account from `clarin@eurac.edu`; NDA via DiDi team for full corpus |
| **LEONIDE** (Eurac, learner trilingual it/de/en) | CLARIN academic login | request CLARIN account from `clarin@eurac.edu` |

These would extend our coverage to:
- ~10,000 Griko narrative sentences (Tier 3a)
- ~600,000 token CMC corpus for South Tyrolean German (Tier 3b)
- ~240,000 token learner corpus (less linguistically representative — pupils
  attempting standard German with errors, not native dialect)

## Why these three languages

Each fills a gap not covered by Wikipedia or FLORES+:
- **Griko**: no Wikipedia edition, not in FLORES+/OLDI. Greek-Romance
  contact zone in Salento — geographically isolated, lexically Greek but
  syntactically influenced by Italian/Salentino.
- **Molise Slavic**: South Slavic (Štokavian-derived) enclave in Italo-Romance
  area. Documented by Breu's EuroSlav project. Tests whether our embedding
  pipeline can detect a non-Romance language inside the Italian linguistic
  space.
- **South Tyrolean**: Bavarian-Alemannic dialect variety. Tests whether
  standard German embeddings (which our pipeline already covers via
  `wiki/languages/deu.csv`) generalize to a non-standard variety in
  contact with Italian.

## Provenance & download method

Tier 1 (git clones, May 2026):
- `griko/uoi/`: `git clone https://github.com/antonisa/griko-italian-parallel-corpus`
  (kept text-only, dropped `wavs/`, `labs/`, `pseudo_phones/`, `silences/`)
- `griko/zrc/`: `git clone https://github.com/mzboito/ZRC_corpora`,
  kept `griko/` subfolder only
- `south_tyrolean/xsid/`: `git clone https://github.com/mainlp/xsid`,
  kept `data/xSID-0.7/de-st.{test,valid}.conll` only
- `south_tyrolean/sid4lr/`: `git clone https://bitbucket.org/robvanderg/sid4lr`,
  kept `tgt_data/{de-st,gsw,nap}.*` only

Tier 2 (Pangloss via Wayback Machine, May 2026):
- 63 XML annotation files harvested through `web.archive.org` because the
  upstream `pangloss.cnrs.fr` and `cocoon.huma-num.fr` returned HTTP 503 at
  the time of acquisition.
- Audio files (WAV, MP3) NOT mirrored — re-fetch from per-text DOIs at
  `https://doi.org/10.24397/pangloss-XXXXXXX` when servers are back online.
- Download script: `/tmp/pangloss_download.sh` (kept as record).
