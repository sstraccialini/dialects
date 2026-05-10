"""
Download FLORES+ per il progetto NLP dialetti italiani.

Scarica entrambi gli split (dev + devtest) per tutte le lingue in LANGS
e li UNISCE subito in un unico file per lingua:
    flores_data/flores_plus/<lang>.txt   -> 2009 frasi totali (997 dev + 1012 devtest)

Produce anche:
    flores_data/flores_plus/parallel.tsv -> 2009 righe, colonne: sentence_id, split, <lang>...
                                            (split = "dev" per le prime 997 righe, "devtest" per le restanti 1012)
    flores_data/stats.csv                -> riepilogo: codice, categoria, n_frasi, n_caratteri, path, note

Ordine delle frasi: prima tutte le 997 di dev, poi tutte le 1012 di devtest.
Le lingue sono allineate per posizione: la riga N di un file e' la
traduzione della riga N di tutti gli altri.

I 3 dialetti extra (lij_Latn, fur_Latn, lld_Latn) possono non essere nella
tua versione di FLORES+: se falliscono, vengono loggati come errore e le
altre lingue proseguono.

NOTA: il napoletano NON e' disponibile in FLORES+ - limite noto del dataset.
"""

import csv
import os
import sys
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import login
from tqdm import tqdm

# --- config --------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent      # .../flores_data
OUT_DIR = BASE_DIR / "flores_plus"
STATS_FILE = BASE_DIR / "stats.csv"

DATASET_ID = "openlanguagedata/flores_plus"
SPLITS = ["dev", "devtest"]

# (codice_flores, categoria, slug per nome file/colonna, descrizione leggibile)
LANGS = [
    # varieta' italiane (core, certe)
    ("vec_Latn", "dialetto_italiano", "veneto",     "Veneto"),
    ("scn_Latn", "dialetto_italiano", "siciliano",  "Siciliano"),
    ("lmo_Latn", "dialetto_italiano", "lombardo",   "Lombardo"),
    ("srd_Latn", "dialetto_italiano", "sardo",      "Sardo"),
    
    # varieta' italiane extra (possono mancare: errore loggato, non blocca)
    ("lij_Latn", "dialetto_italiano", "ligure",     "Ligure (Genovese)"),
    ("fur_Latn", "dialetto_italiano", "friulano",   "Friulano"),
    # ("lld_Latn", "dialetto_italiano", "ladino",     "Ladino"),
    # ("nap_Latn", "dialetto_italiano", "napoletano", "Napoletano"),
    
    # italiano standard
    ("ita_Latn", "italiano",           "italiano",  "Italiano"),
    
    # lingue esterne
    ("eng_Latn", "esterna",            "inglese",   "Inglese"),
    ("spa_Latn", "esterna",            "spagnolo",  "Spagnolo"),
    ("fra_Latn", "esterna",            "francese",  "Francese"),
    ("cat_Latn", "esterna",            "catalano",  "Catalano"),
    ("oci_Latn", "esterna",            "occitano",  "Occitano"),
    ("por_Latn", "esterna",            "portoghese", "Portoghese"),
    ("deu_Latn", "esterna",            "tedesco",   "Tedesco"),
    ("hrv_Latn", "esterna",            "croato",    "Croato"),
    # ("ell_Grek", "esterna",            "greco",     "Greco"),
    # ("arb_Arab", "esterna",            "arabo",     "Arabo"),
    ("slv_Latn", "esterna",            "sloveno",   "Sloveno"),
    ("hun_Latn", "esterna",            "ungherese", "Ungherese"),
]


def authenticate():
    """Usa HF_TOKEN se impostato, altrimenti il token salvato da
    `huggingface-cli login` (in ~/.cache/huggingface/token)."""
    import huggingface_hub
    from huggingface_hub import whoami

    get_token = getattr(huggingface_hub, "get_token", None)
    if get_token is None:
        from huggingface_hub import HfFolder  # noqa: F401
        get_token = HfFolder.get_token

    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        print("ERRORE: nessun token HuggingFace trovato.", file=sys.stderr)
        print("Fai UNA delle due cose:", file=sys.stderr)
        print("  - hf auth login              (o huggingface-cli login, permanente)", file=sys.stderr)
        print("  - export HF_TOKEN=hf_xxx     (solo per questa sessione)", file=sys.stderr)
        sys.exit(1)
    login(token=token, add_to_git_credential=False)
    try:
        user = whoami(token=token).get("name", "?")
        print(f"Autenticato come: {user}\n")
    except Exception:
        pass


def load_split(flores_code, split):
    ds = load_dataset(DATASET_ID, flores_code, split=split, trust_remote_code=False)
    return [row["text"].strip().replace("\n", " ").replace("\r", " ") for row in ds]


def write_stats(rows):
    header = ["codice", "categoria", "n_frasi", "n_caratteri", "path", "note"]
    with STATS_FILE.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main():
    authenticate()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # raccoglie {short: {"dev": [...], "devtest": [...]}} per il TSV parallelo
    collected = {}
    stats_rows = []

    for flores_code, categoria, short, descr in LANGS:
        desc = f"FLORES+ {flores_code:<10} ({descr})"
        try:
            dev = load_split(flores_code, "dev")
            dvt = load_split(flores_code, "devtest")
            combined = dev + dvt

            out_path = OUT_DIR / f"{short}.txt"
            with out_path.open("w", encoding="utf-8") as f:
                for s in tqdm(combined, desc=desc, unit="frasi"):
                    f.write(s + "\n")

            n_chars = sum(len(s) for s in combined)
            collected[short] = {"dev": dev, "devtest": dvt}
            stats_rows.append([
                short, categoria, len(combined), n_chars,
                str(out_path.relative_to(BASE_DIR)),
                f"{descr} | dev={len(dev)} + devtest={len(dvt)}",
            ])
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            hint = ""
            text = str(e).lower()
            if "gated" in text or "403" in text:
                hint = " | accetta i termini su https://huggingface.co/datasets/openlanguagedata/flores_plus"
            elif "not found" in text or "builderconfig" in text:
                hint = " | questa lingua non e' nella versione di FLORES+ installata"
            elif "401" in text or "unauthorized" in text:
                hint = " | rigenera il token HF"
            print(f"  ERRORE {flores_code} ({descr}): {err}{hint}", file=sys.stderr)
            stats_rows.append([
                short, categoria, 0, 0, "",
                f"{descr} | ERRORE: {err}{hint}",
            ])

    # --- TSV parallelo ----------------------------------------------------
    if collected:
        preferred = ["veneto", "siciliano", "lombardo", "sardo",
                     "ligure", "friulano", "ladino", "napoletano",
                     "italiano",
                     "inglese", "spagnolo", "francese", "catalano",
                     "tedesco", "greco", "arabo", "sloveno"]
        col_order = [c for c in preferred if c in collected]
        extra = sorted(set(collected) - set(col_order))
        col_order += extra

        # lunghezze attese (stesse per tutte le lingue scaricate: FLORES e' allineato)
        n_dev = {len(d["dev"]) for d in collected.values()}
        n_dvt = {len(d["devtest"]) for d in collected.values()}
        if len(n_dev) != 1 or len(n_dvt) != 1:
            print("ATTENZIONE: lunghezze non uniformi tra lingue:", file=sys.stderr)
            for lang, d in collected.items():
                print(f"  {lang}: dev={len(d['dev'])}, devtest={len(d['devtest'])}", file=sys.stderr)
        n_dev = min(n_dev) if n_dev else 0
        n_dvt = min(n_dvt) if n_dvt else 0
        total = n_dev + n_dvt

        tsv_path = OUT_DIR / "parallel.tsv"
        with tsv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
            w.writerow(["sentence_id", "split"] + col_order)
            for i in range(n_dev):
                w.writerow([i, "dev"] + [collected[lang]["dev"][i] for lang in col_order])
            for j in range(n_dvt):
                w.writerow([n_dev + j, "devtest"] + [collected[lang]["devtest"][j] for lang in col_order])

        stats_rows.append([
            "ALL", "parallel_tsv", total,
            sum(len(s) for lang in col_order for s in (collected[lang]["dev"] + collected[lang]["devtest"])),
            str(tsv_path.relative_to(BASE_DIR)),
            f"parallelo: {len(col_order)} lingue, {n_dev} dev + {n_dvt} devtest | colonne: {','.join(col_order)}",
        ])
        print(f"\nTSV parallelo: {tsv_path}  ({total} righe, {len(col_order)} lingue)")

    write_stats(stats_rows)

    print(f"\nStats aggiornate: {STATS_FILE}")
    print("\n--- PROMEMORIA ---")
    print("Il napoletano (nap) NON e' presente in FLORES+: e' un limite noto")
    print("del dataset, non un errore.")
    print("------------------")


if __name__ == "__main__":
    main()
