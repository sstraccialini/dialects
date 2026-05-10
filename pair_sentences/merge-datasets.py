#!/usr/bin/env python3
"""Merge two TSV datasets with custom dataset names.

Usage: merge-datasets.py out.tsv input1.tsv input2.tsv name1 name2

Produces a TSV with columns:
 unique_id (global sequential starting at 1)
 dataset (custom dataset name)
 original_id (name_rowindex, 1-based)
 ... followed by the original columns in order
"""
import sys
import csv
from pathlib import Path


def main(argv):
    if len(argv) < 6:
        print("Usage: merge-datasets.py out.tsv input1.tsv input2.tsv name1 name2")
        return 1

    out_path = Path(argv[1])
    inputs = [Path(argv[2]), Path(argv[3])]
    names = [argv[4], argv[5]]

    if not inputs:
        print("No input files")
        return 1

    unique_id = 1
    # Requested output schema: unique_id as the identifier, followed by the
    # dataset/original_id metadata and the language columns in this order.
    fieldnames = [
        "unique_id", "dataset", "original_id",
        "italiano", "veneto", "siciliano", "lombardo", "sardo", "ligure",
        "friulano", "inglese", "spagnolo", "francese", "tedesco", "catalano",
        "sloveno",
    ]

    common_columns = None
    for p in inputs:
        with p.open("r", newline='', encoding='utf-8') as fin:
            reader = csv.DictReader(fin, delimiter='\t')
            if reader.fieldnames:
                file_columns = set(reader.fieldnames)
                if common_columns is None:
                    common_columns = file_columns
                else:
                    common_columns &= file_columns

    dropped_columns = set()
    def is_id_column(name: str) -> bool:
        n = name.lower()
        return n == 'id' or n.endswith('_id')

    if common_columns:
        for p in inputs:
            with p.open("r", newline='', encoding='utf-8') as fin:
                reader = csv.DictReader(fin, delimiter='\t')
                if reader.fieldnames:
                    for name in reader.fieldnames:
                        if is_id_column(name) or name not in common_columns:
                            # drop any id-like columns and any columns not shared by both inputs
                            dropped_columns.add(name)
                    break

    if dropped_columns:
        print("Dropped columns:", ", ".join(sorted(dropped_columns)))

    with out_path.open("w", newline='', encoding='utf-8') as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()

        for p, dataset_name in zip(inputs, names):
            with p.open("r", newline='', encoding='utf-8') as fin:
                reader = csv.DictReader(fin, delimiter='\t')

                for idx, row in enumerate(reader, start=1):
                    out_row = {k: row.get(k, "") for k in fieldnames}
                    out_row.update({
                        "unique_id": str(unique_id),
                        "dataset": dataset_name,
                        "original_id": f"{dataset_name}_{idx}",
                    })
                    # ensure order by writing full fieldnames
                    writer.writerow(out_row)
                    unique_id += 1

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
