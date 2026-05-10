#!/usr/bin/env python3
"""Print sentences from a table file for a given language column.

Usage: print-sentences.py --file PATH --lang LANG

The script reads a delimited table (CSV/TSV) with a header row and prints
the values of the column whose header matches LANG. If LANG is not found,
an error is printed to stderr and the program exits with code 1.
"""
import sys
import argparse
import csv


def detect_dialect(sample: str):
    # If the first column is an id and the header uses tabs, prefer tab-delimited
    try:
        first_line = sample.splitlines()[0] if sample else ''
        if first_line and first_line.split('\t')[0].lower() == 'id':
            return csv.excel_tab
    except Exception:
        pass
    try:
        return csv.Sniffer().sniff(sample, delimiters='\t,;')
    except Exception:
        return csv.get_dialect('excel')


def main():
    p = argparse.ArgumentParser(description='Print sentences from a language column')
    p.add_argument('--file', '-f', required=False, default='llm-translations/oldi_selected.tsv', help='Path to table file')
    p.add_argument('--lang', '-l', required=False, help='Language column header to extract')
    args = p.parse_args()

    try:
        with open(args.file, 'r', encoding='utf-8') as fh:
            sample = fh.read(4096)
            fh.seek(0)
            dialect = detect_dialect(sample)
            reader = csv.DictReader(fh, dialect=dialect)
            fields = reader.fieldnames or []
            # If language not provided, prompt the user with alternatives
            if not args.lang:
                if not fields:
                    sys.stderr.write("No header/columns found in file.\n")
                    sys.exit(1)
                sys.stdout.write("Available columns:\n")
                for i, f in enumerate(fields, 1):
                    sys.stdout.write(f"  {i}. {f}\n")
                choice = input("Select a column by number or name: ").strip()
                # accept numeric selection
                if choice.isdigit():
                    idx = int(choice) - 1
                    if 0 <= idx < len(fields):
                        args.lang = fields[idx]
                else:
                    if choice in fields:
                        args.lang = choice
                if not args.lang:
                    sys.stderr.write("No valid selection made.\n")
                    sys.exit(1)

            if args.lang not in fields:
                sys.stderr.write(f"Column '{args.lang}' not found. Available: {fields}\n")
                sys.exit(1)

            for row in reader:
                val = row.get(args.lang, '')
                if val is None:
                    val = ''
                print(val)
    except FileNotFoundError:
        sys.stderr.write(f"File not found: {args.file}\n")
        sys.exit(1)
    except Exception as e:
        sys.stderr.write(str(e) + "\n")
        sys.exit(1)


if __name__ == '__main__':
    main()
