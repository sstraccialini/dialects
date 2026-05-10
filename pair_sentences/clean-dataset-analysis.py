import pandas as pd
from pathlib import Path

# Load datasets
BASE_DIR = Path('.')
merged_df = pd.read_csv(BASE_DIR / 'merged.tsv', sep='\t')
selected_df = pd.read_csv(BASE_DIR / 'selected.tsv', sep='\t')

print("=" * 80)
print("DATASET COMPARISON: merged.tsv vs selected.tsv")
print("=" * 80)

# Basic statistics
print("\n1. BASIC STATISTICS")
print("-" * 80)
print(f"merged.tsv rows: {len(merged_df)}")
print(f"selected.tsv rows: {len(selected_df)}")
print(f"Rows removed: {len(merged_df) - len(selected_df)} ({100 * (len(merged_df) - len(selected_df)) / len(merged_df):.2f}%)")
print(f"Rows retained: {len(selected_df)} ({100 * len(selected_df) / len(merged_df):.2f}%)")

print(f"\nmerged.tsv columns: {list(merged_df.columns)}")
print(f"selected.tsv columns: {list(selected_df.columns)}")

# Column statistics
print("\n2. COLUMN STATISTICS")
print("-" * 80)
for col in merged_df.columns:
    if col in selected_df.columns:
        print(f"\nColumn: {col}")
        print(f"  merged - dtype: {merged_df[col].dtype}, nulls: {merged_df[col].isnull().sum()}")
        print(f"  selected - dtype: {selected_df[col].dtype}, nulls: {selected_df[col].isnull().sum()}")
        
        # Numeric columns
        if pd.api.types.is_numeric_dtype(merged_df[col]):
            print(f"  merged - mean: {merged_df[col].mean():.2f}, std: {merged_df[col].std():.2f}")
            print(f"  selected - mean: {selected_df[col].mean():.2f}, std: {selected_df[col].std():.2f}")
        
        # String columns
        elif pd.api.types.is_object_dtype(merged_df[col]):
            print(f"  merged - avg length: {merged_df[col].str.len().mean():.2f}")
            print(f"  selected - avg length: {selected_df[col].str.len().mean():.2f}")

# Check for duplicates
print("\n3. DUPLICATE ANALYSIS")
print("-" * 80)
merged_dups = merged_df.duplicated().sum()
selected_dups = selected_df.duplicated().sum()
print(f"merged.tsv duplicates: {merged_dups}")
print(f"selected.tsv duplicates: {selected_dups}")

# Data quality
print("\n4. DATA QUALITY METRICS")
print("-" * 80)
print(f"merged.tsv - null values: {merged_df.isnull().sum().sum()}")
print(f"selected.tsv - null values: {selected_df.isnull().sum().sum()}")
print(f"merged.tsv - completeness: {100 * (1 - merged_df.isnull().sum().sum() / (len(merged_df) * len(merged_df.columns))):.2f}%")
print(f"selected.tsv - completeness: {100 * (1 - selected_df.isnull().sum().sum() / (len(selected_df) * len(selected_df.columns))):.2f}%")

# Intersection and difference
print("\n5. DATASET OVERLAP")
print("-" * 80)
if 'unique_id' in merged_df.columns and 'unique_id' in selected_df.columns:
    merged_ids = set(merged_df['unique_id'])
    selected_ids = set(selected_df['unique_id'])
    intersection = merged_ids & selected_ids
    
    print(f"Unique IDs in merged: {len(merged_ids)}")
    print(f"Unique IDs in selected: {len(selected_ids)}")
    print(f"Overlap: {len(intersection)}")
else:
    print("No 'unique_id' column found for overlap analysis")

# Dataset statistics
print("\n6. DATASET STATISTICS")
print("-" * 80)
dataset_col = None
for candidate in ['dataset', 'Dataset']:
    if candidate in merged_df.columns and candidate in selected_df.columns:
        dataset_col = candidate
        break

if dataset_col:
    for dataset in ['oldi', 'flores']:
        merged_count = (merged_df[dataset_col] == dataset).sum()
        selected_count = (selected_df[dataset_col] == dataset).sum()
        removed_count = merged_count - selected_count
        removed_pct = (100 * removed_count / merged_count) if merged_count else 0
        kept_pct = (100 * selected_count / merged_count) if merged_count else 0

        print(f"{dataset}:")
        print(f"  merged: {merged_count}")
        print(f"  selected: {selected_count}")
        print(f"  removed: {removed_count} ({removed_pct:.2f}%)")
        print(f"  kept: {selected_count} ({kept_pct:.2f}%)")
else:
    print("No 'dataset' column found for per-dataset analysis")

print("\n" + "=" * 80)
