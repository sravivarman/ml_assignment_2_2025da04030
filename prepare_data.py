"""
prepare_data.py

Prepares the Student Dropout and Academic Success dataset for model training.

Tasks Performed:
1. Load the raw dataset.
2. Clean column names.
3. Verify the target column exists.
4. Check for missing values and duplicate records.
5. Save the cleaned dataset.
6. Create an 80:20 stratified train-test split.
7. Save the training and testing datasets.

Generated Files:
- data/full_dataset.csv
- data/train_only.csv
- test_data.csv
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------------
# Project Paths
# ------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_PATH = DATA_DIR / "dataset_raw.csv"
FULL_DATA_PATH = DATA_DIR / "full_dataset.csv"
TRAIN_DATA_PATH = DATA_DIR / "train_only.csv"
TEST_DATA_PATH = PROJECT_ROOT / "test_data.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.20

# ------------------------------------------------------------------
# Load Dataset
# ------------------------------------------------------------------

print("=" * 60)
print("Loading Dataset")
print("=" * 60)

# utf-8-sig removes the Byte Order Mark (BOM) if present
df = pd.read_csv(RAW_DATA_PATH, sep=";", encoding="utf-8-sig")

# ------------------------------------------------------------------
# Clean Column Names
# ------------------------------------------------------------------

df.columns = df.columns.str.strip()

# ------------------------------------------------------------------
# Validate Dataset
# ------------------------------------------------------------------

if "Target" not in df.columns:
    raise ValueError(
        f"Target column not found.\nAvailable columns:\n{list(df.columns)}"
    )

# ------------------------------------------------------------------
# Dataset Information
# ------------------------------------------------------------------

print(f"\nDataset Shape : {df.shape}")

print("\nMissing Values")
print(df.isnull().sum().sum())

print("\nDuplicate Rows")
print(df.duplicated().sum())

print("\nTarget Class Distribution")
print(df["Target"].value_counts())

# ------------------------------------------------------------------
# Save Clean Dataset
# ------------------------------------------------------------------

df.to_csv(FULL_DATA_PATH, index=False)

# ------------------------------------------------------------------
# Train-Test Split
# ------------------------------------------------------------------

train_df, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=df["Target"],
)

# ------------------------------------------------------------------
# Save Files
# ------------------------------------------------------------------

train_df.to_csv(TRAIN_DATA_PATH, index=False)
test_df.to_csv(TEST_DATA_PATH, index=False)

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------

print("\n" + "=" * 60)
print("Dataset Preparation Completed Successfully")
print("=" * 60)

print(f"Full Dataset Shape : {df.shape}")
print(f"Training Set Shape : {train_df.shape}")
print(f"Testing Set Shape  : {test_df.shape}")

print("\nGenerated Files")
print(f"✓ {FULL_DATA_PATH.relative_to(PROJECT_ROOT)}")
print(f"✓ {TRAIN_DATA_PATH.relative_to(PROJECT_ROOT)}")
print(f"✓ {TEST_DATA_PATH.relative_to(PROJECT_ROOT)}")

print("\nClass Distribution (Full Dataset)")
print(df["Target"].value_counts())

print("\nData preparation completed successfully.")