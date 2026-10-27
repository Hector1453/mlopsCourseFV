"""
This module cleans the raw dataset by handling missing values,
removing duplicates and fixing data types.
It outputs a processed dataset ready for machine learning.
"""

import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_PATH = BASE_DIR / "data" / "raw" / "diabetes_dataset.csv"
PROCESSED_PATH = BASE_DIR / "data" / "processed" / "train_v1.csv"




def load_data(path: Path) -> pd.DataFrame:
    """Load a CSV file and return a DataFrame."""
    return pd.read_csv(path)


def copy_data(df: pd.DataFrame) -> pd.DataFrame:
    """Create a copy of the data frame to avoid side effects"""
    df = df.copy()
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values in columns."""
    for col in df.columns:
        df[col] = df[col].fillna(df[col].median())
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicated rows."""
    df = df.drop_duplicates()
    return df

def fix_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Convert data to float, coercing errors to NaN."""
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning steps and return the cleaned DataFrame."""
    df = copy_data(df)
    df = fix_data_types(df)
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    
    return df


def save_data(df: pd.DataFrame, path: Path) -> None:
    """Save DataFrame to CSV without the index."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    """Run the full preprocessing pipeline."""
    print("Loading raw data...")
    df = load_data(RAW_PATH)
    shape_before = df.shape
    print(f"Shape before cleaning: {shape_before}")

    print("Cleaning data...")
    df_cleaned = clean_data(df)
    shape_after = df_cleaned.shape

    print("Saving processed data...")
    save_data(df_cleaned, PROCESSED_PATH)

    print(f"Final shape: {shape_after}")
    print(f"Output path: {PROCESSED_PATH}")


if __name__ == "__main__":
    main()
