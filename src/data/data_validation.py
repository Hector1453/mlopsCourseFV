"""
This module validates the dataset using Pydantic models to ensure data structure
and types before preprocessing.
"""

from typing import Optional

import pandas as pd
import numpy as np
from pydantic import BaseModel, ValidationError


class DiabetesData(BaseModel):
    """Pydantic model representing a single row of diabetes data."""

    age: float                    # Age in years
    sex: float                    # Gender of the patient
    bmi: float                    # Body mass index
    bp: float                     # Average blood pressure
    s1: float                     # Total serum cholesterol (tc)
    s2: float                     # Low-density lipoproteins (ldl)
    s3: float                     # High-density lipoproteins (hdl)
    s4: float                     # Total cholesterol / HDL (tch)
    s5: float                     # Possibly log of serum triglycerides level (ltg)
    s6: float                     # Blood sugar level (glu)
    target: float                 # Measure of disease progression one year after baseline


def validate_row(row: dict, index: int) -> bool:
    """Try to validate a single row against the DiabetesData model.

    Args:
        row: A dictionary representing one row of the dataset.
        index: The row index, used for error messages.

    Returns:
        True if the row is valid, False otherwise.
    """
    try:
        DiabetesData(**row)
        return True
    except ValidationError as e:
        print(f"Row {index} failed validation: {e}")
        return False

    
def pydantic_validation(df: pd.DataFrame) -> None:
    """Validate all rows in the dataset and print a summary.

    Args:
        df: The pandas DataFrame to validate.
    """
    total = len(df)
    valid = 0
    invalid = 0

    for index, row in enumerate(df.to_dict("records")):
        if validate_row(row, index):
            valid += 1
        else:
            invalid += 1

    print("\n--- Pydantic Validation Summary ---")
    print(f"Total rows:   {total}")
    print(f"Valid rows:   {valid}")
    print(f"Invalid rows: {invalid}\n")


def check_target(df: pd.DataFrame) -> None:
    """Checks if target is in the dataset.

    Args:
        df: The pandas DataFrame to validate.
    """
    if 'target' not in df.columns: 
        print("ERROR: Target column missing!")
    else:
        print("OK: Target column found")


def check_features(df: pd.DataFrame) -> None:    
    """Checks if the expected features are in the dataset.

    Args:
        df: The pandas DataFrame to validate.
    """
    required_features = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
    error_number = 0
    for col in required_features:
        if col not in df.columns:
            print(f"Feature {col} is missing!")
            error_number = error_number + 1
    if error_number == 0:
        print("OK: All features correctly found")
    else:
        print(f"ERROR: {error_number} features are absent!")


def check_numeric(df: pd.DataFrame) -> None:    
    """Checks if all columns are numeric.

    Args:
        df: The pandas DataFrame to validate.
    """
    if df.select_dtypes(include=[np.number]).shape[1] == df.shape[1]:
        print("OK: All data are numeric")
    else:
        print("ERROR: Non-numeric data detected!")


def check_missing(df: pd.DataFrame) -> None:    
    """Checks missing values.

    Args:
        df: The pandas DataFrame to validate.
    """
    if df.isnull().sum().sum() == 0: 
        print("OK: No missing values")
    else:
        print("ERROR: Critical missing values found!")


def check_size(df: pd.DataFrame) -> None:
    """Checks dataset size.

    Args:
        df: The pandas DataFrame to validate.
    """
    if len(df) > 0: 
        print("OK: Dataset is not empty")
    else:
        print("ERROR: Dataset is empty!")


def main():
    """Load the raw dataset and run validation."""
    path = "data/raw/diabetes_dataset.csv"
    print("Starting Data Validation")
    print(f"Loading dataset from: {path}")
    df = pd.read_csv(path)
    check_features(df)
    check_target(df)
    check_numeric(df)
    check_missing(df)
    check_size(df)
    pydantic_validation(df)
    print("Data Validation Ended!")


if __name__ == "__main__":
    main()
