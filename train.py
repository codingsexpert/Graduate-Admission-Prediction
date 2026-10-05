"""
Graduate Admission Prediction - Deep Learning Regression
Step 1: Data Loading and Cleaning
"""

import pandas as pd


def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """
    Load dataset from CSV, clean column headers, and remove irrelevant columns.
    
    Parameters:
        filepath (str): Path to the CSV dataset.
        
    Returns:
        pd.DataFrame: Cleaned DataFrame ready for preprocessing.
    """
    # 1. Load the dataset using Pandas
    df = pd.read_csv(filepath)
    print(f"Dataset successfully loaded. Initial shape: {df.shape}")

    # 2. Strip any extra leading or trailing whitespaces from column names
    df.columns = df.columns.str.strip()
    print(f"Cleaned columns: {list(df.columns)}")

    # 3. Drop irrelevant columns ('Serial No.')
    if "Serial No." in df.columns:
        df = df.drop(columns=["Serial No."])
        print("Dropped irrelevant column: 'Serial No.'")

    # 4. Check for missing / null values
    null_counts = df.isnull().sum().sum()
    print(f"Total missing values found: {null_counts}")
    
    print(f"Final shape after cleaning: {df.shape}")
    return df


if __name__ == "__main__":
    dataset_path = "Graduate_Admission_Prediction.csv"
    data = load_and_clean_data(dataset_path)
    print("\nFirst 5 rows of cleaned data:")
    print(data.head())
