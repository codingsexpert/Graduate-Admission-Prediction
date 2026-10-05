import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


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


def preprocess_data(
    df: pd.DataFrame,
    target_column: str = "Chance of Admit",
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Split features and target, perform train-test split, and scale features with MinMaxScaler.
    
    Parameters:
        df (pd.DataFrame): Cleaned input DataFrame.
        target_column (str): Name of the target variable column.
        test_size (float): Proportion of dataset to include in the test split.
        random_state (int): Random state seed for reproducibility.
        
    Returns:
        tuple: (X_train_scaled, X_test_scaled, y_train, y_test, scaler)
    """
    # 1. Separate input features (X) and target variable (y)
    X = df.drop(columns=[target_column])
    y = df[target_column].values

    print(f"\nFeatures shape (X): {X.shape}, Target shape (y): {y.shape}")
    print(f"Feature names: {list(X.columns)}")

    # 2. Split into train and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")

    # 3. Scale input features using MinMaxScaler to normalize into [0, 1] range
    # Fit only on training data to prevent data leakage into the test set
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Features successfully normalized using MinMaxScaler.")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


if __name__ == "__main__":
    dataset_path = "Graduate_Admission_Prediction.csv"
    data = load_and_clean_data(dataset_path)
    
    X_train, X_test, y_train, y_test, scaler = preprocess_data(data)
    print("\nSample scaled feature vector (first training row):")
    print(X_train[0])

