import os
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential


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
        tuple: (X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names)
    """
    # 1. Separate input features (X) and target variable (y)
    X = df.drop(columns=[target_column])
    y = df[target_column].values
    feature_names = list(X.columns)

    print(f"\nFeatures shape (X): {X.shape}, Target shape (y): {y.shape}")
    print(f"Feature names: {feature_names}")

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

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names


def build_model(input_dim: int = 7) -> Sequential:
    """
    Construct an optimized Deep Learning Sequential regression model.
    
    Architecture:
    - Input Layer: Matches the number of features (7 features).
    - Hidden Layer 1: Dense layer with 32 neurons and ReLU activation.
    - Hidden Layer 2: Dense layer with 16 neurons and ReLU activation.
    - Hidden Layer 3: Dense layer with 8 neurons and ReLU activation.
    - Output Layer: Dense layer with 1 neuron and Linear activation.
    
    Parameters:
        input_dim (int): Number of input features (default is 7).
        
    Returns:
        Sequential: Keras Sequential regression model.
    """
    from tensorflow.keras.optimizers import Adam

    model = Sequential([
        Input(shape=(input_dim,), name="input_layer"),
        Dense(32, activation="relu", name="dense_hidden_1"),
        Dense(16, activation="relu", name="dense_hidden_2"),
        Dense(8, activation="relu", name="dense_hidden_3"),
        Dense(1, activation="linear", name="dense_output")
    ], name="Admission_Prediction_NN")

    # Compile model with Adam optimizer and MSE loss
    model.compile(
        optimizer=Adam(learning_rate=0.01),
        loss="mean_squared_error",
        metrics=["mae"]
    )
    return model


def train_and_evaluate(
    model: Sequential,
    X_train_scaled,
    y_train,
    X_test_scaled,
    y_test,
    epochs: int = 120,
    batch_size: int = 16
):
    """
    Train model with callbacks, evaluate on test set, plot curves, and return training history.
    """
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    callbacks = [
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-5, verbose=1),
        EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True, verbose=1)
    ]

    print("\n--- Training Optimized Model ---")
    history = model.fit(
        X_train_scaled,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )

    print("\n--- Evaluating on Test Set ---")
    y_pred = model.predict(X_test_scaled).flatten()

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\nFinal Test Evaluation:")
    print(f"Test MSE: {mse:.6f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test R2 Score: {r2:.4f}")

    # Plot 1: Loss curves
    plt.figure(figsize=(9, 4.5))
    plt.plot(history.history["loss"], label="Train Loss (MSE)", color="#2563EB", linewidth=2)
    plt.plot(history.history["val_loss"], label="Val Loss (MSE)", color="#DC2626", linewidth=2, linestyle="--")
    plt.title("Optimized Model Loss Across Epochs", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Mean Squared Error", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=300)
    plt.close()
    print("Saved training loss curve to 'loss_curve.png'")

    # Plot 2: Actual vs Predicted
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, color="#4F46E5", edgecolors="black", linewidth=0.5)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Ideal Fit (y = x)")
    plt.title(f"Actual vs Predicted Admission Chance (R² = {r2:.2f})", fontsize=13, fontweight="bold")
    plt.xlabel("Actual Chance of Admit", fontsize=11)
    plt.ylabel("Predicted Chance of Admit", fontsize=11)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("actual_vs_predicted.png", dpi=300)
    plt.close()
    print("Saved prediction scatter plot to 'actual_vs_predicted.png'")

    return history, {"mse": mse, "mae": mae, "r2": r2}


if __name__ == "__main__":
    dataset_path = "Graduate_Admission_Prediction.csv"
    data = load_and_clean_data(dataset_path)
    
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names = preprocess_data(data)
    
    print("\nBuilding Deep Learning Model...")
    model = build_model(input_dim=X_train_scaled.shape[1])
    model.summary()

    # Train and evaluate model
    history, metrics = train_and_evaluate(
        model,
        X_train_scaled,
        y_train,
        X_test_scaled,
        y_test,
        epochs=100,
        batch_size=16
    )

    # Save trained model and scaler
    model_save_path = "admission_model.keras"
    scaler_save_path = "scaler.pkl"
    
    model.save(model_save_path)
    joblib.dump(scaler, scaler_save_path)
    print(f"\nModel saved successfully at '{model_save_path}'")
    print(f"Scaler saved successfully at '{scaler_save_path}'")


