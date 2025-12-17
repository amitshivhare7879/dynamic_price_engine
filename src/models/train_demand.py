import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from joblib import dump
import mlflow
import os
from pathlib import Path

# --- Configuration (Paths relative to project root) ---
# Get the absolute path of the current script's directory
SCRIPT_DIR = Path(__file__).resolve().parent
# Navigate up to the project root, then to data/raw/
FILE_PATH = SCRIPT_DIR.parent.parent / 'data' / 'raw' / 'dynamic_pricing.csv'
TARGET_COL = 'Number_of_Riders'
PRICE_COL = 'Historical_Cost_of_Ride' 

# --- 1. Helper Functions and Metrics ---

def mean_absolute_percentage_error(y_true, y_pred):
    """Calculates the MAPE metric, robust to zero actual values."""
    y_true_clean = np.where(y_true == 0, 1, y_true)
    return np.mean(np.abs((y_true - y_pred) / y_true_clean)) * 100

def load_and_preprocess_data(file_path):
    """
    Loads data, performs preprocessing (OHE), and engineers features.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {file_path}.")
        return None, None
    
    # --- Feature Engineering ---
    # Rationale: Ratio captures the current market imbalance (Supply vs. Base Demand)
    df['rider_driver_ratio'] = df['Number_of_Riders'] / df['Number_of_Drivers']
    
    # Rationale: Log transforms often normalize skewed price distributions, aiding model stability.
    df['log_cost'] = np.log1p(df[PRICE_COL])
    
    # Rationale: Squared ratings captures a non-linear benefit of very high ratings.
    df['avg_ratings_sq'] = df['Average_Ratings'] ** 2
    
    # Rationale: One-Hot Encoding (OHE) is required for nominal (unordered) categorical features 
    # to prevent the model from assuming false ordinal relationships.
    categorical_cols = ['Location_Category', 'Customer_Loyalty_Status', 'Time_of_Booking', 'Vehicle_Type']
    df = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols, drop_first=True)
    
    # Define Target and Features
    X = df.drop(columns=[TARGET_COL, PRICE_COL]) # Price is excluded since it's the variable we will manipulate
    y = df[TARGET_COL]
    
    # Save feature columns list for consistency in the API/Engine
    models_dir = SCRIPT_DIR.parent.parent / 'models'
    models_dir.mkdir(exist_ok=True)
    pd.DataFrame({'feature_name': X.columns}).to_csv(models_dir / 'feature_columns.csv', index=False)
    
    return X, y

def train_demand_model(X, y):
    """Performs split, trains, evaluates, and selects/saves the best model."""
    
    # Rationale: Standard 80/10/10 split is used due to the lack of a date column for time-series validation.
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    mlflow.set_experiment("Ride_Sharing_Demand_Model")
    model_results = {}
    
    # --- XGBoost Regressor ---
    # Rationale: Chosen for its high performance and robustness on structured data.
    with mlflow.start_run(run_name="XGBoost_100"):
        xgb_model = XGBRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
        val_mape = mean_absolute_percentage_error(y_val, xgb_model.predict(X_val))
        mlflow.log_metric("val_mape", val_mape)
        model_results['XGBoost'] = {'model': xgb_model, 'mape': val_mape}
        print(f"XGBoost Validation MAPE: {val_mape:.2f}%")

    # --- Robust Random Forest ---
    # Rationale: Required as a complex tree-based alternative for comparison.
    with mlflow.start_run(run_name="RandomForest_200"):
        rf_model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1).fit(X_train, y_train)
        val_mape = mean_absolute_percentage_error(y_val, rf_model.predict(X_val))
        mlflow.log_metric("val_mape", val_mape)
        model_results['RandomForest'] = {'model': rf_model, 'mape': val_mape}
        print(f"RandomForest Validation MAPE: {val_mape:.2f}%")

    # --- Linear Regression (Baseline) ---
    # Rationale: Provides a simple, interpretable baseline to ensure the complex models add value.
    with mlflow.start_run(run_name="LinearRegression_Baseline"):
        lr_model = LinearRegression().fit(X_train, y_train)
        val_mape = mean_absolute_percentage_error(y_val, lr_model.predict(X_val))
        mlflow.log_metric("val_mape", val_mape)
        model_results['LinearRegression'] = {'model': lr_model, 'mape': val_mape}
        print(f"Linear Regression Baseline MAPE: {val_mape:.2f}%")

    # Select and Save Best Model
    best_model_name = min(model_results, key=lambda k: model_results[k]['mape'])
    best_model = model_results[best_model_name]['model']
    best_mape = model_results[best_model_name]['mape']
    
    # Rationale: Joblib is used for efficient serialization of the model object.
    models_dir = SCRIPT_DIR.parent.parent / 'models'
    model_path = models_dir / 'demand_model.joblib'
    dump(best_model, model_path)
    
    print(f"\nBest Model Selected: {best_model_name} (MAPE: {best_mape:.2f}%)")
    print(f"Model saved to {model_path}")
    
    return best_model, X_test, y_test


if __name__ == '__main__':
    print("--- Starting Demand Model Training Script ---")
    X_full, y_full = load_and_preprocess_data(FILE_PATH)
    
    if X_full is not None:
        trained_model, X_test, y_test = train_demand_model(X_full, y_full)
        test_mape = mean_absolute_percentage_error(y_test, trained_model.predict(X_test))
        
        # Rationale: Final Test MAPE provides the true generalization performance.
        print(f"\n--- Final Test Set Evaluation ---")
        print(f"Best Model Test MAPE: {test_mape:.2f}%")