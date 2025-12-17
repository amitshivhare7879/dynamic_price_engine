import numpy as np
import pandas as pd
from joblib import load
from pathlib import Path
import math

# --- Configuration (Model Loading Paths) ---
ENGINE_DIR = Path(__file__).resolve().parent
# Goes up two levels (src/engine/ -> src/ -> project_root/)
MODEL_PATH = ENGINE_DIR.parent.parent / 'models' / 'demand_model.joblib'
FEATURES_PATH = ENGINE_DIR.parent.parent / 'models' / 'feature_columns.csv'

# --- Business Constraints ---
BASE_COST_PER_MINUTE = 2.50 # op cs p/m
REQUIRED_MARGIN = 1.2    

# --- Model and Feature Loading ---
DEMAND_MODEL = None
FEATURE_COLUMNS = []
try:
    DEMAND_MODEL = load(str(MODEL_PATH)) 
    FEATURE_COLUMNS = pd.read_csv(str(FEATURES_PATH))['feature_name'].tolist()
    print("Pricing Engine: Model and features loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model or feature list not found. Checked paths: {MODEL_PATH}")
    DEMAND_MODEL = Exception("Model files missing.") 
except Exception as e:
    DEMAND_MODEL = Exception(f"Failed to load model: {e}")


def estimate_baseline_cost(context: dict) -> float:
    """
    Estimates the standard, non-dynamic cost based on duration and vehicle type.
    """
    duration = context['Expected_Ride_Duration']
    
    baseline_price = 50.0 + (duration * BASE_COST_PER_MINUTE)
    
    if context['Vehicle_Type'] == 'Premium':
        baseline_price *= 1.40 
        
    return math.ceil(baseline_price)


def meets_constraints(new_price: float, baseline_cost: float) -> bool:
    """Checks if the price meets the defined business margin constraint."""
    if new_price < (baseline_cost * REQUIRED_MARGIN):
        return False
    return True
# prepares features and cal the XGboost model
def get_demand_at_price(context: dict, price: float, feature_cols: list, model) -> float:
    """Calculates the predicted demand for a specific price point."""
    sim_context = context.copy()
    sim_context['Historical_Cost_of_Ride'] = price
    
    # Feature Engineering (must match the main loop)
    sim_context['rider_driver_ratio'] = sim_context['Number_of_Riders'] / sim_context['Number_of_Drivers']
    sim_context['log_cost'] = np.log1p(sim_context['Historical_Cost_of_Ride'])
    sim_context['avg_ratings_sq'] = context['Average_Ratings'] ** 2 
    
    # Prepare feature vector
    X_processed = pd.DataFrame(0, index=[0], columns=feature_cols)
    for col in sim_context.keys():
        if col in X_processed.columns:
            X_processed[col] = sim_context[col]
        elif col in ['Location_Category', 'Customer_Loyalty_Status', 'Time_of_Booking', 'Vehicle_Type']:
            dummy_col = f'{col}_{sim_context[col]}'
            if dummy_col in X_processed.columns:
                X_processed[dummy_col] = 1

    pred_demand = model.predict(X_processed[feature_cols])[0]
    return max(0, pred_demand)


def get_baseline_revenue(context: dict, min_price: float, feature_cols: list, model) -> float:
    """Calculates the revenue if we only charged the minimum required price."""
    pred_demand = get_demand_at_price(context, min_price, feature_cols, model)
    return pred_demand * min_price

#optimization engine
def recommend_price(context: dict) -> dict:
    """
    Implements the price optimization algorithm using the predicted demand.
    """
    if isinstance(DEMAND_MODEL, Exception):
        return {"error": f"Model initialization failed: {DEMAND_MODEL}"}
        
    baseline_cost = estimate_baseline_cost(context)
    min_price_required = baseline_cost * REQUIRED_MARGIN
    base_price_to_search = baseline_cost

    simulated_results = []
    
    # --- 1. Define Dynamic Price Grid ---
    ratio = context['Number_of_Riders'] / context['Number_of_Drivers']
    dynamic_factor = np.clip(ratio * 0.10, 0.15, 1.00) # Max range 100%
    
    price_grid = np.linspace(base_price_to_search * (1 - dynamic_factor), base_price_to_search * (1 + dynamic_factor), 15) 
    
    best_price = None
    best_revenue = -1.0 
    predicted_demand_at_optimal = 0.0
    
    # --- 2. Iterate through Price Grid and Optimize ---
    for p in price_grid:
        p = round(p, 2)
        
        sim_context = context.copy()
        sim_context['Historical_Cost_of_Ride'] = p
        
        # Apply Feature Engineering transformations 
        sim_context['rider_driver_ratio'] = sim_context['Number_of_Riders'] / sim_context['Number_of_Drivers']
        sim_context['log_cost'] = np.log1p(sim_context['Historical_Cost_of_Ride'])
        sim_context['avg_ratings_sq'] = context['Average_Ratings'] ** 2
        
        # Prepare feature vector for model prediction
        X_processed = pd.DataFrame(0, index=[0], columns=FEATURE_COLUMNS)
        
        for col in sim_context.keys():
            if col in X_processed.columns:
                X_processed[col] = sim_context[col]
            elif col in ['Location_Category', 'Customer_Loyalty_Status', 'Time_of_Booking', 'Vehicle_Type']:
                dummy_col = f'{col}_{sim_context[col]}'
                if dummy_col in X_processed.columns:
                    X_processed[dummy_col] = 1

        is_valid_price = meets_constraints(p, baseline_cost)
        current_revenue = 0.0

        if is_valid_price:
            pred_demand = DEMAND_MODEL.predict(X_processed[FEATURE_COLUMNS])[0]
            pred_demand = max(0, pred_demand) 
            current_revenue = pred_demand * p

            if current_revenue > best_revenue:
                best_price = p
                best_revenue = current_revenue
                predicted_demand_at_optimal = pred_demand 
        
        simulated_results.append({
            'Price': p, 
            'Revenue': current_revenue if is_valid_price else None,
            'Valid': is_valid_price
        })
                
    if best_price is None:
        # --- FALLBACK LOGIC: Default to Min Required Price ---
        default_price = min_price_required
        
        pred_demand_at_default = get_demand_at_price(
            context, default_price, FEATURE_COLUMNS, DEMAND_MODEL
        )
        default_revenue = default_price * pred_demand_at_default
        
        # Set all values to the default floor value
        return {
            "best_price": float(default_price),
            "best_revenue": float(default_revenue),
            "predicted_demand": float(pred_demand_at_default),
            "baseline_cost": float(baseline_cost),
            "simulated_data": simulated_results, 
            "baseline_revenue": float(default_revenue)
        }
    
    # --- SUCCESSFUL OPTIMIZATION PATH ---
    baseline_revenue_final = get_baseline_revenue(context, min_price_required, FEATURE_COLUMNS, DEMAND_MODEL)

    return {
        "best_price": float(best_price), 
        "best_revenue": float(best_revenue), 
        "predicted_demand": float(predicted_demand_at_optimal),
        "baseline_cost": float(baseline_cost),
        "simulated_data": simulated_results,
        "baseline_revenue": float(baseline_revenue_final)
    }