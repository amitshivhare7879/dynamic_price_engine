from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal

# Final stable import using the simple absolute package path
from engine.pricing_engine import recommend_price

# --- 1. Define the Input Data Schema (Pydantic Model) ---
class PricingContext(BaseModel):
    Number_of_Riders: int
    Number_of_Drivers: int
    Number_of_Past_Rides: int
    Average_Ratings: float
    Expected_Ride_Duration: int
    Historical_Cost_of_Ride: float # Placeholder/Ignored in final logic

    Location_Category: Literal['Urban', 'Suburban', 'Rural']
    Customer_Loyalty_Status: Literal['Silver', 'Regular', 'Gold']
    Time_of_Booking: Literal['Night', 'Evening', 'Afternoon', 'Morning']
    Vehicle_Type: Literal['Premium', 'Economy']

# --- 2. Initialize FastAPI Application ---
app = FastAPI(
    title="Dynamic Pricing Engine API",
    description="Provides optimal price recommendations.",
    version="1.0.0"
)

# --- 3. Define the Recommendation Endpoint ---
@app.post("/recommend_price", tags=["Recommendations"])
def get_recommendation(context: PricingContext):
    try:
        context_dict = context.model_dump()
        recommendation = recommend_price(context_dict)
        
        if "error" in recommendation:
            return {"status": "error", "message": recommendation["error"]}

        return recommendation
    
    except Exception as e:
        print(f"An error occurred during recommendation: {e}")
        return {"status": "error", "message": "An internal processing error occurred."}

# --- 4. Root Endpoint (Optional) ---
@app.get("/", tags=["Status"])
def read_root():
    return {"status": "Pricing Engine API is operational"}