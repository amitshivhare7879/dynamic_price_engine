# DYNAMIC PRICING ENGINE

## Project Overview

This is a production-grade Dynamic Pricing Engine built using Python (FastAPI, XGBoost, Streamlit). The system is designed to maximize revenue for a ride-sharing service by adapting pricing in real-time based on market dynamics (demand elasticity, supply/demand ratio) while ensuring a minimum profit margin is always met.

**Business Goal:** To overcome the 8-15% revenue loss associated with static pricing by introducing a data-driven, adaptive strategy.

### Key Deliverables

1.  **Demand Prediction Model:** An XGBoost Regressor trained to predict customer demand for any given price and context.
2.  **Optimization Engine:** A service that executes a constrained search ($\max(\text{Price} \times \text{Predicted Demand})$).
3.  **Real-time API:** A high-performance FastAPI service to deliver price recommendations.
4.  **Interactive Dashboard:** A Streamlit front-end for simulation and visualization of the optimization curve.

-----

## PROJECT ARCHITECTURE

The system is split into two primary, separate services:

1.  **BACKEND API:** Serves the optimization logic and ML model.
2.  **FRONTEND DASHBOARD:** Provides the interactive user interface.

### Project Structure

```
dynamic_price_engine/
├── data/
│   └── raw/
│       └── dynamic_pricing.csv       # Training dataset
├── models/
│   ├── demand_model.joblib           # The trained XGBoost model artifact
│   └── feature_columns.csv           # Feature contract (input order for API)
├── src/
│   ├── api/
│   │   └── main.py                   # FastAPI application definition
│   ├── engine/
│   │   └── pricing_engine.py         # Core optimization logic
│   └── models/
│       └── train_demand.py           # ML model training script
├── dashboard/
│   └── app.py                        # Streamlit dashboard interface
├── Procfile                          # Render/Deployment instructions
└── requirements.txt                  # Project dependencies
```

-----

## INSTALLATION AND SETUP

### Prerequisites

  * Python 3.9+
  * Git
  * Windows PowerShell (Recommended for simplified command execution)

### 1. Clone the Repository

```bash
git clone YOUR_REPOSITORY_URL
cd dynamic_price_engine
```

### 2. Create and Activate Virtual Environment

```bash
# Create the environment
python -m venv venv

# Activate the environment (PowerShell)
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

Install all necessary packages, ensuring the environment is complete for both model serving and visualization.

```bash
pip install -r requirements.txt
```

### 4. Training the Model (Crucial Step)

The API requires the model files to exist. You must run the training script once to generate the artifacts.

```bash
python src/models/train_demand.py
```

  * **Result:** This creates `demand_model.joblib` and `feature_columns.csv` in the `models/` directory.

-----

## LOCAL DEPLOYMENT (TWO-TERMINAL LAUNCH)

The application must be run using two separate, simultaneous processes.

### Terminal 1: Launch the FastAPI Backend (API)

The following command uses a failsafe method (`$env:PYTHONPATH`) to ensure the server correctly finds Python files inside the `src` folder.

1.  **Ensure `(venv)` is active.**
2.  **Ensure you are in the Project Root.**
3.  Execute the launch command:

```bash
$env:PYTHONPATH="./src" ; .\venv\Scripts\python -m uvicorn api.main:app --reload --port 8001
```

  * **Verification:** The server is running on `http://127.0.0.1:8001`. Look for the message: `Pricing Engine: Model and features loaded successfully.` 

### Terminal 2: Launch the Streamlit Frontend (Dashboard)

1.  **Open a NEW Terminal and activate `(venv)`**
2.  **Ensure you are in the Project Root.**
3.  Execute the Streamlit launch command:

```bash
streamlit run dashboard/app.py --server.port 8502
```

  * **Verification:** The application opens in your web browser at `http://localhost:8502`.

-----

## BUSINESS LOGIC AND TECHNICAL HIGHLIGHTS

### Optimization Strategy

The core logic (`pricing_engine.py`) performs a constrained revenue maximization:

$$\text{Optimal Price} = \underset{P}{\arg \max} \left( P \times \text{Demand}(P) \right) \quad \text{subject to } P \ge \text{Min Required Price}$$

**Key Features:**

  * **Dynamic Price Grid:** The range of prices searched is dynamically scaled by the Rider-to-Driver Ratio, allowing for adaptive surging/discounting.
  * **Profit Constraint:** Every price tested is filtered by the `meets_constraints` function, guaranteeing a minimum margin (e.g., 20%) is met.
  * **Fallback Logic:** In low-demand scenarios where no tested price is profitable, the engine defaults its recommendation to the **Required Minimum Price** to protect the financial floor.

### API Endpoints

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/` | `GET` | API Health Check (Returns status operational). |
| `/recommend_price` | `POST` | **Main Endpoint.** Receives the market context and returns the optimal price, revenue, and simulation data. |
| `/docs` | `GET` | Interactive OpenAPI documentation (Swagger UI). |
