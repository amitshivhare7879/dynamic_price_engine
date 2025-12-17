import streamlit as st
import requests
import json
import pandas as pd
import altair as alt

# --- Configuration ---
API_URL = "http://127.0.0.1:8001/recommend_price" 
REQUIRED_MARGIN = 1.2 

st.set_page_config(
    page_title="Dynamic Pricing Engine Dashboard",
    layout="centered"
)

# Inject custom CSS for better styling and visual separation
st.markdown("""
<style>
/* Centering titles and enhancing headers */
h1 { text-align: center; color: #FF4B4B; }
h2 { color: #6C757D; font-size: 1.5rem; border-bottom: 2px solid #E9ECEF; padding-bottom: 5px; }

/* Styling the results container */
.stMetric {
    background-color: #262730; 
    border-radius: 10px;
    padding: 10px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    min-height: 100px;
}
.stMetric > div:first-child {
    font-size: 1rem;
    color: #FF4B4B; 
}
.stMetric > div:nth-child(2) {
    font-size: 2.5rem;
    font-weight: bold;
    color: #28a745;
}
</style>
""", unsafe_allow_html=True)


# --- App Title and Description ---
st.title("💰 Dynamic Pricing Engine Demo")
st.markdown("### Optimal Price Calculation")
st.markdown("Adjust the market conditions below to receive a real-time price recommendation maximizing revenue.")
st.markdown("---")


# --- 1. Input Form (Centered) ---
st.subheader("📊 Market Context")

col_num_1, col_num_2 = st.columns(2)

with col_num_1:
    number_of_riders = st.slider("👥 Number of Riders (Demand)", min_value=10, max_value=200, value=100, step=1)
    number_of_drivers = st.slider("🚕 Number of Drivers (Supply)", min_value=1, max_value=100, value=50, step=1)
    
with col_num_2:
    past_rides = st.number_input("🔢 Historical Rides (Customer Experience)", min_value=0, value=50)
    avg_ratings = st.slider("⭐ Average Driver Ratings", min_value=3.0, max_value=5.0, value=4.5, step=0.01)
    duration = st.number_input("⏱️ Expected Ride Duration (minutes)", min_value=5, value=30)

st.markdown("---")

# Categorical features in a single row
st.subheader("📍 Environmental Factors")
col_cat_1, col_cat_2, col_cat_3, col_cat_4 = st.columns(4)

with col_cat_1:
    location = st.selectbox("Location Category", ('Urban', 'Suburban', 'Rural'))
with col_cat_2:
    loyalty = st.selectbox("Customer Loyalty Status", ('Regular', 'Silver', 'Gold'))
with col_cat_3:
    time = st.selectbox("Time of Booking", ('Afternoon', 'Morning', 'Evening', 'Night'))
with col_cat_4:
    vehicle = st.selectbox("Vehicle Type", ('Economy', 'Premium'))


# Button to trigger API call
st.markdown("<br>", unsafe_allow_html=True)
col_btn_1, col_btn_2, col_btn_3 = st.columns([1, 2, 1])
if col_btn_2.button("GET OPTIMAL PRICE", type="primary", use_container_width=True):
    payload = {
        "Number_of_Riders": number_of_riders,
        "Number_of_Drivers": number_of_drivers,
        "Number_of_Past_Rides": past_rides,
        "Average_Ratings": avg_ratings,
        "Expected_Ride_Duration": duration,
        "Historical_Cost_of_Ride": 100.0, # Placeholder value
        "Location_Category": location,
        "Customer_Loyalty_Status": loyalty,
        "Time_of_Booking": time,
        "Vehicle_Type": vehicle
    }
    st.session_state['payload'] = payload

# --- 3. Main Display Area (Show Results) ---

if 'payload' in st.session_state:
    st.markdown("---")
    st.subheader("💡 Recommendation Results Flow")
    
    with st.spinner('Calculating optimal price and predicting demand...'):
        try:
            response = requests.post(API_URL, json=st.session_state['payload'])
            
            if response.status_code == 200:
                result = response.json()
                
                if result.get("status") == "error":
                    st.error(f"API Error: {result['message']}")
                elif result.get("best_price") is None:
                    st.warning(f"Optimization failed: {result.get('message', 'No valid price found.')} Please adjust the inputs.")
                else:
                    baseline_cost = result['baseline_cost']
                    min_price_required = baseline_cost * REQUIRED_MARGIN
                    baseline_revenue = result.get('baseline_revenue', 0)
                    
                    # --- Step 1: Baseline Cost & Margin ---
                    col_base_1, col_base_2, col_base_3 = st.columns(3)
                    
                    col_base_1.metric(
                        "1. Estimated Baseline Cost",
                        f"💲 {baseline_cost:.2f}",
                        help="Calculated based on Duration and Vehicle Type."
                    )
                    
                    col_base_2.metric(
                        "Required Minimum Price",
                        f"💲 {min_price_required:.2f}",
                        help=f"Minimum price required to meet the {REQUIRED_MARGIN * 100:.0f}% profit margin."
                    )
                    
                    col_base_3.metric(
                        "Predicted Demand at Optimal Price", 
                        f"👥 {int(result['predicted_demand']):,}",
                        delta=f"Ratio: {st.session_state['payload']['Number_of_Riders'] / st.session_state['payload']['Number_of_Drivers']:.1f}"
                    )

                    st.markdown("---")

                    # --- Step 2 & 3: Final Price & Revenue ---
                    col_final_1, col_final_2 = st.columns(2)

                    # Calculate the growth percentage
                    revenue_growth_perc = 0
                    if baseline_revenue > 0:
                        revenue_growth_perc = ((result['best_revenue'] - baseline_revenue) / baseline_revenue) * 100

                    # 1. Optimal Price
                    col_final_1.metric(
                        "2. Optimal Recommended Price",
                        f"💲 {result['best_price']:.2f}", 
                        delta=f"{(result['best_price'] - min_price_required) / min_price_required * 100:.1f}% above Min Price"
                    )
                                
                    # 2. Maximized Revenue and Growth Percentage
                    col_final_2.metric(
                        "3. Maximized Total Revenue", 
                        f"💰 {result['best_revenue']:.2f}",
                        delta=f"+{revenue_growth_perc:.1f}% Revenue Growth", 
                        delta_color="normal",
                        help=f"Revenue achieved by charging the optimal price versus charging only the minimum required price (💲{baseline_revenue:.2f})."
                    )
                    
                    st.success("Recommendation successful. Price optimized for maximum revenue.")
                    
                    st.markdown("---")
                    st.subheader("📊 Optimization Curve: Revenue vs. Price")

                    # --- ENHANCED CHART VISUALIZATION ---
                    chart_data = pd.DataFrame(result['simulated_data'])
                    chart_data = chart_data.dropna(subset=['Revenue'])
                    
                    optimal_price_df = chart_data[chart_data['Price'] == result['best_price']]
                    
                    # DataFrames for the vertical constraint lines
                    min_price_df = pd.DataFrame({'Min_Price': [min_price_required]})
                    base_cost_df = pd.DataFrame({'Base_Cost': [baseline_cost]})
                    
                    # Base Chart Setup
                    base = alt.Chart(chart_data).encode(
                        x=alt.X('Price', title='Hypothetical Price ($)'),
                        y=alt.Y('Revenue', title='Projected Revenue ($)'),
                        tooltip=['Price', 'Revenue']
                    ).properties(
                        title="Revenue Simulation Across Price Grid"
                    )

                    # 1. Line and Points showing the Revenue Curve
                    line = base.mark_line(color='#4c78a8').encode()
                    points = base.mark_circle().encode()
                    
                    # 2. Highlight the Optimal Price point (the peak)
                    optimal_point = alt.Chart(optimal_price_df).mark_point(
                        filled=True, 
                        color='#28a745', 
                        size=150,
                        strokeWidth=2
                    ).encode(
                        x='Price',
                        y='Revenue',
                        tooltip=['Price', 'Revenue']
                    )

                    # 3. Vertical Line for Required Minimum Price (The Constraint)
                    min_price_line = alt.Chart(min_price_df).mark_rule(color='orange', strokeDash=[5, 5]).encode(
                        x=alt.X('Min_Price', title='Minimum Profitable Price'),
                        tooltip=[alt.Tooltip('Min_Price', title='Min Price Required')]
                    )

                    # 4. Vertical Line for Estimated Baseline Cost (The Floor)
                    base_cost_line = alt.Chart(base_cost_df).mark_rule(color='red', strokeDash=[1, 1]).encode(
                        x=alt.X('Base_Cost', title='Baseline Operational Cost'),
                        tooltip=[alt.Tooltip('Base_Cost', title='Baseline Cost')]
                    )

                    # Combine all layers
                    st.altair_chart(base_cost_line + min_price_line + line + points + optimal_point, use_container_width=True)

            else:
                st.error(f"API Connection Error: Status Code {response.status_code}. Ensure the FastAPI service is running on port 8001.")

        except requests.exceptions.ConnectionError:
            st.error("Connection Failed. The FastAPI service is not reachable. Please ensure you have run the failsafe command in the project root.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")