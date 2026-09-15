import streamlit as st
import pickle
import numpy as np
import os

# 1. Page Configuration (Must be the first Streamlit command)
st.set_page_config(
    page_title="SaaS Revenue Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Cache the models to prevent reloading on every button click (Professional Practice)
@st.cache_resource
def load_models():
    try:
        # Load the scaler and model
        scaler_obj = pickle.load(open('scaler.pkl', 'rb'))
        model_obj = pickle.load(open('model.pkl', 'rb'))
        return scaler_obj, model_obj
    except FileNotFoundError:
        st.error("System Error: Model files not found. Please ensure 'scaler.pkl' and 'model.pkl' exist in the directory.")
        return None, None

scaler, model = load_models()

# 3. Main Dashboard Header
st.title("SaaS Startup Revenue Predictor")
st.markdown("---")
st.markdown(
    """
    Welcome to the Revenue Prediction Dashboard. This tool utilizes a **Lasso Regression Model (L1 Regularization)** 
    to forecast corporate profitability based on departmental expenditures and geographical location.
    """
)

# 4. Sidebar for Inputs (Makes it look like a real SaaS Dashboard)
st.sidebar.header("Financial Inputs")
st.sidebar.markdown("Enter the startup's financial data below:")

rd_spend = st.sidebar.number_input("R&D Expenditure (₹)", min_value=0.0, value=100000.0, step=1000.0)
admin_spend = st.sidebar.number_input("Administration Costs (₹)", min_value=0.0, value=50000.0, step=1000.0)
market_spend = st.sidebar.number_input("Marketing Budget (₹)", min_value=0.0, value=150000.0, step=1000.0)

st.sidebar.markdown("---")
st.sidebar.header("Geographical Data")
state = st.sidebar.selectbox("Operating State", ("California", "Florida", "New York"))

# 5. Process the categorical data
state_florida = 1 if state == "Florida" else 0
state_new_york = 1 if state == "New York" else 0

# 6. Prediction Logic and Output Display
st.markdown("### Prediction Results")

if model and scaler:
    # Using type="primary" makes the button stand out professionally
    if st.button("Generate Forecast", type="primary"):
        
        # Prepare data array
        user_data = np.array([[rd_spend, admin_spend, market_spend, state_florida, state_new_york]])
        
        # Transform data
        scaled_data = scaler.transform(user_data)
        
        # Predict
        prediction = model.predict(scaled_data)
        
        # Display using Streamlit's professional Metric component
        st.success("Analysis Complete.")
        
        # Creating columns for a clean visual layout
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(label="Estimated Annual Profit", value=f"₹{prediction[0]:,.2f}")
        with col2:
            st.metric(label="Total Expenditure", value=f"₹{(rd_spend + admin_spend + market_spend):,.2f}")
        with col3:
            st.metric(label="Primary Location", value=state)
            
        st.markdown("---")
        st.info("Note: This forecast is generated using automated feature selection, prioritizing R&D and Marketing data over administrative costs.")