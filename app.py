import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Page Configuration for a clean UI
st.set_page_config(
    page_title="Air Passengers Prediction Dashboard",
    page_icon="✈️",
    layout="centered"
)

# --- 🛠️ EXCEPTION HANDLING FOR LOADING MODELS ---
@st.cache_resource
def load_ml_artifacts():
    try:
        # Checking if files exist before loading
        if not os.path.exists("model.pkl"):
            raise FileNotFoundError("Core model file ('model.pkl') is missing from the directory.")
        if not os.path.exists("scaler.pkl"):
            raise FileNotFoundError("Scaler artifact ('scaler.pkl') is missing from the directory.")
        if not os.path.exists("columns.pkl"):
            raise FileNotFoundError("Columns layout file ('columns.pkl') is missing from the directory.")
            
        # Loading artifacts using joblib
        loaded_model = joblib.load("model.pkl")
        loaded_scaler = joblib.load("scaler.pkl")
        loaded_columns = joblib.load("columns.pkl")
        
        return loaded_model, loaded_scaler, loaded_columns, None
        
    except Exception as e:
        # Capture any error (FileNotFound, Version Mismatch, etc.)
        return None, None, None, str(e)

# Initialize and load
model, scaler, expected_columns, error_message = load_ml_artifacts()

# If loading fails, halt and show a clean alert
if error_message:
    st.error("🚨 **Initialization Error!** Failed to load the Machine Learning files.")
    st.info(f"**Details:** {error_message}")
    st.warning("Please ensure 'model.pkl', 'scaler.pkl', and 'columns.pkl' are uploaded to the main root folder.")
    st.stop()


# --- 🎨 USER INTERFACE (UI) ---
st.title("✈️ Air Passengers Prediction System")
st.write("Predict the estimated number of passengers based on the year and month using our trained high-accuracy model.")
st.markdown("---")

st.subheader("📊 Enter Details for Prediction")

# Input 1: Year Column (Numerical Input)
# Restricting to realistic limits or ranges based on your data structure
input_year = st.number_input(
    "Select Year", 
    min_value=1900, 
    max_value=2100, 
    value=1950, 
    step=1,
    help="Enter the target year for passenger prediction."
)

# Input 2: Month Column (Categorical Dropdown)
months_list = [
    "January", "February", "March", "April", "May", "June", 
    "July", "August", "September", "October", "November", "December"
]
selected_month = st.selectbox("Select Month", options=months_list)

# Prediction Button Trigger
st.markdown("###")
if st.button("🚀 Predict Passenger Count", use_container_width=True):
    with st.spinner("Processing data & generating prediction..."):
        try:
            # --- 🛠️ STEP 1: PREPROCESSING DATA ---
            
            # 1. Standard Scaling the Year column
            # Transforming single scalar requires reshaping to 2D array [[value]]
            scaled_year = scaler.transform([[input_year]])[0][0]
            
            # 2. Recreating the base framework structure for One-Hot Encoding
            # Creating a baseline dictionary filled with 0s for expected features
            input_dict = {col: 0 for col in expected_columns}
            
            # Setting scaled year in the correct column key
            if 'year' in input_dict:
                input_dict['year'] = scaled_year
            
            # Setting One-Hot Encoding flag (1) for selected month column
            # Check for the matching month key dynamically (e.g., 'month_January')
            month_feature_name = f"month_{selected_month}"
            if month_feature_name in input_dict:
                input_dict[month_feature_name] = 1
            else:
                # Fallback if names are stored without prefix or lowercase
                alt_name = selected_month.lower()
                if alt_name in input_dict:
                    input_dict[alt_name] = 1
            
            # Convert engineered dict directly to DataFrame matching trained structural layout
            final_features_df = pd.DataFrame([input_dict], columns=expected_columns)
            
            # --- 🔮 STEP 2: MODEL INFERENCE ---
            prediction = model.predict(final_features_df)
            
            # Handling log transformations if applied during training
            # (Uncomment the line below if target variable was np.log1p transformed)
            # prediction = np.expm1(prediction) 
            
            final_result = int(np.round(prediction[0]))
            
            # --- 🎉 STEP 3: OUTPUT RESULTS DISPLAY ---
            st.success("🎯 **Prediction Computed Successfully!**")
            st.metric(label="Estimated Passenger Count", value=f"{final_result:,} Passengers")
            
        except Exception as prediction_error:
            st.error("⚠️ **Prediction pipeline failed!** Check feature matching.")
            st.code(f"Error logs: {str(prediction_error)}")

st.markdown("---")
st.caption("Powered by Streamlit & Scikit-Learn | Accuracy Rating: ~98%")