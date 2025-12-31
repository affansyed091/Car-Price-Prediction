import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Streamlit app configuration
st.set_page_config(
    page_title="Car Price Prediction App",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Car Price Prediction Dashboard")
st.markdown("Predict car prices using Machine Learning models")

# Load and preprocess data
@st.cache_data
def load_and_preprocess_data():
    # Load data
    df = pd.read_csv("CAR DATA1.csv")
    
    # Preprocessing (same as your original code)
    df["Car_Age"] = 2025 - df["Year"]
    df.drop(["Year", "Car_Name"], axis=1, inplace=True)
    df = pd.get_dummies(
        df,
        columns=['Fuel_Type', 'Selling_type', 'Transmission'],
        drop_first=True
    )
    
    return df

# Load data
df_processed = load_and_preprocess_data()

# Sidebar navigation
st.sidebar.header("Navigation")
option = st.sidebar.selectbox(
    "Choose what you want to do:",
    ["📊 Data Overview", "🤖 Train Models", "🔮 Make Predictions"]
)

if option == "📊 Data Overview":
    st.header("📊 Data Overview")
    
    # Show original data
    st.subheader("Original Dataset")
    df_original = pd.read_csv("CAR DATA1.csv")
    st.dataframe(df_original.head())
    
    st.metric("Total Records", len(df_original))
    st.metric("Features", len(df_original.columns))
    
    # Show processed data
    st.subheader("Processed Dataset (After Feature Engineering)")
    st.dataframe(df_processed.head())
    
    # Data information
    st.subheader("Data Information")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Original Columns:**")
        st.write(list(df_original.columns))
    with col2:
        st.write("**Processed Columns:**")
        st.write(list(df_processed.columns))

elif option == "🤖 Train Models":
    st.header("🤖 Train Machine Learning Models")
    
    # Prepare features and target
    X = df_processed.drop('Selling_Price', axis=1)
    y = df_processed['Selling_Price']
    
    # Train-test split
    test_size = st.slider("Test Size (%)", 10, 40, 20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size/100, random_state=42
    )
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Training Samples", len(X_train))
    with col2:
        st.metric("Testing Samples", len(X_test))
    
    # Train models
    if st.button("🚀 Train Both Models", type="primary"):
        with st.spinner("Training models..."):
            # Linear Regression
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            y_pred_lr = lr.predict(X_test)
            
            # Random Forest
            rf = RandomForestRegressor(random_state=42)
            rf.fit(X_train, y_train)
            y_pred_rf = rf.predict(X_test)
            
            # Calculate metrics
            lr_mae = mean_absolute_error(y_test, y_pred_lr)
            lr_mse = mean_squared_error(y_test, y_pred_lr)
            lr_rmse = np.sqrt(lr_mse)
            lr_r2 = r2_score(y_test, y_pred_lr)
            
            rf_mae = mean_absolute_error(y_test, y_pred_rf)
            rf_mse = mean_squared_error(y_test, y_pred_rf)
            rf_rmse = np.sqrt(rf_mse)
            rf_r2 = r2_score(y_test, y_pred_rf)
            
            # Display results
            st.subheader("📊 Model Performance")
            
            # Create comparison table
            results_df = pd.DataFrame({
                'Metric': ['MAE', 'MSE', 'RMSE', 'R² Score'],
                'Linear Regression': [lr_mae, lr_mse, lr_rmse, lr_r2],
                'Random Forest': [rf_mae, rf_mse, rf_rmse, rf_r2]
            })
            
            # Format the table
            display_df = results_df.copy()
            for col in ['Linear Regression', 'Random Forest']:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
            
            st.dataframe(display_df.set_index('Metric'), use_container_width=True)
            
            # Best model
            if rf_r2 > lr_r2:
                st.success("🎯 Random Forest performs better!")
                st.session_state.best_model = rf
            else:
                st.info("📈 Linear Regression performs better!")
                st.session_state.best_model = lr
            
            # Save models to session state
            st.session_state.lr_model = lr
            st.session_state.rf_model = rf
            st.session_state.X_columns = X.columns.tolist()
            
            st.success("✅ Models trained successfully!")

elif option == "🔮 Make Predictions":
    st.header("🔮 Predict Car Price")
    
    if 'X_columns' not in st.session_state:
        st.warning("⚠️ Please train models first in the 'Train Models' section!")
    else:
        st.subheader("Enter Car Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            present_price = st.number_input("Present Price (lakhs)", 
                                           min_value=0.0, max_value=100.0, value=5.0)
            driven_kms = st.number_input("Kilometers Driven", 
                                        min_value=0, max_value=500000, value=50000)
            car_age = st.number_input("Car Age (years)", 
                                     min_value=0, max_value=50, value=5)
        
        with col2:
            owner = st.selectbox("Number of Previous Owners", [0, 1, 2, 3])
            fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
            selling_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
            transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
        
        # Model selection
        model_choice = st.radio(
            "Select prediction model:",
            ["Linear Regression", "Random Forest", "Both"],
            horizontal=True
        )
        
        if st.button("Predict Price", type="primary"):
            # Prepare input data
            input_data = {
                'Present_Price': present_price,
                'Driven_kms': driven_kms,
                'Owner': owner,
                'Car_Age': car_age,
                'Fuel_Type_Diesel': 1 if fuel_type == 'Diesel' else 0,
                'Fuel_Type_Petrol': 1 if fuel_type == 'Petrol' else 0,
                'Selling_type_Individual': 1 if selling_type == 'Individual' else 0,
                'Transmission_Manual': 1 if transmission == 'Manual' else 0
            }
            
            # Add missing columns with 0
            for col in st.session_state.X_columns:
                if col not in input_data:
                    input_data[col] = 0
            
            # Convert to DataFrame
            input_df = pd.DataFrame([input_data])
            input_df = input_df[st.session_state.X_columns]
            
            # Make predictions
            predictions = {}
            
            if model_choice in ["Linear Regression", "Both"]:
                lr_pred = st.session_state.lr_model.predict(input_df)[0]
                predictions["Linear Regression"] = lr_pred
            
            if model_choice in ["Random Forest", "Both"]:
                rf_pred = st.session_state.rf_model.predict(input_df)[0]
                predictions["Random Forest"] = rf_pred
            
            # Display results
            st.subheader("📊 Prediction Results")
            
            for model_name, price in predictions.items():
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.metric(
                        f"{model_name}",
                        f"₹{price:.2f}L",
                        delta="Good Deal" if price > present_price else "Market Price"
                    )
                with col2:
                    # Visual indicator
                    progress_val = min(price / 50, 1.0)
                    st.progress(progress_val)
                    st.caption(f"Predicted: ₹{price:.2f} lakhs | Actual: ₹{present_price:.2f} lakhs")
            
            # Deal analysis
            st.subheader("💡 Deal Analysis")
            if predictions:
                avg_price = np.mean(list(predictions.values()))
                if present_price < avg_price * 0.9:
                    st.success("✅ **GOOD DEAL!** The car is significantly undervalued.")
                elif present_price > avg_price * 1.1:
                    st.warning("⚠️ **OVERPRICED!** The car is priced above market value.")
                else:
                    st.info("📊 **MARKET PRICE** - The car is fairly priced.")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | Car Price Prediction App")
