import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Car Price Prediction App",
    page_icon="🚗",
    layout="wide"
)

# ---------------- Sidebar ----------------
st.sidebar.title("🚗 Navigation")
app_mode = st.sidebar.selectbox(
    "Choose a page:",
    [
        "📊 Data Overview",
        "🤖 Model Evaluation",
        "📈 Model Comparison",
        "🔮 Good Deal Analysis",
        "🧮 Price Calculator"
    ]
)

# ---------------- Load Data ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("CAR DATA1.csv")
    # Strip whitespace from column headers
    df.columns = df.columns.str.strip()
    return df

df_raw = load_data()

# ---------------- Preprocessing ----------------
df = df_raw.copy()
df["Car_Age"] = 2025 - df["Year"]

# Keep a copy of Car_Name for selection feature
car_names = df_raw["Car_Name"].unique()

# Drop columns for modeling
df.drop(["Year", "Car_Name"], axis=1, inplace=True)

# Convert categorical variables to dummy variables
df = pd.get_dummies(
    df,
    columns=["Fuel_Type", "Selling_type", "Transmission"],
    drop_first=True
)

# Features and target
X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

# ---------------- Train Models ----------------
@st.cache_resource
def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    lr = LinearRegression()
    rf = RandomForestRegressor(n_estimators=200, random_state=42)

    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    return lr, rf, X_train, X_test, y_train, y_test

lr, rf, X_train, X_test, y_train, y_test = train_models(X, y)

# Predictions
y_lr = lr.predict(X_test)
y_rf = rf.predict(X_test)

df["Predicted_LR"] = lr.predict(X)
df["Predicted_RF"] = rf.predict(X)
df["Good_Deal_LR"] = (df["Selling_Price"] < df["Predicted_LR"]).astype(int)
df["Good_Deal_RF"] = (df["Selling_Price"] < df["Predicted_RF"]).astype(int)

# ================== DATA OVERVIEW ==================
if app_mode == "📊 Data Overview":
    st.title("📊 Data Overview")
    st.subheader("Dataset Preview")
    st.dataframe(df_raw.head())
    st.subheader("Statistical Summary")
    st.dataframe(df.describe())
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), cmap="coolwarm", annot=True)
    st.pyplot(fig)
    st.subheader("Selling Price Distribution")
    fig = px.histogram(df_raw, x="Selling_Price", nbins=30)
    st.plotly_chart(fig, use_container_width=True)

# ================== MODEL EVALUATION ==================
elif app_mode == "🤖 Model Evaluation":
    st.title("🤖 Model Evaluation")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Linear Regression")
        st.metric("MAE", round(mean_absolute_error(y_test, y_lr), 3))
        st.metric("RMSE", round(mean_squared_error(y_test, y_lr)**0.5, 3))
        st.metric("R²", round(r2_score(y_test, y_lr), 3))
    with col2:
        st.subheader("Random Forest")
        st.metric("MAE", round(mean_absolute_error(y_test, y_rf), 3))
        st.metric("RMSE", round(mean_squared_error(y_test, y_rf)**0.5, 3))
        st.metric("R²", round(r2_score(y_test, y_rf), 3))
    st.subheader("Actual vs Predicted (Random Forest)")
    fig = px.scatter(x=y_test, y=y_rf, labels={"x": "Actual", "y": "Predicted"})
    st.plotly_chart(fig, use_container_width=True)

# ================== MODEL COMPARISON ==================
elif app_mode == "📈 Model Comparison":
    st.title("📈 Model Comparison")
    metrics = pd.DataFrame({
        "Metric": ["MAE", "RMSE", "R²"],
        "Linear Regression": [
            mean_absolute_error(y_test, y_lr),
            mean_squared_error(y_test, y_lr)**0.5,
            r2_score(y_test, y_lr)
        ],
        "Random Forest": [
            mean_absolute_error(y_test, y_rf),
            mean_squared_error(y_test, y_rf)**0.5,
            r2_score(y_test, y_rf)
        ]
    })
    st.dataframe(metrics)
    fig = px.bar(metrics, x="Metric", y=["Linear Regression", "Random Forest"], barmode="group")
    st.plotly_chart(fig, use_container_width=True)

# ================== GOOD DEAL ANALYSIS ==================
elif app_mode == "🔮 Good Deal Analysis":
    st.title("🔮 Good Deal Analysis")
    col1, col2 = st.columns(2)
    col1.metric("Good Deals (LR)", df["Good_Deal_LR"].sum())
    col2.metric("Good Deals (RF)", df["Good_Deal_RF"].sum())
    fig = px.pie(
        names=["Good Deal", "Not Good Deal"],
        values=[df["Good_Deal_RF"].sum(), len(df) - df["Good_Deal_RF"].sum()]
    )
    st.plotly_chart(fig)

# ================== PRICE CALCULATOR ==================
elif app_mode == "🧮 Price Calculator":
    st.title("🧮 Car Price Calculator")
    st.write("Select a car and check whether its asking price is reasonable")

    # ---- Car Name Selection ----
    car_name = st.selectbox("Select Car Name", sorted(car_names))
    car_df = df_raw[df_raw["Car_Name"] == car_name]

    # Use historical averages safely
    avg_year = int(car_df["Year"].mean())
    kms_col = [c for c in car_df.columns if "Kms" in c][0]  
    avg_kms = int(car_df[kms_col].mean())
    avg_owner = int(car_df["Owner"].mode()[0])
    avg_price = float(car_df["Selling_Price"].mean())

    st.info(f"📊 Historical Average Price: {avg_price:.2f} Lakhs")

    col1, col2 = st.columns(2)
    with col1:
        asking_price = st.number_input("Asking Price (Lakhs)", 0.0, 50.0, round(avg_price, 2))
        kms = st.number_input("Kilometers Driven", 0, 500000, avg_kms)
        car_age = st.slider("Car Age (Years)", 0, 25, 2025 - avg_year)
    with col2:
        owner = st.selectbox("Owner Type", [0, 1, 3], index=0)
        fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel"])
        seller = st.selectbox("Seller Type", ["Dealer", "Individual"])
        transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

    if st.button("🚀 Predict Price"):
        input_data = pd.DataFrame([{
            "Present_Price": asking_price,
            "Kms_Driven": kms,
            "Owner": owner,
            "Car_Age": car_age,
            "Fuel_Type_Diesel": 1 if fuel == "Diesel" else 0,
            "Selling_type_Individual": 1 if seller == "Individual" else 0,
            "Transmission_Manual": 1 if transmission == "Manual" else 0
        }])
        input_data = input_data.reindex(columns=X.columns, fill_value=0)
        predicted_price = rf.predict(input_data)[0]

        st.success(f"💰 Predicted Fair Price: {predicted_price:.2f} Lakhs")
        if asking_price <= predicted_price:
            st.success("🟢 Reasonable Price / Good Deal")
        else:
            st.error("🔴 Overpriced Car")

        fig, ax = plt.subplots()
        ax.bar(["Asking Price", "Predicted Price", "Historical Avg"], [asking_price, predicted_price, avg_price])
        ax.set_ylabel("Price (Lakhs)")
        ax.set_title(f"Price Comparison for {car_name}")
        st.pyplot(fig)
