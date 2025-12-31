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
        "🤖 Model Evaluation & Comparison",
        "🔮 Good Deal Analysis",
        "🧮 Price Calculator"
    ]
)

# ---------------- Load Data ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("CAR DATA1.csv")
    df.columns = df.columns.str.strip()  # strip whitespace from column names
    return df

df_raw = load_data()

# ---------------- Preprocessing ----------------
df = df_raw.copy()
df["Car_Age"] = 2025 - df["Year"]
car_names = df_raw["Car_Name"].unique()
df.drop(["Year", "Car_Name"], axis=1, inplace=True)

df = pd.get_dummies(
    df,
    columns=["Fuel_Type", "Selling_type", "Transmission"],
    drop_first=True
)

X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

# ---------------- Train Models ----------------
@st.cache_resource
def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lr = LinearRegression()
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    return lr, rf, X_train, X_test, y_train, y_test

lr, rf, X_train, X_test, y_train, y_test = train_models(X, y)
y_lr = lr.predict(X_test)
y_rf = rf.predict(X_test)

df["Predicted_LR"] = lr.predict(X)
df["Predicted_RF"] = rf.predict(X)
df["Good_Deal_LR"] = (df["Selling_Price"] < df["Predicted_LR"]).astype(int)
df["Good_Deal_RF"] = (df["Selling_Price"] < df["Predicted_RF"]).astype(int)

# ================== DATA OVERVIEW ==================
if app_mode == "📊 Data Overview":
    st.title("📊 Data Overview")
    
    st.subheader("Complete Dataset")
    st.dataframe(df_raw)

    st.subheader("Statistical Summary")
    st.dataframe(df.describe())

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(df.corr(), cmap="coolwarm", annot=True)
    st.pyplot(fig)

    st.subheader("Selling Price Distribution")
    fig = px.histogram(df_raw, x="Selling_Price", nbins=30, title="Distribution of Selling Price")
    st.plotly_chart(fig, use_container_width=True)

# ================== MODEL EVALUATION & COMPARISON ==================
elif app_mode == "🤖 Model Evaluation & Comparison":
    st.title("🤖 Model Evaluation & Comparison")

    st.markdown("### Metrics for Linear Regression and Random Forest")
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

    st.markdown("### Visualizations")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Actual vs Predicted - Linear Regression")
        fig = px.scatter(x=y_test, y=y_lr, labels={"x": "Actual", "y": "Predicted"})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Actual vs Predicted - Random Forest")
        fig = px.scatter(x=y_test, y=y_rf, labels={"x": "Actual", "y": "Predicted"})
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Comparison of Model Performance")
    fig = px.bar(metrics, x="Metric", y=["Linear Regression", "Random Forest"], barmode="group",
                 title="Model Comparison on Key Metrics")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Methodology Note:**  
    - Linear Regression captures linear trends between features and car price.  
    - Random Forest, an ensemble method, handles nonlinear relationships and feature interactions better.  
    - Random Forest generally provides more accurate and robust predictions, as reflected in higher R² and lower error metrics.
    """)

# ================== GOOD DEAL ANALYSIS ==================
elif app_mode == "🔮 Good Deal Analysis":
    st.title("🔮 Good Deal Analysis")
    col1, col2 = st.columns(2)
    col1.metric("Good Deals (LR)", df["Good_Deal_LR"].sum())
    col2.metric("Good Deals (RF)", df["Good_Deal_RF"].sum())

    fig = px.pie(
        names=["Good Deal", "Not Good Deal"],
        values=[df["Good_Deal_RF"].sum(), len(df) - df["Good_Deal_RF"].sum()],
        title="Proportion of Good Deals (Random Forest)"
    )
    st.plotly_chart(fig)

# ================== PRICE CALCULATOR ==================
elif app_mode == "🧮 Price Calculator":
    st.title("🧮 Car Price Calculator")
    st.write("Select a car and check whether its asking price is reasonable")

    car_name = st.selectbox("Select Car Name", sorted(df_raw["Car_Name"].unique()))
    car_df = df_raw[df_raw["Car_Name"] == car_name]

    avg_year = int(car_df["Year"].mean())

    # Detect kilometers column dynamically
    kms_col = None
    for col in df_raw.columns:
        if "kms" in col.lower() or "kilometer" in col.lower():
            kms_col = col
            break
    avg_kms = 30000 if kms_col is None else int(car_df[kms_col].mean())

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
        "🤖 Model Evaluation & Comparison",
        "🔮 Good Deal Analysis",
        "🧮 Price Calculator"
    ]
)

# ---------------- Load Data ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("CAR DATA1.csv")
    df.columns = df.columns.str.strip()  # strip whitespace from column names
    return df

df_raw = load_data()

# ---------------- Preprocessing ----------------
df = df_raw.copy()
df["Car_Age"] = 2025 - df["Year"]
car_names = df_raw["Car_Name"].unique()
df.drop(["Year", "Car_Name"], axis=1, inplace=True)

df = pd.get_dummies(
    df,
    columns=["Fuel_Type", "Selling_type", "Transmission"],
    drop_first=True
)

X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

# ---------------- Train Models ----------------
@st.cache_resource
def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lr = LinearRegression()
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    return lr, rf, X_train, X_test, y_train, y_test

lr, rf, X_train, X_test, y_train, y_test = train_models(X, y)
y_lr = lr.predict(X_test)
y_rf = rf.predict(X_test)

df["Predicted_LR"] = lr.predict(X)
df["Predicted_RF"] = rf.predict(X)
df["Good_Deal_LR"] = (df["Selling_Price"] < df["Predicted_LR"]).astype(int)
df["Good_Deal_RF"] = (df["Selling_Price"] < df["Predicted_RF"]).astype(int)

# ================== DATA OVERVIEW ==================
if app_mode == "📊 Data Overview":
    st.title("📊 Data Overview")
    
    st.subheader("Complete Dataset")
    st.dataframe(df_raw)

    st.subheader("Statistical Summary")
    st.dataframe(df.describe())

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(df.corr(), cmap="coolwarm", annot=True)
    st.pyplot(fig)

    st.subheader("Selling Price Distribution")
    fig = px.histogram(df_raw, x="Selling_Price", nbins=30, title="Distribution of Selling Price")
    st.plotly_chart(fig, use_container_width=True)

# ================== MODEL EVALUATION & COMPARISON ==================
elif app_mode == "🤖 Model Evaluation & Comparison":
    st.title("🤖 Model Evaluation & Comparison")

    st.markdown("### Metrics for Linear Regression and Random Forest")
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

    st.markdown("### Visualizations")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Actual vs Predicted - Linear Regression")
        fig = px.scatter(x=y_test, y=y_lr, labels={"x": "Actual", "y": "Predicted"})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Actual vs Predicted - Random Forest")
        fig = px.scatter(x=y_test, y=y_rf, labels={"x": "Actual", "y": "Predicted"})
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Comparison of Model Performance")
    fig = px.bar(metrics, x="Metric", y=["Linear Regression", "Random Forest"], barmode="group",
                 title="Model Comparison on Key Metrics")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Methodology Note:**  
    - Linear Regression captures linear trends between features and car price.  
    - Random Forest, an ensemble method, handles nonlinear relationships and feature interactions better.  
    - Random Forest generally provides more accurate and robust predictions, as reflected in higher R² and lower error metrics.
    """)

# ================== GOOD DEAL ANALYSIS ==================
elif app_mode == "🔮 Good Deal Analysis":
    st.title("🔮 Good Deal Analysis")
    col1, col2 = st.columns(2)
    col1.metric("Good Deals (LR)", df["Good_Deal_LR"].sum())
    col2.metric("Good Deals (RF)", df["Good_Deal_RF"].sum())

    fig = px.pie(
        names=["Good Deal", "Not Good Deal"],
        values=[df["Good_Deal_RF"].sum(), len(df) - df["Good_Deal_RF"].sum()],
        title="Proportion of Good Deals (Random Forest)"
    )
    st.plotly_chart(fig)

# ================== PRICE CALCULATOR ==================
elif app_mode == "🧮 Price Calculator":
    st.title("🧮 Car Price Calculator")
    st.write("Select a car and check whether its asking price is reasonable")

    car_name = st.selectbox("Select Car Name", sorted(df_raw["Car_Name"].unique()))
    car_df = df_raw[df_raw["Car_Name"] == car_name]

    avg_year = int(car_df["Year"].mean())

    # Detect kilometers column dynamically
    kms_col = None
    for col in df_raw.columns:
        if "kms" in col.lower() or "kilometer" in col.lower():
            kms_col = col
            break
    avg_kms = 30000 if kms_col is None else int(car_df[kms_col].mean())

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

