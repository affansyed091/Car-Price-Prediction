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


st.set_page_config(
    page_title="Car Price Prediction App",
    page_icon="🚗",
    layout="wide"
)


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


@st.cache_data
def load_data():
    return pd.read_csv("CAR DATA1.csv")

df_raw = load_data()


df = df_raw.copy()
df["Car_Age"] = 2025 - df["Year"]
df.drop(["Year", "Car_Name"], axis=1, inplace=True)

df = pd.get_dummies(
    df,
    columns=['Fuel_Type', 'Selling_type', 'Transmission'],
    drop_first=True
)

X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


lr = LinearRegression()
rf = RandomForestRegressor(random_state=42)

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

y_lr = lr.predict(X_test)
y_rf = rf.predict(X_test)


df["Predicted_LR"] = lr.predict(X)
df["Predicted_RF"] = rf.predict(X)
df["Good_Deal_LR"] = (df["Selling_Price"] < df["Predicted_LR"]).astype(int)
df["Good_Deal_RF"] = (df["Selling_Price"] < df["Predicted_RF"]).astype(int)


if app_mode == "📊 Data Overview":
    st.title("📊 Data Overview")

    st.subheader("Dataset Preview")
    st.dataframe(df_raw.head())

    st.subheader("Statistical Summary")
    st.dataframe(df.describe())

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), cmap="coolwarm")
    st.pyplot(fig)

    st.subheader("Selling Price Distribution")
    fig = px.histogram(df_raw, x="Selling_Price", nbins=30)
    st.plotly_chart(fig, use_container_width=True)


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

    st.subheader("Actual vs Predicted")
    fig = px.scatter(x=y_test, y=y_rf, labels={"x":"Actual","y":"Predicted"})
    st.plotly_chart(fig, use_container_width=True)


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

    fig = px.bar(
        metrics,
        x="Metric",
        y=["Linear Regression", "Random Forest"],
        barmode="group",
        title="Model Performance Comparison"
    )
    st.plotly_chart(fig, use_container_width=True)


elif app_mode == "🔮 Good Deal Analysis":
    st.title("🔮 Good Deal Analysis")

    col1, col2 = st.columns(2)

    col1.metric("Good Deals (LR)", df["Good_Deal_LR"].sum())
    col2.metric("Good Deals (RF)", df["Good_Deal_RF"].sum())

    fig = px.pie(
        names=["Good Deal", "Not Good Deal"],
        values=[df["Good_Deal_RF"].sum(), len(df)-df["Good_Deal_RF"].sum()],
        title="Random Forest Good Deal Ratio"
    )
    st.plotly_chart(fig)


elif app_mode == "🧮 Price Calculator":
    st.title("🧮 Car Price Calculator")

    st.markdown("Enter car details to predict price & check if it's reasonable")

    present_price = st.number_input("Current Market Price (Lakhs)", 0.0, 50.0, 5.0)
    kms = st.number_input("Kilometers Driven", 0, 500000, 30000)
    owner = st.selectbox("Owner Type", [0, 1, 3])
    fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel"])
    seller = st.selectbox("Seller Type", ["Dealer", "Individual"])
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
    car_age = st.slider("Car Age (Years)", 0, 25, 5)

    if st.button("🚀 Predict Price"):
        input_data = pd.DataFrame([{
            "Present_Price": present_price,
            "Kms_Driven": kms,
            "Owner": owner,
            "Car_Age": car_age,
            "Fuel_Type_Diesel": 1 if fuel == "Diesel" else 0,
            "Selling_type_Individual": 1 if seller == "Individual" else 0,
            "Transmission_Manual": 1 if transmission == "Manual" else 0
        }])

        input_data = input_data.reindex(columns=X.columns, fill_value=0)

        predicted_price = rf.predict(input_data)[0]

        st.success(f"💰 Predicted Price: {predicted_price:.2f} Lakhs")

        diff = present_price - predicted_price

        if diff < 0:
            st.success("🟢 Reasonable / Good Deal")
        else:
            st.error("🔴 Overpriced")

        # Visualization
        fig, ax = plt.subplots()
        ax.bar(
            ["Entered Price", "Predicted Price", "Average Market"],
            [present_price, predicted_price, df["Selling_Price"].mean()]
        )
        ax.set_ylabel("Price (Lakhs)")
        ax.set_title("Price Comparison")
        st.pyplot(fig)
