# app.py

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


st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose a page:",
                                ["📊 Data Overview",
                                 "🤖 Model Evaluation",
                                 "📈 Model Comparison",
                                 "🔮 Good Deal Analysis"])


@st.cache_data
def load_data():
    df = pd.read_csv("CAR DATA1.csv")
    return df

df = load_data()


df["Car_Age"] = 2025 - df["Year"]
df.drop(["Year", "Car_Name"], axis=1, inplace=True)
df = pd.get_dummies(df, columns=['Fuel_Type', 'Selling_type', 'Transmission'], drop_first=True)

X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


lr = LinearRegression()
rf = RandomForestRegressor(random_state=42)

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

y_lr = lr.predict(X_test)
y_rf = rf.predict(X_test)


df['Predicted_Price_LR'] = lr.predict(X)
df['Predicted_Price_RF'] = rf.predict(X)
df['Good_Deal_LR'] = (df['Selling_Price'] < df['Predicted_Price_LR']).astype(int)
df['Good_Deal_RF'] = (df['Selling_Price'] < df['Predicted_Price_RF']).astype(int)


if app_mode == "📊 Data Overview":
    st.title("📊 Data Overview")
    st.dataframe(df.head())
    st.markdown("### Dataset Statistics")
    st.dataframe(df.describe())
    
    st.markdown("### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    st.pyplot(fig)
    
    st.markdown("### Selling Price Distribution")
    fig = px.histogram(df, x='Selling_Price', nbins=30, color_discrete_sequence=['#636EFA'])
    st.plotly_chart(fig)

elif app_mode == "🤖 Model Evaluation":
    st.title("🤖 Model Evaluation")
    
   
    st.markdown("### Linear Regression")
    st.write("MAE:", mean_absolute_error(y_test, y_lr))
    st.write("MSE:", mean_squared_error(y_test, y_lr))
    st.write("RMSE:", mean_squared_error(y_test, y_lr)**0.5)
    st.write("R2 Score:", r2_score(y_test, y_lr))
    
    
    st.markdown("### Random Forest Regressor")
    st.write("MAE:", mean_absolute_error(y_test, y_rf))
    st.write("MSE:", mean_squared_error(y_test, y_rf))
    st.write("RMSE:", mean_squared_error(y_test, y_rf)**0.5)
    st.write("R2 Score:", r2_score(y_test, y_rf))
    

    st.markdown("### Predicted vs Actual Selling Prices")
    fig = px.scatter(x=y_test, y=y_lr, labels={'x':'Actual','y':'Predicted'}, title='Linear Regression')
    st.plotly_chart(fig)
    fig = px.scatter(x=y_test, y=y_rf, labels={'x':'Actual','y':'Predicted'}, title='Random Forest')
    st.plotly_chart(fig)

elif app_mode == "📈 Model Comparison":
    st.title("📈 Model Comparison")
    
    metrics = pd.DataFrame({
        'Metric':['MAE','MSE','RMSE','R2'],
        'Linear Regression':[mean_absolute_error(y_test, y_lr),
                             mean_squared_error(y_test, y_lr),
                             mean_squared_error(y_test, y_lr)**0.5,
                             r2_score(y_test, y_lr)],
        'Random Forest':[mean_absolute_error(y_test, y_rf),
                         mean_squared_error(y_test, y_rf),
                         mean_squared_error(y_test, y_rf)**0.5,
                         r2_score(y_test, y_rf)]
    })
    st.dataframe(metrics)
    
   
    fig = px.bar(metrics, x='Metric', y=['Linear Regression','Random Forest'], barmode='group', title="Model Comparison")
    st.plotly_chart(fig)

elif app_mode == "🔮 Good Deal Analysis":
    st.title("🔮 Good Deal Analysis")
    st.write("Good deals (Linear Regression):", df['Good_Deal_LR'].sum())
    st.write("Good deals (Random Forest):", df['Good_Deal_RF'].sum())
    
    st.markdown("### Good Deal Distribution")
    fig = px.histogram(df, x='Good_Deal_LR', color_discrete_sequence=['#636EFA'], title="Linear Regression Good Deals")
    st.plotly_chart(fig)
    fig = px.histogram(df, x='Good_Deal_RF', color_discrete_sequence=['#EF553B'], title="Random Forest Good Deals")
    st.plotly_chart(fig)
