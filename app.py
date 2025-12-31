import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,StandardScaler

st.set_page_config(
    page_title="Car Price Prediction App",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Car Price Prediction")
st.markdown("Machine Learning based car price analysis")
st.subheader("Dataset Preview")
st.dataframe(df.head())
st.subheader("Model Evaluation")

st.markdown("### Linear Regression")
st.write("MAE:", mean_absolute_error(y_test, y_lr))
st.write("MSE:", mean_squared_error(y_test, y_lr))
st.write("RMSE:", mean_squared_error(y_test, y_lr)**0.5)
st.write("R2 Score:", r2_score(y_test, y_lr))

st.markdown("### Random Forest")
st.write("MAE:", mean_absolute_error(y_test, y_rf))
st.write("MSE:", mean_squared_error(y_test, y_rf))
st.write("RMSE:", mean_squared_error(y_test, y_rf)**0.5)
st.write("R2 Score:", r2_score(y_test, y_rf))
st.subheader("Good Deal Analysis")

st.write("Good deals (Linear Regression):", df['Good_Deal_LR'].sum())
st.write("Good deals (Random Forest):", df['Good_Deal_RF'].sum())





df = pd.read_csv("CAR DATA1.csv")
df





df.isnull().sum()





df.duplicated()




df["Car_Age"] = 2025 - df["Year"] 





df.drop(["Year", "Car_Name"], axis=1, inplace=True) 





df





df = pd.get_dummies(
    df,
    columns=['Fuel_Type', 'Selling_type', 'Transmission'],
    drop_first=True
)





df




X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']





X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)





from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor





lr = LinearRegression()
rf = RandomForestRegressor()





lr.fit(X_train, y_train)
rf.fit(X_train, y_train)





y_lr = lr.predict(X_test)
y_rf = rf.predict(X_test)





from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error,accuracy_score,confusion_matrix
print("/////////// Evaluation Report ///////////")
print("------ LINEAR REGRESSION ------")

print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_lr):.3f}")
print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_lr):.3f}")
print(f"Root Mean Squared Error (RMSE): {(mean_squared_error(y_test, y_lr))**0.5:.3f}")
print(f"R2 Score: {r2_score(y_test, y_lr):.3f}")





print("\n------ RANDOM FOREST REGRESSOR ------")
print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_rf):.3f}")
print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_rf):.3f}")
print(f"Root Mean Squared Error (RMSE): {(mean_squared_error(y_test, y_rf))**0.5:.3f}")
print(f"R2 Score: {r2_score(y_test, y_rf):.3f}")





df['Predicted_Price_LR'] = lr.predict(X)


df['Good_Deal_LR'] = (df['Selling_Price'] < df['Predicted_Price_LR']).astype(int)






df['Predicted_Price_RF'] = rf.predict(X)


df['Good_Deal_RF'] = (df['Selling_Price'] < df['Predicted_Price_RF']).astype(int)





print("Good deals (Linear Regression):", df['Good_Deal_LR'].sum())
print("Good deals (Random Forest):", df['Good_Deal_RF'].sum())
