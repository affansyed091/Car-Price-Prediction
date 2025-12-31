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

st.title("🚗 Car Price Prediction App")
st.markdown("Using Linear Regression and Random Forest models")

# Function to load and preprocess data (same as your code)
@st.cache_data
def load_and_preprocess_data():
    # Load data
    df = pd.read_csv("CAR DATA1.csv")
    
    # Preprocessing (EXACTLY as in your code)
    df["Car_Age"] = 2025 - df["Year"]
    df.drop(["Year", "Car_Name"], axis=1, inplace=True)
    df = pd.get_dummies(
        df,
        columns=['Fuel_Type', 'Selling_type', 'Transmission'],
        drop_first=True
    )
    
    return df

# Sidebar navigation
st.sidebar.header("Navigation")
option = st.sidebar.selectbox(
    "Choose what you want to do:",
    ["📊 View Original Code", "🤖 Run Analysis", "📈 View Results", "🔮 Predict New Price"]
)

if option == "📊 View Original Code":
    st.header("📊 Your Original Python Code")
  # In[2]:


df = pd.read_csv("CAR DATA1.csv")
df


# In[3]:


df.isnull().sum()


# In[4]:


df.duplicated()


# In[5]:


df["Car_Age"] = 2025 - df["Year"] 


# In[6]:


df.drop(["Year", "Car_Name"], axis=1, inplace=True) 


# In[7]:


df


# In[8]:


df = pd.get_dummies(
    df,
    columns=['Fuel_Type', 'Selling_type', 'Transmission'],
    drop_first=True
)


# In[9]:


df


# In[10]:


X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']


# In[11]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# In[12]:


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


# In[13]:


lr = LinearRegression()
rf = RandomForestRegressor()


# In[14]:


lr.fit(X_train, y_train)
rf.fit(X_train, y_train)


# In[15]:


y_lr = lr.predict(X_test)
y_rf = rf.predict(X_test)


# In[16]:



from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error,accuracy_score,confusion_matrix
print("/////////// Evaluation Report ///////////")
print("------ LINEAR REGRESSION ------")

print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_lr):.3f}")
print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_lr):.3f}")
print(f"Root Mean Squared Error (RMSE): {(mean_squared_error(y_test, y_lr))**0.5:.3f}")
print(f"R2 Score: {r2_score(y_test, y_lr):.3f}")


# In[17]:


print("\n------ RANDOM FOREST REGRESSOR ------")
print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_rf):.3f}")
print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_rf):.3f}")
print(f"Root Mean Squared Error (RMSE): {(mean_squared_error(y_test, y_rf))**0.5:.3f}")
print(f"R2 Score: {r2_score(y_test, y_rf):.3f}")


# In[18]:



df['Predicted_Price_LR'] = lr.predict(X)


df['Good_Deal_LR'] = (df['Selling_Price'] < df['Predicted_Price_LR']).astype(int)


# In[19]:



df['Predicted_Price_RF'] = rf.predict(X)


df['Good_Deal_RF'] = (df['Selling_Price'] < df['Predicted_Price_RF']).astype(int)


# In[20]:


print("Good deals (Linear Regression):", df['Good_Deal_LR'].sum())
print("Good deals (Random Forest):", df['Good_Deal_RF'].sum())


    
