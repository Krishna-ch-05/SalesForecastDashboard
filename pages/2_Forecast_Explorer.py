import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -------------------------------------------------------
# Page Configuration
# -------------------------------------------------------

st.set_page_config(
    page_title="Forecast Explorer",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Forecast Explorer")

# -------------------------------------------------------
# Load Dataset
# -------------------------------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv("train.csv")

    df["Order Date"] = pd.to_datetime(
        df["Order Date"],
        dayfirst=True,
        errors="coerce"
    )

    df = df.dropna(subset=["Order Date"])

    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    df["Quarter"] = df["Order Date"].dt.quarter

    return df

df = load_data()

# -------------------------------------------------------
# Sidebar
# -------------------------------------------------------

st.sidebar.header("Forecast Settings")

forecast_type = st.sidebar.selectbox(
    "Forecast By",
    ["Category", "Region"]
)

# -------------------------------------------------------
# Category / Region Selection
# -------------------------------------------------------

if forecast_type == "Category":

    selected = st.sidebar.selectbox(
        "Select Category",
        sorted(df["Category"].unique())
    )

    data = df[df["Category"] == selected]

else:

    selected = st.sidebar.selectbox(
        "Select Region",
        sorted(df["Region"].unique())
    )

    data = df[df["Region"] == selected]

# -------------------------------------------------------
# Monthly Sales
# -------------------------------------------------------

monthly = (
    data.groupby(
        pd.Grouper(key="Order Date", freq="ME")
    )["Sales"]
    .sum()
    .reset_index()
)

monthly["Month_Number"] = np.arange(len(monthly))

# -------------------------------------------------------
# Lag Features
# -------------------------------------------------------

monthly["Lag1"] = monthly["Sales"].shift(1)
monthly["Lag2"] = monthly["Sales"].shift(2)
monthly["Lag3"] = monthly["Sales"].shift(3)

monthly["RollingMean"] = (
    monthly["Sales"]
    .rolling(3)
    .mean()
)

monthly = monthly.dropna()

# -------------------------------------------------------
# Prepare Data
# -------------------------------------------------------

X = monthly[
    [
        "Month_Number",
        "Lag1",
        "Lag2",
        "Lag3",
        "RollingMean"
    ]
]

y = monthly["Sales"]

# -------------------------------------------------------
# Train/Test Split
# -------------------------------------------------------

split = int(len(monthly) * 0.80)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

# -------------------------------------------------------
# Train XGBoost Model
# -------------------------------------------------------

model = XGBRegressor(

    n_estimators=100,

    learning_rate=0.1,

    max_depth=3,

    random_state=42
)

model.fit(X_train, y_train)

st.success("✅ XGBoost model trained successfully!")

# -------------------------------------------------------
# Model Evaluation
# -------------------------------------------------------

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

st.subheader("📊 Model Performance")

col1, col2 = st.columns(2)

col1.metric("MAE", f"{mae:.2f}")
col2.metric("RMSE", f"{rmse:.2f}")

# -------------------------------------------------------
# Forecast Horizon
# -------------------------------------------------------

forecast_horizon = st.sidebar.slider(
    "Forecast Horizon (Months)",
    min_value=1,
    max_value=3,
    value=3
)

# -------------------------------------------------------
# Predict Next Months
# -------------------------------------------------------

future_predictions = []

last_sales = monthly["Sales"].tolist()

month_number = monthly["Month_Number"].iloc[-1]

for i in range(forecast_horizon):

    lag1 = last_sales[-1]
    lag2 = last_sales[-2]
    lag3 = last_sales[-3]

    rolling = np.mean(last_sales[-3:])

    month_number += 1

    future = pd.DataFrame({

        "Month_Number":[month_number],
        "Lag1":[lag1],
        "Lag2":[lag2],
        "Lag3":[lag3],
        "RollingMean":[rolling]

    })

    pred = model.predict(future)[0]

    future_predictions.append(pred)

    last_sales.append(pred)

# -------------------------------------------------------
# Forecast Table
# -------------------------------------------------------

future_dates = pd.date_range(

    start=monthly["Order Date"].max(),

    periods=forecast_horizon + 1,

    freq="ME"

)[1:]

forecast_df = pd.DataFrame({

    "Forecast Month":future_dates,

    "Predicted Sales":future_predictions

})

st.subheader("📅 Forecast Output")

st.dataframe(forecast_df)
