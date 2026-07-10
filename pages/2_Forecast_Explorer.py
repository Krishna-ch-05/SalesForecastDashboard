import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Forecast Explorer", layout="wide")

st.title("🔮 Forecast Explorer")

# Load Dataset
df = pd.read_csv("SampleSuperstore.csv", encoding="latin1")
df.columns = df.columns.str.strip()

# Convert Date
df["Order Date"] = pd.to_datetime(
    df["Order Date"],
    format="%d/%m/%Y"
)
# Load Trained XGBoost Model
with open("xgboost_model.pkl", "rb") as file:
    model = pickle.load(file)

# ==============================
# Sidebar Filters
# ==============================

forecast_type = st.selectbox(
    "Forecast By",
    ["Category", "Region"]
)

if forecast_type == "Category":
    selected = st.selectbox(
        "Select Category",
        sorted(df["Category"].unique())
    )
    filtered_df = df[df["Category"] == selected]

else:
    selected = st.selectbox(
        "Select Region",
        sorted(df["Region"].unique())
    )
    filtered_df = df[df["Region"] == selected]

# Forecast Horizon
months_ahead = st.slider(
    "Forecast Horizon (Months)",
    min_value=1,
    max_value=3,
    value=1
)

# ==============================
# Monthly Sales
# ==============================

monthly_df = (
    filtered_df
    .groupby(pd.Grouper(key="Order Date", freq="M"))["Sales"]
    .sum()
    .reset_index()
)

monthly_df.rename(columns={"Order Date": "Date"}, inplace=True)

monthly_df = monthly_df.sort_values("Date")

# ==============================
# Feature Engineering
# ==============================

monthly_df["Lag1"] = monthly_df["Sales"].shift(1)
monthly_df["Lag2"] = monthly_df["Sales"].shift(2)
monthly_df["Lag3"] = monthly_df["Sales"].shift(3)

monthly_df["RollingMean3"] = (
    monthly_df["Sales"]
    .rolling(3)
    .mean()
)

monthly_df["Month"] = monthly_df["Date"].dt.month
monthly_df["Quarter"] = monthly_df["Date"].dt.quarter

monthly_df["Season"] = (
    monthly_df["Month"] % 12 // 3 + 1
)

monthly_df.dropna(inplace=True)

# ==============================
# Historical Prediction
# ==============================

X = monthly_df[
    [
        "Lag1",
        "Lag2",
        "Lag3",
        "RollingMean3",
        "Month",
        "Quarter",
        "Season"
    ]
]

y = monthly_df["Sales"]

historical_pred = model.predict(X)

# =====================================
# Recursive Forecast
# =====================================

forecast_values = []

lag1 = monthly_df["Sales"].iloc[-1]
lag2 = monthly_df["Sales"].iloc[-2]
lag3 = monthly_df["Sales"].iloc[-3]

last_date = monthly_df["Date"].iloc[-1]

future_dates = []

for i in range(months_ahead):

    next_date = last_date + pd.DateOffset(months=1)

    month = next_date.month
    quarter = next_date.quarter
    season = month % 12 // 3 + 1

    rolling_mean = (lag1 + lag2 + lag3) / 3

    X_future = pd.DataFrame({
        "Lag1":[lag1],
        "Lag2":[lag2],
        "Lag3":[lag3],
        "RollingMean3":[rolling_mean],
        "Month":[month],
        "Quarter":[quarter],
        "Season":[season]
    })

    prediction = model.predict(X_future)[0]

    forecast_values.append(prediction)
    future_dates.append(next_date)

    lag3 = lag2
    lag2 = lag1
    lag1 = prediction

    last_date = next_date

forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Forecast Sales": forecast_values
})

# =====================================
# Forecast Chart
# =====================================

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=monthly_df["Date"],
        y=monthly_df["Sales"],
        mode="lines+markers",
        name="Historical Sales"
    )
)

fig.add_trace(
    go.Scatter(
        x=forecast_df["Date"],
        y=forecast_df["Forecast Sales"],
        mode="lines+markers",
        name="Forecast"
    )
)

fig.update_layout(
    title=f"{forecast_type} Forecast : {selected}",
    xaxis_title="Date",
    yaxis_title="Sales",
    template="plotly_white",
    height=550
)

st.plotly_chart(fig, use_container_width=True)

# =====================================
# Forecast Table
# =====================================

st.subheader("Forecast Results")

st.dataframe(
    forecast_df.style.format({
        "Forecast Sales":"${:,.2f}"
    }),
    use_container_width=True
)

# =====================================
# Model Performance
# =====================================

mae = mean_absolute_error(
    y,
    historical_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y,
        historical_pred
    )
)

st.subheader("Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "MAE",
        f"{mae:.2f}"
    )

with col2:
    st.metric(
        "RMSE",
        f"{rmse:.2f}"
    )
