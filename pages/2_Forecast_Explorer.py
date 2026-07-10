import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.title("🔮 Forecast Explorer")

# Load data
df = pd.read_csv("SampleSuperstore.csv", encoding="latin1")
df.columns = df.columns.str.strip()

df["Order Date"] = pd.to_datetime(df["Order Date"])

# Load trained model
with open("best_sales_model.pkl", "rb") as f:
    model = pickle.load(f)

# -----------------------------
# Select forecast type
# -----------------------------
forecast_type = st.selectbox(
    "Forecast Based On",
    ["Category", "Region"]
)

if forecast_type == "Category":
    value = st.selectbox(
        "Select Category",
        sorted(df["Category"].unique())
    )
    filtered = df[df["Category"] == value]

else:
    value = st.selectbox(
        "Select Region",
        sorted(df["Region"].unique())
    )
    filtered = df[df["Region"] == value]

# -----------------------------
# Forecast horizon
# -----------------------------
months = st.slider(
    "Forecast Horizon (Months)",
    1,
    3,
    1
)

# Monthly sales
sales = (
    filtered.groupby(
        pd.Grouper(
            key="Order Date",
            freq="M"
        )
    )["Sales"]
    .sum()
    .reset_index()
)

sales["Month_No"] = np.arange(len(sales))

# Predictions
future = np.arange(
    len(sales),
    len(sales) + months
).reshape(-1,1)

forecast = model.predict(future)

future_dates = pd.date_range(
    sales["Order Date"].max(),
    periods=months+1,
    freq="M"
)[1:]

forecast_df = pd.DataFrame({
    "Order Date": future_dates,
    "Forecast Sales": forecast
})

# Plot
fig = px.line(
    sales,
    x="Order Date",
    y="Sales",
    title="Historical Sales"
)

fig.add_scatter(
    x=forecast_df["Order Date"],
    y=forecast_df["Forecast Sales"],
    mode="lines+markers",
    name="Forecast"
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Model Evaluation
# -----------------------------
pred = model.predict(
    sales["Month_No"].values.reshape(-1,1)
)

mae = mean_absolute_error(
    sales["Sales"],
    pred
)

rmse = np.sqrt(
    mean_squared_error(
        sales["Sales"],
        pred
    )
)

st.subheader("📈 Model Performance")

c1, c2 = st.columns(2)

with c1:
    st.metric("MAE", f"{mae:,.2f}")

with c2:
    st.metric("RMSE", f"{rmse:,.2f}")
