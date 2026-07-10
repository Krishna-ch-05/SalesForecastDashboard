import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="Anomaly Report", page_icon="🚨", layout="wide")

st.title("🚨 Sales Anomaly Report")

# -----------------------------
# Load Dataset
# -----------------------------

@st.cache_data
def load_data():

    df = pd.read_csv("train.csv")

    df["Order Date"] = pd.to_datetime(
    df["Order Date"],
    format="%d/%m/%Y"
)
    )

    df = df.dropna(subset=["Order Date"])

    return df

df = load_data()

# -----------------------------
# Weekly Sales
# -----------------------------

weekly_sales = (
    df.groupby(pd.Grouper(key="Order Date", freq="W"))["Sales"]
    .sum()
    .reset_index()
)

# -----------------------------
# Isolation Forest
# -----------------------------

iso = IsolationForest(
    contamination=0.05,
    random_state=42
)

weekly_sales["Anomaly"] = iso.fit_predict(
    weekly_sales[["Sales"]]
)

anomalies = weekly_sales[
    weekly_sales["Anomaly"] == -1
]

# -----------------------------
# Plot
# -----------------------------

fig = px.line(
    weekly_sales,
    x="Order Date",
    y="Sales",
    title="Weekly Sales with Anomalies"
)

fig.add_scatter(
    x=anomalies["Order Date"],
    y=anomalies["Sales"],
    mode="markers",
    marker=dict(
        color="red",
        size=10
    ),
    name="Anomaly"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# -----------------------------
# Table
# -----------------------------

st.subheader("Detected Anomalies")

table = anomalies[["Order Date", "Sales"]]

table.columns = [
    "Anomaly Date",
    "Sales Value"
]

st.dataframe(table)

st.success(
    f"Total Anomalies Detected : {len(table)}"
)
