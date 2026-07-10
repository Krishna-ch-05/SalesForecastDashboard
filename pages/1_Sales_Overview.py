import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Sales Overview", layout="wide")

st.title("📊 Sales Overview Dashboard")

# Load dataset
df = pd.read_csv("train.csv", encoding="latin1")

# Clean column names
df.columns = df.columns.str.strip()

# Convert date column
df["Order Date"] = pd.to_datetime(
    df["Order Date"],
    format="%d/%m/%Y"
)

# Create Year and Month columns
df["Year"] = df["Order Date"].dt.year
df["Month"] = df["Order Date"].dt.to_period("M").astype(str)

# Sidebar filters
st.sidebar.header("Filters")

regions = st.sidebar.multiselect(
    "Select Region",
    options=sorted(df["Region"].dropna().unique()),
    default=sorted(df["Region"].dropna().unique())
)

categories = st.sidebar.multiselect(
    "Select Category",
    options=sorted(df["Category"].dropna().unique()),
    default=sorted(df["Category"].dropna().unique())
)

filtered_df = df[
    (df["Region"].isin(regions)) &
    (df["Category"].isin(categories))
]

# KPI
st.metric("Total Sales", f"${filtered_df['Sales'].sum():,.2f}")

# ----------------------------
# Total Sales by Year
# ----------------------------
year_sales = (
    filtered_df.groupby("Year")["Sales"]
    .sum()
    .reset_index()
)

fig1 = px.bar(
    year_sales,
    x="Year",
    y="Sales",
    title="Total Sales by Year",
    text_auto=True
)

st.plotly_chart(fig1, use_container_width=True)

# ----------------------------
# Monthly Sales Trend
# ----------------------------
monthly_sales = (
    filtered_df.groupby("Month")["Sales"]
    .sum()
    .reset_index()
)

fig2 = px.line(
    monthly_sales,
    x="Month",
    y="Sales",
    markers=True,
    title="Monthly Sales Trend"
)

st.plotly_chart(fig2, use_container_width=True)

# ----------------------------
# Sales by Region
# ----------------------------
region_sales = (
    filtered_df.groupby("Region")["Sales"]
    .sum()
    .reset_index()
)

fig3 = px.bar(
    region_sales,
    x="Region",
    y="Sales",
    title="Sales by Region",
    text_auto=True
)

st.plotly_chart(fig3, use_container_width=True)

# ----------------------------
# Sales by Category
# ----------------------------
category_sales = (
    filtered_df.groupby("Category")["Sales"]
    .sum()
    .reset_index()
)

fig4 = px.pie(
    category_sales,
    names="Category",
    values="Sales",
    title="Sales by Category"
)

st.plotly_chart(fig4, use_container_width=True)
