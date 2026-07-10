import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Sales Overview", layout="wide")

st.title("📈 Sales Overview Dashboard")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")
    df["Order Date"] = pd.to_datetime(
    df["Order Date"],
    format="%d/%m/%Y"
)
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month_name()
    return df

df = load_data()

# Sidebar Filters
st.sidebar.header("Filters")

regions = st.sidebar.multiselect(
    "Select Region",
    df["Region"].unique(),
    default=df["Region"].unique()
)

categories = st.sidebar.multiselect(
    "Select Category",
    df["Category"].unique(),
    default=df["Category"].unique()
)

filtered = df[
    (df["Region"].isin(regions)) &
    (df["Category"].isin(categories))
]

# KPI Cards
col1, col2, col3 = st.columns(3)

col1.metric("💰 Total Sales",
            f"${filtered['Sales'].sum():,.2f}")

col2.metric("📦 Total Orders",
            filtered["Order ID"].nunique())

col3.metric("💵 Total Profit",
            f"${filtered['Profit'].sum():,.2f}")

st.divider()

# Sales by Year
sales_year = filtered.groupby("Year")["Sales"].sum().reset_index()

fig = px.bar(
    sales_year,
    x="Year",
    y="Sales",
    text_auto=True,
    title="Total Sales by Year"
)

st.plotly_chart(fig, use_container_width=True)

# Monthly Trend
monthly = filtered.groupby(
    pd.Grouper(key="Order Date", freq="ME")
)["Sales"].sum().reset_index()

fig2 = px.line(
    monthly,
    x="Order Date",
    y="Sales",
    markers=True,
    title="Monthly Sales Trend"
)

st.plotly_chart(fig2, use_container_width=True)

# Region & Category Charts
left, right = st.columns(2)

with left:
    region_sales = filtered.groupby("Region")["Sales"].sum().reset_index()

    fig3 = px.pie(
        region_sales,
        values="Sales",
        names="Region",
        title="Sales by Region"
    )

    st.plotly_chart(fig3, use_container_width=True)

with right:
    cat_sales = filtered.groupby("Category")["Sales"].sum().reset_index()

    fig4 = px.bar(
        cat_sales,
        x="Category",
        y="Sales",
        color="Category",
        text_auto=True,
        title="Sales by Category"
    )

    st.plotly_chart(fig4, use_container_width=True)

# Data Preview
st.subheader("Filtered Dataset")

st.dataframe(filtered)

st.success("Sales Overview Loaded Successfully!")
