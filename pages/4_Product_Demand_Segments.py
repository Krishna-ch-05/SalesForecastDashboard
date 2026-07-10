import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# -------------------------------------------------------
# Page Configuration
# -------------------------------------------------------

st.set_page_config(
    page_title="Product Demand Segments",
    page_icon="📦",
    layout="wide"
)

st.title("📦 Product Demand Segments")

# -------------------------------------------------------
# Load Dataset
# -------------------------------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv("train.csv")

    df["Order Date"] = pd.to_datetime(
    df["Order Date"],
    format="%d/%m/%Y"

    )

    df = df.dropna(subset=["Order Date"])

    df["Year"] = df["Order Date"].dt.year

    return df

df = load_data()

# -------------------------------------------------------
# Monthly Sales
# -------------------------------------------------------

monthly = (
    df.groupby(
        ["Sub-Category", "Year"]
    )["Sales"]
    .sum()
    .reset_index()
)

# -------------------------------------------------------
# Feature Engineering
# -------------------------------------------------------

summary = (
    df.groupby("Sub-Category")
    .agg(
        Total_Sales=("Sales", "sum"),
        Average_Order_Value=("Sales", "mean")
    )
)

growth = (
    monthly.groupby("Sub-Category")["Sales"]
    .pct_change()
    .groupby(monthly["Sub-Category"])
    .mean()
)

volatility = (
    monthly.groupby("Sub-Category")["Sales"]
    .std()
)

summary["Growth_Rate"] = growth
summary["Volatility"] = volatility

summary = summary.fillna(0)

# -------------------------------------------------------
# Standardization
# -------------------------------------------------------

scaler = StandardScaler()

scaled = scaler.fit_transform(summary)

# -------------------------------------------------------
# KMeans Clustering
# -------------------------------------------------------

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

summary["Cluster"] = kmeans.fit_predict(scaled)

# -------------------------------------------------------
# Cluster Names
# -------------------------------------------------------

cluster_names = {

    0: "High Volume",
    1: "Growing Demand",
    2: "Stable Demand",
    3: "High Volatility"

}

summary["Demand Cluster"] = summary["Cluster"].map(cluster_names)

# -------------------------------------------------------
# PCA
# -------------------------------------------------------

pca = PCA(n_components=2)

points = pca.fit_transform(scaled)

summary["PCA1"] = points[:,0]
summary["PCA2"] = points[:,1]

# -------------------------------------------------------
# Scatter Plot
# -------------------------------------------------------

fig = px.scatter(

    summary,

    x="PCA1",

    y="PCA2",

    color="Demand Cluster",

    hover_name=summary.index,

    title="Product Demand Clusters"

)

st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------
# Cluster Table
# -------------------------------------------------------

st.subheader("Sub-Category Cluster Table")

table = summary.reset_index()[

    [

        "Sub-Category",

        "Total_Sales",

        "Average_Order_Value",

        "Growth_Rate",

        "Volatility",

        "Demand Cluster"

    ]

]

st.dataframe(table)

# -------------------------------------------------------
# Stocking Strategy
# -------------------------------------------------------

st.subheader("Recommended Stocking Strategy")

st.markdown("""

**High Volume**
- Maintain high inventory.
- Replenish frequently.

**Growing Demand**
- Increase stock gradually.
- Monitor future demand closely.

**Stable Demand**
- Maintain regular inventory levels.
- Predictable restocking schedule.

**High Volatility**
- Keep safety stock.
- Monitor sales frequently to avoid stockouts.

""")
