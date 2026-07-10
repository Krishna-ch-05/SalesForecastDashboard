import streamlit as st

st.set_page_config(
    page_title="Sales Forecast Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Sales Forecast Dashboard")

st.markdown("""
Welcome to the **Sales Forecast Dashboard**

Use the navigation panel on the left to explore:

- 📈 Sales Overview
- 🔮 Forecast Explorer
- 🚨 Anomaly Report
- 📦 Product Demand Segments
""")

st.success("Dashboard Loaded Successfully!")
