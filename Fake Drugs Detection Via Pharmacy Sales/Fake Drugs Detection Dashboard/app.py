# file: app.py (updated with extra research question sections)

import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Fake Drugs Detection Dashboard", layout="wide")

# ----------------------------
# Load Data
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\tech\Fake Drugs Detection\pharma_anomaly_scored.csv", parse_dates=["Date", "Expiry_Date"])
    return df

df = load_data()

st.title("💊 Fake Drugs Detection & Risk Dashboard")

# ----------------------------
# Sidebar Filters
# ----------------------------
st.sidebar.header("Filters")
location_filter = st.sidebar.multiselect("Select Locations", df["Location"].unique(), default=df["Location"].unique())
pharmacy_filter = st.sidebar.multiselect("Select Pharmacies", df["Pharmacy"].unique(), default=df["Pharmacy"].unique())

df_filtered = df[(df["Location"].isin(location_filter)) & (df["Pharmacy"].isin(pharmacy_filter))]

st.markdown("### Dataset Overview")
st.dataframe(df_filtered.head(20))

# ----------------------------
# 1. Suspicious Pharmacies & Suppliers
# ----------------------------
st.markdown("## 🏥 Suspicious Pharmacies & Suppliers")

pharmacy_risk = df_filtered.groupby("Pharmacy").agg(
    total_records=("Batch_Number", "count"),
    anomalies=("ensemble_flag", "sum"),
    avg_risk=("risk_score", "mean")
).sort_values("anomalies", ascending=False).head(10)

supplier_risk = df_filtered.groupby("Supplier_Name").agg(
    total_records=("Batch_Number", "count"),
    anomalies=("ensemble_flag", "sum"),
    avg_risk=("risk_score", "mean")
).sort_values("anomalies", ascending=False).head(10)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Top 10 Suspicious Pharmacies")
    st.plotly_chart(px.bar(pharmacy_risk, x=pharmacy_risk.index, y="anomalies", color="avg_risk",
                           title="Pharmacies by Anomalies", text="anomalies"), use_container_width=True)

with col2:
    st.subheader("Top 10 Suspicious Suppliers")
    st.plotly_chart(px.bar(supplier_risk, x=supplier_risk.index, y="anomalies", color="avg_risk",
                           title="Suppliers by Anomalies", text="anomalies"), use_container_width=True)

# ----------------------------
# 2. Pharmacies with Unusual Price Drops
# ----------------------------
st.markdown("## 💰 Pharmacies Selling at Unusually Low Prices")

price_dev = df_filtered.groupby("Pharmacy").agg(
    avg_price=("Price", "mean"),
    avg_brand_price=("brand_avg_price", "mean")
).reset_index()
price_dev["price_ratio"] = price_dev["avg_price"] / price_dev["avg_brand_price"]

fig_price = px.bar(price_dev.sort_values("price_ratio").head(10), 
                   x="Pharmacy", y="price_ratio", 
                   title="Pharmacies with Lowest Price Ratios",
                   labels={"price_ratio": "Pharmacy Price / Brand Avg Price"})
st.plotly_chart(fig_price, use_container_width=True)

# ----------------------------
# 3. Brands with High Sales (Volume)
# ----------------------------
st.markdown("## 🏷️ Top Brands by Sales Volume")

brand_sales = df_filtered.groupby("Brand").agg(
    total_sales=("Quantity", "sum"),
    anomalies=("ensemble_flag", "sum")
).sort_values("total_sales", ascending=False).head(10)

fig_brand_sales = px.bar(brand_sales, x=brand_sales.index, y="total_sales", color="anomalies",
                         title="Top Brands by Sales Volume", text="total_sales")
st.plotly_chart(fig_brand_sales, use_container_width=True)

# ----------------------------
# 4. Urban vs Rural Anomaly Comparison
# ----------------------------
st.markdown("## 🌆 Urban vs Rural Anomaly Rates")

urban_rural = df_filtered.groupby("Location").agg(
    anomalies=("ensemble_flag", "mean"),
    total=("ensemble_flag", "count")
).reset_index()

fig_urban = px.bar(urban_rural, x="Location", y="anomalies",
                   title="Anomaly Rate by Location",
                   labels={"anomalies": "Avg Anomaly Rate"})
st.plotly_chart(fig_urban, use_container_width=True)

# ----------------------------
# 5. Expiry Date Monitoring
# ----------------------------
st.markdown("## ⏳ Near Expiry vs Normal Drugs")

expiry_counts = df_filtered["near_expiry"].value_counts().rename({0: "Normal", 1: "Near Expiry"}).reset_index()
expiry_counts.columns = ["Status", "Count"]

st.plotly_chart(px.pie(expiry_counts, names="Status", values="Count", 
                       title="Near Expiry vs Normal Drugs"), use_container_width=True)

# ----------------------------
# Extra: Risk Score Distribution
# ----------------------------
st.markdown("## 📊 Risk Score Distribution")

fig6, ax = plt.subplots(figsize=(7, 4))
sns.histplot(df_filtered["risk_score"], bins=10, kde=True, ax=ax)
ax.set_title("Distribution of Risk Scores")
st.pyplot(fig6)
