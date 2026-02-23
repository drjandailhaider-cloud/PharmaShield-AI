import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
import requests

st.set_page_config(page_title="RxGuard AI", layout="wide")

# ===============================
# SIDEBAR
# ===============================
st.sidebar.title("💊 RxGuard AI")
page = st.sidebar.radio("Navigation", [
    "Executive Dashboard",
    "Doctor Intelligence",
    "Territory Intelligence",
    "Predictive Intelligence",
    "Upload Data"
])

# ===============================
# DATA STORAGE
# ===============================
if "data" not in st.session_state:
    st.session_state.data = None

# ===============================
# UPLOAD PAGE
# ===============================
if page == "Upload Data":
    st.title("📂 Upload Excel File")

    file = st.file_uploader("Upload Sales Excel (.xlsx)", type=["xlsx"])

    if file:
        df = pd.read_excel(file)
        st.session_state.data = df
        st.success("Data Uploaded Successfully ✅")
        st.dataframe(df.head())

# ===============================
# IF DATA NOT UPLOADED
# ===============================
if st.session_state.data is None and page != "Upload Data":
    st.warning("Please upload data first from Upload Page.")
    st.stop()

df = st.session_state.data

# Basic Required Columns:
# Doctor | Month | Units | Territory

# ===============================
# EXECUTIVE DASHBOARD
# ===============================
if page == "Executive Dashboard":

    st.title("📊 Executive Dashboard")

    total_revenue = df["Units"].sum()
    doctors = df["Doctor"].nunique()
    territories = df["Territory"].nunique()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Units", total_revenue)
    col2.metric("Total Doctors", doctors)
    col3.metric("Total Territories", territories)

    monthly = df.groupby("Month")["Units"].sum().reset_index()

    fig = px.line(monthly, x="Month", y="Units",
                  title="Revenue Trend")
    st.plotly_chart(fig, use_container_width=True)

# ===============================
# DOCTOR INTELLIGENCE
# ===============================
if page == "Doctor Intelligence":

    st.title("👨‍⚕ Doctor Leakage Analysis")

    doctor_data = df.groupby(["Doctor", "Month"])["Units"].sum().reset_index()
    pivot = doctor_data.pivot(index="Doctor", columns="Month", values="Units").fillna(0)

    if pivot.shape[1] >= 2:
        pivot["Drop %"] = ((pivot.iloc[:, -2] - pivot.iloc[:, -1]) /
                           pivot.iloc[:, -2]) * 100
        pivot["Risk"] = np.where(pivot["Drop %"] > 20, "High",
                          np.where(pivot["Drop %"] > 10, "Medium", "Low"))

        st.dataframe(pivot.sort_values("Drop %", ascending=False))

# ===============================
# TERRITORY INTELLIGENCE
# ===============================
if page == "Territory Intelligence":

    st.title("🗺 Territory Performance")

    territory = df.groupby("Territory")["Units"].sum().reset_index()

    fig = px.bar(territory, x="Territory", y="Units",
                 title="Territory Performance",
                 color="Units")
    st.plotly_chart(fig, use_container_width=True)

# ===============================
# PREDICTIVE INTELLIGENCE
# ===============================
if page == "Predictive Intelligence":

    st.title("🤖 Revenue Forecast (Free AI)")

    monthly = df.groupby("Month")["Units"].sum().reset_index()
    monthly["Month_Index"] = range(len(monthly))

    X = monthly[["Month_Index"]]
    y = monthly["Units"]

    model = LinearRegression()
    model.fit(X, y)

    next_month = np.array([[len(monthly)]])
    prediction = model.predict(next_month)[0]

    st.metric("Predicted Next Month Units", int(prediction))

    fig = px.line(monthly, x="Month", y="Units",
                  title="Forecast Trend")
    st.plotly_chart(fig, use_container_width=True)

    # OPTIONAL: FREE HuggingFace AI API
    st.subheader("AI Insight Summary")

    if st.button("Generate AI Insight"):
        prompt = f"Analyze pharma sales trend. Current monthly units: {list(monthly['Units'])}. Give business insight."

        response = requests.post(
            "https://api-inference.huggingface.co/models/google/flan-t5-base",
            headers={"Authorization": "Bearer YOUR_FREE_HF_TOKEN"},
            json={"inputs": prompt}
        )

        if response.status_code == 200:
            st.write(response.json()[0]["generated_text"])
        else:
            st.error("Add your free HuggingFace token.")
