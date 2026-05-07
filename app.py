import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from datetime import date

st.set_page_config(page_title="Dashboard Harga Rumah", layout="wide")

@st.cache_resource
def load_model():
    with open("model_rumah.pkl", "rb") as f:
        return pickle.load(f)

skl_model = load_model()

# --- Sidebar ---
st.sidebar.header("Input Fitur")

x1 = st.sidebar.date_input(
    "X1 - Tanggal Transaksi",
    value=date(2013, 6, 1),
    min_value=date(2012, 1, 1),
    max_value=date(2014, 12, 31),
)
x2 = st.sidebar.slider("X2 - Usia Rumah (tahun)", 0.0, 50.0, 17.8, step=0.1)
x3 = st.sidebar.number_input(
    "X3 - Jarak ke Stasiun MRT (meter)", 0.0, 7000.0, 1096.96, step=10.0
)
x4 = st.sidebar.slider("X4 - Jumlah Minimarket Terdekat", 0, 15, 4)
x5 = st.sidebar.number_input("X5 - Latitude", 24.0, 26.0, 24.969, step=0.001, format="%.3f")
x6 = st.sidebar.number_input("X6 - Longitude", 121.0, 122.5, 121.533, step=0.001, format="%.3f")

# --- Header ---
st.title("Dashboard Analitik DJPb")
st.subheader("Prediksi Harga Unit Rumah — Model AdaBoost Regressor")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("#### Data Input")
    input_df = pd.DataFrame({
        "Fitur": [
            "X1 Tanggal Transaksi",
            "X2 Usia Rumah (thn)",
            "X3 Jarak MRT (m)",
            "X4 Jumlah Minimarket",
            "X5 Latitude",
            "X6 Longitude",
        ],
        "Nilai": [
            str(x1), x2, x3, x4, x5, x6,
        ],
    })
    st.dataframe(input_df, use_container_width=True, hide_index=True)

with col2:
    st.markdown("#### Info Model")
    st.info(
        f"**Tipe:** AdaBoost Regressor\n\n"
        f"**Base estimator:** Decision Tree\n\n"
        f"**Jumlah Estimator:** {len(skl_model.estimators_)}\n\n"
        f"**Target:** Harga Unit Rumah"
    )

# --- Prediksi ---
if st.button("Jalankan Prediksi", type="primary"):
    import datetime as dt
    ts = dt.datetime(x1.year, x1.month, x1.day).timestamp()
    X = np.array([[ts, x2, x3, x4, x5, x6]])
    hasil = skl_model.predict(X)[0]

    st.success(f"Prediksi Harga: **{hasil:,.0f} NTD**")
    st.metric(
        label="Estimasi Harga Unit Area",
        value=f"{hasil:,.0f} NTD",
        delta=f"{hasil / 1_000_000:.2f} juta NTD",
    )

# --- Feature Importance ---
st.markdown("---")
st.markdown("#### Feature Importance")
feature_names = [
    "X1 Tgl Transaksi", "X2 Usia Rumah", "X3 Jarak MRT",
    "X4 Minimarket", "X5 Latitude", "X6 Longitude",
]
importances = skl_model.feature_importances_

fig, ax = plt.subplots(figsize=(8, 3))
bars = ax.barh(feature_names, importances, color="#4C72B0")
ax.set_xlabel("Importance")
ax.set_title("Kontribusi Fitur terhadap Prediksi")
ax.bar_label(bars, fmt="%.3f", padding=3)
fig.tight_layout()
st.pyplot(fig)
