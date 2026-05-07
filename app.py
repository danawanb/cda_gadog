import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

st.set_page_config(page_title="Dashboard Gap Kompetensi DJPb", layout="wide")

KOMPETENSI = [
    "Olahraga", "Seni", "Komunikasi", "Olah Data",
    "Public Speaking", "Sulap", "Programming",
    "Teamwork", "Networking",
]
GAP_COLS = [f"Gap {k}" for k in KOMPETENSI]


@st.cache_data
def load_data():
    """Replikasi pipeline Orange:
    1. File loader
    2. Formula: Gap_X = Realisasi - Ekspektasi
    3. Select Rows: AND semua Gap_X != 0
    """
    df = pd.read_excel("Data Latihan CDA.xlsx", sheet_name="Rekap Adm")
    for k in KOMPETENSI:
        df[f"Gap {k}"] = df[f"{k} Realisasi"] - df[f"{k} Ekspektasi"]
    df_filtered = df[(df[GAP_COLS] != 0).all(axis=1)].copy()
    return df, df_filtered


df_full, df = load_data()

# --- Sidebar ---
st.sidebar.header("Menu Filter")
pakai_filter = st.sidebar.checkbox(
    "Aktifkan filter Select Rows (semua Gap ≠ 0)", value=True
)
data = df if pakai_filter else df_full

jenis_pilih = st.sidebar.multiselect(
    "Jenis Kantor",
    options=sorted(data["Jenis kantor penugasan"].unique()),
    default=sorted(data["Jenis kantor penugasan"].unique()),
)
status_pilih = st.sidebar.multiselect(
    "Status Remote",
    options=sorted(data["Status Remote"].unique()),
    default=sorted(data["Status Remote"].unique()),
)
data = data[
    data["Jenis kantor penugasan"].isin(jenis_pilih)
    & data["Status Remote"].isin(status_pilih)
].copy()

# --- Header ---
st.title("Dashboard Analitik DJPb")
st.subheader("Penugasan CDA Kelompok 3 — Gap Kompetensi Manajerial")


c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Data Mentah", f"{len(df_full):,}")
c2.metric("Setelah Filter Orange", f"{len(df):,}")
c3.metric("Data Tampil", f"{len(data):,}")
c4.metric("Rata-rata Gap", f"{data[GAP_COLS].mean().mean():.3f}")

tab_data, tab_dist, tab_bar, tab_line, tab_scatter = st.tabs(
    ["Data Table", "Distributions", "Bar Plot", "Line Plot", "Scatter (K-Means)"]
)

# ============ Tab 1: Data Table ============
with tab_data:
    st.markdown("#### Pratinjau Data (setelah Formula + Select Rows)")
    st.dataframe(data.head(50), use_container_width=True)

    st.markdown("#### Column Statistics — Gap Kompetensi")
    stats = data[GAP_COLS].describe().T[["mean", "std", "min", "max"]].round(3)
    stats["std (population)"] = data[GAP_COLS].std(ddof=0).round(6)
    st.dataframe(stats, use_container_width=True)

# ============ Tab 2: Distributions (Orange-style) ============
with tab_dist:
    st.markdown("#### Distributions — Replika Widget Orange")
    col_a, col_b, col_c = st.columns([2, 2, 1])
    with col_a:
        var_options = GAP_COLS + [f"{k} Realisasi" for k in KOMPETENSI] \
                      + [f"{k} Ekspektasi" for k in KOMPETENSI]
        pilih_var = st.selectbox("Variable:", var_options,
                                 index=GAP_COLS.index("Gap Teamwork"))
    with col_b:
        split_by = st.selectbox(
            "Split by:",
            ["(none)", "Status Remote", "Jenis kantor penugasan", "Role dalam unit"],
            index=1,
        )
    with col_c:
        n_bins = st.slider("Bins", 5, 30, 10)

    series = data[pilih_var].dropna()
    bin_edges = np.linspace(series.min(), series.max(), n_bins + 1)
    bin_w = bin_edges[1] - bin_edges[0]

    fig, ax = plt.subplots(figsize=(11, 5))
    palette = ["#4FC3F7", "#EF5350", "#9CCC65", "#FFB74D",
               "#BA68C8", "#4DB6AC", "#FF8A65"]

    if split_by == "(none)":
        ax.hist(series, bins=bin_edges, alpha=0.65, color="#4C72B0",
                edgecolor="white")
        mu, sigma = series.mean(), series.std(ddof=0)
        if sigma > 0:
            xs = np.linspace(bin_edges[0] - bin_w, bin_edges[-1] + bin_w, 250)
            pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * \
                  np.exp(-0.5 * ((xs - mu) / sigma) ** 2)
            ax.plot(xs, pdf * len(series) * bin_w, color="#1f4e79",
                    linewidth=2, label=f"All (μ={mu:g}, σ={sigma:g})")
    else:
        groups = sorted(data[split_by].dropna().unique())
        for i, grp in enumerate(groups):
            sub = data[data[split_by] == grp][pilih_var].dropna()
            if len(sub) == 0:
                continue
            color = palette[i % len(palette)]
            ax.hist(sub, bins=bin_edges, alpha=0.55, color=color,
                    edgecolor="white")
            mu, sigma = sub.mean(), sub.std(ddof=0)
            if sigma > 0:
                xs = np.linspace(bin_edges[0] - bin_w,
                                 bin_edges[-1] + bin_w, 250)
                pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * \
                      np.exp(-0.5 * ((xs - mu) / sigma) ** 2)
                ax.plot(xs, pdf * len(sub) * bin_w, color=color, linewidth=2.5,
                        label=f"{grp} (μ={mu:g}, σ={sigma:g})")

    ax.set_xlabel(pilih_var)
    ax.set_ylabel("Frequency")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)

# ============ Tab 3: Bar Plot ============
with tab_bar:
    st.markdown("#### Rata-rata Gap per Kompetensi (Group by Status Remote)")
    by_status = data.groupby("Status Remote")[GAP_COLS].mean()
    by_status.columns = [c.replace("Gap ", "") for c in by_status.columns]
    st.bar_chart(by_status.T, height=380)

    st.markdown("#### Rata-rata Gap per Jenis Kantor")
    by_jenis = data.groupby("Jenis kantor penugasan")[GAP_COLS].mean()
    by_jenis.columns = [c.replace("Gap ", "") for c in by_jenis.columns]
    st.bar_chart(by_jenis.T, height=360)

    st.markdown("#### Rata-Rata Gap per Lokasi (10 paling rendah / paling negatif)")
    by_lokasi = (
        data.groupby("Lokasi kantor penugasan")[GAP_COLS]
        .mean().sum(axis=1) / 8
    ).sort_values().head(10)
    st.bar_chart(by_lokasi, color="#C44E52", height=320)

# ============ Tab 4: Line Plot ============
with tab_line:
    st.markdown("#### Realisasi vs Ekspektasi (rata-rata) per Kompetensi")
    real_cols = [f"{k} Realisasi" for k in KOMPETENSI]
    eksp_cols = [f"{k} Ekspektasi" for k in KOMPETENSI]
    line_df = pd.DataFrame({
        "Realisasi": data[real_cols].mean().values,
        "Ekspektasi": data[eksp_cols].mean().values,
    }, index=KOMPETENSI)
    st.line_chart(line_df, height=380)

    st.markdown("#### Profil Gap per Status Remote")
    by_status_t = data.groupby("Status Remote")[GAP_COLS].mean().T
    by_status_t.index = [c.replace("Gap ", "") for c in by_status_t.index]
    st.line_chart(by_status_t, height=380)

# ============ Tab 5: Scatter (K-Means) — replika Orange ============
with tab_scatter:
    st.markdown("#### Scatter Plot — Rata-Rata Gap per Lokasi (K-Means Cluster)")
    st.caption(
        "Pipeline: Group by Lokasi (Mean semua Gap + Mode Status Remote) → "
        "Formula `Rata-Rata Gap = (sum 9 gap mean) / 8` → K-Means → Scatter."
    )

    k = st.slider("Jumlah Cluster (k)", 2, 6, 3)

    # Group by Lokasi: mean semua gap + mode Status Remote (replika Orange Group by)
    grouped = data.groupby("Lokasi kantor penugasan").agg({
        **{c: "mean" for c in GAP_COLS},
        "Status Remote": lambda s: s.mode().iloc[0],
    }).reset_index()
    grouped.rename(columns={"Status Remote": "Status Remote - Mode"}, inplace=True)

    # Formula: Rata-Rata Gap = sum(9 gap means) / 8 (sesuai .ows file)
    grouped["Rata-Rata Gap"] = grouped[GAP_COLS].sum(axis=1) / 8

    # K-Means (sesuai Orange owkmeans default: pakai semua fitur Gap_X mean)
    if len(grouped) >= k:
        X = grouped[GAP_COLS].values
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        grouped["Cluster"] = km.fit_predict(X)

        # Scatter: x=Rata-Rata Gap, y=Lokasi, color=Cluster, shape=Status Remote Mode
        shape_map = {"Kantor Pusat": "o", "REMOTE": "x", "TIDAK": "^"}
        cluster_palette = ["#4FC3F7", "#EF5350", "#9CCC65", "#FFB74D",
                           "#BA68C8", "#4DB6AC"]
        clusters = sorted(grouped["Cluster"].unique())

        sorted_lokasi = grouped.sort_values("Lokasi kantor penugasan",
                                            ascending=False)["Lokasi kantor penugasan"].tolist()

        fig, ax = plt.subplots(figsize=(11, 11))
        for _, row in grouped.iterrows():
            color = cluster_palette[int(row["Cluster"]) % len(cluster_palette)]
            marker = shape_map.get(row["Status Remote - Mode"], "s")
            y = sorted_lokasi.index(row["Lokasi kantor penugasan"])
            ax.scatter(row["Rata-Rata Gap"], y, c=color, marker=marker,
                       s=120, edgecolors="white", linewidths=0.8)

        ax.set_yticks(range(len(sorted_lokasi)))
        ax.set_yticklabels(sorted_lokasi, fontsize=9)
        ax.set_xlabel("Rata-Rata Gap")
        ax.set_ylabel("Lokasi kantor penugasan")
        ax.grid(True, axis="x", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        legend_color = [plt.Line2D([], [], marker="o", color=cluster_palette[c],
                                    linestyle="", markersize=10, label=f"C{c+1}")
                        for c in clusters]
        legend_shape = [plt.Line2D([], [], marker=shape_map[s], color="black",
                                    linestyle="", markersize=10, label=s)
                        for s in shape_map if s in grouped["Status Remote - Mode"].values]
        leg1 = ax.legend(handles=legend_shape, loc="upper right", title=None,
                          framealpha=0.9, bbox_to_anchor=(1.0, 1.0))
        ax.add_artist(leg1)
        ax.legend(handles=legend_color, loc="lower right", title=None,
                   framealpha=0.9, bbox_to_anchor=(1.0, 0.0))
        fig.tight_layout()
        st.pyplot(fig)

        st.markdown("#### Tabel Hasil — Group by + Cluster")
        show_cols = ["Lokasi kantor penugasan", "Status Remote - Mode",
                     "Rata-Rata Gap", "Cluster"] + GAP_COLS
        st.dataframe(
            grouped[show_cols].sort_values("Rata-Rata Gap").round(4),
            use_container_width=True,
        )
    else:
        st.warning("Data hasil filter terlalu sedikit untuk clustering.")
