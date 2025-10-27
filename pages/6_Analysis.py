# pages/6_Analysis.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import LocalOutlierFactor
from scipy.fftpack import dct, idct

from utils import download_open_meteo, get_selected_price_area
from utils_elhub import fetch_line_df, list_groups

st.set_page_config(layout="wide")
st.title("Avansert analyse: Outliers & Anomalier")

# ----------------------- Datasettvalg -----------------------
dataset = st.radio("Analyser datasett", ["Production (Elhub)", "Meteorology (Open-Meteo)"], horizontal=True)

if dataset == "Production (Elhub)":
    pa = get_selected_price_area()
    years = [2021, 2022]  # tilpass etter hva som finnes hos deg
    year = st.selectbox("År", years, index=0)
    month = st.slider("Måned", 1, 12, 7)
    all_groups = list_groups(pa) or ["hydro", "wind", "solar", "thermal", "other"]
    group = st.selectbox("Produksjonsgruppe", all_groups, index=0)

    @st.cache_data(show_spinner=True)
    def _load_prod(pa, grp, y, m):
        df = fetch_line_df(pa, [grp], m, y)
        return df

    df = _load_prod(pa, group, year, month)
    if df.empty:
        st.warning("Ingen produksjonsdata funnet for valget.")
        st.stop()

    series_name = "quantityKwh"
    ts = pd.Series(df[series_name].values, index=pd.to_datetime(df["startTime"]))
    ts = ts.asfreq("H").interpolate(limit_direction="both")

else:
    pa = get_selected_price_area()
    years = list(range(2019, 2025))
    year = st.selectbox("År", years, index=years.index(2021))
    start_date, end_date = f"{year}-01-01", f"{year}-12-31"
    vars_met = ["temperature_2m","precipitation","wind_speed_10m","relative_humidity_2m","surface_pressure","cloud_cover"]
    var = st.selectbox("Variabel", vars_met, index=0)

    @st.cache_data(show_spinner=True)
    def _load_met(pa, sd, ed, var):
        df = download_open_meteo(price_area=pa, start_date=sd, end_date=ed, hourly=[var])
        return df

    df = _load_met(pa, start_date, end_date, var)
    if df.empty:
        st.warning("Ingen meteorologidata funnet for valget.")
        st.stop()

    series_name = var
    ts = pd.Series(df[series_name].values, index=pd.to_datetime(df["time"]))
    ts = ts.asfreq("H").interpolate(limit_direction="both")

# ----------------------- Hjelpefunksjoner -----------------------
def satv_dct(x: np.ndarray, keep_low_k: int = 24):
    """
    Høy-pass via DCT: Behold 'keep_low_k' lavfrekvente koeffisienter som sesong/trend,
    trekk dem fra for å få SATV (seasonally adjusted temperature/production variations).
    """
    x = np.asarray(x, dtype=float)
    c = dct(x, norm="ortho")
    c_hp = c.copy()
    c_hp[:keep_low_k] = 0.0  # fjern lavfrekvent (beholdes i trend)
    satv = idct(c_hp, norm="ortho")
    trend = x - satv
    return satv, trend

def robust_bounds(y: np.ndarray, k_sigma: float = 3.5):
    """
    SPC-grenser med robuste mål (median + MAD).
    """
    med = np.median(y)
    mad = np.median(np.abs(y - med)) + 1e-9
    # approx std fra MAD
    std = 1.4826 * mad
    return med - k_sigma*std, med + k_sigma*std

# -------------------------- TABS --------------------------
tab_outlier, tab_lof = st.tabs(["Outlier / SPC", "Anomaly / LOF"])

# --------------------- Outlier/SPC TAB ---------------------
with tab_outlier:
    st.subheader("Outlier / SPC (robust, basert på SATV)")

    col1, col2 = st.columns(2)
    with col1:
        keep_low_k = st.slider("Frekvens-cutoff (antall lavfrekvente DCT-koeff.)", 4, 336, 48, step=4,
                               help="Større tall → glatter mer sesong/trend; outliers blir mer kortsiktige")
    with col2:
        k_sigma = st.slider("Antall 'σ' (robust)", 2.0, 6.0, 3.5, 0.1)

    satv, trend = satv_dct(ts.values, keep_low_k=keep_low_k)
    low, high = robust_bounds(satv, k_sigma=k_sigma)
    outlier_mask = (satv < low) | (satv > high)

    # Plot original serie og grenser på SATV-projisert nivå (vis original, farg outliers)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(ts.index, ts.values, lw=1.0, label=series_name)
    ax.scatter(ts.index[outlier_mask], ts.values[outlier_mask], s=15, color="tab:red", label="Outliers")
    ax.set_title(f"Outliers i {series_name} – {pa}")
    ax.set_xlabel("Tid")
    ax.set_ylabel(series_name)
    ax.grid(alpha=0.3)
    ax.legend()
    st.pyplot(fig, use_container_width=True)

    st.write({
        "total_points": int(len(ts)),
        "outliers": int(outlier_mask.sum()),
        "outlier_fraction": float(outlier_mask.mean()),
        "upper_bound_SATV": float(high),
        "lower_bound_SATV": float(low),
    })

# ----------------------- LOF TAB -----------------------
with tab_lof:
    st.subheader("Anomaly detection – Local Outlier Factor (LOF)")

    col1, col2 = st.columns(2)
    with col1:
        n_neighbors = st.slider("n_neighbors", 5, 100, 35, 1)
    with col2:
        contamination = st.slider("Forventet andel anomalier", 0.005, 0.10, 0.02, 0.005)

    # Enkle 2D-features: verdi + rullende gjennomsnitt → gir LOF litt kontekst
    x = pd.DataFrame({
        "val": ts.values,
        "roll": pd.Series(ts.values).rolling(24, min_periods=1).mean().values
    }).values

    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    y_pred = lof.fit_predict(x)       # -1 = outlier
    scores = -lof.negative_outlier_factor_
    is_out = (y_pred == -1)

    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.plot(ts.index, ts.values, lw=1.0, label=series_name)
    ax2.scatter(ts.index[is_out], ts.values[is_out], s=15, color="tab:orange", label="LOF outliers")
    ax2.set_title(f"LOF anomalier i {series_name} – {pa}")
    ax2.set_xlabel("Tid")
    ax2.set_ylabel(series_name)
    ax2.grid(alpha=0.3)
    ax2.legend()
    st.pyplot(fig2, use_container_width=True)

    st.write({
        "total_points": int(len(ts)),
        "anomalies": int(is_out.sum()),
        "anomaly_fraction": float(is_out.mean()),
        "score_p95": float(np.percentile(scores, 95)),
        "score_p99": float(np.percentile(scores, 99)),
    })
