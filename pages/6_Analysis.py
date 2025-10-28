# pages/6_Analysis.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import LocalOutlierFactor
from scipy.fftpack import dct, idct

from utils import download_open_meteo, get_selected_price_area

st.set_page_config(layout="wide")
st.title("Avansert analyse: Outliers & Anomalier (Open-Meteo)")

# ------------------------------------------------------------
# Konfigurasjon / valg
# ------------------------------------------------------------
pa = get_selected_price_area()
years = list(range(2019, 2026))
year = st.selectbox("År", years, index=years.index(2019))
start_date, end_date = f"{year}-01-01", f"{year}-12-31"

# Vi trenger kun temperatur og nedbør til denne siden (new B)
REQ_VARS = ("temperature_2m", "precipitation")

@st.cache_data(show_spinner=True)
def _load_met(pa_code: str, sd: str, ed: str, vars_):
    return download_open_meteo(price_area=pa_code, start_date=sd, end_date=ed, hourly=tuple(vars_))

df = _load_met(pa, start_date, end_date, REQ_VARS)
if df.empty:
    st.warning("Ingen meteorologidata funnet for valget.")
    st.stop()

df["time"] = pd.to_datetime(df["time"])

# ------------------------------------------------------------
# Hjelpefunksjoner
# ------------------------------------------------------------
def satv_dct(x: np.ndarray, keep_low_k: int = 48):
    """
    Høy-pass via DCT: behold lavfrekvent komponent som trend (første keep_low_k koeffisienter),
    og trekk den fra originalen for å få SATV (sesongjusterte variasjoner).
    """
    x = np.asarray(x, dtype=float)
    C = dct(x, norm="ortho")
    # rekonstruer trend fra lavfrekvent del
    C_low = C.copy()
    C_low[keep_low_k:] = 0.0
    trend = idct(C_low, norm="ortho")
    satv = x - trend
    return satv, trend

def robust_bounds(y: np.ndarray, k_sigma: float = 3.5):
    """
    SPC-grenser basert på median og MAD (robuste mål).
    """
    med = np.median(y)
    mad = np.median(np.abs(y - med)) + 1e-12
    std = 1.4826 * mad  # approx std
    return med - k_sigma * std, med + k_sigma * std

# ------------------------------------------------------------
# Tabs: Outlier/SPC (Temperature) og Anomaly/LOF (Precipitation)
# ------------------------------------------------------------
tab_outlier, tab_lof = st.tabs(["Outlier / SPC (Temperature)", "Anomaly / LOF (Precipitation)"])

# --------------------- Outlier/SPC TAB ---------------------
with tab_outlier:
    st.subheader("Outlier / SPC (robust, basert på SATV) — Temperature")

    # Temperaturserie som timeserie
    ts_temp = (
        pd.Series(df["temperature_2m"].values, index=df["time"])
        .asfreq("H")
        .interpolate(limit_direction="both")
    )

    col1, col2 = st.columns(2)
    with col1:
        keep_low_k = st.slider(
            "Frekvens-cutoff (antall lavfrekvente DCT-koeff.)",
            min_value=4, max_value=336, value=48, step=4,
            help="Større tall → mer glatting i trend; outliers blir mer kortsiktige"
        )
    with col2:
        k_sigma = st.slider(
            "Antall 'σ' (robust)",
            min_value=2.0, max_value=6.0, value=3.5, step=0.1
        )

    satv, trend = satv_dct(ts_temp.values, keep_low_k=keep_low_k)
    low_satv, high_satv = robust_bounds(satv, k_sigma=k_sigma)

    # Prosjekter robuste grenser tilbake på originalskala
    lower_curve = trend + low_satv
    upper_curve = trend + high_satv
    outlier_mask = (ts_temp.values < lower_curve) | (ts_temp.values > upper_curve)

    # Plot: originalserie + SPC-grenser + outliers
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(ts_temp.index, ts_temp.values, lw=1.0, label="temperature_2m")
    ax.plot(ts_temp.index, lower_curve, ls="--", alpha=0.8, label="Nedre grense")
    ax.plot(ts_temp.index, upper_curve, ls="--", alpha=0.8, label="Øvre grense")
    ax.scatter(ts_temp.index[outlier_mask], ts_temp.values[outlier_mask], s=15, label="Outliers")
    ax.set_title(f"Outliers i temperature_2m – {pa} ({year})")
    ax.set_xlabel("Tid")
    ax.set_ylabel("temperature_2m [°C]")
    ax.grid(alpha=0.3)
    ax.legend()
    st.pyplot(fig, use_container_width=True)

    st.write({
        "total_points": int(len(ts_temp)),
        "outliers": int(outlier_mask.sum()),
        "outlier_fraction": float(outlier_mask.mean()),
        "lower_bound_SATV": float(low_satv),
        "upper_bound_SATV": float(high_satv),
    })

# ----------------------- LOF TAB -----------------------
with tab_lof:
    st.subheader("Anomaly detection — Local Outlier Factor (LOF) — Precipitation")

    ts_prec = (
        pd.Series(df["precipitation"].values, index=df["time"])
        .asfreq("H")
        .interpolate(limit_direction="both")
    )

    col1, col2 = st.columns(2)
    with col1:
        n_neighbors = st.slider("n_neighbors", 5, 100, 35, 1)
    with col2:
        contamination = st.slider("Forventet andel anomalier", 0.005, 0.10, 0.01, 0.005)  # default 1%

    # Enkle 2D-features: verdi + rullende gjennomsnitt (gir LOF litt lokal kontekst)
    X = pd.DataFrame({
        "val": ts_prec.values,
        "roll": pd.Series(ts_prec.values).rolling(24, min_periods=1).mean().values
    }).values

    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    y_pred = lof.fit_predict(X)   # -1 = outlier
    scores = -lof.negative_outlier_factor_
    is_out = (y_pred == -1)

    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.plot(ts_prec.index, ts_prec.values, lw=1.0, label="precipitation")
    ax2.scatter(ts_prec.index[is_out], ts_prec.values[is_out], s=15, label="LOF anomalies")
    ax2.set_title(f"LOF anomalier i precipitation – {pa} ({year})")
    ax2.set_xlabel("Tid")
    ax2.set_ylabel("precipitation [mm]")
    ax2.grid(alpha=0.3)
    ax2.legend()
    st.pyplot(fig2, use_container_width=True)

    st.write({
        "total_points": int(len(ts_prec)),
        "anomalies": int(is_out.sum()),
        "anomaly_fraction": float(is_out.mean()),
        "score_p95": float(np.percentile(scores, 95)),
        "score_p99": float(np.percentile(scores, 99)),
    })
