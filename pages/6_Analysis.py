# pages/6_Analysis.py
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.neighbors import LocalOutlierFactor
from scipy.fftpack import dct, idct
import plotly.graph_objects as go

from utils import download_open_meteo, get_selected_price_area

st.set_page_config(layout="wide")
st.title("Advanced analysis: Outliers & Anomalies (Open-Meteo)")

# ------------------------------------------------------------
# Configuration / choices
# ------------------------------------------------------------
pa = get_selected_price_area()
years = list(range(2019, 2026))
year = st.selectbox("Year", years, index=years.index(2019))
start_date, end_date = f"{year}-01-01", f"{year}-12-31"

# We only need temperature and precipitation for this page (new B)
REQ_VARS = ("temperature_2m", "precipitation")


@st.cache_data(show_spinner=True)
def _load_met(pa_code: str, sd: str, ed: str, vars_):
    return download_open_meteo(
        price_area=pa_code,
        start_date=sd,
        end_date=ed,
        hourly=tuple(vars_),
    )


df = _load_met(pa, start_date, end_date, REQ_VARS)
if df.empty:
    st.warning("No meteorological data found for the selection.")
    st.stop()

df["time"] = pd.to_datetime(df["time"])

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def satv_dct(x: np.ndarray, keep_low_k: int = 48):
    """
    High-pass via DCT: keep the low-frequency component as trend (first keep_low_k coefficients),
    and subtract it from the original to obtain SATV (seasonally adjusted transient variations).
    """
    x = np.asarray(x, dtype=float)
    C = dct(x, norm="ortho")
    # Reconstruct trend from low-frequency part
    C_low = C.copy()
    C_low[keep_low_k:] = 0.0
    trend = idct(C_low, norm="ortho")
    satv = x - trend
    return satv, trend


def robust_bounds(y: np.ndarray, k_sigma: float = 3.5):
    """
    SPC bounds based on median and MAD (robust measures).
    """
    med = np.median(y)
    mad = np.median(np.abs(y - med)) + 1e-12
    std = 1.4826 * mad  # approx std
    return med - k_sigma * std, med + k_sigma * std


# ------------------------------------------------------------
# Tabs: Outlier/SPC (Temperature) and Anomaly/LOF (Precipitation)
# ------------------------------------------------------------
tab_outlier, tab_lof = st.tabs(
    ["Outlier / SPC (Temperature)", "Anomaly / LOF (Precipitation)"]
)

# --------------------- Outlier/SPC TAB ---------------------
with tab_outlier:
    st.subheader("Outlier / SPC (robust, based on SATV) — Temperature")

    # Temperature series as hourly time series
    ts_temp = (
        pd.Series(df["temperature_2m"].values, index=df["time"])
        .asfreq("H")
        .interpolate(limit_direction="both")
    )

    col1, col2 = st.columns(2)
    with col1:
        keep_low_k = st.slider(
            "Frequency cutoff (number of low-frequency DCT coeffs.)",
            min_value=4,
            max_value=336,
            value=48,
            step=4,
            help="Larger value → more smoothing in the trend; outliers become more short-term",
        )
    with col2:
        k_sigma = st.slider(
            "Number of 'σ' (robust)",
            min_value=2.0,
            max_value=6.0,
            value=3.5,
            step=0.1,
        )

    satv, trend = satv_dct(ts_temp.values, keep_low_k=keep_low_k)
    low_satv, high_satv = robust_bounds(satv, k_sigma=k_sigma)

    # Project robust bounds back to the original scale
    lower_curve = trend + low_satv
    upper_curve = trend + high_satv
    outlier_mask = (ts_temp.values < lower_curve) | (ts_temp.values > upper_curve)

    # --- Plot with Plotly: original series + SPC bounds + outliers ---
    fig_out = go.Figure()

    fig_out.add_trace(
        go.Scatter(
            x=ts_temp.index,
            y=ts_temp.values,
            mode="lines",
            name="temperature_2m",
        )
    )
    fig_out.add_trace(
        go.Scatter(
            x=ts_temp.index,
            y=lower_curve,
            mode="lines",
            name="Lower bound",
            line=dict(dash="dash"),
        )
    )
    fig_out.add_trace(
        go.Scatter(
            x=ts_temp.index,
            y=upper_curve,
            mode="lines",
            name="Upper bound",
            line=dict(dash="dash"),
        )
    )
    fig_out.add_trace(
        go.Scatter(
            x=ts_temp.index[outlier_mask],
            y=ts_temp.values[outlier_mask],
            mode="markers",
            name="Outliers",
            marker=dict(size=6, color="red", symbol="x"),
        )
    )

    fig_out.update_layout(
        title=f"Outliers in temperature_2m – {pa} ({year})",
        xaxis_title="Time",
        yaxis_title="temperature_2m [°C]",
        hovermode="x unified",
        margin=dict(l=40, r=20, t=40, b=40),
    )

    st.plotly_chart(fig_out, use_container_width=True)

    st.write(
        {
            "total_points": int(len(ts_temp)),
            "outliers": int(outlier_mask.sum()),
            "outlier_fraction": float(outlier_mask.mean()),
            "lower_bound_SATV": float(low_satv),
            "upper_bound_SATV": float(high_satv),
        }
    )

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
        contamination = st.slider(
            "Expected anomaly fraction",
            0.005,
            0.10,
            0.01,
            0.005,
        )  # default 1%

    # --- LOF only on strictly positive precipitation to avoid 0-valued "anomalies" ---
    values = ts_prec.values
    mask_positive = values > 0.0  # ignore dry hours for anomaly detection

    if mask_positive.sum() < n_neighbors + 1:
        st.warning(
            "Not enough non-zero precipitation values for LOF with the current "
            "n_neighbors setting. Try a different year or reduce n_neighbors."
        )
        st.stop()

    # Simple 2D features: value + rolling mean (provides local context to LOF)
    full_features = pd.DataFrame(
        {
            "val": values,
            "roll": pd.Series(values).rolling(24, min_periods=1).mean().values,
        }
    )

    X = full_features[mask_positive].values

    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
    )
    y_pred = lof.fit_predict(X)  # -1 = outlier
    scores_valid = -lof.negative_outlier_factor_

    # Map back to full time index: only positive-precip hours can be anomalies
    is_out = np.zeros(len(values), dtype=bool)
    is_out[mask_positive] = (y_pred == -1)

    scores = np.full(len(values), np.nan, dtype=float)
    scores[mask_positive] = scores_valid

    # --- Plot with Plotly: precipitation + LOF anomalies ---
    fig_lof = go.Figure()

    fig_lof.add_trace(
        go.Scatter(
            x=ts_prec.index,
            y=ts_prec.values,
            mode="lines",
            name="precipitation",
        )
    )
    fig_lof.add_trace(
        go.Scatter(
            x=ts_prec.index[is_out],
            y=ts_prec.values[is_out],
            mode="markers",
            name="LOF anomalies",
            marker=dict(
                size=7,
                color="red",      # high contrast
                symbol="x",       # clearly different from line
                line=dict(width=1),
            ),
        )
    )

    fig_lof.update_layout(
        title=f"LOF anomalies in precipitation – {pa} ({year})",
        xaxis_title="Time",
        yaxis_title="precipitation [mm]",
        hovermode="x unified",
        margin=dict(l=40, r=20, t=40, b=40),
    )

    st.plotly_chart(fig_lof, use_container_width=True)

    # Summary only over non-NaN scores (i.e., non-zero precipitation)
    valid_scores = scores[~np.isnan(scores)]

    st.write(
        {
            "total_points": int(len(ts_prec)),
            "nonzero_points": int(mask_positive.sum()),
            "anomalies": int(is_out.sum()),
            "anomaly_fraction_over_nonzero": float(is_out[mask_positive].mean()),
            "score_p95": float(np.percentile(valid_scores, 95)),
            "score_p99": float(np.percentile(valid_scores, 99)),
        }
    )

    st.caption(
        "Note: LOF is applied only to hours with positive precipitation, "
        "to avoid flagging dry periods (0 mm) as anomalies."
    )
