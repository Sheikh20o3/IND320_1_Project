# pages/3_Meteorology.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from scipy.signal import spectrogram

from utils import download_open_meteo, get_selected_price_area, get_price_area_table

st.set_page_config(layout="wide")
st.title("Meteorology (ERA5 via Open-Meteo)")

# ---- Area selection: use global selector from page 2, but allow local override ----
colA, colB = st.columns([1, 3])
with colA:
    use_local = st.checkbox("Override price area locally", value=False)
with colB:
    if use_local:
        df_pa = get_price_area_table()
        pa = st.selectbox("Choose price area", df_pa["price_area"].tolist(), index=0)
    else:
        pa = get_selected_price_area()

st.caption(f"Active price area: **{pa}**")

# ---- Time range and variables ----
years = list(range(2019, 2025))
year = st.selectbox("Year", years, index=years.index(2021))
start_date = f"{year}-01-01"
end_date   = f"{year}-12-31"

# Recommended ERA5 variables
default_hourly = [
    "temperature_2m",
    "precipitation",
    "wind_speed_10m",
    "relative_humidity_2m",
    "surface_pressure",
    "cloud_cover",
]
vars_chosen = st.multiselect("Choose variables (ERA5)", default_hourly, default=default_hourly)

@st.cache_data(show_spinner=True)
def _load_met(pa, sd, ed, hourly):
    return download_open_meteo(price_area=pa, start_date=sd, end_date=ed, hourly=hourly)

df = _load_met(pa, start_date, end_date, tuple(vars_chosen))
if df.empty:
    st.warning("No meteorological data returned for this selection.")
    st.stop()

# Note: New A should not display raw data tables; hence no st.dataframe here.

tab_stl, tab_spec = st.tabs(["STL", "Spectrogram"])

# -------------------------- STL TAB --------------------------
with tab_stl:
    st.subheader("STL decomposition")
    target = st.selectbox("Choose series for STL", [c for c in df.columns if c != "time"], index=0)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        period = st.number_input(
            "Period (hours)",
            min_value=2, value=168, step=1,
            help="Weekly rhythm = 24*7=168"
        )
    with col2:
        seasonal = st.number_input(
            "Seasonal (LOESS window, odd)",
            min_value=7, value=13, step=2,
            help="Must be odd and at least 7"
        )
    with col3:
        trend = st.number_input(
            "Trend (LOESS window, odd)",
            min_value=3, value=301, step=2,
            help="Must be odd and at least 3"
        )
    with col4:
        robust = st.checkbox("Robust", value=True)

    # --- Build hourly time series without NaN ---
    ts = pd.Series(pd.to_numeric(df[target], errors="coerce").values, index=pd.to_datetime(df["time"]))
    ts = ts.asfreq("H").interpolate(limit_direction="both")
    nobs = int(ts.size)

    # --- Helpers to ensure safe STL parameters ---
    def _to_odd(x: int) -> int:
        x = int(x)
        return x if x % 2 == 1 else x + 1

    def _clamp(v: int, lo: int, hi: int) -> int:
        return int(max(lo, min(hi, v)))

    # Minimum requirements
    period = int(max(2, int(period)))
    seasonal = _to_odd(max(7, int(seasonal)))
    trend = _to_odd(max(3, int(trend)))

    # STL dislikes series too short relative to periods/windows
    # 1) ensure we have at least 2*period observations (otherwise lower period)
    if nobs < 2 * period:
        period = max(2, nobs // 2)

    # 2) windows cannot be >= nobs
    seasonal = _clamp(seasonal, 7, max(7, nobs - 1))
    seasonal = _to_odd(seasonal)
    trend = _clamp(trend, 3, max(3, nobs - 1))
    trend = _to_odd(trend)

    # 3) trend should be > period (otherwise increase)
    if trend <= period:
        trend = _to_odd(period + 1)

    # 4) extra safety: if still very large windows, scale down proportionally
    #    (e.g., short selections during testing)
    if seasonal >= nobs:
        seasonal = _to_odd(max(7, nobs // 5 * 2 + 1))
    if trend >= nobs:
        trend = _to_odd(max(3, nobs // 3 * 2 + 1))
    if trend <= period:
        trend = _to_odd(period + 1)

    # 5) final fallback if init still fails
    try:
        stl = STL(ts, period=int(period), seasonal=int(seasonal), trend=int(trend), robust=robust)
    except Exception:
        # choose more conservative windows based on dataset length
        seasonal = _to_odd(_clamp(max(7, period + 5), 7, max(7, nobs - 3)))
        trend = _to_odd(_clamp(max(3, period * 3 + 1), 3, max(3, nobs - 3)))
        st.info("Parameters for STL were automatically adjusted to avoid errors.")
        stl = STL(ts, period=int(period), seasonal=int(seasonal), trend=int(trend), robust=robust)

    res = stl.fit()
    fig = res.plot()
    fig.set_size_inches(12, 8)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

# ----------------------- SPECTROGRAM TAB ----------------------
with tab_spec:
    st.subheader("Spectrogram")
    target2 = st.selectbox(
        "Choose series for spectrogram",
        [c for c in df.columns if c != "time"],
        index=0,
        key="spec_target"
    )

    # Choose window length first, then set overlap based on the window (overlap < nperseg)
    window_length = st.slider("Window length (nperseg, hours)", min_value=32, max_value=1024, value=256, step=32)
    max_overlap = max(0, window_length - 2)
    overlap = st.slider("Overlap (hours)", min_value=0, max_value=max_overlap, value=min(128, max_overlap), step=16)

    sig = pd.Series(pd.to_numeric(df[target2], errors="coerce").values, index=pd.to_datetime(df["time"])).asfreq("H")
    sig = sig.interpolate(limit_direction="both").values

    f, t, Sxx = spectrogram(
        sig, fs=1.0, nperseg=int(window_length), noverlap=int(overlap),
        scaling="density", mode="magnitude"
    )
    Sxx_log = 10 * np.log10(Sxx + 1e-12)

    fig2, ax = plt.subplots(figsize=(12, 5))
    pcm = ax.pcolormesh(t, f, Sxx_log, shading="auto")
    ax.set_ylabel("Frequency [cycles/hour]")
    ax.set_xlabel("Time [hours from start]")
    fig2.colorbar(pcm, ax=ax, label="dB")
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)

    st.caption("Tip: A clear ‘stripe’ at ~1/24 indicates a diurnal cycle; ~1/168 indicates a weekly rhythm.")
