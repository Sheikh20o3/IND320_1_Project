# pages/3_Meteorology.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from scipy.signal import spectrogram

from utils import download_open_meteo, get_selected_price_area, get_price_area_table

st.set_page_config(layout="wide")
st.title("Meteorologi (ERA5 via Open-Meteo)")

# ---- Områdevalg: bruk global velger fra side 2, men tillat lokal override ----
colA, colB = st.columns([1, 3])
with colA:
    use_local = st.checkbox("Overstyr prisområde lokalt", value=False)
with colB:
    if use_local:
        df_pa = get_price_area_table()
        pa = st.selectbox("Velg prisområde", df_pa["price_area"].tolist(), index=0)
    else:
        pa = get_selected_price_area()

st.caption(f"Aktivt prisområde: **{pa}**")

# ---- Tidsrom og variabler ----
years = list(range(2019, 2025))
year = st.selectbox("År", years, index=years.index(2021))
start_date = f"{year}-01-01"
end_date   = f"{year}-12-31"

# Anbefalte ERA5-variabler
default_hourly = [
    "temperature_2m",
    "precipitation",
    "wind_speed_10m",
    "relative_humidity_2m",
    "surface_pressure",
    "cloud_cover",
]
vars_chosen = st.multiselect("Velg variabler (ERA5)", default_hourly, default=default_hourly)

@st.cache_data(show_spinner=True)
def _load_met(pa, sd, ed, hourly):
    return download_open_meteo(price_area=pa, start_date=sd, end_date=ed, hourly=hourly)

df = _load_met(pa, start_date, end_date, tuple(vars_chosen))
if df.empty:
    st.warning("Ingen meteorologidata returnert for dette valget.")
    st.stop()

st.dataframe(df.head(10), use_container_width=True)

tab_stl, tab_spec = st.tabs(["STL", "Spectrogram"])

# -------------------------- STL TAB --------------------------
with tab_stl:
    st.subheader("STL-dekomponering")
    target = st.selectbox("Velg serie for STL", [c for c in df.columns if c != "time"], index=0)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        period = st.number_input("Period (timer)", min_value=2, value=168, help="Ukerytme = 24*7=168")
    with col2:
        seasonal = st.number_input("Seasonal smooth", min_value=7, value=13)
    with col3:
        trend = st.number_input("Trend smooth", min_value=15, value=301)
    with col4:
        robust = st.checkbox("Robust", value=True)

    ts = pd.Series(df[target].values, index=pd.to_datetime(df["time"]))
    ts = ts.asfreq("H")  # jevn sampling
    ts = ts.interpolate(limit_direction="both")

    stl = STL(ts, period=int(period), seasonal=int(seasonal), trend=int(trend), robust=robust)
    res = stl.fit()

    fig = res.plot()
    fig.set_size_inches(12, 8)
    st.pyplot(fig, use_container_width=True)

# ----------------------- SPECTROGRAM TAB ----------------------
with tab_spec:
    st.subheader("Spektrogram")
    target2 = st.selectbox("Velg serie for spektrogram", [c for c in df.columns if c != "time"], index=0, key="spec_target")

    col1, col2 = st.columns(2)
    with col1:
        window_length = st.slider("Vindulengde (nperseg, timer)", min_value=32, max_value=1024, value=256, step=32)
    with col2:
        overlap = st.slider("Overlap (timer)", min_value=0, max_value=768, value=128, step=16)

    sig = pd.Series(df[target2].values, index=pd.to_datetime(df["time"])).asfreq("H")
    sig = sig.interpolate(limit_direction="both").values

    f, t, Sxx = spectrogram(sig, fs=1.0, nperseg=window_length, noverlap=overlap, scaling="density", mode="magnitude")
    Sxx_log = 10 * np.log10(Sxx + 1e-12)

    fig2, ax = plt.subplots(figsize=(12, 5))
    pcm = ax.pcolormesh(t, f, Sxx_log, shading="auto")
    ax.set_ylabel("Frekvens [sykluser/time]")
    ax.set_xlabel("Tid [timer fra start]")
    fig2.colorbar(pcm, ax=ax, label="dB")
    st.pyplot(fig2, use_container_width=True)
    st.caption("Tips: En tydelig ‘stripe’ ved ~1/24 indikerer døgnsyklus; ~1/168 indikerer ukerytme.")
