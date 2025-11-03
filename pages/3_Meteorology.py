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

# NB: New A skal ikke vise rådatatabeller, derfor ingen st.dataframe her.

tab_stl, tab_spec = st.tabs(["STL", "Spectrogram"])

# -------------------------- STL TAB --------------------------
with tab_stl:
    st.subheader("STL-dekomponering")
    target = st.selectbox("Velg serie for STL", [c for c in df.columns if c != "time"], index=0)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        period = st.number_input(
            "Period (timer)",
            min_value=2, value=168, step=1,
            help="Ukerytme = 24*7=168"
        )
    with col2:
        seasonal = st.number_input(
            "Seasonal (LOESS-vindu, oddetall)",
            min_value=7, value=13, step=2,
            help="Må være oddetall og minst 7"
        )
    with col3:
        trend = st.number_input(
            "Trend (LOESS-vindu, oddetall)",
            min_value=3, value=301, step=2,
            help="Må være oddetall og minst 3"
        )
    with col4:
        robust = st.checkbox("Robust", value=True)

    # --- Bygg timeserie (timelig) uten NaN ---
    ts = pd.Series(pd.to_numeric(df[target], errors="coerce").values, index=pd.to_datetime(df["time"]))
    ts = ts.asfreq("H").interpolate(limit_direction="both")
    nobs = int(ts.size)

    # --- Hjelpere for sikre STL-parametre ---
    def _to_odd(x: int) -> int:
        x = int(x)
        return x if x % 2 == 1 else x + 1

    def _clamp(v: int, lo: int, hi: int) -> int:
        return int(max(lo, min(hi, v)))

    # Minstekrav
    period = int(max(2, int(period)))
    seasonal = _to_odd(max(7, int(seasonal)))
    trend = _to_odd(max(3, int(trend)))

    # STL liker ikke for korte serier relativt til perioder/vinduer
    # 1) sørg for at vi har minst 2*period observasjoner (ellers senk period)
    if nobs < 2 * period:
        period = max(2, nobs // 2)

    # 2) vinduer kan ikke være >= nobs
    seasonal = _clamp(seasonal, 7, max(7, nobs - 1))
    seasonal = _to_odd(seasonal)
    trend = _clamp(trend, 3, max(3, nobs - 1))
    trend = _to_odd(trend)

    # 3) trend bør være > period (ellers øk)
    if trend <= period:
        trend = _to_odd(period + 1)

    # 4) ekstra sikkerhet: hvis fortsatt svært langt vindu, skaler ned proporsjonalt
    #    (f.eks. korte utvalg ved testing)
    if seasonal >= nobs:
        seasonal = _to_odd(max(7, nobs // 5 * 2 + 1))
    if trend >= nobs:
        trend = _to_odd(max(3, nobs // 3 * 2 + 1))
    if trend <= period:
        trend = _to_odd(period + 1)

    # 5) endelig fallback hvis init fortsatt feiler
    try:
        stl = STL(ts, period=int(period), seasonal=int(seasonal), trend=int(trend), robust=robust)
    except Exception:
        # velg mer konservative vinduer basert på datasettets lengde
        seasonal = _to_odd(_clamp(max(7, period + 5), 7, max(7, nobs - 3)))
        trend = _to_odd(_clamp(max(3, period * 3 + 1), 3, max(3, nobs - 3)))
        st.info("Parametre for STL ble automatisk justert for å unngå feil.")
        stl = STL(ts, period=int(period), seasonal=int(seasonal), trend=int(trend), robust=robust)

    res = stl.fit()
    fig = res.plot()
    fig.set_size_inches(12, 8)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

# ----------------------- SPECTROGRAM TAB ----------------------
with tab_spec:
    st.subheader("Spektrogram")
    target2 = st.selectbox(
        "Velg serie for spektrogram",
        [c for c in df.columns if c != "time"],
        index=0,
        key="spec_target"
    )

    # Velg vindulengde først, så sett overlap basert på vindu (overlap < nperseg)
    window_length = st.slider("Vindulengde (nperseg, timer)", min_value=32, max_value=1024, value=256, step=32)
    max_overlap = max(0, window_length - 2)
    overlap = st.slider("Overlap (timer)", min_value=0, max_value=max_overlap, value=min(128, max_overlap), step=16)

    sig = pd.Series(pd.to_numeric(df[target2], errors="coerce").values, index=pd.to_datetime(df["time"])).asfreq("H")
    sig = sig.interpolate(limit_direction="both").values

    f, t, Sxx = spectrogram(
        sig, fs=1.0, nperseg=int(window_length), noverlap=int(overlap),
        scaling="density", mode="magnitude"
    )
    Sxx_log = 10 * np.log10(Sxx + 1e-12)

    fig2, ax = plt.subplots(figsize=(12, 5))
    pcm = ax.pcolormesh(t, f, Sxx_log, shading="auto")
    ax.set_ylabel("Frekvens [sykluser/time]")
    ax.set_xlabel("Tid [timer fra start]")
    fig2.colorbar(pcm, ax=ax, label="dB")
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)

    st.caption("Tips: En tydelig ‘stripe’ ved ~1/24 indikerer døgnsyklus; ~1/168 indikerer ukerytme.")
