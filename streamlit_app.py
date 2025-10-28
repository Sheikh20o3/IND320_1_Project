# streamlit_app.py
import streamlit as st
from utils import get_selected_price_area

st.set_page_config(page_title="Energy & Weather", page_icon="🌤️", layout="wide")

st.title("Open-Meteo: Oversikt")
st.caption(
    "Appen bruker **Open-Meteo API** (ikke CSV). Bruk menyen til venstre. "
    "Rekkefølge: 1 (denne), 4, new A, 2, 3, new B, 5."
)

# Viser valgt prisområde (velges på siden «Prisområde (Elhub)»)
area = get_selected_price_area()
st.info(f"Valgt prisområde: {area}")

# Sidebar-navigasjon i ønsket rekkefølge
st.sidebar.header("Navigasjon")
st.sidebar.page_link("pages/4_Data_Table.py",         label="📄 Data Table")
st.sidebar.page_link("pages/3_Meteorology.py",        label="🧪 new A — STL & Spectrogram")
st.sidebar.page_link("pages/2_Elhub_Production.py",   label="⚡ Prisområde (Elhub)")
st.sidebar.page_link("pages/5_Plot.py",               label="📈 Plot")
st.sidebar.page_link("pages/6_Analysis.py",           label="🧭 new B — Outlier & Anomaly")
st.sidebar.page_link("pages/7_About.py",              label="ℹ️ About")
