# streamlit_app.py
import streamlit as st
from utils import get_selected_price_area
from utils import load_data

st.set_page_config(page_title="Energy & Weather", page_icon="🌤️", layout="wide")

st.title("Open-Meteo: Overview")
st.caption(
    "The app uses the **Open-Meteo API** (not CSV). Use the menu on the left. "
    "Order: 1 (this one), 4, new A, 2, 3, new B, 5."
)

# Shows selected price area (chosen on the "Price area (Elhub)" page)
area = get_selected_price_area()
st.info(f"Selected price area: {area}")

# Sidebar navigation in desired order
st.sidebar.header("Navigation")
st.sidebar.page_link("pages/4_Data_Table.py",         label="📄 Data Table")
st.sidebar.page_link("pages/3_Meteorology.py",        label="🧪 new A — STL & Spectrogram")
st.sidebar.page_link("pages/2_Elhub_Production.py",   label="⚡ Price area (Elhub)")
st.sidebar.page_link("pages/5_Plot.py",               label="📈 Plot")
st.sidebar.page_link("pages/6_Analysis.py",           label="🧭 new B — Outlier & Anomaly")
st.sidebar.page_link("pages/7_About.py",              label="ℹ️ About")

# Quick preview
df = load_data()
st.subheader("Quick look at the data")
st.dataframe(df.head(), use_container_width=True)
