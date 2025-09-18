import streamlit as st
from utils import load_data

st.set_page_config(page_title="Streamlit MWE", page_icon="🌤️", layout="wide")

st.title("Streamlit Minimum Working Example")
st.write("Denne appen leser filen 'open-meteo-subset.csv', viser en tabell med minigraf og et plott.")

# Sidebar-navigasjon til andre sider
st.sidebar.header("Navigasjon")
st.sidebar.page_link("pages/2_Data_Table.py", label="📄 Data table")
st.sidebar.page_link("pages/3_Plot.py", label="📈 Plot")
st.sidebar.page_link("pages/4_About.py", label="ℹ️ About")

# Vis litt data
df = load_data()
st.success(f"Lastet {len(df)} rader × {len(df.columns)} kolonner.")
st.dataframe(df.head(), use_container_width=True)
