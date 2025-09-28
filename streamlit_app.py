import streamlit as st
from utils import load_data

st.set_page_config(page_title="Streamlit MWE", page_icon="🌤️", layout="wide")

st.title("Open-Meteo: Overview")
st.caption("Explore the `open-meteo-subset.csv` dataset. Use the menu on the left to view the data table with sparklines or an interactive plot.")

# Sidebar navigation
st.sidebar.header("Navigation")
st.sidebar.page_link("pages/2_Data_Table.py", label="📄 Data table")
st.sidebar.page_link("pages/3_Plot.py", label="📈 Plot")
st.sidebar.page_link("pages/4_About.py", label="ℹ️ About")

# Quick preview
df = load_data()
st.subheader("Quick look at the data")
st.dataframe(df.head(), use_container_width=True)
st.success(f"Loaded {len(df)} rows × {len(df.columns)} columns.")
