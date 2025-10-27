import streamlit as st
from utils import load_data
from utils import get_selected_price_area
import pathlib as _pl, streamlit as _st

st.set_page_config(page_title="Streamlit MWE", page_icon="🌤️", layout="wide")


# NÅ DET CHAT SA
area = get_selected_price_area()
from utils import download_open_meteo
df_met = download_open_meteo(price_area=area, start_date="2021-01-01", end_date="2021-12-31")


st.title("Open-Meteo: Overview")
st.caption("Explore the `open-meteo-subset.csv` dataset. Use the menu on the left to view the data table with sparklines or an interactive plot.")

# Sidebar navigation
st.sidebar.header("Navigation")
st.sidebar.page_link("pages/2_Data_Table.py", label="📄 Data table")
st.sidebar.page_link("pages/3_Plot.py",        label="📈 Plot")
st.sidebar.page_link("pages/4_Elhub_Production.py", label="⚡ Elhub Production")
st.sidebar.page_link("pages/5_About.py",       label="ℹ️ About")


# Quick preview
df = load_data()
st.subheader("Quick look at the data")
st.dataframe(df.head(), use_container_width=True)
st.success(f"Loaded {len(df)} rows × {len(df.columns)} columns.")


_st.sidebar.header("Navigation")
for fname, label in [("2_Data_Table.py","📄 Data table"),
                     ("3_Plot.py","📈 Plot"),
                     ("4_Elhub_Production.py","⚡ Elhub Production (MongoDB)"),
                     ("5_About.py","ℹ️ About (moved)")]:
    if (_pl.Path(__file__).parent / fname).exists():
        _st.sidebar.page_link(fname, label=label)
