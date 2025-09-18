import streamlit as st
st.set_page_config(page_title="About", page_icon="ℹ️")
st.title("About")
st.markdown(
"""
Denne multi-side appen viser:
- **Caching** av CSV med `st.cache_data`
- **Tabell** med `LineChartColumn` for første måned
- **Plott** med kolonnevalg + månedssubsett
"""
)
