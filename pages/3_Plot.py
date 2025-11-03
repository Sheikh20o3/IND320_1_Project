# pages/3_Plot.py
import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.express as px

st.set_page_config(page_title="Data plot (Plotly)", page_icon="📈", layout="wide")
st.title("Data plot (Plotly)")

def _find_file(name: str) -> Path | None:
    """Søk etter en fil i repo-rot, i pages/-mappen og i nåværende katalog."""
    here = Path(__file__).resolve()
    candidates = [
        here.parents[1] / name,   # repo-rot
        here.parent / name,       # pages/
        Path.cwd() / name,        # current working dir (Streamlit Cloud bruker ofte /mount/src/<repo>)
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    # 1) Prøv å finne en committed fil i repoet
    target = _find_file("open-meteo-subset.csv")
    if target is not None:
        return pd.read_csv(target)

    # 2) Fallback: la brukeren laste opp fila ved kjøring
    uploaded = st.file_uploader(
        "Fant ikke 'open-meteo-subset.csv'. Last opp CSV her, eller legg den i repoet.",
        type=["csv"]
    )
    if uploaded is not None:
        return pd.read_csv(uploaded)

    # 3) Ingen fil tilgjengelig → stopp pent
    st.warning(
        "Jeg finner ikke 'open-meteo-subset.csv'. "
        "Legg fila i repoets rotmappe **(ikke en lokal /Users/... sti)** og push til Git, "
        "eller last den opp via knappen over."
    )
    st.stop()

# ---- Les data
df = load_data()

# Valgfri kvittering
st.success(f"Lastet {len(df):,} rader.")

# En enkel interaktiv plott-UI (tilpass etter dine kolonner)
num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
if not num_cols:
    st.info("Fant ingen numeriske kolonner å plotte.")
else:
    y = st.selectbox("Velg Y-kolonne", num_cols, index=0)
    x_candidates = [c for c in df.columns if c != y]
    x = st.selectbox("Velg X-kolonne", x_candidates, index=0 if x_candidates else None)
    fig = px.line(df, x=x, y=y, title=f"{y} vs {x}")
    st.plotly_chart(fig, use_container_width=True)
