import json
import datetime as dt
from pathlib import Path

import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium

from utils_elhub import (
    get_client,
    list_price_areas,
    list_groups,
)

st.set_page_config(page_title="Map – Price Areas", page_icon="🗺️", layout="wide")
st.title("Map and Energy Statistics – Norwegian Price Areas (NO1–NO5)")

# -------------------------------------------------------------
# 1. Finn root og last GeoJSON (NVE Elspot / ElSpot_omraade)
# -------------------------------------------------------------
# Root = repo-mappen (en over pages/)
ROOT_DIR = Path(__file__).resolve().parent.parent
GEOJSON_PATH = ROOT_DIR / "/Users/a.h.sheikh/Desktop/IND320_Git_Job/IND320_1_Project/file.geojson"

if not GEOJSON_PATH.exists():
    st.error(
        f"GeoJSON-fil ble ikke funnet.\n\n"
        f"Forventet sti: `{GEOJSON_PATH}`\n\n"
        "Sjekk at `file.geojson` ligger i rotmappen til repoet (samme nivå som `streamlit_app.py`) "
        "og at den er commit’et og pushet til samme branch som appen kjører på."
    )
    st.stop()

with GEOJSON_PATH.open("r", encoding="utf-8") as f:
    geojson = json.load(f)

# -------------------------------------------------------------
# 2. Kontroller: dataset, gruppe, tidsintervall
# -------------------------------------------------------------
areas_from_db = list_price_areas() or ["NO1", "NO2", "NO3", "NO4", "NO5"]
default_area = st.session_state.get("price_area", areas_from_db[0])

col_top1, col_top2, col_top3 = st.columns([1, 1, 1.5])

with col_top1:
    mode = st.radio(
        "Dataset",
        ["Production", "Consumption"],
        horizontal=True,
        help="Choose whether to visualise production or consumption.",
    )

with col_top2:
    try:
        base_groups = list_groups(price_area=default_area)
    except Exception:
        base_groups = []

    group_options = ["All groups"] + base_groups
    group = st.selectbox("Energy group", group_options, index=0)

with col_top3:
    start_date = st.date_input("Start date", value=dt.date(2021, 1, 1))
    end_date = st.date_input("End date", value=dt.date(2021, 1, 15))

if start_date > end_date:
    st.error("Start date must be before or equal to end date.")
    st.stop()

st.caption(
    "The mean value is computed over the selected date interval for the chosen "
    "production/consumption group in each price area."
)

# -------------------------------------------------------------
# 3. Hent gjennomsnitt per prisområde fra MongoDB
# -------------------------------------------------------------
@st.cache_data(show_spinner=True)
def fetch_means(mode: str, group: str, start_d: dt.date, end_d: dt.date) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
        priceArea, meanValue
    mode: 'Production' or 'Consumption'
    group: group name from DB or 'All groups'
    """
    cli = get_client()
    db = cli["elhub"]

    if mode == "Production":
        coll_name = "production_2021_by_hour"
        group_field = "productionGroup"
    else:
        coll_name = "consumption_2021_by_hour"
        group_field = "consumptionGroup"

    coll = db[coll_name]

    start_dt = dt.datetime.combine(start_d, dt.time.min)
    end_dt_excl = dt.datetime.combine(end_d + dt.timedelta(days=1), dt.time.min)

    match_filter = {
        "startTime": {"$gte": start_dt, "$lt": end_dt_excl},
    }
    if group != "All groups":
        match_filter[group_field] = group

    pipeline = [
        {"$match": match_filter},
        {
            "$group": {
                "_id": "$priceArea",
                "meanValue": {"$avg": "$quantityKwh"},
            }
        },
        {
            "$project": {
                "_id": 0,
                "priceArea": "$_id",
                "meanValue": 1,
            }
        },
    ]

    docs = list(coll.aggregate(pipeline))
    if not docs:
        return pd.DataFrame(columns=["priceArea", "meanValue"])
    return pd.DataFrame(docs)


df_stats = fetch_means(mode, group, start_date, end_date)

if df_stats.empty:
    st.warning(
        "No data returned for this combination of dataset, group and date interval. "
        "Check that your MongoDB collections contain data for the selected years."
    )
    st.stop()

value_map = df_stats.set_index("priceArea")["meanValue"].to_dict()

st.info(
    f"Computed mean **{mode.lower()}** for group **{group}** "
    f"from **{start_date}** to **{end_date}** in each price area."
)

# -------------------------------------------------------------
# 4. Bygg Folium-kart
# -------------------------------------------------------------
center = [64.5, 11.0]
m = folium.Map(location=center, zoom_start=4.7, tiles="CartoDB positron")

selected_area = st.session_state.get("price_area", default_area)


def normalize_area_code(raw: str) -> str:
    """
    GeoJSON bruker gjerne 'NO 1', 'NO 2', ... → vi vil ha 'NO1', 'NO2', ...
    """
    if not isinstance(raw, str):
        return ""
    raw = raw.strip()
    return raw.replace(" ", "")


def style_function(feature):
    props = feature.get("properties", {})
    raw_code = props.get("ElSpotOmr", "")
    area_code = normalize_area_code(raw_code)

    if area_code == selected_area:
        outline_color = "red"
        weight = 4
    else:
        outline_color = "black"
        weight = 1

    val = value_map.get(area_code)
    if val is None:
        fill_color = "#cccccc"
        fill_opacity = 0.2
    else:
        vmax = max(value_map.values())
        vmin = min(value_map.values())
        norm = (val - vmin) / (vmax - vmin + 1e-9)

        r = int(255 * norm)
        g = 0
        b = int(255 * (1 - norm))
        fill_color = f"rgba({r},{g},{b},0.45)"
        fill_opacity = 0.45

    return {
        "fillColor": fill_color,
        "color": outline_color,
        "weight": weight,
        "fillOpacity": fill_opacity,
    }


folium.GeoJson(
    geojson,
    name="Elspot Areas",
    style_function=style_function,
    tooltip=folium.GeoJsonTooltip(
        fields=["ElSpotOmr"],
        aliases=["Elspot area:"],
        localize=True,
    ),
).add_to(m)

current_coord = st.session_state.get("map_coord")
if current_coord is not None:
    folium.Marker(
        location=[current_coord[0], current_coord[1]],
        popup=f"Selected coordinate\n({current_coord[0]:.4f}, {current_coord[1]:.4f})",
        icon=folium.Icon(color="red", icon="map-marker"),
    ).add_to(m)

# -------------------------------------------------------------
# 5. Klikk-håndtering og lagring av koordinat
# -------------------------------------------------------------
result = st_folium(m, height=600, width="100%", returned_objects=["last_clicked"])

last_clicked = result.get("last_clicked") if result else None
if last_clicked is not None:
    lat = last_clicked["lat"]
    lon = last_clicked["lng"]
    st.session_state["map_coord"] = (lat, lon)
    st.success(f"Clicked coordinate stored: **({lat:.4f}, {lon:.4f})**")
else:
    st.caption("Click anywhere on the map to store a coordinate for use on other pages (e.g. snow drift).")

# -------------------------------------------------------------
# 6. Ekstra info
# -------------------------------------------------------------
with st.expander("Details and debug info"):
    st.markdown(
        """
        - **GeoJSON source:** NVE Elspot areas (`ElSpot_omraade` / `ElSpotOmr`).
        - **Outline:** The price area stored in `st.session_state["price_area"]`
          is highlighted with a thicker red border.
        - **Fill color:** Transparent choropleth based on mean `quantityKwh`
          for the selected production/consumption group over the chosen date interval.
        - **Clicked coordinate:** Stored in `st.session_state["map_coord"]`
          and used by other pages (e.g. snow drift).
        """
    )
    st.write("Raw statistics per price area:")
    st.dataframe(df_stats, use_container_width=True)
    st.write("GeoJSON path in this environment:")
    st.code(str(GEOJSON_PATH))
