# pages/5_Map_PriceAreas.py
import json
import datetime as dt
from pathlib import Path
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium

from utils_elhub import get_client, list_price_areas, list_groups

# ------------------------------------------------------------
# Sideoppsett
# ------------------------------------------------------------
st.set_page_config(page_title="Map – Price Areas", page_icon="🗺️", layout="wide")
st.title("Map and Energy Statistics – Norwegian Price Areas (NO1–NO5)")

# ------------------------------------------------------------
# 1. Last GeoJSON
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
GEOJSON_PATH = BASE_DIR / "file.geojson"

if not GEOJSON_PATH.exists():
    st.error(f"GeoJSON file not found at `{GEOJSON_PATH}`.")
    st.stop()

with GEOJSON_PATH.open("r", encoding="utf-8") as f:
    geojson_data = json.load(f)

# ------------------------------------------------------------
# 2. UI-kontroller
# ------------------------------------------------------------
try:
    areas_from_db = list_price_areas()
except Exception:
    areas_from_db = ["NO1", "NO2", "NO3", "NO4", "NO5"]

default_area = areas_from_db[0]

col1, col2, col3 = st.columns([1, 1, 1.5])

with col1:
    mode = st.radio(
        "Dataset",
        ["Production", "Consumption"],
        horizontal=True,
    )

with col2:
    try:
        base_groups = list_groups(price_area=default_area)
    except Exception:
        base_groups = []
    group_options = ["All groups"] + base_groups
    group = st.selectbox("Energy group", group_options, index=0)

with col3:
    start_date = st.date_input("Start date", dt.date(2021, 1, 1))
    end_date = st.date_input("End date", dt.date(2021, 1, 15))

if start_date > end_date:
    st.error("Start date must be before end date.")
    st.stop()

st.caption(
    f"The mean value is computed over the selected date interval for the chosen {mode.lower()} group."
)

# ------------------------------------------------------------
# 3. Hent gjennomsnitt per prisområde fra MongoDB
# ------------------------------------------------------------
@st.cache_data(show_spinner=True)
def fetch_means(mode: str, group: str, start_d: dt.date, end_d: dt.date) -> pd.DataFrame:
    """Return mean quantity per price area."""
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

    match_filter = {"startTime": {"$gte": start_dt, "$lt": end_dt_excl}}
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
        {"$project": {"_id": 0, "priceArea": "$_id", "meanValue": 1}},
    ]

    docs = list(coll.aggregate(pipeline))
    return pd.DataFrame(docs) if docs else pd.DataFrame(columns=["priceArea", "meanValue"])


df_stats = fetch_means(mode, group, start_date, end_date)

if df_stats.empty:
    st.warning("No data found for this selection.")
    st.stop()

value_map = df_stats.set_index("priceArea")["meanValue"].to_dict()

st.info(
    f"Computed mean **{mode.lower()}** for group **{group}** "
    f"from **{start_date}** to **{end_date}**."
)

# ------------------------------------------------------------
# 4. Bygg kart
# ------------------------------------------------------------
center = [64.5, 11.0]
m = folium.Map(location=center, zoom_start=5, tiles="CartoDB positron")

selected_area = st.session_state.get("price_area")  # valgt globalt på andre sider
clicked_coord = st.session_state.get("map_coord")

def normalize_area(raw: str) -> str:
    return raw.strip().replace(" ", "") if isinstance(raw, str) else ""

# Dynamisk fargestil
def style_function(feature):
    props = feature.get("properties", {})
    raw_code = props.get("ElSpotOmr", "")
    area_code = normalize_area(raw_code)

    # Outline
    if area_code == selected_area:
        outline_color, weight = "red", 4
    else:
        outline_color, weight = "black", 1

    # Fyllfarge etter meanValue
    val = value_map.get(area_code)
    if val is None:
        fill_color, fill_opacity = "#cccccc", 0.2
    else:
        vmin, vmax = min(value_map.values()), max(value_map.values())
        norm = (val - vmin) / (vmax - vmin + 1e-9)
        r = int(255 * norm)
        g = 0
        b = int(255 * (1 - norm))
        fill_color, fill_opacity = f"rgba({r},{g},{b},0.45)", 0.5

    return {
        "fillColor": fill_color,
        "color": outline_color,
        "weight": weight,
        "fillOpacity": fill_opacity,
    }

# Tegn hovedkart
folium.GeoJson(
    geojson_data,
    name="Elspot Areas",
    style_function=style_function,
    tooltip=folium.GeoJsonTooltip(
        fields=["ElSpotOmr"],
        aliases=["Elspot area:"],
    ),
).add_to(m)

# ------------------------------------------------------------
# 5. Klikk-håndtering (én klikk nok)
# ------------------------------------------------------------
click_data = st_folium(m, height=600, width="100%", returned_objects=["last_clicked"])

if click_data and click_data.get("last_clicked"):
    lat = click_data["last_clicked"]["lat"]
    lon = click_data["last_clicked"]["lng"]
    st.session_state["map_coord"] = (lat, lon)
    st.success(f"📍 Coordinate selected: ({lat:.4f}, {lon:.4f})")

    # Marker posisjonen på kartet
    folium.Marker(
        [lat, lon],
        icon=folium.Icon(color="red", icon="map-marker"),
        popup=f"Selected coordinate\n({lat:.4f}, {lon:.4f})",
    ).add_to(m)

    # Oppdater valgt område basert på nærmeste polygon (enkel sjekk)
    for feature in geojson_data["features"]:
        area_name = normalize_area(feature["properties"].get("ElSpotOmr", ""))
        geom = feature["geometry"]
        if geom["type"] == "Polygon":
            for coords in geom["coordinates"]:
                poly = folium.vector_layers.Polygon(locations=[(y, x) for x, y in coords])
                if any(abs(lat - y) < 1 and abs(lon - x) < 1 for x, y in coords):
                    st.session_state["price_area"] = area_name
                    selected_area = area_name
                    break

    st.write(f"🔹 Selected price area: **{selected_area}**")

    # Tegn kartet på nytt med oppdatert omriss
    st_folium(m, height=600, width="100%", key="updated_map")

else:
    st.caption("Click once on the map to select a coordinate and price area.")

# ------------------------------------------------------------
# 6. Info/debug
# ------------------------------------------------------------
with st.expander("Details and debug info"):
    st.write("Mean values per price area:")
    st.dataframe(df_stats, use_container_width=True)
    st.write(f"GeoJSON path: `{GEOJSON_PATH}`")
    st.write(f"Stored coordinate: {st.session_state.get('map_coord')}")
    st.write(f"Selected price area: {st.session_state.get('price_area')}")
