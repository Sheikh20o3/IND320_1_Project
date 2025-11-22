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
# 1. Paths (GeoJSON)
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
GEOJSON_PATH = PROJECT_ROOT / "file.geojson"

if not GEOJSON_PATH.exists():
    st.error(f"GeoJSON file not found at `{GEOJSON_PATH}`.")
    st.stop()

with GEOJSON_PATH.open("r", encoding="utf-8") as f:
    geojson_data = json.load(f)

# ------------------------------------------------------------
# 2. Hjelpefunksjoner for Mongo
# ------------------------------------------------------------
@st.cache_data(show_spinner=True)
def get_mongo_collections() -> list[str]:
    cli = get_client()
    db = cli["elhub"]
    return db.list_collection_names()


@st.cache_data(show_spinner=True)
def list_consumption_groups() -> list[str]:
    """Distinct consumptionGroup-verdier fra Mongo."""
    cli = get_client()
    db = cli["elhub"]
    collection_names = db.list_collection_names()

    if "consumption_2021_2024_by_hour" in collection_names:
        coll = db["consumption_2021_2024_by_hour"]
    else:
        candidates = [n for n in collection_names if n.startswith("consumption")]
        if not candidates:
            return []
        coll = db[candidates[0]]

    groups = coll.distinct("consumptionGroup")
    return sorted(g for g in groups if g)


# ------------------------------------------------------------
# 3. UI-kontroller
# ------------------------------------------------------------
try:
    areas_from_db = list_price_areas()
except Exception:
    areas_from_db = ["NO1", "NO2", "NO3", "NO4", "NO5"]

if not areas_from_db:
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
    if mode == "Production":
        try:
            base_groups = list_groups(price_area=default_area)
        except Exception:
            base_groups = []
    else:
        try:
            base_groups = list_consumption_groups()
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
# 4. Hent gjennomsnitt per prisområde fra MongoDB
# ------------------------------------------------------------
@st.cache_data(show_spinner=True)
def fetch_means(
    mode: str,
    group: str,
    start_d: dt.date,
    end_d: dt.date,
) -> pd.DataFrame:
    """
    Return mean quantity per price area from MongoDB.

    - Production: production_2021_2024_by_hour / production_2021_hourly_by_group
    - Consumption: consumption_2021_2024_by_hour
    """
    cli = get_client()
    db = cli["elhub"]
    collection_names = db.list_collection_names()

    if mode == "Production":
        group_field = "productionGroup"
        # Foretrukket rekkefølge
        if "production_2021_2024_by_hour" in collection_names:
            coll_name = "production_2021_2024_by_hour"
        elif "production_2021_hourly_by_group" in collection_names:
            coll_name = "production_2021_hourly_by_group"
        else:
            candidates = [n for n in collection_names if n.startswith("production")]
            if not candidates:
                return pd.DataFrame(columns=["priceArea", "meanValue"])
            coll_name = candidates[0]
    else:
        group_field = "consumptionGroup"
        if "consumption_2021_2024_by_hour" in collection_names:
            coll_name = "consumption_2021_2024_by_hour"
        else:
            candidates = [n for n in collection_names if n.startswith("consumption")]
            if not candidates:
                return pd.DataFrame(columns=["priceArea", "meanValue"])
            coll_name = candidates[0]

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


try:
    df_stats = fetch_means(mode, group, start_date, end_date)
except Exception as e:
    st.error(f"Could not fetch {mode.lower()} data from MongoDB: {e}")
    st.stop()

if df_stats.empty:
    st.warning(f"No {mode.lower()} data found for this selection.")
    st.stop()

value_map = df_stats.set_index("priceArea")["meanValue"].to_dict()

st.info(
    f"Computed mean **{mode.lower()}** for group **{group}** "
    f"from **{start_date}** to **{end_date}**."
)

# ------------------------------------------------------------
# 5. Kart + klikk-håndtering i ett
# ------------------------------------------------------------
center = [64.5, 11.0]
m = folium.Map(location=center, zoom_start=5, tiles="CartoDB positron")

def normalize_area(raw: str) -> str:
    return raw.strip().replace(" ", "") if isinstance(raw, str) else ""

# Les nåværende valg fra session_state (kan komme fra andre sider)
selected_area = st.session_state.get("price_area")
stored_coord = st.session_state.get("map_coord")

# Dynamisk fargestil (farge = meanValue, omriss = valgt område)
def style_function(feature):
    props = feature.get("properties", {})
    raw_code = props.get("ElSpotOmr", "")
    area_code = normalize_area(raw_code)

    # Outline (bare valgt område får rød kant)
    if area_code == selected_area:
        outline_color, weight = "red", 4
    else:
        outline_color, weight = "black", 1

    # Fyllfarge etter meanValue
    val = value_map.get(area_code)
    if val is None:
        fill_color, fill_opacity = "#cccccc", 0.2
    else:
        vmin = min(value_map.values())
        vmax = max(value_map.values())
        denom = (vmax - vmin) if vmax != vmin else 1.0
        norm = (val - vmin) / denom
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

# GeoJSON-lag (choropleth + hover)
folium.GeoJson(
    geojson_data,
    name="Elspot Areas",
    style_function=style_function,
    tooltip=folium.GeoJsonTooltip(
        fields=["ElSpotOmr"],
        aliases=["Elspot area:"],
    ),
).add_to(m)

# Hvis vi allerede har en koordinat, legg inn marker + vis den til brukeren
if stored_coord:
    lat, lon = stored_coord
    folium.Marker(
        [lat, lon],
        icon=folium.Icon(color="red", icon="map-marker"),
        popup=f"Selected coordinate\n({lat:.4f}, {lon:.4f})",
    ).add_to(m)

    st.info(
        f"📍 Stored coordinate from map: **({lat:.4f}, {lon:.4f})**\n\n"
        "These coordinates will be used on the Snow Drift page."
    )

    # Lett å kopiere rett inn i Snow Drift-siden
    st.code(f"{lat:.6f}, {lon:.6f}", language="text")

    if selected_area:
        st.write(f"🔹 Selected price area: **{selected_area}**")

else:
    st.caption("Click once on the map to select a coordinate and price area.")

# Vis kartet (ÉN gang)
map_data = st_folium(
    m,
    height=600,
    width="100%",
    key="main_map",
    returned_objects=["last_clicked"],
)

# ------------------------------------------------------------
# 6. Oppdater session_state ved nytt klikk (slipper dobbeltklikk)
# ------------------------------------------------------------
def find_price_area(lat, lon, geojson):
    """Enkel bounding box-sjekk for å finne prisområde for et punkt."""
    for feature in geojson["features"]:
        area_name = normalize_area(feature.get("properties", {}).get("ElSpotOmr", ""))
        geom = feature.get("geometry", {})
        gtype = geom.get("type")
        coords_root = geom.get("coordinates", [])

        # Normaliser til liste over ringer
        if gtype == "Polygon":
            rings = coords_root
        elif gtype == "MultiPolygon":
            rings = [ring for poly in coords_root for ring in poly]
        else:
            continue

        for ring in rings:
            lons = [c[0] for c in ring]
            lats = [c[1] for c in ring]
            # Enkel "inne i bounding box"-test
            if (min(lats) <= lat <= max(lats)) and (min(lons) <= lon <= max(lons)):
                return area_name

    return None


if map_data and map_data.get("last_clicked"):
    last_clicked = map_data["last_clicked"]
    lat = last_clicked["lat"]
    lon = last_clicked["lng"]

    old_coord = st.session_state.get("map_coord")

    # Bare gjør noe hvis koordinaten faktisk er ny (hindrer evig rerun-loop)
    if (not old_coord) or (abs(old_coord[0] - lat) > 1e-6 or abs(old_coord[1] - lon) > 1e-6):
        area_name = find_price_area(lat, lon, geojson_data)

        st.session_state["map_coord"] = (lat, lon)
        if area_name:
            st.session_state["price_area"] = area_name

        # Tving ny kjøring slik at kartet tegnes opp igjen med oppdatert outline + marker
        st.rerun()

# ------------------------------------------------------------
# 7. Info/debug
# ------------------------------------------------------------
with st.expander("Details and debug info"):
    st.write("Mean values per price area:")
    st.dataframe(df_stats, use_container_width=True)
    st.write(f"GeoJSON path: `{GEOJSON_PATH}`")
    st.write("Mongo collections in 'elhub':")
    try:
        st.write(get_mongo_collections())
    except Exception as e:
        st.write(f"(could not list collections: {e})")
    st.write(f"Stored coordinate: {st.session_state.get('map_coord')}")
    st.write(f"Selected price area: {st.session_state.get('price_area')}")
