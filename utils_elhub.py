# utils_elhub.py
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Iterable, List, Optional

import pandas as pd
import pymongo
import streamlit as st


#  Mongo connection 
def _get_uri() -> str:
    """
    Hent MongoDB URI fra Streamlit secrets eller miljøvariabler.
    Bygger automatisk URI fra bruker/pass/host hvis full URI mangler.
    """
    import os
    import urllib.parse
    import streamlit as st

    # 1️⃣ Sjekk om full URI finnes
    if hasattr(st, "secrets"):
        s = st.secrets
        if "MONGODB_URI" in s:
            return s["MONGODB_URI"]

        # 2️⃣ Bygg URI manuelt fra bruker/pass/host
        user = s.get("MONGODB_USER", "")
        pw = urllib.parse.quote_plus(s.get("MONGODB_PASSWORD", ""))
        host = s.get("MONGODB_HOST", "ahs786student.qh8rsrb.mongodb.net")
        app = s.get("MONGODB_APPNAME", "AHS786Student")
        if user and pw:
            return (
                f"mongodb+srv://{user}:{pw}@{host}/elhub?"
                f"retryWrites=true&w=majority&tls=true&appName={app}"
            )

    # 3️⃣ Fallback til miljøvariabel
    uri = os.getenv("MONGODB_URI")
    if uri:
        return uri

    # 4️⃣ Hvis alt feiler
    raise RuntimeError("Fant ingen MongoDB-kredentialer i secrets eller miljøvariabler.")





@st.cache_resource(show_spinner=False)
def get_client() -> pymongo.MongoClient:
    """
    Create a global MongoClient with proper TLS settings.
    Also validates the connection with `ping()`.
    """
    uri = _get_uri()
    client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=30_000)
    # Light sanity check — raises if the connection is invalid/misconfigured
    client.admin.command("ping")
    return client


def _coll(db: str = "elhub", coll: str = "production_2021_by_hour") -> pymongo.collection.Collection:
    # Convenience accessor: return a handle to the desired collection
    return get_client()[db][coll]


#  List choices for the UI 
@st.cache_data(ttl=300, show_spinner=False)
def list_price_areas(db_name: str = "elhub", coll_name: str = "production_2021_by_hour") -> List[str]:
    """
    Return a distinct, sorted list of `priceArea` values.
    Cached for 300s to reduce round-trips to MongoDB.
    """
    c = _coll(db_name, coll_name)
    res = c.aggregate([
        {"$group": {"_id": "$priceArea"}},
        {"$project": {"_id": 0, "priceArea": "$_id"}},
        {"$sort": {"priceArea": 1}},
    ])
    return [d["priceArea"] for d in res]


@st.cache_data(ttl=300, show_spinner=False)
def list_groups(price_area: Optional[str] = None,
                db_name: str = "elhub", coll_name: str = "production_2021_by_hour") -> List[str]:
    """
    Return a distinct, sorted list of `productionGroup` values,
    optionally filtered by a given `price_area`.
    """
    c = _coll(db_name, coll_name)
    pipeline = []
    if price_area:
        pipeline.append({"$match": {"priceArea": price_area}})
    pipeline += [
        {"$group": {"_id": "$productionGroup"}},
        {"$project": {"_id": 0, "productionGroup": "$_id"}},
        {"$sort": {"productionGroup": 1}},
    ]
    res = c.aggregate(pipeline)
    return [d["productionGroup"] for d in res]


#  Data for charts 
@st.cache_data(ttl=300, show_spinner=False)
def fetch_pie_df(price_area: str,
                 db_name: str = "elhub", coll_name: str = "production_2021_by_hour") -> pd.DataFrame:
    """
    Sum total production per group in 2021 for the selected price area.
    Returns a DataFrame with columns: ['productionGroup', 'quantityKwh'].
    """
    if not price_area:
        return pd.DataFrame(columns=["productionGroup", "quantityKwh"])

    c = _coll(db_name, coll_name)
    pipeline = [
        {"$match": {"priceArea": price_area}},
        {"$group": {
            "_id": "$productionGroup",
            "quantityKwh": {"$sum": "$quantityKwh"},
        }},
        {"$project": {
            "_id": 0,
            "productionGroup": "$_id",
            "quantityKwh": 1,
        }},
        {"$sort": {"quantityKwh": -1}},
    ]
    rows = list(c.aggregate(pipeline))
    return pd.DataFrame(rows, columns=["productionGroup", "quantityKwh"])


def _month_bounds(year: int, month: int) -> tuple[datetime, datetime]:
    """
    Build [start, end) UTC bounds for a given month.
    Note: raises ValueError on invalid month; message kept in Norwegian to avoid changing user-facing text.
    """
    if month < 1 or month > 12:
        raise ValueError("month må være 1..12")
    start = datetime(year, month, 1, 0, 0, 0, tzinfo=timezone.utc)
    if month == 12:
        end = datetime(year + 1, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    else:
        end = datetime(year, month + 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    return start, end


@st.cache_data(ttl=300, show_spinner=False)
def fetch_line_df(price_area: str,
                  groups: Optional[Iterable[str]],
                  month: int,
                  year: int = 2021,
                  db_name: str = "elhub",
                  coll_name: str = "production_2021_by_hour") -> pd.DataFrame:
    """
    Fetch a time series for the selected month/price area and (optionally) selected groups.
    Returns a DataFrame with columns:
      - startTime (datetime64[ns, UTC])
      - productionGroup
      - quantityKwh
    The rows are sorted ascending by time.
    """
    if not price_area:
        return pd.DataFrame(columns=["startTime", "productionGroup", "quantityKwh"])

    start, end = _month_bounds(year, month)
    match = {"priceArea": price_area, "startTime": {"$gte": start, "$lt": end}}
    groups = list(groups or [])
    if groups:
        match["productionGroup"] = {"$in": groups}

    c = _coll(db_name, coll_name)
    pipeline = [
        {"$match": match},
        {"$project": {
            "_id": 0,
            "priceArea": 1,
            "productionGroup": 1,
            "startTime": 1,
            "quantityKwh": 1,
        }},
        {"$sort": {"startTime": 1, "productionGroup": 1}},
    ]
    rows = list(c.aggregate(pipeline))
    if not rows:
        return pd.DataFrame(columns=["startTime", "productionGroup", "quantityKwh"])

    df = pd.DataFrame(rows)
    # Ensure tz-aware datetimes in pandas (required for consistent plotting/merging)
    df["startTime"] = pd.to_datetime(df["startTime"], utc=True)
    # We only need these columns downstream for charts
    return df[["startTime", "productionGroup", "quantityKwh"]]


def uri_preview() -> str:
    try:
        s = getattr(__import__("streamlit"), "secrets", {})
        if "MONGODB_USER" in s:
            user = str(s.get("MONGODB_USER","")).strip()
            host = str(s.get("MONGODB_HOST","ahs786student.qh8rsrb.mongodb.net")).strip()
            app  = str(s.get("MONGODB_APPNAME","AHS786Student")).strip()
            return f"user={user}, host={host}, appName={app}, authSource=admin"
        elif "MONGODB_URI" in s:
            uri = str(s["MONGODB_URI"]).strip()
            safe = uri.split("@")[-1]
            return f"uri=***:***@{safe}"
    except Exception:
        pass
    return "Ingen MongoDB-secrets funnet."
