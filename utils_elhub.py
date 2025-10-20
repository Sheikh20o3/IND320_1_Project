# utils_elhub.py
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Iterable, List, Optional

import pandas as pd
import pymongo
import streamlit as st


# ---- Mongo tilkobling --------------------------------------------------------

def _get_uri() -> str:
    """
    Henter MongoDB-URI fra Streamlit secrets eller miljøvariabel.
    """
    if hasattr(st, "secrets") and "MONGODB_URI" in st.secrets:
        return st.secrets["MONGODB_URI"]
    uri = os.getenv("MONGODB_URI")
    if not uri:
        raise RuntimeError(
            "MONGODB_URI mangler. Legg den i .streamlit/secrets.toml eller miljøvariabel."
        )
    return uri


@st.cache_resource(show_spinner=False)
def get_client() -> pymongo.MongoClient:
    """
    Oppretter en global MongoClient med riktig TLS-oppsett.
    Validerer også tilkoblingen med ping().
    """
    uri = _get_uri()
    client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=30_000)
    # Liten sanity check – kaster exception hvis noe er feil
    client.admin.command("ping")
    return client


def _coll(db: str = "elhub", coll: str = "production_2021_by_hour") -> pymongo.collection.Collection:
    return get_client()[db][coll]


# ---- Listevalg til UI --------------------------------------------------------

@st.cache_data(ttl=300, show_spinner=False)
def list_price_areas(db_name: str = "elhub", coll_name: str = "production_2021_by_hour") -> List[str]:
    """
    Distinct liste over priceArea, sortert.
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
    Distinct liste over productionGroup, globalt eller filtrert på price_area.
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


# ---- Data til figurer --------------------------------------------------------

@st.cache_data(ttl=300, show_spinner=False)
def fetch_pie_df(price_area: str,
                 db_name: str = "elhub", coll_name: str = "production_2021_by_hour") -> pd.DataFrame:
    """
    Summerer total produksjon per gruppe i 2021 for valgt prisområde.
    Returnerer DataFrame: columns = ['productionGroup', 'quantityKwh'].
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
    Lager [start, end) grenser i UTC for en måned.
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
    Henter time-serie for valgt måned/prisområde og (valgte) grupper.
    Returnerer DataFrame med kolonner: startTime (datetime64[ns, UTC]),
    productionGroup, quantityKwh – sortert på tid stigende.
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
    # Sørg for tz-aware i pandas
    df["startTime"] = pd.to_datetime(df["startTime"], utc=True)
    # Vi trenger bare disse kolonnene videre i plott
    return df[["startTime", "productionGroup", "quantityKwh"]]
