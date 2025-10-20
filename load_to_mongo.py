# load_to_mongo.py
import os
from datetime import datetime
from pyspark.sql import SparkSession, functions as F

# --- 1) Hent Mongo-URI ---
# Prøv miljøvariabel først, deretter .streamlit/secrets.toml, til slutt “hard fail”.
MONGO_URI = os.environ.get("MONGODB_URI")

if not MONGO_URI:
    # Les .streamlit/secrets.toml hvis finnes
    secrets_path = os.path.join(".streamlit", "secrets.toml")
    if os.path.isfile(secrets_path):
        try:
            import toml
            MONGO_URI = toml.load(secrets_path).get("MONGODB_URI")
        except Exception as e:
            raise RuntimeError(f"Klarte ikke å lese {secrets_path}: {e}")

if not MONGO_URI:
    raise RuntimeError("Fant ikke MONGODB_URI (verken i env eller .streamlit/secrets.toml).")

# Sikker: sørg for TLS og OCSP-workaround i tilfelle campusnett blokkerer OCSP
if "tls=" not in MONGO_URI:
    joiner = "&" if "?" in MONGO_URI else "?"
    MONGO_URI = f"{MONGO_URI}{joiner}tls=true"
if "tlsDisableOCSPEndpointCheck=" not in MONGO_URI:
    joiner = "&" if "?" in MONGO_URI else "?"
    MONGO_URI = f"{MONGO_URI}{joiner}tlsDisableOCSPEndpointCheck=true"

# --- 2) Start Spark med nødvendige connectors ---
spark = (
    SparkSession.builder
    .appName("Elhub Cassandra → Mongo")
    # MongoDB Spark Connector + Cassandra Connector
    .config(
        "spark.jars.packages",
        "org.mongodb.spark:mongo-spark-connector_2.12:10.4.0,"
        "com.datastax.spark:spark-cassandra-connector_2.12:3.5.1"
    )
    # Cassandra (lokal)
    .config("spark.cassandra.connection.host", "127.0.0.1")
    .config("spark.cassandra.connection.port", "9042")
    # (valgfritt) litt større fetch
    .config("spark.cassandra.input.fetch.size_in_rows", "2000")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

# --- 3) Les fra Cassandra ---
src_df = (
    spark.read.format("org.apache.spark.sql.cassandra")
    .options(table="production_hourly_by_group", keyspace="elhub_data")
    .load()
)

# Kolonnene i Cassandra er: pricearea, productiongroup, starttime, quantitykwh
# Oppgaven krever CamelCase til Mongo: priceArea, productionGroup, startTime, quantityKwh
df = (
    src_df
    .select(
        F.col("pricearea").alias("priceArea"),
        F.col("productiongroup").alias("productionGroup"),
        F.col("starttime").alias("startTime"),
        F.col("quantitykwh").alias("quantityKwh"),
    )
)

# --- 4) Filtrer til hele 2021 (inkludert), ekskluder 2022 ---
start_utc = datetime(2021, 1, 1, 0, 0, 0)
end_utc   = datetime(2022, 1, 1, 0, 0, 0)
df_2021 = df.where((F.col("startTime") >= F.lit(start_utc)) & (F.col("startTime") < F.lit(end_utc)))

# (Valgfritt) sanity-check: tell og vis ett eksempel
count_2021 = df_2021.count()
print(f"Skal skrive {count_2021:,} dokumenter til MongoDB...")

# --- 5) Skriv til Mongo (DB: elhub, Collection: production_2021_by_hour) ---
(
    df_2021.write.format("mongodb")
    .mode("overwrite")   # bruk "append" hvis du vil legge til i stedet
    .option("spark.mongodb.write.connection.uri", MONGO_URI)
    .option("spark.mongodb.write.database", "elhub")
    .option("spark.mongodb.write.collection", "production_2021_by_hour")
    .save()
)

# --- 6) Verifiser ved å lese tilbake fra Mongo med Spark ---
verify_df = (
    spark.read.format("mongodb")
    .option("spark.mongodb.read.connection.uri", MONGO_URI)
    .option("spark.mongodb.read.database", "elhub")
    .option("spark.mongodb.read.collection", "production_2021_by_hour")
    .load()
)

vcount = verify_df.count()
print(f"Verifisering: {vcount:,} dokumenter i Mongo (elhub.production_2021_by_hour).")

# Vis ett eksempel
verify_df.orderBy(F.col("startTime").asc()).show(3, truncate=False)

spark.stop()
print("Ferdig ✅")
