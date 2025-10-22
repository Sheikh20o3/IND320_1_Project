# load_to_mongo.py
import os
from datetime import datetime
from pyspark.sql import SparkSession, functions as F

# --- 1) Fetch Mongo URI ---
# Try environment variable first, then .streamlit/secrets.toml, and finally “hard fail”.
MONGO_URI = os.environ.get("MONGODB_URI")

if not MONGO_URI:
    # Read .streamlit/secrets.toml if it exists
    secrets_path = os.path.join(".streamlit", "secrets.toml")
    if os.path.isfile(secrets_path):
        try:
            import toml
            MONGO_URI = toml.load(secrets_path).get("MONGODB_URI")
        except Exception as e:
            raise RuntimeError(f"Did not read {secrets_path}: {e}")

if not MONGO_URI:
    raise RuntimeError("Did not find MONGODB_URI.")

# Safety: ensure TLS and an OCSP workaround in case the campus network blocks OCSP
if "tls=" not in MONGO_URI:
    joiner = "&" if "?" in MONGO_URI else "?"
    MONGO_URI = f"{MONGO_URI}{joiner}tls=true"
if "tlsDisableOCSPEndpointCheck=" not in MONGO_URI:
    joiner = "&" if "?" in MONGO_URI else "?"
    MONGO_URI = f"{MONGO_URI}{joiner}tlsDisableOCSPEndpointCheck=true"

# 2) Start Spark with the required connectors
spark = (
    SparkSession.builder
    .appName("Elhub Cassandra → Mongo")
    # MongoDB Spark Connector + Cassandra Connector (Scala 2.12 binaries)
    .config(
        "spark.jars.packages",
        "org.mongodb.spark:mongo-spark-connector_2.12:10.4.0,"
        "com.datastax.spark:spark-cassandra-connector_2.12:3.5.1"
    )
    # Cassandra (local instance)
    .config("spark.cassandra.connection.host", "127.0.0.1")
    .config("spark.cassandra.connection.port", "9042")
    # (optional) slightly larger fetch size
    .config("spark.cassandra.input.fetch.size_in_rows", "2000")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

# 3) Read from Cassandra
src_df = (
    spark.read.format("org.apache.spark.sql.cassandra")
    .options(table="production_hourly_by_group", keyspace="elhub_data")
    .load()
)

# Columns in Cassandra are: pricearea, productiongroup, starttime, quantitykwh
# The task requires CamelCase for Mongo: priceArea, productionGroup, startTime, quantityKwh
df = (
    src_df
    .select(
        F.col("pricearea").alias("priceArea"),
        F.col("productiongroup").alias("productionGroup"),
        F.col("starttime").alias("startTime"),
        F.col("quantitykwh").alias("quantityKwh"),
    )
)

# 4) Filter to all of 2021 (inclusive), exclude 2022 ---
start_utc = datetime(2021, 1, 1, 0, 0, 0)
end_utc   = datetime(2022, 1, 1, 0, 0, 0)
df_2021 = df.where((F.col("startTime") >= F.lit(start_utc)) & (F.col("startTime") < F.lit(end_utc)))

# Sanity check: count and show one example
count_2021 = df_2021.count()
print(f"Will write {count_2021:,} and document toMongoDB...")

# --- 5) Write to Mongo (DB: elhub, Collection: production_2021_by_hour) ---
(
    df_2021.write.format("mongodb")
    .mode("overwrite")   # use "append" if you want to add instead of replace
    .option("spark.mongodb.write.connection.uri", MONGO_URI)
    .option("spark.mongodb.write.database", "elhub")
    .option("spark.mongodb.write.collection", "production_2021_by_hour")
    .save()
)

# --- 6) Verify by reading back from Mongo with Spark ---
verify_df = (
    spark.read.format("mongodb")
    .option("spark.mongodb.read.connection.uri", MONGO_URI)
    .option("spark.mongodb.read.database", "elhub")
    .option("spark.mongodb.read.collection", "production_2021_by_hour")
    .load()
)

vcount = verify_df.count()
print(f"Verification: {vcount:,} Documentation in Mongo (elhub.production_2021_by_hour).")

# Show one example
verify_df.orderBy(F.col("startTime").asc()).show(3, truncate=False)

spark.stop()
print("Done ✅")
