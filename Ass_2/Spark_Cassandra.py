from pyspark.sql import SparkSession, functions as F

CONNECTOR = "com.datastax.spark:spark-cassandra-connector_2.12:3.5.1"  # matcher Spark 3.5.x

spark = (
    SparkSession.builder
    .appName("IND320-Cassandra")
    .config("spark.master", "local[*]")
    .config("spark.jars.packages", CONNECTOR)
    .config("spark.sql.extensions", "com.datastax.spark.connector.CassandraSparkExtensions")
    .config("spark.cassandra.connection.host", "127.0.0.1")
    .config("spark.cassandra.connection.port", "9042")
    .getOrCreate()
)

print("Spark version:", spark.version)

df = spark.createDataFrame(
    [("oslo-02", "2025-01-01 12:00:00", 3.2),
     ("oslo-02", "2025-01-01 13:00:00", 3.6)],
    ["station_id", "ts_str", "temp_c"]
).withColumn("ts", F.to_timestamp("ts_str")).drop("ts_str")

(df.write
 .format("org.apache.spark.sql.cassandra")
 .options(table="weather_raw", keyspace="ind320")
 .mode("append").save())

read_df = (spark.read
    .format("org.apache.spark.sql.cassandra")
    .options(table="weather_raw", keyspace="ind320").load())

read_df.orderBy("station_id","ts").show(10, truncate=False)


from pyspark.sql import functions as F

# write a couple rows
df = spark.createDataFrame(
    [("oslo-02","2025-01-01 12:00:00",3.2),
     ("oslo-02","2025-01-01 13:00:00",3.6)],
    ["station_id","ts_str","temp_c"]
).withColumn("ts", F.to_timestamp("ts_str")).drop("ts_str")

(df.write
 .format("org.apache.spark.sql.cassandra")
 .options(table="weather_raw", keyspace="ind320")
 .mode("append").save())

# read back
read_df = (spark.read
    .format("org.apache.spark.sql.cassandra")
    .options(table="weather_raw", keyspace="ind320")
    .load())

read_df.orderBy("station_id","ts").show(10, truncate=False)
