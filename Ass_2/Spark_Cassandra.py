# --- Tving Java 17 (macOS + Homebrew) før PySpark lastes ---
import os
os.environ["JAVA_HOME"] = "/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"
os.environ["PATH"] = "/opt/homebrew/opt/openjdk@17/bin:" + os.environ.get("PATH", "")
# -----------------------------------------------------------

from pyspark.sql import SparkSession

def get_spark():
    return (SparkSession.builder
        .appName("IND320 Cassandra test")
        # Viktig: hent inn connectoren når du kjører som Python-script
        .config("spark.jars.packages", "com.datastax.spark:spark-cassandra-connector_2.12:3.5.1")
        # Cassandra-tilkobling (din lokale node)
        .config("spark.cassandra.connection.host", "127.0.0.1")
        .config("spark.cassandra.connection.port", "9042")
        .config("spark.cassandra.connection.localDC", "datacenter1")
        .getOrCreate())

if __name__ == "__main__":
    spark = get_spark()
    df = (spark.read.format("org.apache.spark.sql.cassandra")
          .options(keyspace="ind320", table="ping").load())
    df.show(truncate=False)
