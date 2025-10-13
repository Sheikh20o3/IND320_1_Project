from .mongo_client import get_collection
coll = get_collection("ind320", "ping")
print("Document inserted successfully.")
coll = get_collection("ind320", "ping")
coll.insert_one({"_id": 123, "note": "ok"})

import os
uri = os.environ["MONGODB_URI"]

(spark.read.format("org.apache.spark.sql.cassandra")
     .options(keyspace="ind320", table="ping").load()
     .write.format("mongodb")
     .option("uri", uri)
     .option("database", "ind320")
     .option("collection", "ping")
     .mode("append").save())
