from pymongo import MongoClient
from datetime import datetime, UTC

MONGO_URI = "mongodb+srv://AHS_db_user:<db_password>@ahs786student.qh8rsrb.mongodb.net/?retryWrites=true&w=majority&appName=AHS786Student"

client = MongoClient(MONGO_URI)
db = client.ind320
col = db.weather_curated

doc = {"station_id":"oslo-01","ts":datetime.now(UTC),"temp_c":7.3,"source":"smoketest"}
res = col.insert_one(doc)
print("Inserted _id:", res.inserted_id)
print("One doc:", col.find_one({}, {"_id":0}))
print("Total docs:", col.count_documents({}))
