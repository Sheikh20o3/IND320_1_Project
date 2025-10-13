import os, certifi
from pymongo import MongoClient

uri = os.environ["MONGODB_URI"]  # må være satt fra før
client = MongoClient(uri, tlsCAFile=certifi.where())

coll = client["ind320"]["ping"]
coll.insert_one({"_id": 901, "note": "hello from python (TLS OK)"})
print(coll.find_one({"_id": 901}))
print("✅ MongoDB fungerer fra Python med riktig CA")
