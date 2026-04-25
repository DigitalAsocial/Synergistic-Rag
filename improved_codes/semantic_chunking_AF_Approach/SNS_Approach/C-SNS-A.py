import json
from chromadb import PersistentClient

DB_DIR = r" Enter your chromadb body database location "
INPUT_COLLECTION = "semantic_body_chunks"

client = PersistentClient(path=DB_DIR)
in_col = client.get_or_create_collection(name=INPUT_COLLECTION, embedding_function=None)

items = in_col.get()
print(f"📘 Found {len(items['ids'])} chunks in input collection.")

data = []
for idx, doc, meta in zip(items["ids"], items["documents"], items["metadatas"]):
    data.append({"id": idx, "doc": doc, "meta": meta})

with open("sns_input.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("✅ Saved input chunks to sns_input.json")
