import json
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

QUERIES_FILE = r" Enter your queries.jsonl location "
OUTPUT_PATH = r" Choose your Output.json destination "

CHROMA_ABS_PATH = r" Enter your chromadb abstracts database location "
EMBED_MODEL_PATH = r" Enter your embedding model location "
TOP_A = 100

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
embed_model = SentenceTransformer(EMBED_MODEL_PATH, device=DEVICE)

client_abs = PersistentClient(path=CHROMA_ABS_PATH)
abs_collection = client_abs.get_collection("abstracts")

def retrieve_from_abstracts(query, top_k=TOP_A):
    q_emb = embed_model.encode([query], normalize_embeddings=True).tolist()[0]
    res = abs_collection.query(query_embeddings=[q_emb], n_results=top_k)
    return res["ids"][0]

queries = [json.loads(line) for line in open(QUERIES_FILE, "r", encoding="utf-8") if line.strip()]

results = []
for item in queries:
    abs_ids = retrieve_from_abstracts(item["query"])
    results.append({"id": item["id"], "query": item["query"], "abstract_ids": abs_ids})

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"✅ Abstract retrieval done, saved to {OUTPUT_PATH}")

