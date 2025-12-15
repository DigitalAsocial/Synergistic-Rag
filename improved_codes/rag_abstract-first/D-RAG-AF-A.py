import json, torch
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

QUERIES_FILE = r" Enter your .jsonl queries list file location "
OUTPUT_PATH  = r" Enter .json output destination "

CHROMA_ABS_PATH  = r" Enter your ChromaDB Abstracts Database location "
EMBED_MODEL_PATH = r" Enter your finetuned embedding model location "

TOP_A = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

embed_model = SentenceTransformer(EMBED_MODEL_PATH, device=DEVICE)
client_abs  = PersistentClient(path=CHROMA_ABS_PATH)
abs_collection = client_abs.get_collection("abstracts")

def retrieve_from_abstracts(query, top_k=TOP_A):
    q_emb = embed_model.encode([query], normalize_embeddings=True).tolist()[0]
    res = abs_collection.query(query_embeddings=[q_emb], n_results=top_k)
    return res["ids"][0], res["documents"][0]

queries = [json.loads(l) for l in open(QUERIES_FILE, "r", encoding="utf-8") if l.strip()]

results = []
for item in queries:
    doc_ids, abs_texts = retrieve_from_abstracts(item["query"])
    results.append({
        "id": item["id"],
        "query": item["query"],
        "abstract_ids": doc_ids,
        "abstract_contexts": abs_texts
    })

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"✅ Abstract retrieval done → {OUTPUT_PATH}")
