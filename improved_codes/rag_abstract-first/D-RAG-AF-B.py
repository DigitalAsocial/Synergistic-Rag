import json, torch
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

INPUT_PATH  = r" Enter the A stage .json output location "
OUTPUT_PATH = r" Enter retrieved .json chunks output destination "

CHROMA_BODY_PATH = r" Enter your ChromaDB Body Database location "
EMBED_MODEL_PATH = r" Enter your finetuned embedding model location "

TOP_B = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

embed_model = SentenceTransformer(EMBED_MODEL_PATH, device=DEVICE)
client_body = PersistentClient(path=CHROMA_BODY_PATH)
body_collection = client_body.get_collection("semantic_body_chunks")

def retrieve_from_bodies(query, allowed_doc_ids, top_k=TOP_B):
    where_clause = {"source": {"$in": [f"{d}.xml" for d in allowed_doc_ids]}}
    q_emb = embed_model.encode([query], normalize_embeddings=True).tolist()[0]
    res = body_collection.query(query_embeddings=[q_emb], n_results=top_k, where=where_clause)
    return res["documents"][0]

abs_results = json.load(open(INPUT_PATH, "r", encoding="utf-8"))

body_results = []
for item in abs_results:
    body_chunks = retrieve_from_bodies(item["query"], item["abstract_ids"])
    body_results.append({
        **item,
        "contexts": body_chunks
    })

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(body_results, f, ensure_ascii=False, indent=2)

print(f"✅ Body retrieval done → {OUTPUT_PATH}")
