import json
import torch
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from tqdm import tqdm

INPUT_PATH = r" Enter your extracted_abstracts .json location "
OUTPUT_PATH = r" Enter your extracted sns chunks .json destination "

CHROMA_SNS_PATH = r" Your ChromaDB database location "
EMBED_MODEL_PATH = r" Your embedding model location "
TOP_C = 40

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
embed_model = SentenceTransformer(EMBED_MODEL_PATH, device=DEVICE)

client_sns = PersistentClient(path=CHROMA_SNS_PATH)
sns_collection = client_sns.get_collection("TB17-G-AF-SNS")

def retrieve_sns_nodes(query, allowed_docs, top_k=TOP_C):
    normalized_docs = [doc_id.replace(".tei", "") for doc_id in allowed_docs]
    all_sns = sns_collection.get(include=["metadatas", "documents"])
    ids, metas = all_sns["ids"], all_sns["metadatas"]

    valid_ids = []
    for sns_id, meta in zip(ids, metas):
        src = meta.get("sns_source", "")
        if any(src.startswith(f"{doc_id}.tei_chunk_") for doc_id in normalized_docs):
            valid_ids.append(sns_id)

    if not valid_ids:
        return []

    q_emb = embed_model.encode([query], normalize_embeddings=True).tolist()[0]
    res = sns_collection.query(query_embeddings=[q_emb], n_results=top_k, ids=valid_ids)

    # برگرداندن هم متن و هم متادیتا
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    return [{"text": d, **m} for d, m in zip(docs, metas)]

def sns_aware_selection(question, contexts, emb_model, max_contexts=6, per_parent_k=1):
    from collections import defaultdict
    import numpy as np

    if not contexts:
        return []

    # Extract texts and robust parent ids
    texts = []
    parents = []
    for c in contexts:
        t = c.get("text", "").strip()
        if not t:
            continue
        texts.append(t)
        pid = c.get("sns_source") or c.get("doc_id") or "unknown"
        parents.append(pid)

    if not texts:
        return []

    # Embeddings (normalized cosine -> dot product)
    q_emb = emb_model.encode([question], normalize_embeddings=True)[0]  # shape: (d,)
    c_embs = emb_model.encode(texts, normalize_embeddings=True)         # shape: (n, d)
    sims = np.dot(c_embs, q_emb)                                       # shape: (n,)

    # Group by parent
    grouped = defaultdict(list)
    for text, pid, sim in zip(texts, parents, sims):
        grouped[pid].append((text, float(sim)))

    # Score parents by 0.7*best + 0.3*mean; allow top-k per parent
    parent_candidates = []
    for pid, items in grouped.items():
        # sort by sim desc
        items.sort(key=lambda x: x[1], reverse=True)
        best_sim = items[0][1]
        mean_sim = float(np.mean([s for _, s in items]))
        parent_score = 0.7 * best_sim + 0.3 * mean_sim

        # select up to per_parent_k texts
        take = items[:per_parent_k]
        for text, sim in take:
            parent_candidates.append({
                "text": text,
                "sns_source": pid,
                "score": parent_score  # parent-level score for global ranking
            })

    # Global ranking and clip
    parent_candidates.sort(key=lambda x: x["score"], reverse=True)
    k = min(max_contexts, len(parent_candidates))
    return parent_candidates[:k]


with open(INPUT_PATH, "r", encoding="utf-8") as f:
    abs_results = json.load(f)

sns_results = []
for item in tqdm(abs_results, desc="Processing queries", unit="query"):
    sns_nodes = retrieve_sns_nodes(item["query"], item["abstract_ids"])
    selected_nodes = sns_aware_selection(item["query"], sns_nodes, embed_model, max_contexts=6)
    sns_results.append({
        "id": item["id"],
        "query": item["query"],
        "sns_nodes": selected_nodes 
    })

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(sns_results, f, ensure_ascii=False, indent=2)

print(f"✅ SNS retrieval done, saved to {OUTPUT_PATH}")
