import json
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import chromadb
import torch
BATCH_SIZE = 128
CHROMA_SAFE_LIMIT = 5400
EMB_MODEL = Path(r" Enter your embedding model location ")
CHROMA_BODY = Path(r" Enter your chromadb body database destination here ")
CHROMA_BODY.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"

print("🔹 Loading embedding model:", EMB_MODEL)
model = SentenceTransformer(str(EMB_MODEL), device=DEVICE)
client = chromadb.PersistentClient(path=str(CHROMA_BODY))
collection = client.get_or_create_collection(name="semantic_body_chunks")

# ENTER YOUR CHUNKS FILE HERE - FROM PREVIOUS STEP:
with open("chunks.json", "r", encoding="utf-8") as f:
    all_chunks = json.load(f)

buffer_docs, buffer_embs, buffer_meta, buffer_ids = [], [], [], []
total_chunks = 0

def flush_to_db():
    global buffer_docs, buffer_embs, buffer_meta, buffer_ids
    while len(buffer_docs) > 0:
        chunk_size = min(len(buffer_docs), CHROMA_SAFE_LIMIT)
        collection.add(
            documents=buffer_docs[:chunk_size],
            embeddings=buffer_embs[:chunk_size],
            metadatas=buffer_meta[:chunk_size],
            ids=buffer_ids[:chunk_size]
        )
        buffer_docs = buffer_docs[chunk_size:]
        buffer_embs = buffer_embs[chunk_size:]
        buffer_meta = buffer_meta[chunk_size:]
        buffer_ids = buffer_ids[chunk_size:]

for i in tqdm(range(0, len(all_chunks), BATCH_SIZE), desc="Embedding chunks"):
    batch = all_chunks[i:i+BATCH_SIZE]
    batch_chunks = [c["chunk"] for c in batch]
    batch_ids = [c["id"] for c in batch]
    batch_metas = [{"source": c["source"], "chunk_id": c["chunk_id"]} for c in batch]

    with torch.no_grad():
        batch_embs = model.encode(
            batch_chunks,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            convert_to_tensor=True,
            show_progress_bar=False
        ).cpu().numpy()



    buffer_docs.extend(batch_chunks)
    buffer_embs.extend(batch_embs.tolist())
    buffer_meta.extend(batch_metas)
    buffer_ids.extend(batch_ids)
    total_chunks += len(batch_chunks)

    if len(buffer_docs) >= 5000:
        flush_to_db()

flush_to_db()

print(f"✅ Done: processed {len(all_chunks)} chunks, stored in {CHROMA_BODY}")
