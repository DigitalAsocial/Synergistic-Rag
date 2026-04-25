import json
from pathlib import Path
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util
import torch
import nltk

nltk.download('punkt', quiet=True)

SIM_THRESHOLD = 0.55
MIN_CHUNK_LEN = 3
MAX_CHUNK_LEN = 15
BATCH_SIZE = 128
EMB_MODEL = Path(r" Enter your embedding model location here ")
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"

print("🔹 Loading embedding model:", EMB_MODEL)
model = SentenceTransformer(str(EMB_MODEL), device=DEVICE)

def semantic_chunk_text(text, threshold=SIM_THRESHOLD, min_len=MIN_CHUNK_LEN, max_len=MAX_CHUNK_LEN):
    sents = [s.strip() for s in sent_tokenize(text) if len(s.strip()) > 15]
    if len(sents) <= min_len:
        return [" ".join(sents)]

    with torch.no_grad():
        embs = model.encode(
            sents,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False
        )

    sims = util.cos_sim(embs[:-1], embs[1:]).diagonal().cpu().numpy()

    chunks, cur_chunk = [], [sents[0]]

    for i in range(1, len(sents)):
        if sims[i-1] < threshold or len(cur_chunk) >= max_len:
            chunks.append(" ".join(cur_chunk))
            cur_chunk = [sents[i]]
        else:
            cur_chunk.append(sents[i])

    if cur_chunk:
        chunks.append(" ".join(cur_chunk))

    return chunks

# ENTER YOUR tei_parsed.json HERE:
with open("tei_parsed.json", "r", encoding="utf-8") as f:
    parsed_texts = json.load(f)

all_chunks = []
for item in tqdm(parsed_texts, desc="Chunking bodies"):
    chunks = semantic_chunk_text(item["body"])
    for i, chunk in enumerate(chunks):
        all_chunks.append({
            "chunk": chunk,
            "source": item["name"],
            "chunk_id": i,
            "id": f"{Path(item['name']).stem}_chunk_{i}"
        })

with open("chunks.json", "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, ensure_ascii=False, indent=2)

print(f"✅ Created {len(all_chunks)} chunks and saved to chunks.json")
