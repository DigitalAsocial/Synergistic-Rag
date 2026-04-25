import json
from pathlib import Path
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
import nltk
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util

nltk.download('punkt', quiet=True)

EMBED_MODEL = r" Enter your embedding model location "
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SNS_THRESHOLD = 0.55
MAX_NODE_LEN = 8
MIN_NODE_LEN = 2
SENT_BATCH = 96

print("🔹 Loading embedding model:", EMBED_MODEL)
model = SentenceTransformer(EMBED_MODEL, device=DEVICE)

def sns_split(text):
    sents = [s.strip() for s in sent_tokenize(text) if len(s.strip()) > 0]
    if len(sents) <= MIN_NODE_LEN:
        return [text]

    with torch.no_grad():
        embs = model.encode(
            sents,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=SENT_BATCH
        )

    sims = util.cos_sim(embs[:-1], embs[1:]).diagonal().cpu().numpy()

    nodes, cur = [], [sents[0]]

    for i in range(1, len(sents)):
        if sims[i-1] < SNS_THRESHOLD or len(cur) >= MAX_NODE_LEN:
            nodes.append(" ".join(cur))
            cur = [sents[i]]
        else:
            cur.append(sents[i])

    if cur:
        nodes.append(" ".join(cur))

    return nodes

# ENTER YOUR SNS INPUTS FROM PREVIOUS STEP HERE:
with open("sns_input.json", "r", encoding="utf-8") as f:
    input_data = json.load(f)

output_nodes = []
for item in tqdm(input_data, desc="SNS splitting"):
    nodes = sns_split(item["doc"])
    for j, n in enumerate(nodes):
        output_nodes.append({
            "id": f"{item['id']}_sns_{j}",
            "doc": n,
            "meta": {**item["meta"], "sns_index": j, "sns_source": item["id"]}
        })

with open("sns_nodes.json", "w", encoding="utf-8") as f:
    json.dump(output_nodes, f, ensure_ascii=False, indent=2)

print(f"✅ Created {len(output_nodes)} SNS nodes and saved to sns_nodes.json")
