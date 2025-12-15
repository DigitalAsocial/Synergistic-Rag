import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import torch

SNS_THRESHOLD = 0.55
MAX_NODE_LEN = 8
MIN_NODE_LEN = 2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = r" Enter your original embedding model location "
model = SentenceTransformer(MODEL_PATH, device=DEVICE)

# بارگذاری جمله‌ها از خروجی مرحله اول
with open("parsed_tb17-g.json", "r", encoding="utf-8") as f:
    all_sentences = json.load(f)

def sns_split(sents):
    if len(sents) <= MIN_NODE_LEN:
        return [" ".join(sents)]
    embs = model.encode(sents, normalize_embeddings=True, show_progress_bar=False)
    sims = util.cos_sim(embs[:-1], embs[1:]).diagonal().cpu().numpy()
    nodes, cur = [], [sents[0]]
    for i in range(1, len(sents)):
        cur.append(sents[i])
        if sims[i - 1] < SNS_THRESHOLD or len(cur) >= MAX_NODE_LEN:
            nodes.append(" ".join(cur))
            cur = []
    if cur:
        nodes.append(" ".join(cur))
    return nodes

parsed_nodes = {}
for name, sents in tqdm(all_sentences.items(), desc="SNS splitting"):
    parsed_nodes[name] = sns_split(sents)

with open("parsed_tb17-sns.json", "w", encoding="utf-8") as out:
    json.dump(parsed_nodes, out, ensure_ascii=False, indent=2)

print(f"✅ SNS completed and saved to parsed_tb17-sns.json")
