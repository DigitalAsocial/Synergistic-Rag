import os
from pathlib import Path
from tqdm import tqdm
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import chromadb
import nltk
nltk.download('punkt', quiet=True)

DATA_DIR = Path(r" Enter your input tei.xml files location ")
EMB_MODEL = Path(r" Enter your embedding model location ")
CHROMA_ABS = Path(r" Enter your chromadb abstracts database destination ")
CHROMA_ABS.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 256
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"

print("🔹 Loading embedding model:", EMB_MODEL)
model = SentenceTransformer(str(EMB_MODEL), device=DEVICE)

client = chromadb.PersistentClient(path=str(CHROMA_ABS))
collection = client.get_or_create_collection(name="abstracts")

def extract_abstract_from_tei(path: Path) -> str:
    txt = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(txt, "xml")
    abs_nodes = soup.find_all("abstract")
    if not abs_nodes:
        abs_nodes = soup.find_all(lambda tag: tag.name == "div" and tag.get("type") == "abstract")
    abs_text = " ".join([n.get_text(" ", strip=True) for n in abs_nodes]).strip()
    return abs_text

file_list = sorted(DATA_DIR.glob("*.tei.xml"))
docs, ids, metas = [], [], []

for f in tqdm(file_list, desc="Extracting abstracts"):
    a = extract_abstract_from_tei(f)
    if a and len(a.split()) > 5:
        docs.append(a)
        ids.append(f.stem)
        metas.append({"source": f.name})

print(f"📘 Found {len(docs)} abstracts to index.")

for i in tqdm(range(0, len(docs), BATCH_SIZE), desc="Indexing abstracts"):
    chunk_docs = docs[i:i+BATCH_SIZE]
    chunk_ids = ids[i:i+BATCH_SIZE]
    chunk_metas = metas[i:i+BATCH_SIZE]
    embs = model.encode(chunk_docs, normalize_embeddings=True)
    collection.add(
        documents=chunk_docs,
        embeddings=embs.tolist(),
        metadatas=chunk_metas,
        ids=chunk_ids
    )

print("✅ Abstract indexing complete. Chroma path:", CHROMA_ABS)


