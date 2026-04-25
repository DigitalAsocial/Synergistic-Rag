import json
from pathlib import Path
from tqdm import tqdm
from bs4 import BeautifulSoup
import nltk

nltk.download('punkt', quiet=True)

DATA_DIR = Path(r" Enter your tei.xml data files location here ")
SELECTED_IDS_FILE = Path(r" Enter your selected_ids.txt list here - if it exists ")
SELECT_ALL = True  

def extract_body_from_tei(path: Path) -> str:
    txt = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(txt, "lxml-xml")
    bodies = soup.find_all("body")
    if not bodies:
        bodies = soup.find_all(lambda tag: tag.name == "div" and tag.get("type") == "body")
    body_text = " ".join([b.get_text(" ", strip=True) for b in bodies]).strip()
    if not body_text:
        for t in soup.find_all(["abstract", "front"]):
            t.decompose()
        body_text = soup.get_text(" ", strip=True)
    return body_text

if SELECT_ALL or not SELECTED_IDS_FILE.exists():
    to_process = sorted(DATA_DIR.glob("*.tei.xml"))
else:
    ids = [line.strip() for line in SELECTED_IDS_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    to_process = [DATA_DIR / (i + ".tei.xml") for i in ids if (DATA_DIR / (i + ".tei.xml")).exists()]

print(f"📘 Will process {len(to_process)} articles for parsing.")

parsed_texts = []
for path in tqdm(to_process, desc="Parsing TEI files"):
    body = extract_body_from_tei(path)
    if body and len(body.split()) >= 50:
        parsed_texts.append({"name": path.name, "body": body})

with open("tei_parsed.json", "w", encoding="utf-8") as f:
    json.dump(parsed_texts, f, ensure_ascii=False, indent=2)

print(f"✅ Parsed {len(parsed_texts)} valid articles and saved to parsed.json")
