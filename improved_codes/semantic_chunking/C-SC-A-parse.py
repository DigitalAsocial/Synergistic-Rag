import json
from pathlib import Path
from tqdm import tqdm
import nltk

nltk.download('punkt', quiet=True)

DATA_DIR = Path(r" Enter your tei.xml files location ")

txt_files = sorted(DATA_DIR.glob("*.tei.xml"))
print(f"📘 Found {len(txt_files)} TEI files for parsing.")

parsed_texts = []
for path in tqdm(txt_files, desc="Parsing TEI files"):
    text = path.read_text(encoding="utf-8", errors="ignore")
    text = text.replace("\n", " ").replace("<", " <").replace(">", "> ").replace("  ", " ")
    text = " ".join([t for t in text.split() if not t.startswith("<")])
    if len(text.split()) >= 50:
        parsed_texts.append({"name": path.name, "body": text})

with open("parsed.json", "w", encoding="utf-8") as f:
    json.dump(parsed_texts, f, ensure_ascii=False, indent=2)

print(f"✅ Parsed {len(parsed_texts)} valid TEI files and saved to parsed.json")
