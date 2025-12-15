import json
from pathlib import Path
from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt', quiet=True)

DATA_ROOT = Path(r" Enter your data location ")
CACHE_DIR = Path("cache_sents")
CACHE_DIR.mkdir(exist_ok=True)

MAX_SENT_PER_FILE = 6000
MIN_LEN = 30

def cache_path_for(f):
    return CACHE_DIR / f"{f.name}.json"

def file_to_sentences_cached(path):
    cache_file = cache_path_for(path)
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text(encoding="utf-8"))
        except:
            pass
    text = path.read_text(encoding="utf-8", errors="ignore")
    sents = [s.strip() for s in sent_tokenize(text) if len(s.strip()) >= MIN_LEN]
    sents = sents[:MAX_SENT_PER_FILE]
    cache_file.write_text(json.dumps(sents, ensure_ascii=False), encoding="utf-8")
    return sents

files = sorted(DATA_ROOT.glob("*.txt"))
print(f"📘 Found {len(files)} text files in {DATA_ROOT}")

parsed = {}
for f in tqdm(files, desc="Parsing sentences"):
    parsed[f.name] = file_to_sentences_cached(f)

with open("parsed.json", "w", encoding="utf-8") as out:
    json.dump(parsed, out, ensure_ascii=False, indent=2)

print(f"✅ Parsed {len(parsed)} files and saved to parsed.json")

