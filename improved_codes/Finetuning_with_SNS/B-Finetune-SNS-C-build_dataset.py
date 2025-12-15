import json
import random
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import InputExample

WINDOW_SIZE = 3
NUM_THREADS = 12


with open("parsed.json", "r", encoding="utf-8") as f:
    all_nodes = json.load(f)

def process_nodes_for_examples(name):
    examples = []
    nodes = all_nodes[name]
    for i in range(len(nodes) - 1):
       
        examples.append(InputExample(texts=[nodes[i], nodes[i + 1]]))
        
        for j in range(2, WINDOW_SIZE + 1):
            if i + j < len(nodes):
                examples.append(InputExample(texts=[nodes[i], nodes[i + j]]))
    return examples

examples = []
with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
    for ex_list in tqdm(executor.map(process_nodes_for_examples, all_nodes.keys()),
                        total=len(all_nodes), desc="Building training pairs (SNS)"):
        examples.extend(ex_list)

random.shuffle(examples)
split = int(len(examples) * 0.95)
train_examples = examples[:split]
val_examples = examples[split:]

dataset = {
    "train": [(ex.texts[0], ex.texts[1]) for ex in train_examples],
    "val": [(ex.texts[0], ex.texts[1]) for ex in val_examples]
}

with open("dataset.json", "w", encoding="utf-8") as out:
    json.dump(dataset, out, ensure_ascii=False, indent=2)

print(f"✅ Dataset (SNS) built with {len(train_examples)} train and {len(val_examples)} val pairs")
