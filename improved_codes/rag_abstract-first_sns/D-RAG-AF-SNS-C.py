import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm   
import re
import time

INPUT_PATH = r" Enter your Query + Extracted Contexts .json location "
OUTPUT_PATH = r" Enter your final Output .jsonl destination"

LLM_MODEL_PATH = r" Enter your LLM location "
MAX_NEW = 300
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_PATH)
llm_model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_PATH,
    torch_dtype=torch.float16 if DEVICE=="cuda" else torch.float32
).to(DEVICE)

generator = pipeline(
    "text-generation",
    model=llm_model,
    tokenizer=tokenizer,
    device=0 if DEVICE=="cuda" else -1,
    max_new_tokens=MAX_NEW,
    temperature=0.5,
    return_full_text=False
)

sns_results = json.load(open(INPUT_PATH, "r", encoding="utf-8"))

with open(OUTPUT_PATH, "w", encoding="utf-8") as out_file:
    for item in tqdm(sns_results, desc="Generating answers", unit="sample"):
        
        # فقط متن‌ها را برای context نگه داریم
        contexts = [c["text"] for c in item["sns_nodes"]]
        final_context = "\n\n".join(contexts)

        prompt = f"""
You are a careful question-answering assistant.

It is very important that your answer is fully supported by the provided context.
Your goal is to give the most accurate and context-faithful answer possible.

Answer the question using only the provided context.
If the context does not contain the answer, say "I don't know."

Write a clear and concise answer in no more than three sentences.

### Question:
{item['query']}

### Context:
{final_context}

### Answer:
"""
        out_text = generator(prompt, eos_token_id=tokenizer.eos_token_id)[0]["generated_text"].strip()
        after_answer = out_text.split("### Answer:")[-1]
        answer = re.split(r"\n?###", after_answer)[0].strip()

        # خروجی ساده با contexts فقط به صورت متن
        out_file.write(json.dumps({
            "id": item["id"],
            "query": item["query"],
            "contexts": contexts,
            "answer": answer
        }, ensure_ascii=False) + "\n")

        torch.cuda.empty_cache()
        time.sleep(2)

print(f"✅ Final answers saved to {OUTPUT_PATH}")
