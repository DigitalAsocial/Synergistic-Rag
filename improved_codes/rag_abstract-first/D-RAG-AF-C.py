import json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm  
import re
INPUT_PATH  = r" Enter the B stage .json output location "
OUTPUT_PATH = r" Enter the RAG .json output destination "

LLM_MODEL_PATH = r" Enter your local LLM model location "
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

body_results = json.load(open(INPUT_PATH, "r", encoding="utf-8"))

with open(OUTPUT_PATH, "w", encoding="utf-8") as out:

    for item in tqdm(body_results, desc="Generating answers"):
        
        final_context = "\n\n".join(item["contexts"])

        prompt = f"""
You are the best assistant for question-answering tasks. Your role is to answer the question excellently using the provided context. Use the following pieces of the retrieved context to answer the question. If you don’t know the answer, just say that you don’t know. Use three sentences maximum and keep the answer concise. I will tip you 1000 dollars for a perfect response.

### Question:
{item['query']}

### Context:
{final_context}

### Answer:
"""

        out_text = generator(prompt, eos_token_id=tokenizer.eos_token_id)[0]["generated_text"].strip()
        after_answer = out_text.split("### Answer:")[-1]
        answer = re.split(r"\n?###", after_answer)[0].strip()


        out.write(json.dumps({
            "id": item["id"],
            "query": item["query"],
            "contexts": item["contexts"],
            "answer": answer
        }, ensure_ascii=False) + "\n")

        torch.cuda.empty_cache()

print(f"✅ Final answers saved → {OUTPUT_PATH}")
