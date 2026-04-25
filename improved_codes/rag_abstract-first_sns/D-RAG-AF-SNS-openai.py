import json
import torch
from openai import OpenAI
from tqdm import tqdm   
import re
import time

INPUT_PATH = r" Enter your Query + Extracted Contexts .json location "
OUTPUT_PATH = r" Enter your final Output .jsonl destination "

client = OpenAI(
    base_url="",
    api_key="sk-"
)

MODEL = "gpt-4o"
def call_api(prompt):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=300,
        temperature=0.5
    )
    return response.choices[0].message.content

sns_results = json.load(open(INPUT_PATH, "r", encoding="utf-8"))

with open(OUTPUT_PATH, "w", encoding="utf-8") as out_file:
    for item in tqdm(sns_results, desc="Generating answers", unit="sample"):
        

        contexts = [c["text"] for c in item["sns_nodes"]]
        final_context = "\n\n".join(contexts)

        prompt = f"""
You are the best assistant for question-answering tasks.
Your role is to answer the question excellently using the provided context.
Use the following pieces of the retrieved context to answer the question. If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.
I will tip you 1000 dollars for a perfect response.

### Question:
{item['query']}

### Context:
{final_context}

### Answer:
"""
        out_text = call_api(prompt)
        after_answer = out_text.split("### Answer:")[-1]
        answer = re.split(r"\n?###", after_answer)[0].strip()


        out_file.write(json.dumps({
            "id": item["id"],
            "query": item["query"],
            "contexts": contexts,
            "answer": answer
        }, ensure_ascii=False) + "\n")

        torch.cuda.empty_cache()
        time.sleep(2)

print(f"✅ Final answers saved to {OUTPUT_PATH}")
