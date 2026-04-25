import os, json, torch
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import nltk
nltk.download('punkt', quiet=True)

INPUT_QUERIES_FILE = r" Enter your queries .jsonl file location"  
OUTPUT_FILE = r" Enter your output .jsonl destination "

CHROMA_PATH = r" Enter your chromadb database location "              
EMB_MODEL_PATH = r" Enter your embedding model location "         
LLM_MODEL_PATH = r" Enter your LLM location  "

TOP_K = 5
MAX_NEW_TOKENS = 300
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🔹 Loading embedding model: {EMB_MODEL_PATH}")
embed_model = SentenceTransformer(EMB_MODEL_PATH, device=DEVICE)

print(f"🔹 Loading LLM model: {LLM_MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_PATH)
llm_model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_PATH, torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
).to(DEVICE)

generator = pipeline(
    "text-generation",
    model=llm_model,
    tokenizer=tokenizer,
    device=0 if DEVICE == "cuda" else -1,
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=0.5,
)

client = PersistentClient(path=CHROMA_PATH)
collection = client.get_collection("semantic_chunks_articles")

def rag_query(query: str, top_k=5):
    query_emb = embed_model.encode([query], normalize_embeddings=True).tolist()[0]

    results = collection.query(query_embeddings=[query_emb], n_results=top_k)
    retrieved_texts = results["documents"][0]

    context = "\n\n".join(retrieved_texts)

    prompt = f"""
You are a careful question-answering assistant.

It is very important that your answer is fully supported by the provided context.
Your goal is to give the most accurate and context-faithful answer possible.

Answer the question using only the provided context.
If the context does not contain the answer, say "I don't know."

Write a clear and concise answer in no more than three sentences.

### Question:
{query}

### Context:
{context}

### Answer:"""

    output = generator(prompt)[0]["generated_text"]
    answer = output.split("### Answer:")[-1].strip()

    return retrieved_texts, answer

def read_queries(path: str):
    queries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))
    return queries

input_queries = read_queries(INPUT_QUERIES_FILE)

with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
    for item in tqdm(input_queries, desc="Running RAG Simple"):
        q_id = item["id"]
        q_text = item["query"]

        contexts, answer = rag_query(q_text, top_k=TOP_K)

        output_obj = {
            "id": q_id,
            "query": q_text,
            "contexts": contexts,
            "answer": answer
        }

        outfile.write(json.dumps(output_obj, ensure_ascii=False) + "\n")

print(f"✅ Finished! Results saved to {OUTPUT_FILE}")
