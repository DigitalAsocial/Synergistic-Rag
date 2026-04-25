# RAG output evaluation ,benchmark ,and comparison with RAGAS 0.1.1
# python evaluation_with_ragas.py --input RAG-output.jsonl --output RAG-evaluation.csv
import argparse
import json
import time
from typing import List

import pandas as pd
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_relevancy  
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas.run_config import RunConfig

import json
import re
from datasets import Dataset


def clean_text(text: str) -> str:
    if not text:
        return ""

    text = re.sub(r"</?think>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"#{2,}", " ", text)
    text = re.sub(r"`+", " ", text)

    sentences = []
    for s in re.split(r"(?<=[.!?])\s+", text):
        s = s.strip()
        if s and s not in sentences:
            sentences.append(s)

    text = " ".join(sentences)
    text = re.sub(r"\s+", " ", text)

    return text.strip()



def load_jsonl(path: str) -> Dataset:
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)

            answer = clean_text(obj.get("answer", ""))

            contexts = [
                clean_text(c)
                for c in obj.get("contexts", [])
                if c and c.strip()
            ]

            records.append({
                "id": obj["id"],
                "question": obj["query"].strip(),
                "answer": answer,
                "contexts": contexts,
            })

    return Dataset.from_list(records)


def build_llm():
    return ChatOpenAI(
        model="lmstudio",
        base_url="http://localhost:1234/v1",
        api_key="lm-studio",
        temperature=0.0,
        timeout=300,
    )


def build_embeddings():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )


def chunk_dataset(ds: Dataset, batch_size: int) -> List[Dataset]:
    chunks = []
    for i in range(0, len(ds), batch_size):
        chunks.append(ds.select(range(i, min(i + batch_size, len(ds)))))
    return chunks


def main(input_path: str, output_path: str, batch_size: int, pause_seconds: float):
    print("Loading dataset...")
    dataset = load_jsonl(input_path)
    print(f"Total samples: {len(dataset)}")

    print("Loading LLM...")
    llm = build_llm()

    print("Loading embeddings...")
    emb = build_embeddings()

    print("Configuring executor...")
    run_cfg = RunConfig(
        timeout=240,
        max_retries=5,
        max_wait=300,
        max_workers=1,
        thread_timeout=180,
        log_tenacity=True
    )

    print(f"Splitting dataset into batches of size {batch_size}...")
    batches = chunk_dataset(dataset, batch_size)
    print(f"Total batches: {len(batches)}")

    all_dfs = []

    for idx, batch in enumerate(batches, start=1):
        print(f"\n=== Evaluating batch {idx}/{len(batches)} (size={len(batch)}) ===")

        try:
            result = evaluate(
                batch,
                metrics=[faithfulness, answer_relevancy, context_relevancy],
                llm=llm,
                embeddings=emb,
                run_config=run_cfg,
                raise_exceptions=False  
            )

            df = result.to_pandas()
            df = df[["id", "faithfulness","answer_relevancy", "context_relevancy"]]
            all_dfs.append(df)

        except Exception as e:
            print(f"⚠️ Error in batch {idx}: {e}")
            continue

        if pause_seconds > 0 and idx != len(batches):
            print(f"Sleeping for {pause_seconds} seconds to let GPU/LLM rest...")
            time.sleep(pause_seconds)

    if not all_dfs:
        print("❌ No successful batches. Exiting without writing output.")
        return

    final_df = pd.concat(all_dfs, ignore_index=True)
    final_df.to_csv(output_path, index=False)
    print(f"\n✅ Done. Final results saved to: {output_path}")
    print(f"Total evaluated samples: {len(final_df)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batched RAGAS Evaluation Script")
    parser.add_argument("--input", required=True, help="Path to input JSONL file")
    parser.add_argument("--output", required=True, help="Path to output CSV file")
    parser.add_argument("--batch-size", type=int, default=2, help="Number of samples per batch")
    parser.add_argument("--pause", type=float, default=2.0, help="Seconds to sleep between batches")
    args = parser.parse_args()

    main(args.input, args.output, args.batch_size, args.pause)
