import json
from pathlib import Path
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
import torch

EPOCHS = 6
BATCH_SIZE = 16
LR = 2e-5

BASE_MODEL = Path(r" Enter The original embedding model location ")
SAVE_DIR = Path(r" Enter your fine-tuned embedding model destination ")
SAVE_DIR.mkdir(parents=True, exist_ok=True)


with open("dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

train_examples = [InputExample(texts=[a, b]) for a, b in dataset["train"]]
val_examples = [InputExample(texts=[a, b]) for a, b in dataset["val"]]

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE, pin_memory=True)
torch.backends.cudnn.benchmark = True

val_sentences1 = [ex.texts[0] for ex in val_examples]
val_sentences2 = [ex.texts[1] for ex in val_examples]
val_scores = [1.0] * len(val_examples)

val_evaluator = evaluation.EmbeddingSimilarityEvaluator(val_sentences1, val_sentences2, val_scores, name="val")

print(f"🔹 Loading base model: {BASE_MODEL}")
model = SentenceTransformer(str(BASE_MODEL), device="cuda")
model._target_device = torch.device("cuda")

train_loss = losses.MultipleNegativesRankingLoss(model)
warmup_steps = max(100, int(len(train_dataloader) * EPOCHS * 0.1))

print(f"🚀 Starting fine-tuning... (epochs={EPOCHS}, batch={BATCH_SIZE})")
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=val_evaluator,
    epochs=EPOCHS,
    warmup_steps=warmup_steps,
    optimizer_params={'lr': LR},
    output_path=str(SAVE_DIR),
    save_best_model=True,
    use_amp=True,
    show_progress_bar=True
)

print(f"✅ Best fine-tuned model (SNS) saved to: {SAVE_DIR}/best_model")
