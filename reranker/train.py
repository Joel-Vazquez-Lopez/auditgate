import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

TRAIN_PATH = Path(
    "data/eval/evidence_selector_train.jsonl"
)

OUTPUT_DIR = Path(
    "models/evidence-selector"
)

RANDOM_SEED = 42


# --------------------------------------------------
# Reproducibility
# --------------------------------------------------

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# --------------------------------------------------
# Load training examples
# --------------------------------------------------

def load_jsonl(path):
    with open(path) as f:
        return [
            json.loads(line)
            for line in f
        ]


examples = load_jsonl(TRAIN_PATH)

print("=" * 60)
print("AUDITGATE EVIDENCE SELECTOR")
print("=" * 60)

print(f"Training examples: {len(examples)}")

positives = sum(
    example["label"] == 1
    for example in examples
)

negatives = sum(
    example["label"] == 0
    for example in examples
)

print(f"Positive: {positives}")
print(f"Negative: {negatives}")


# --------------------------------------------------
# Tokenizer
# --------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL)


# --------------------------------------------------
# Dataset
# --------------------------------------------------

class EvidenceDataset(Dataset):

    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):

        example = self.examples[index]

        encoding = tokenizer(
            example["claim"],
            example["passage"],
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids":
                encoding["input_ids"].squeeze(0),

            "attention_mask":
                encoding["attention_mask"].squeeze(0),

            "labels":
                torch.tensor(
                    example["label"],
                    dtype=torch.long,
                ),
        }


train_dataset = EvidenceDataset(examples)


# --------------------------------------------------
# Model
# --------------------------------------------------

# MS-MARCO MiniLM normally has a single relevance
# output. We replace that head with a binary
# classification head:
#
# 0 = irrelevant
# 1 = relevant
#
# The transformer body still starts from the
# pretrained MS-MARCO relevance model.
# --------------------------------------------------

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL,
    num_labels=2,
    ignore_mismatched_sizes=True,
)

model.config.id2label = {
    0: "irrelevant",
    1: "relevant",
}

model.config.label2id = {
    "irrelevant": 0,
    "relevant": 1,
}


# --------------------------------------------------
# Training
# --------------------------------------------------

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),

    num_train_epochs=2,

    per_device_train_batch_size=8,

    learning_rate=2e-5,

    weight_decay=0.01,

    logging_steps=25,

    # Important after our previous 1.8 GB
    # checkpoint adventure:
    save_strategy="no",

    report_to="none",

    seed=RANDOM_SEED,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)


print()
print("Starting evidence-selector fine-tuning...")
print(
    "MPS available:",
    torch.backends.mps.is_available(),
)


trainer.train()


# --------------------------------------------------
# Save ONLY final model
# --------------------------------------------------

trainer.save_model(
    str(OUTPUT_DIR)
)

tokenizer.save_pretrained(
    str(OUTPUT_DIR)
)


print()
print(
    f"Saved evidence selector to {OUTPUT_DIR}"
)