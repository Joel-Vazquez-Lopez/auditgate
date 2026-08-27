import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

MODEL = "cross-encoder/nli-deberta-v3-base"

CLAIMS_PATH = "data/normalized/scifact/claims_train.jsonl"
DOCUMENTS_PATH = "data/normalized/scifact/documents.jsonl"
OUTPUT_DIR = "models/scifact-verifier"


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


# --------------------------------------------------
# Build training examples
# --------------------------------------------------

documents = {
    doc["document_id"]: doc
    for doc in load_jsonl(DOCUMENTS_PATH)
}

examples = []

for claim in load_jsonl(CLAIMS_PATH):

    if claim["gold_label"] not in ["supported", "contradicted"]:
        continue

    label = 0 if claim["gold_label"] == "supported" else 1

    for evidence_set in claim["gold_evidence"]:
        document = documents[evidence_set["document_id"]]
        passage_ids = set(evidence_set["passage_ids"])

        evidence = " ".join(
            passage["text"]
            for passage in document["passages"]
            if passage["passage_id"] in passage_ids
        )

        if evidence:
            examples.append({
                "evidence": evidence,
                "claim": claim["text"],
                "label": label,
            })


print(f"Training examples: {len(examples)}")

supported = sum(x["label"] == 0 for x in examples)
contradicted = sum(x["label"] == 1 for x in examples)

print(f"Supported: {supported}")
print(f"Contradicted: {contradicted}")


# --------------------------------------------------
# Dataset
# --------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL)


class SciFactDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        encoding = tokenizer(
            example["evidence"],
            example["claim"],
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(example["label"]),
        }


train_dataset = SciFactDataset(examples)


# --------------------------------------------------
# Model
# --------------------------------------------------

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL,
    num_labels=2,
    ignore_mismatched_sizes=True,
)

model.config.id2label = {
    0: "supported",
    1: "contradicted",
}

model.config.label2id = {
    "supported": 0,
    "contradicted": 1,
}


# --------------------------------------------------
# Training
# --------------------------------------------------

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=20,
    save_strategy="no",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

print("\nStarting SciFact fine-tuning...")
print("MPS available:", torch.backends.mps.is_available())

trainer.train()

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\nSaved verifier to {OUTPUT_DIR}")