import json
import random
from pathlib import Path

import torch
from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA_PATH = Path("data/alignment/train.jsonl")
MODEL_OUTPUT = Path("models/alignment")

BASE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

RANDOM_SEED = 42

BATCH_SIZE = 16
EPOCHS = 2
LEARNING_RATE = 2e-5

# Keep a small validation split for monitoring.
VALIDATION_FRACTION = 0.15


# --------------------------------------------------
# Load data
# --------------------------------------------------

def load_jsonl(path):
    with open(path) as f:
        return [
            json.loads(line)
            for line in f
        ]


print("Loading alignment training data...")

rows = load_jsonl(DATA_PATH)

print(f"Examples: {len(rows)}")


# --------------------------------------------------
# IMPORTANT:
# Split by CLAIM, not by pair.
#
# Otherwise passages belonging to the same claim
# could appear in both train and validation data.
# --------------------------------------------------

random.seed(RANDOM_SEED)

claims = sorted({
    row["claim"]
    for row in rows
})

random.shuffle(claims)

validation_size = max(
    1,
    int(
        len(claims)
        * VALIDATION_FRACTION
    ),
)

validation_claims = set(
    claims[:validation_size]
)

train_rows = [
    row
    for row in rows
    if row["claim"] not in validation_claims
]

validation_rows = [
    row
    for row in rows
    if row["claim"] in validation_claims
]


print()
print("=" * 60)
print("ALIGNMENT DATA SPLIT")
print("=" * 60)

print(
    f"Train claims:       "
    f"{len(set(r['claim'] for r in train_rows))}"
)

print(
    f"Validation claims:  "
    f"{len(set(r['claim'] for r in validation_rows))}"
)

print(
    f"Train examples:     "
    f"{len(train_rows)}"
)

print(
    f"Validation examples:"
    f" {len(validation_rows)}"
)


# --------------------------------------------------
# Convert to CrossEncoder examples
# --------------------------------------------------

train_examples = [
    InputExample(
        texts=[
            row["claim"],
            row["evidence"],
        ],
        label=float(row["label"]),
    )
    for row in train_rows
]


train_loader = DataLoader(
    train_examples,
    shuffle=True,
    batch_size=BATCH_SIZE,
)


# --------------------------------------------------
# Load MS-MARCO cross-encoder
# --------------------------------------------------

print()
print(
    f"Loading base model: {BASE_MODEL}"
)

model = CrossEncoder(
    BASE_MODEL,
    num_labels=1,
    max_length=512,
)


# --------------------------------------------------
# Device information
# --------------------------------------------------

if torch.backends.mps.is_available():
    print("MPS available: True")
elif torch.cuda.is_available():
    print("CUDA available: True")
else:
    print("Using CPU")


# --------------------------------------------------
# Training
# --------------------------------------------------

warmup_steps = max(
    1,
    int(
        len(train_loader)
        * EPOCHS
        * 0.1
    ),
)

print(
    f"Training batches per epoch: "
    f"{len(train_loader)}"
)

print(
    f"Warmup steps: {warmup_steps}"
)

print()
print(
    "Starting proposition-alignment fine-tuning..."
)


model.fit(
    train_dataloader=train_loader,
    epochs=EPOCHS,
    warmup_steps=warmup_steps,
    optimizer_params={
        "lr": LEARNING_RATE,
    },
    show_progress_bar=True,
)


# --------------------------------------------------
# Explicitly save final trained model
# --------------------------------------------------

MODEL_OUTPUT.mkdir(
    parents=True,
    exist_ok=True,
)

print()
print("Saving trained alignment model...")

model.save_pretrained(
    str(MODEL_OUTPUT)
)


# --------------------------------------------------
# Verify that the model was actually written
# --------------------------------------------------

expected_config = MODEL_OUTPUT / "config.json"

if not expected_config.exists():
    raise RuntimeError(
        f"Model save failed: "
        f"{expected_config} does not exist."
    )


print()
print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(
    "Saved alignment model to:"
)

print(
    MODEL_OUTPUT.resolve()
)