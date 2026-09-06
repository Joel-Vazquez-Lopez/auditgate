import json
from pathlib import Path

import numpy as np


DATA_PATH = Path("data/tacer/train.jsonl")

OUTPUT_DIR = Path("data/tacer/splits")

TRAIN_PATH = OUTPUT_DIR / "train_claim_ids.json"
VALIDATION_PATH = OUTPUT_DIR / "validation_claim_ids.json"

SEED = 42
TRAIN_FRACTION = 0.80


# --------------------------------------------------
# Load TACER states
# --------------------------------------------------

with DATA_PATH.open() as f:
    rows = [
        json.loads(line)
        for line in f
        if line.strip()
    ]


claim_ids = sorted({
    row["claim_id"]
    for row in rows
})


# --------------------------------------------------
# Deterministic shuffle
# --------------------------------------------------

rng = np.random.default_rng(SEED)

shuffled_ids = np.asarray(
    claim_ids,
    dtype=object,
)

rng.shuffle(shuffled_ids)


split_index = int(
    TRAIN_FRACTION
    * len(shuffled_ids)
)


train_ids = (
    shuffled_ids[:split_index]
    .tolist()
)

validation_ids = (
    shuffled_ids[split_index:]
    .tolist()
)


# --------------------------------------------------
# Sanity checks
# --------------------------------------------------

assert (
    set(train_ids)
    .isdisjoint(
        set(validation_ids)
    )
)

assert (
    len(train_ids)
    + len(validation_ids)
    == len(claim_ids)
)


# --------------------------------------------------
# Save
# --------------------------------------------------

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


with TRAIN_PATH.open("w") as f:
    json.dump(
        train_ids,
        f,
        indent=2,
    )


with VALIDATION_PATH.open("w") as f:
    json.dump(
        validation_ids,
        f,
        indent=2,
    )


print("=" * 60)
print("TACER DEVELOPMENT SPLIT")
print("=" * 60)

print(
    f"Total supervised claims: {len(claim_ids)}"
)

print(
    f"Train claims:            {len(train_ids)}"
)

print(
    f"Validation claims:       {len(validation_ids)}"
)

print()
print(f"Train IDs:      {TRAIN_PATH}")
print(f"Validation IDs: {VALIDATION_PATH}")

print()
print(
    "SciFact dev remains untouched "
    "for final end-to-end evaluation."
)