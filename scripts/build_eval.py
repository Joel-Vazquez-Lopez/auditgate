import json
import random
from collections import defaultdict

random.seed(42)

INPUT = "data/normalized/scifact/claims_dev.jsonl"
OUTPUT = "data/eval/auditgate_eval.jsonl"

groups = defaultdict(list)

with open(INPUT) as f:
    for line in f:
        claim = json.loads(line)
        groups[claim["gold_label"]].append(claim)

examples = []

for label in [
    "supported",
    "contradicted",
    "insufficient_evidence",
]:
    examples.extend(
        random.sample(groups[label], 50)
    )

random.shuffle(examples)

with open(OUTPUT, "w") as f:
    for example in examples:
        f.write(json.dumps(example) + "\n")

print(f"Saved {len(examples)} evaluation examples")

for label in groups:
    count = sum(
        example["gold_label"] == label
        for example in examples
    )
    print(label, count)
