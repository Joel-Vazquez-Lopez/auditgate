import json
import sys

from transformers import pipeline


MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

classifier = pipeline(
    "zero-shot-classification",
    model=MODEL,
)

data = json.load(sys.stdin)

candidate_labels = [
    "an externally verifiable factual claim",
    "an opinion, question, command, preference, or subjective judgement",
]

results = []

for case in data["cases"]:
    output = classifier(
        case["claim"],
        candidate_labels,
        hypothesis_template="This statement is {}.",
        multi_label=False,
    )

    results.append({
        "name": case["name"],
        "expected": case["expected"],
        "labels": output["labels"],
        "scores": output["scores"],
    })

print(json.dumps({"results": results}, indent=2))