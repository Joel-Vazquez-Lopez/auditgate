import sys
import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

MODEL = "cross-encoder/nli-deberta-v3-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
model.eval()

data = json.load(sys.stdin)

claim = data["claim"]
evidence_list = data["evidence"]

# Repeat the claim once for every evidence passage.
claims = [claim] * len(evidence_list)

inputs = tokenizer(
    evidence_list,
    claims,
    return_tensors="pt",
    truncation=True,
    padding=True,
)

with torch.no_grad():
    logits = model(**inputs).logits
    probabilities = torch.softmax(logits, dim=1)

results = []

for probs in probabilities:
    # cross-encoder/nli-deberta-v3-base:
    # 0 = contradiction
    # 1 = entailment
    # 2 = neutral

    scores = {
        "contradicted": probs[0].item(),
        "supported": probs[1].item(),
        "insufficient_evidence": probs[2].item(),
    }

    label = max(scores, key=scores.get)

    results.append({
        "label": label,
        "confidence": scores[label],
        "scores": scores,
    })

print(json.dumps({
    "results": results
}))