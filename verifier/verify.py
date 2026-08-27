import sys
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL = "cross-encoder/nli-deberta-v3-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
model.eval()

data = json.load(sys.stdin)

evidence = data["evidence"]
claim = data["claim"]

inputs = tokenizer(
    evidence,
    claim,
    return_tensors="pt",
    truncation=True,
)

with torch.no_grad():
    logits = model(**inputs).logits[0]
    probabilities = torch.softmax(logits, dim=0)

# This model uses:
# 0 = contradiction
# 1 = entailment
# 2 = neutral

scores = {
    "contradicted": probabilities[0].item(),
    "supported": probabilities[1].item(),
    "insufficient_evidence": probabilities[2].item(),
}

label = max(scores, key=scores.get)

print(json.dumps({
    "label": label,
    "confidence": scores[label],
    "scores": scores,
}))