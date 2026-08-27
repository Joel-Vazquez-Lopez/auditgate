import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

CLAIMS = "data/normalized/scifact/claims_dev.jsonl"
DOCUMENTS = "data/normalized/scifact/documents.jsonl"


tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
model.eval()


# Load documents
documents = {}

with open(DOCUMENTS) as f:
    for line in f:
        doc = json.loads(line)
        documents[doc["document_id"]] = doc


# Load evidence-bearing claims
claims = []

with open(CLAIMS) as f:
    for line in f:
        claim = json.loads(line)

        if claim["gold_label"] in ["supported", "contradicted"]:
            claims.append(claim)


correct = 0
total = 0

confusion = {
    "supported": {
        "supported": 0,
        "contradicted": 0,
        "insufficient_evidence": 0,
    },
    "contradicted": {
        "supported": 0,
        "contradicted": 0,
        "insufficient_evidence": 0,
    },
}

shown_failures = 0
for claim in claims:

    # Use the first gold evidence set.
    evidence_set = claim["gold_evidence"][0]

    document = documents[evidence_set["document_id"]]

    passage_ids = set(evidence_set["passage_ids"])

    evidence_text = " ".join(
        passage["text"]
        for passage in document["passages"]
        if passage["passage_id"] in passage_ids
    )

    inputs = tokenizer(
        evidence_text,
        claim["text"],
        return_tensors="pt",
        truncation=True,
    )

    with torch.no_grad():
        logits = model(**inputs).logits[0]
        probabilities = torch.softmax(logits, dim=0)

    scores = {
        "supported": probabilities[0].item(),
        "insufficient_evidence": probabilities[1].item(),
        "contradicted": probabilities[2].item(),
    }

    prediction = max(scores, key=scores.get)
    gold = claim["gold_label"]

    if prediction != gold and shown_failures < 10:
        print("\n" + "=" * 70)
        print("GOLD:", gold)
        print("PREDICTED:", prediction)
        print("CONFIDENCE:", round(scores[prediction], 3))

        print("\nCLAIM:")
        print(claim["text"])

        print("\nEVIDENCE:")
        print(evidence_text)

        print("\nSCORES:")
        print(scores)

    shown_failures += 1

    confusion[gold][prediction] += 1

    if prediction == gold:
        correct += 1

    total += 1


print("\nGold-evidence verifier evaluation")
print("---------------------------------")
print(f"Claims: {total}")
print(f"Accuracy: {correct / total:.3f}")

print("\nConfusion matrix")

print(json.dumps(confusion, indent=2))