import json
import sys
from pathlib import Path

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


MODEL_DIR = Path(__file__).parent / "models" / "distilbert" / "best"

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR
).to(device)
model.eval()


def classify(text: str) -> tuple[str, float]:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits

    probabilities = torch.softmax(logits, dim=-1)[0]
    predicted_id = int(torch.argmax(probabilities).item())

    kind = model.config.id2label[predicted_id]
    confidence = float(probabilities[predicted_id].item())

    return kind, confidence


def main() -> None:
    request = json.load(sys.stdin)

    kind, confidence = classify(request["text"])

    json.dump(
        {
            "kind": kind,
            "confidence": confidence,
        },
        sys.stdout,
    )


if __name__ == "__main__":
    main()