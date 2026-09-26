import json
import sys

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


MODEL = "cross-encoder/nli-deberta-v3-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
model.eval()


def evaluate(source: str, candidate: str) -> dict:
    inputs = tokenizer(
        source,
        candidate,
        return_tensors="pt",
        truncation=True,
    )

    with torch.no_grad():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=1)[0]

    return {
        "contradiction": probabilities[0].item(),
        "entailment": probabilities[1].item(),
        "neutral": probabilities[2].item(),
    }

def loses_conditional_structure(source: str, candidate: str) -> bool:
    source_lower = source.lower()
    candidate_lower = candidate.lower()

    source_is_conditional = source_lower.startswith("if ")
    candidate_is_conditional = candidate_lower.startswith("if ")

    return source_is_conditional and not candidate_is_conditional

def decide(source: str, candidate: str, scores: dict) -> str:
    if loses_conditional_structure(source, candidate):
        return "unsafe"

    if scores["entailment"] >= 0.90:
        return "faithful"

    return "unsafe"

def main() -> None:
    request = json.load(sys.stdin)

    if "cases" in request:
        results = []

        for case in request["cases"]:
            scores = evaluate(
                case["source"],
                case["candidate"],
            )

            results.append(
            {
                "name": case.get("name"),
                "decision": decide(
                    case["source"],
                    case["candidate"],
                    scores,
                ),
                "scores": scores,
            }
        )
            

        print(json.dumps({"results": results}))
        return

    scores = evaluate(
    request["source"],
    request["candidate"],
    )

    print(
        json.dumps(
            {
                "decision": decide(
                    request["source"],
                    request["candidate"],
                    scores,
                ),
                "scores": scores,
            }
        )
    )
        
if __name__ == "__main__":
    main()