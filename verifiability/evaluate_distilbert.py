from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models" / "distilbert" / "best"

ID2LABEL = {
    0: "NON_VERIFIABLE",
    1: "VERIFIABLE",
}


def predict(
    texts: list[str],
    model,
    tokenizer,
    device,
    batch_size: int = 32,
) -> list[str]:
    predictions = []

    model.eval()

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits

        predicted_ids = torch.argmax(logits, dim=-1).cpu().tolist()
        predictions.extend(ID2LABEL[index] for index in predicted_ids)

    return predictions


def evaluate(name: str, dataset: pd.DataFrame, model, tokenizer, device) -> None:
    predictions = predict(
        dataset["text"].tolist(),
        model,
        tokenizer,
        device,
    )

    print(f"\n=== {name} ===")
    print(f"Rows: {len(dataset):,}")
    print(f"Accuracy: {accuracy_score(dataset['label'], predictions):.3f}")
    print(
        f"Macro-F1: "
        f"{f1_score(dataset['label'], predictions, average='macro'):.3f}"
    )

    print(
        classification_report(
            dataset["label"],
            predictions,
            digits=3,
        )
    )

    print("Confusion matrix:")
    print(
        confusion_matrix(
            dataset["label"],
            predictions,
            labels=["NON_VERIFIABLE", "VERIFIABLE"],
        )
    )


def main() -> None:
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    print("Device:", device)
    print("Model:", MODEL_DIR)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR
    ).to(device)

    validation = pd.read_json(
        DATA_DIR / "validation.jsonl",
        lines=True,
    )
    expert_test = pd.read_json(
        DATA_DIR / "expert_test.jsonl",
        lines=True,
    )

    evaluate(
        "Validation",
        validation,
        model,
        tokenizer,
        device,
    )

    for source in sorted(validation["source_dataset"].unique()):
        subset = validation[
            validation["source_dataset"] == source
        ]

        evaluate(
            f"Validation — {source}",
            subset,
            model,
            tokenizer,
            device,
        )

    evaluate(
        "Expert test",
        expert_test,
        model,
        tokenizer,
        device,
    )


if __name__ == "__main__":
    main()