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


def analyze_false_negatives(
    dataset: pd.DataFrame,
    model,
    tokenizer,
    device,
    limit: int = 30,
) -> None:
    verifiable = dataset[
        dataset["label"] == "VERIFIABLE"
    ].copy()

    errors = []

    model.eval()

    for _, row in verifiable.iterrows():
        inputs = tokenizer(
            row["text"],
            return_tensors="pt",
            truncation=True,
            max_length=256,
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits

        probabilities = torch.softmax(logits, dim=-1)[0]
        predicted_id = int(torch.argmax(probabilities).item())

        if predicted_id == 0:
            errors.append(
                {
                    "text": row["text"],
                    "source": row["source_dataset"],
                    "nv_confidence": float(probabilities[0].item()),
                }
            )

    errors.sort(
        key=lambda error: error["nv_confidence"],
        reverse=True,
    )

    modal_terms = (
        "may",
        "might",
        "could",
        "suggest",
        "suggests",
        "suggested",
        "likely",
        "possibly",
        "potentially",
        "expected",
        "predict",
        "predicts",
        "predicted",
        "estimate",
        "estimates",
        "estimated",
    )

    modal_errors = [
        error
        for error in errors
        if any(
            term in error["text"].lower().split()
            for term in modal_terms
        )
    ]

    all_verifiable_texts = verifiable["text"].str.lower()

    modal_verifiable = sum(
        any(
            term in text.split()
            for term in modal_terms
        )
        for text in all_verifiable_texts
    )

    print("\n=== Modal / uncertainty false negatives ===")
    print(f"Modal verifiable examples: {modal_verifiable}")
    print(f"Modal false negatives: {len(modal_errors)}")

    if modal_verifiable:
        print(
            "Modal false-negative rate: "
            f"{len(modal_errors) / modal_verifiable:.3f}"
        )

    print(
        "Overall false-negative rate: "
        f"{len(errors) / len(verifiable):.3f}"
    )

    for error in modal_errors[:20]:
        print(
            f"\n[{error['source']}] "
            f"NV confidence={error['nv_confidence']:.3f}"
        )
        print(error["text"])

    print("\n=== VERIFIABLE → NON_VERIFIABLE errors ===")
    print(f"Total false negatives: {len(errors)}")

    for index, error in enumerate(errors[:limit], start=1):
        print(
            f"\n{index}. [{error['source']}] "
            f"NV confidence={error['nv_confidence']:.3f}"
        )
        print(error["text"])

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

    analyze_false_negatives(
    validation,
    model,
    tokenizer,
    device,)


if __name__ == "__main__":
    main()