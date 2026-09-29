import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


DATA_DIR = Path(__file__).parent / "data" / "processed"


def main() -> None:
    train = pd.read_json(DATA_DIR / "train.jsonl", lines=True)
    validation = pd.read_json(DATA_DIR / "validation.jsonl", lines=True)
    expert_test = pd.read_json(DATA_DIR / "expert_test.jsonl", lines=True)

    print(f"Train:       {len(train):,}")
    print(f"Validation:  {len(validation):,}")
    print(f"Expert test: {len(expert_test):,}")
    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_features=50_000,
                sublinear_tf=True,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])

    print("\nTraining TF-IDF + logistic regression...")
    model.fit(train["text"], train["label"])
    print("Training complete.")
    
    for name, dataset in [
        ("Validation", validation),
        ("Expert test", expert_test),
    ]:
        predictions = model.predict(dataset["text"])

        print(f"\n=== {name} ===")
        print(f"Accuracy: {accuracy_score(dataset['label'], predictions):.3f}")
        print(
            f"Macro-F1: "
            f"{f1_score(dataset['label'], predictions, average='macro'):.3f}"
        )

        print("\nClassification report:")
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
    print("\n=== Validation by source dataset ===")

    for source_dataset in sorted(validation["source_dataset"].unique()):
        subset = validation[
            validation["source_dataset"] == source_dataset
        ]

        predictions = model.predict(subset["text"])

        print(f"\n{source_dataset} ({len(subset):,} rows)")
        print(
            f"Accuracy: "
            f"{accuracy_score(subset['label'], predictions):.3f}"
        )
        print(
            f"Macro-F1: "
            f"{f1_score(subset['label'], predictions, average='macro'):.3f}"
        )

        print(
            classification_report(
                subset["label"],
                predictions,
                digits=3,
            )
        )


if __name__ == "__main__":
    main()