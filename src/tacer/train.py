import json
import pickle
from pathlib import Path

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA_PATH = Path(
    "data/tacer/train.jsonl"
)

MODEL_DIR = Path(
    "models/tacer"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# --------------------------------------------------
# Runtime-observable TACER features
# --------------------------------------------------

FEATURES = [
    "bm25_document_top",
    "bm25_document_mean",
    "bm25_document_std",
    "bm25_document_gap_1_2",
    "bm25_document_gap_1_3",
    "bm25_document_entropy",

    "msmarco_top",
    "msmarco_mean",
    "msmarco_std",
    "msmarco_gap_1_2",
    "msmarco_gap_1_3",
    "msmarco_entropy",

    "alignment_top",
    "alignment_mean",
    "alignment_std",
    "alignment_gap_1_2",
    "alignment_gap_1_3",
    "alignment_entropy",

    "exact_overlap",
    "rare_overlap",
    "phrase_overlap",
    "density",

    "ranker_top1_agreement",
]


# --------------------------------------------------
# Targets
# --------------------------------------------------

TARGETS = {
    "candidate": "gold_in_candidates",
    "top3": "gold_top3",
}


# --------------------------------------------------
# Load data
# --------------------------------------------------

with DATA_PATH.open() as f:
    rows = [
        json.loads(line)
        for line in f
        if line.strip()
    ]


print(
    f"Loaded {len(rows)} TACER states."
)


# --------------------------------------------------
# Persistent development split
# --------------------------------------------------

SPLIT_DIR = Path(
    "data/tacer/splits"
)

TRAIN_IDS_PATH = (
    SPLIT_DIR
    / "train_claim_ids.json"
)

VALIDATION_IDS_PATH = (
    SPLIT_DIR
    / "validation_claim_ids.json"
)

with TRAIN_IDS_PATH.open() as f:
    train_claim_ids = set(
        json.load(f)
    )

with VALIDATION_IDS_PATH.open() as f:
    validation_claim_ids = set(
        json.load(f)
    )

assert train_claim_ids.isdisjoint(
    validation_claim_ids
)

known_ids = {
    row["claim_id"]
    for row in rows
}

assert (
    train_claim_ids
    | validation_claim_ids
) == known_ids

train_rows = [
    row
    for row in rows
    if row["claim_id"]
    in train_claim_ids
]

validation_rows = [
    row
    for row in rows
    if row["claim_id"]
    in validation_claim_ids
]


print()
print("=" * 70)
print("TACER DATA SPLIT")
print("=" * 70)

print(
    f"Train claims:      "
    f"{len(train_claim_ids)}"
)

print(
    f"Validation claims: "
    f"{len(validation_claim_ids)}"
)

print(
    f"Train states:      "
    f"{len(train_rows)}"
)

print(
    f"Validation states: "
    f"{len(validation_rows)}"
)


# --------------------------------------------------
# Feature matrix
# --------------------------------------------------

def make_x(rows):

    return np.asarray(
        [
            [
                float(row[feature])
                for feature in FEATURES
            ]
            for row in rows
        ],
        dtype=np.float64,
    )


def make_y(rows, target):

    return np.asarray(
        [
            int(row[target])
            for row in rows
        ],
        dtype=np.int64,
    )


x_train = make_x(
    train_rows
)

x_validation = make_x(
    validation_rows
)


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

def evaluate(
    name,
    model,
    x,
    y,
):

    predictions = model.predict(
        x
    )

    probabilities = (
        model.predict_proba(x)[:, 1]
    )

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print(
        f"Accuracy:          "
        f"{accuracy_score(y, predictions):.3f}"
    )

    print(
        f"Balanced accuracy: "
        f"{balanced_accuracy_score(y, predictions):.3f}"
    )

    print(
        f"ROC-AUC:           "
        f"{roc_auc_score(y, probabilities):.3f}"
    )

    print(
        f"PR-AUC positive:   "
        f"{average_precision_score(y, probabilities):.3f}"
    )

    print()
    print("Confusion matrix")
    print(
        confusion_matrix(
            y,
            predictions,
        )
    )

    print()
    print(
        classification_report(
            y,
            predictions,
            digits=3,
        )
    )


    # --------------------------------------------------
    # Safety-oriented metric
    # --------------------------------------------------
    #
    # Negative = retrieval/ranking insufficient
    # Positive = sufficient
    #
    # False sufficient:
    #
    # actual 0
    # predicted 1
    #
    # This is the dangerous TACER error.
    # --------------------------------------------------

    negatives = (
        y == 0
    )

    false_sufficient = (
        (predictions == 1)
        & negatives
    )

    negative_count = int(
        negatives.sum()
    )

    false_sufficient_count = int(
        false_sufficient.sum()
    )

    if negative_count > 0:

        false_sufficient_rate = (
            false_sufficient_count
            / negative_count
        )

    else:
        false_sufficient_rate = 0.0

    print(
        "False-sufficient rate: "
        f"{false_sufficient_rate:.3f} "
        f"({false_sufficient_count}/"
        f"{negative_count})"
    )


    return probabilities


# --------------------------------------------------
# Train one model per target
# --------------------------------------------------

for model_name, target in TARGETS.items():

    print()
    print()
    print("#" * 70)
    print(
        f"TRAINING TACER MODEL: {model_name}"
    )
    print(
        f"Target: {target}"
    )
    print("#" * 70)

    y_train = make_y(
        train_rows,
        target,
    )

    y_validation = make_y(
        validation_rows,
        target,
    )


    print()
    print("TRAIN LABEL DISTRIBUTION")

    unique, counts = np.unique(
        y_train,
        return_counts=True,
    )

    for label, count in zip(
        unique,
        counts,
    ):
        print(
            f"  {label}: {count}"
        )


    # --------------------------------------------------
    # Model
    # --------------------------------------------------
    #
    # class_weight="balanced" is important because
    # retrieval failures are substantially rarer.
    # --------------------------------------------------

    model = Pipeline([
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                max_iter=5000,
                random_state=42,
            ),
        ),
    ])


    model.fit(
        x_train,
        y_train,
    )


    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    evaluate(
        f"TACER {model_name} — validation",
        model,
        x_validation,
        y_validation,
    )


    # --------------------------------------------------
    # Feature coefficients
    # --------------------------------------------------

    classifier = model.named_steps[
        "classifier"
    ]

    coefficients = (
        classifier.coef_[0]
    )

    ranked_features = sorted(
        zip(
            FEATURES,
            coefficients,
        ),
        key=lambda item:
            abs(item[1]),
        reverse=True,
    )


    print()
    print("FEATURE COEFFICIENTS")
    print("-" * 55)

    print(
        f"{'FEATURE':<35}"
        f"{'COEFFICIENT':>15}"
    )

    for feature, coefficient in ranked_features:

        print(
            f"{feature:<35}"
            f"{coefficient:>15.3f}"
        )


    # --------------------------------------------------
    # Save model
    # --------------------------------------------------

    model_path = (
        MODEL_DIR
        / f"{model_name}.pkl"
    )

    with model_path.open(
        "wb"
    ) as f:

        pickle.dump(
            {
                "model": model,
                "features": FEATURES,
                "target": target,
            },
            f,
        )


    print()
    print(
        f"Saved: {model_path}"
    )


print()
print("=" * 70)
print("TACER TRAINING COMPLETE")
print("=" * 70)