import json
from pathlib import Path

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# Paths
# --------------------------------------------------

TRAIN_PATH = Path(
    "data/tacer/decision_train.jsonl"
)

DEV_PATH = Path(
    "data/tacer/decision_dev.jsonl"
)


# --------------------------------------------------
# Load data
# --------------------------------------------------

def load_jsonl(path):
    with path.open() as f:
        return [
            json.loads(line)
            for line in f
        ]


train_rows = load_jsonl(
    TRAIN_PATH
)

dev_rows = load_jsonl(
    DEV_PATH
)


print(
    f"Train states: {len(train_rows)}"
)

print(
    f"Dev states:   {len(dev_rows)}"
)

# --------------------------------------------------
# Feature groups
# --------------------------------------------------

# A. Minimal verifier-confidence baseline.
#
# Tests whether TACER-B can do anything beyond
# looking at the strongest verifier probabilities.
CONFIDENCE_FEATURES = [
    "verifier_max_support",
    "verifier_max_contradiction",
]


# B. Full verifier-state representation.
#
# Tests whether the distribution of verifier
# responses across evidence passages contains
# useful information beyond maximum confidence.
VERIFIER_FEATURES = [
    "verifier_max_support",
    "verifier_max_contradiction",
    "verifier_mean_support",
    "verifier_mean_contradiction",
    "verifier_margin",
    "verifier_strong_support_count",
    "verifier_strong_contradiction_count",
]


# C. Retrieval / evidence-state representation.
#
# These are observable before the final decision.
RETRIEVAL_FEATURES = [
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


# D. Full TACER-B state.
FULL_FEATURES = (
    RETRIEVAL_FEATURES
    + VERIFIER_FEATURES
)


EXPERIMENTS = {
    "confidence": CONFIDENCE_FEATURES,
    "verifier": VERIFIER_FEATURES,
    "retrieval": RETRIEVAL_FEATURES,
    "full": FULL_FEATURES,
}


# --------------------------------------------------
# Matrix helpers
# --------------------------------------------------

def make_x(
    rows,
    features,
):
    return np.asarray(
        [
            [
                float(row[feature])
                for feature in features
            ]
            for row in rows
        ],
        dtype=np.float64,
    )


def make_y(rows):
    return np.asarray(
        [
            int(row["decision_sufficient"])
            for row in rows
        ],
        dtype=np.int64,
    )


y_train = make_y(
    train_rows
)

y_dev = make_y(
    dev_rows
)


# --------------------------------------------------
# Train and evaluate one feature group
# --------------------------------------------------

def run_experiment(
    name,
    features,
):
    x_train = make_x(
        train_rows,
        features,
    )

    x_dev = make_x(
        dev_rows,
        features,
    )

    model = Pipeline([
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])

    model.fit(
        x_train,
        y_train,
    )

    predictions = model.predict(
        x_dev
    )

    probabilities = (
        model.predict_proba(
            x_dev
        )[:, 1]
    )

    accuracy = accuracy_score(
        y_dev,
        predictions,
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y_dev,
            predictions,
        )
    )

    roc_auc = roc_auc_score(
        y_dev,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_dev,
        probabilities,
    )

    print()
    print("=" * 70)
    print(
        f"TACER-B EXPERIMENT: {name}"
    )
    print("=" * 70)

    print(
        f"Features:          {len(features)}"
    )
    print(
        f"Accuracy:          {accuracy:.3f}"
    )
    print(
        f"Balanced accuracy: {balanced_accuracy:.3f}"
    )
    print(
        f"ROC-AUC:           {roc_auc:.3f}"
    )
    print(
        f"PR-AUC positive:   {pr_auc:.3f}"
    )

    print()
    print("Confusion matrix")
    print(
        confusion_matrix(
            y_dev,
            predictions,
        )
    )

    print()
    print(
        classification_report(
            y_dev,
            predictions,
            digits=3,
        )
    )

    return {
        "name": name,
        "features": features,
        "model": model,
        "accuracy": accuracy,
        "balanced_accuracy":
            balanced_accuracy,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }


# --------------------------------------------------
# Run ablation
# --------------------------------------------------

results = []

for name, features in EXPERIMENTS.items():
    results.append(
        run_experiment(
            name,
            features,
        )
    )


# --------------------------------------------------
# Final comparison
# --------------------------------------------------

print()
print("=" * 70)
print("TACER-B ABLATION SUMMARY")
print("=" * 70)

print(
    f"{'MODEL':<15}"
    f"{'FEATURES':>10}"
    f"{'ACC':>10}"
    f"{'BAL ACC':>10}"
    f"{'ROC-AUC':>10}"
    f"{'PR-AUC':>10}"
)

print("-" * 65)

for result in results:
    print(
        f"{result['name']:<15}"
        f"{len(result['features']):>10}"
        f"{result['accuracy']:>10.3f}"
        f"{result['balanced_accuracy']:>10.3f}"
        f"{result['roc_auc']:>10.3f}"
        f"{result['pr_auc']:>10.3f}"
    )