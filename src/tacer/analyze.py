import json
from pathlib import Path

import numpy as np


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA_PATH = Path(
    "data/tacer/train.jsonl"
)


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
# Observable runtime features
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
# Targets we want to understand
# --------------------------------------------------

TARGETS = [
    "gold_in_candidates",
    "gold_top1",
    "gold_top3",
    "gold_top5",
]


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def values_for(rows, feature):
    return np.asarray(
        [
            float(row[feature])
            for row in rows
        ],
        dtype=np.float64,
    )


def summarize(values):
    if len(values) == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
        }

    return {
        "mean": float(
            np.mean(values)
        ),
        "median": float(
            np.median(values)
        ),
        "std": float(
            np.std(values)
        ),
    }


def standardized_difference(
    positive_values,
    negative_values,
):
    """
    Difference in group means divided by pooled
    standard deviation.

    Positive:
        feature tends to be higher when retrieval
        succeeds.

    Negative:
        feature tends to be higher when retrieval
        fails.
    """

    if (
        len(positive_values) < 2
        or len(negative_values) < 2
    ):
        return 0.0

    positive_variance = np.var(
        positive_values,
        ddof=1,
    )

    negative_variance = np.var(
        negative_values,
        ddof=1,
    )

    pooled_variance = (
        (
            (len(positive_values) - 1)
            * positive_variance
        )
        +
        (
            (len(negative_values) - 1)
            * negative_variance
        )
    ) / (
        len(positive_values)
        + len(negative_values)
        - 2
    )

    if pooled_variance <= 0:
        return 0.0

    pooled_std = np.sqrt(
        pooled_variance
    )

    return float(
        (
            np.mean(positive_values)
            - np.mean(negative_values)
        )
        / pooled_std
    )


# --------------------------------------------------
# Basic label distribution
# --------------------------------------------------

print()
print("=" * 72)
print("TACER STATE ANALYSIS")
print("=" * 72)

for target in TARGETS:

    positives = sum(
        int(row[target])
        for row in rows
    )

    negatives = (
        len(rows) - positives
    )

    print(
        f"{target:<22} "
        f"yes={positives:<4} "
        f"no={negatives:<4} "
        f"rate={positives / len(rows):.3f}"
    )


# --------------------------------------------------
# Feature separation for each target
# --------------------------------------------------

for target in TARGETS:

    positive_rows = [
        row
        for row in rows
        if int(row[target]) == 1
    ]

    negative_rows = [
        row
        for row in rows
        if int(row[target]) == 0
    ]

    print()
    print("=" * 72)
    print(
        f"TARGET: {target}"
    )
    print("=" * 72)

    print(
        f"Positive states: {len(positive_rows)}"
    )

    print(
        f"Negative states: {len(negative_rows)}"
    )

    results = []

    for feature in FEATURES:

        positive_values = values_for(
            positive_rows,
            feature,
        )

        negative_values = values_for(
            negative_rows,
            feature,
        )

        positive_summary = summarize(
            positive_values
        )

        negative_summary = summarize(
            negative_values
        )

        effect = standardized_difference(
            positive_values,
            negative_values,
        )

        results.append({
            "feature": feature,
            "positive_mean":
                positive_summary["mean"],
            "negative_mean":
                negative_summary["mean"],
            "positive_median":
                positive_summary["median"],
            "negative_median":
                negative_summary["median"],
            "effect":
                effect,
        })

    # Strongest separation first.
    results.sort(
        key=lambda result:
            abs(result["effect"]),
        reverse=True,
    )

    print()
    print(
        f"{'FEATURE':<30}"
        f"{'SUCCESS':>10}"
        f"{'FAIL':>10}"
        f"{'EFFECT':>10}"
    )

    print("-" * 60)

    for result in results:

        print(
            f"{result['feature']:<30}"
            f"{result['positive_mean']:>10.3f}"
            f"{result['negative_mean']:>10.3f}"
            f"{result['effect']:>10.3f}"
        )


# --------------------------------------------------
# Agreement diagnostic
# --------------------------------------------------

print()
print("=" * 72)
print("RANKER AGREEMENT")
print("=" * 72)

agreement_rows = [
    row
    for row in rows
    if int(
        row["ranker_top1_agreement"]
    ) == 1
]

disagreement_rows = [
    row
    for row in rows
    if int(
        row["ranker_top1_agreement"]
    ) == 0
]

print(
    f"Agreement:    {len(agreement_rows)}"
)

print(
    f"Disagreement: {len(disagreement_rows)}"
)

for target in TARGETS:

    if agreement_rows:
        agreement_rate = np.mean([
            row[target]
            for row in agreement_rows
        ])
    else:
        agreement_rate = 0.0

    if disagreement_rows:
        disagreement_rate = np.mean([
            row[target]
            for row in disagreement_rows
        ])
    else:
        disagreement_rate = 0.0

    print()
    print(target)

    print(
        "  when rankers agree:    "
        f"{agreement_rate:.3f}"
    )

    print(
        "  when rankers disagree: "
        f"{disagreement_rate:.3f}"
    )


# --------------------------------------------------
# Failure examples
# --------------------------------------------------

print()
print("=" * 72)
print("EXAMPLE CANDIDATE-RETRIEVAL FAILURES")
print("=" * 72)

failures = [
    row
    for row in rows
    if int(
        row["gold_in_candidates"]
    ) == 0
]

# Show failures where TACER might nevertheless
# look confident according to Alignment.
failures.sort(
    key=lambda row:
        float(row["alignment_top"]),
    reverse=True,
)

for row in failures[:10]:

    print()
    print(row["claim"])

    print(
        "Alignment top: ",
        f"{row['alignment_top']:.3f}",
    )

    print(
        "Alignment entropy: ",
        f"{row['alignment_entropy']:.3f}",
    )

    print(
        "MS-MARCO top: ",
        f"{row['msmarco_top']:.3f}",
    )

    print(
        "Exact overlap: ",
        f"{row['exact_overlap']:.3f}",
    )

    print(
        "Ranker agreement: ",
        row["ranker_top1_agreement"],
    )

    print("-" * 72)