import json
from pathlib import Path


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DEV_PATH = Path(
    "data/tacer/decision_dev.jsonl"
)

THRESHOLDS = [
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
]


# --------------------------------------------------
# Load states
# --------------------------------------------------

with DEV_PATH.open() as f:
    rows = [
        json.loads(line)
        for line in f
    ]


# --------------------------------------------------
# Counterfactual decision policy
# --------------------------------------------------

def decision_at_threshold(
    row,
    threshold,
):
    support = row[
        "verifier_max_support"
    ]

    contradiction = row[
        "verifier_max_contradiction"
    ]

    strong_support = (
        support >= threshold
    )

    strong_contradiction = (
        contradiction >= threshold
    )

    if (
        strong_support
        and not strong_contradiction
    ):
        return "ALLOW"

    if (
        strong_contradiction
        and not strong_support
    ):
        return "BLOCK"

    return "REVIEW"


def decision_correct(
    row,
    decision,
):
    return (
        (
            decision == "ALLOW"
            and row["gold_label"]
            == "supported"
        )
        or
        (
            decision == "BLOCK"
            and row["gold_label"]
            == "contradicted"
        )
    )


# --------------------------------------------------
# Evaluate thresholds
# --------------------------------------------------

print()
print("=" * 78)
print(
    "TACER-B VERIFIER RISK-COVERAGE — "
    "SCIFACT DEV"
)
print("=" * 78)

print()
print(
    f"{'THRESH':>7}"
    f"{'COVERAGE':>12}"
    f"{'COMMIT':>9}"
    f"{'CORRECT':>10}"
    f"{'WRONG':>8}"
    f"{'RISK':>10}"
    f"{'REVIEW':>9}"
)

print("-" * 78)


results = []

for threshold in THRESHOLDS:

    committed = 0
    correct = 0
    wrong = 0
    review = 0

    for row in rows:

        decision = (
            decision_at_threshold(
                row,
                threshold,
            )
        )

        if decision == "REVIEW":
            review += 1
            continue

        committed += 1

        if decision_correct(
            row,
            decision,
        ):
            correct += 1
        else:
            wrong += 1

    coverage = (
        committed / len(rows)
    )

    risk = (
        wrong / committed
        if committed
        else 0.0
    )

    results.append({
        "threshold": threshold,
        "coverage": coverage,
        "committed": committed,
        "correct": correct,
        "wrong": wrong,
        "risk": risk,
        "review": review,
    })

    print(
        f"{threshold:>7.2f}"
        f"{coverage:>12.3f}"
        f"{committed:>9}"
        f"{correct:>10}"
        f"{wrong:>8}"
        f"{risk:>10.3f}"
        f"{review:>9}"
    )


# --------------------------------------------------
# Wrong commitments
# --------------------------------------------------

print()
print("=" * 78)
print("WRONG COMMITMENTS AT THRESHOLD 0.90")
print("=" * 78)

for row in rows:

    decision = decision_at_threshold(
        row,
        0.90,
    )

    if (
        decision != "REVIEW"
        and not decision_correct(
            row,
            decision,
        )
    ):
        print()
        print("-" * 78)

        print(
            f"Claim ID: {row['claim_id']}"
        )

        print(
            f"Gold:     {row['gold_label']}"
        )

        print(
            f"Decision: {decision}"
        )

        print(
            "Support:  "
            f"{row['verifier_max_support']:.3f}"
        )

        print(
            "Contrad.: "
            f"{row['verifier_max_contradiction']:.3f}"
        )

        print(
        "Gold in candidates:",
        row["gold_in_candidates"],
        )

        print(
            "Alignment gold rank:",
            row["alignment_gold_rank"],
        )

        print(
            "MS-MARCO gold rank:",
            row["msmarco_gold_rank"],
        )

        print(
            "Gold top-5:",
            row["gold_top5"],
        )

        print()
        print(row["claim"])