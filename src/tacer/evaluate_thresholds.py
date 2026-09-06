import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


DATA_PATH = Path("data/tacer/train.jsonl")
MODEL_PATH = Path("models/tacer/candidate.pkl")

# --------------------------------------------------
# Load
# --------------------------------------------------

with DATA_PATH.open() as f:
    rows = [
        json.loads(line)
        for line in f
        if line.strip()
    ]

with MODEL_PATH.open("rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]
features = bundle["features"]
target = bundle["target"]


# --------------------------------------------------
# Load persistent validation split
# --------------------------------------------------

VALIDATION_IDS_PATH = Path(
    "data/tacer/splits/validation_claim_ids.json"
)

with VALIDATION_IDS_PATH.open() as f:
    validation_ids = set(
        json.load(f)
    )

validation_rows = [
    row
    for row in rows
    if row["claim_id"]
    in validation_ids
]

# Sanity checks
assert len(validation_ids) == 101

assert {
    row["claim_id"]
    for row in validation_rows
} == validation_ids


x = np.asarray(
    [
        [
            float(row[feature])
            for feature in features
        ]
        for row in validation_rows
    ],
    dtype=np.float64,
)

y = np.asarray(
    [
        int(row[target])
        for row in validation_rows
    ],
    dtype=np.int64,
)

probabilities = model.predict_proba(x)[:, 1]


print("=" * 90)
print("TACER CANDIDATE-SUFFICIENCY OPERATING CURVE")
print("=" * 90)

print(f"Validation states: {len(y)}")
print(f"Sufficient:        {(y == 1).sum()}")
print(f"Insufficient:      {(y == 0).sum()}")
print(
    f"ROC-AUC:           "
    f"{roc_auc_score(y, probabilities):.3f}"
)


# --------------------------------------------------
# Threshold sweep
# --------------------------------------------------

print()
print(
    f"{'THRESH':>7} "
    f"{'ACCEPT%':>8} "
    f"{'FAIL DET':>9} "
    f"{'FALSE SUFF':>11} "
    f"{'UNNEC EXP':>10} "
    f"{'ACCEPT PREC':>12}"
)
print("-" * 75)

results = []

for threshold in np.arange(
    0.05,
    1.00,
    0.05,
):

    accept = probabilities >= threshold

    sufficient = y == 1
    insufficient = y == 0

    # Fraction of all requests allowed through
    # without expansion.
    accept_rate = np.mean(accept)

    # Insufficient states correctly caught and
    # therefore NOT accepted.
    failure_detection = np.mean(
        ~accept[insufficient]
    )

    # Dangerous error:
    # insufficient evidence accepted as sufficient.
    false_sufficient = np.mean(
        accept[insufficient]
    )

    # Sufficient evidence unnecessarily expanded.
    unnecessary_expansion = np.mean(
        ~accept[sufficient]
    )

    # Of requests we accept, how many truly contain
    # the annotated evidence?
    if accept.sum() > 0:
        accept_precision = np.mean(
            y[accept]
        )
    else:
        accept_precision = 1.0

    result = {
        "threshold": float(threshold),
        "accept_rate": float(accept_rate),
        "failure_detection": float(
            failure_detection
        ),
        "false_sufficient": float(
            false_sufficient
        ),
        "unnecessary_expansion": float(
            unnecessary_expansion
        ),
        "accept_precision": float(
            accept_precision
        ),
    }

    results.append(result)

    print(
        f"{threshold:>7.2f} "
        f"{accept_rate:>8.3f} "
        f"{failure_detection:>9.3f} "
        f"{false_sufficient:>11.3f} "
        f"{unnecessary_expansion:>10.3f} "
        f"{accept_precision:>12.3f}"
    )


# --------------------------------------------------
# Find useful operating points
# --------------------------------------------------

print()
print("=" * 90)
print("SAFETY-CONSTRAINED OPERATING POINTS")
print("=" * 90)

for maximum_false_sufficient in [
    0.20,
    0.15,
    0.10,
    0.05,
]:

    eligible = [
        result
        for result in results
        if result["false_sufficient"]
        <= maximum_false_sufficient
    ]

    if not eligible:
        print(
            f"\nFalse-sufficient <= "
            f"{maximum_false_sufficient:.0%}: "
            "no threshold found"
        )
        continue

    # Among thresholds satisfying the safety
    # constraint, maximize immediate acceptance.
    best = max(
        eligible,
        key=lambda result:
            result["accept_rate"],
    )

    print(
        f"\nFalse-sufficient <= "
        f"{maximum_false_sufficient:.0%}"
    )

    print(
        f"  threshold:             "
        f"{best['threshold']:.2f}"
    )

    print(
        f"  immediate acceptance:  "
        f"{best['accept_rate']:.1%}"
    )

    print(
        f"  failure detection:      "
        f"{best['failure_detection']:.1%}"
    )

    print(
        f"  unnecessary expansion: "
        f"{best['unnecessary_expansion']:.1%}"
    )

    print(
        f"  accepted precision:     "
        f"{best['accept_precision']:.1%}"
    )


# --------------------------------------------------
# Individual validation states
# --------------------------------------------------

print()
print("=" * 90)
print("MOST CONFIDENT FAILURES")
print("=" * 90)

failures = []

for row, probability, actual in zip(
    validation_rows,
    probabilities,
    y,
):
    if actual == 0:
        failures.append(
            (
                float(probability),
                row["claim"],
            )
        )

failures.sort(
    key=lambda item: item[0],
    reverse=True,
)

for probability, claim in failures[:10]:
    print()
    print(
        f"P(sufficient) = {probability:.3f}"
    )
    print(claim)