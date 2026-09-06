import json
from pathlib import Path


RESULTS_PATH = Path(
    "results/tacer/adaptive_v1_scifact_dev.json"
)


# --------------------------------------------------
# Load adaptive evaluation
# --------------------------------------------------

with RESULTS_PATH.open() as f:
    data = json.load(f)

claims = data["claims"]


# --------------------------------------------------
# Find rescuable top-5 failures
#
# Gold absent at depth 5,
# but present at depth 20.
# --------------------------------------------------

rescuable = [
    row
    for row in claims
    if (
        not row["fixed_5"]["candidate_success"]
        and row["fixed_20"]["candidate_success"]
    )
]


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def first_gold(candidates):
    for rank, candidate in enumerate(
        candidates,
        start=1,
    ):
        if candidate["is_gold"]:
            return rank, candidate

    return None, None


def first_false(candidates):
    for rank, candidate in enumerate(
        candidates,
        start=1,
    ):
        if not candidate["is_gold"]:
            return rank, candidate

    return None, None


def rank_bucket(rank):
    if rank is None:
        return ">20 / missing"

    if rank == 1:
        return "1"

    if rank <= 3:
        return "2-3"

    if rank <= 5:
        return "4-5"

    if rank <= 10:
        return "6-10"

    return "11-20"


# --------------------------------------------------
# Aggregate diagnostics
# --------------------------------------------------

buckets = {
    "1": 0,
    "2-3": 0,
    "4-5": 0,
    "6-10": 0,
    "11-20": 0,
    ">20 / missing": 0,
}

expanded = 0
gold_top5_after_expansion = 0
gold_top3_after_expansion = 0
gold_top1_after_expansion = 0

score_margins = []


print()
print("=" * 80)
print("TACER EXPANSION FAILURE ANALYSIS")
print("=" * 80)

print()
print(
    f"Rescuable top-5 failures: "
    f"{len(rescuable)}"
)


# --------------------------------------------------
# Individual cases
# --------------------------------------------------

for case_index, row in enumerate(
    rescuable,
    start=1,
):
    ranked_20 = row[
        "top_alignment_20"
    ]

    gold_rank, gold = first_gold(
        ranked_20
    )

    false_rank, false = first_false(
        ranked_20
    )

    bucket = rank_bucket(
        gold_rank
    )

    buckets[bucket] += 1

    if row["expanded"]:
        expanded += 1

    if gold_rank is not None:
        if gold_rank <= 1:
            gold_top1_after_expansion += 1

        if gold_rank <= 3:
            gold_top3_after_expansion += 1

        if gold_rank <= 5:
            gold_top5_after_expansion += 1

    margin = None

    if (
        gold is not None
        and false is not None
    ):
        margin = (
            gold["alignment_score"]
            - false["alignment_score"]
        )

        score_margins.append(
            margin
        )

    print()
    print("=" * 80)
    print(
        f"CASE {case_index}"
    )
    print("=" * 80)

    print()
    print("CLAIM:")
    print(
        row["claim"]
    )

    print()
    print(
        "TACER P(sufficient): "
        f"{row['probability_sufficient']:.3f}"
    )

    print(
        "Expanded: "
        f"{'YES' if row['expanded'] else 'NO'}"
    )

    print()
    print(
        "Gold Alignment rank @20: "
        f"{gold_rank}"
    )

    if gold is not None:
        print(
            "Gold Alignment score:   "
            f"{gold['alignment_score']:.3f}"
        )

        print()
        print("GOLD EVIDENCE:")
        print(
            gold["text"]
        )

    if false is not None:
        print()
        print(
            "Highest false rank:      "
            f"{false_rank}"
        )

        print(
            "Highest false score:     "
            f"{false['alignment_score']:.3f}"
        )

        if margin is not None:
            print(
                "Gold - false margin:   "
                f"{margin:+.3f}"
            )

        print()
        print("TOP FALSE EVIDENCE:")
        print(
            false["text"]
        )

    print()
    print(
        "OUTCOME: ",
        end="",
    )

    if not row["expanded"]:
        print(
            "ROUTING MISS "
            "(TACER did not expand)"
        )

    elif gold_rank == 1:
        print(
            "FULL RESCUE "
            "(gold becomes rank 1)"
        )

    elif (
        gold_rank is not None
        and gold_rank <= 5
    ):
        print(
            "RETRIEVAL RESCUE + "
            "RANKING PARTIAL SUCCESS"
        )

    else:
        print(
            "RETRIEVAL RESCUE + "
            "RANKING DILUTION"
        )


# --------------------------------------------------
# Summary
# --------------------------------------------------

print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)

print()
print(
    f"Rescuable failures:          "
    f"{len(rescuable)}"
)

print(
    f"TACER expanded:              "
    f"{expanded}/{len(rescuable)}"
)

print()
print("GOLD ALIGNMENT RANK @20")
print("-" * 40)

for bucket, count in buckets.items():
    print(
        f"{bucket:<15}"
        f"{count:>5}"
    )

print()
print(
    f"Gold reaches R@1:            "
    f"{gold_top1_after_expansion}"
    f"/{len(rescuable)}"
)

print(
    f"Gold reaches R@3:            "
    f"{gold_top3_after_expansion}"
    f"/{len(rescuable)}"
)

print(
    f"Gold reaches R@5:            "
    f"{gold_top5_after_expansion}"
    f"/{len(rescuable)}"
)

if score_margins:
    mean_margin = (
        sum(score_margins)
        / len(score_margins)
    )

    print()
    print(
        "Mean gold-vs-top-false "
        f"margin: {mean_margin:+.3f}"
    )


# --------------------------------------------------
# Compact machine-readable output
# --------------------------------------------------

OUTPUT_PATH = Path(
    "results/tacer/"
    "expansion_analysis_v1.json"
)

summary = {
    "rescuable_failures":
        len(rescuable),

    "expanded":
        expanded,

    "gold_rank_buckets":
        buckets,

    "gold_r1":
        gold_top1_after_expansion,

    "gold_r3":
        gold_top3_after_expansion,

    "gold_r5":
        gold_top5_after_expansion,

    "mean_gold_false_margin":
        (
            sum(score_margins)
            / len(score_margins)
            if score_margins
            else None
        ),
}

with OUTPUT_PATH.open(
    "w"
) as f:
    json.dump(
        summary,
        f,
        indent=2,
    )

print()
print(
    f"Summary saved to: "
    f"{OUTPUT_PATH}"
)