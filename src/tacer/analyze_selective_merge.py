import json
from pathlib import Path
from collections import Counter


RESULTS_PATH = Path(
    "results/tacer/adaptive_v1_scifact_dev.json"
)


# --------------------------------------------------
# Load results
# --------------------------------------------------

with RESULTS_PATH.open() as f:
    data = json.load(f)

claims = data["claims"]


# --------------------------------------------------
# Counters
# --------------------------------------------------

counts = Counter()

lost_cases = []


# --------------------------------------------------
# Compare TACER vs selective merge
# --------------------------------------------------

for row in claims:

    fixed5 = row["fixed_5"]
    adaptive = row["adaptive"]
    selective3 = row["selective_3"]
    selective5 = row["selective_5"]

    counts["claims"] += 1

    if row["expanded"]:
        counts["expanded"] += 1

    # ----------------------------------------------
    # Candidate recall changes
    # ----------------------------------------------

    if (
        adaptive["candidate_success"]
        and not selective3["candidate_success"]
    ):
        counts["lost_candidate_s3"] += 1

    if (
        adaptive["candidate_success"]
        and not selective5["candidate_success"]
    ):
        counts["lost_candidate_s5"] += 1

    # ----------------------------------------------
    # Cases where gold existed before expansion
    # but Selective-3 lost it
    # ----------------------------------------------

    if (
        fixed5["candidate_success"]
        and not selective3["candidate_success"]
    ):

        initial_gold_ranks = [
            rank
            for rank, candidate
            in enumerate(
                row["top_alignment_5"],
                start=1,
            )
            if candidate["is_gold"]
        ]

        gold_rank = (
            initial_gold_ranks[0]
            if initial_gold_ranks
            else None
        )

        if gold_rank is None:
            reason = (
                "gold_below_logged_top10"
            )

        elif gold_rank > 5:
            reason = (
                "gold_below_initial_top5"
            )

        else:
            reason = (
                "unexpected_loss"
            )

        counts[reason] += 1

        lost_cases.append({
            "claim_id":
                row["claim_id"],

            "claim":
                row["claim"],

            "gold_rank_initial":
                gold_rank,

            "reason":
                reason,

            "probability_sufficient":
                row["probability_sufficient"],
        })


# --------------------------------------------------
# Ranking comparison
# --------------------------------------------------

for k in (1, 3, 5):

    key = f"top{k}"

    for selective_name in (
        "selective_3",
        "selective_5",
    ):

        adaptive_success = (
            row_success
            for row_success in []
        )

        improved = sum(
            (
                not row["adaptive"][key]
                and row[selective_name][key]
            )
            for row in claims
        )

        harmed = sum(
            (
                row["adaptive"][key]
                and not row[selective_name][key]
            )
            for row in claims
        )

        unchanged = (
            len(claims)
            - improved
            - harmed
        )

        counts[
            f"{selective_name}_{key}_improved"
        ] = improved

        counts[
            f"{selective_name}_{key}_harmed"
        ] = harmed

        counts[
            f"{selective_name}_{key}_unchanged"
        ] = unchanged


# --------------------------------------------------
# Print summary
# --------------------------------------------------

print()
print("=" * 78)
print("SELECTIVE MERGE V1 DIAGNOSTIC")
print("=" * 78)

print()
print(
    f"Claims:                         "
    f"{counts['claims']}"
)

print(
    f"Expanded:                       "
    f"{counts['expanded']}"
)

print()
print("CANDIDATE RECALL LOSSES")
print("-" * 50)

print(
    f"TACER → Selective-3 losses:     "
    f"{counts['lost_candidate_s3']}"
)

print(
    f"TACER → Selective-5 losses:     "
    f"{counts['lost_candidate_s5']}"
)

print()
print("WHY SELECTIVE-3 LOST INITIAL GOLD")
print("-" * 50)

print(
    f"Gold below initial top 5:       "
    f"{counts['gold_below_initial_top5']}"
)

print(
    f"Gold below logged top 10:       "
    f"{counts['gold_below_logged_top10']}"
)

print(
    f"Unexpected losses:              "
    f"{counts['unexpected_loss']}"
)


print()
print("PAIRWISE RANKING CHANGES VS TACER")
print("-" * 78)

print(
    f"{'STRATEGY':<15}"
    f"{'METRIC':<10}"
    f"{'BETTER':>10}"
    f"{'WORSE':>10}"
    f"{'SAME':>10}"
)

for strategy in (
    "selective_3",
    "selective_5",
):
    for k in (1, 3, 5):

        key = f"top{k}"

        print(
            f"{strategy:<15}"
            f"{'R@' + str(k):<10}"
            f"{counts[f'{strategy}_{key}_improved']:>10}"
            f"{counts[f'{strategy}_{key}_harmed']:>10}"
            f"{counts[f'{strategy}_{key}_unchanged']:>10}"
        )


# --------------------------------------------------
# Print lost cases
# --------------------------------------------------

print()
print("=" * 78)
print("INITIAL EVIDENCE LOST BY SELECTIVE-3")
print("=" * 78)

for index, case in enumerate(
    lost_cases,
    start=1,
):
    print()
    print("-" * 78)

    print(
        f"{index}. {case['claim_id']}"
    )

    print(
        case["claim"]
    )

    print(
        "Initial gold alignment rank: "
        f"{case['gold_rank_initial']}"
    )

    print(
        "Reason: "
        f"{case['reason']}"
    )

    print(
        "TACER P(sufficient): "
        f"{case['probability_sufficient']:.3f}"
    )


# --------------------------------------------------
# Save compact diagnostic
# --------------------------------------------------

OUTPUT_PATH = Path(
    "results/tacer/"
    "selective_merge_v1_diagnostic.json"
)

output = {
    "counts":
        dict(counts),

    "lost_cases":
        lost_cases,
}

with OUTPUT_PATH.open("w") as f:
    json.dump(
        output,
        f,
        indent=2,
    )

print()
print(
    f"Saved to: {OUTPUT_PATH}"
)