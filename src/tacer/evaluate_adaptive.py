import json
import pickle
import random
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from state import (
    compute_tacer_state,
    tokenize,
)


# --------------------------------------------------
# Frozen experimental configuration
# --------------------------------------------------

CLAIMS_PATH = Path(
    "data/normalized/scifact/claims_dev.jsonl"
)

DOCUMENTS_PATH = Path(
    "data/normalized/scifact/documents.jsonl"
)

ALIGNMENT_MODEL = Path(
    "models/alignment"
)

TACER_MODEL = Path(
    "models/tacer/candidate.pkl"
)

MSMARCO_MODEL = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

TACER_THRESHOLD = 0.55

DEPTHS = (
    5,
    10,
    20,
)

RANDOM_SEED = 42


# --------------------------------------------------
# Utilities
# --------------------------------------------------

def load_jsonl(path):
    with path.open() as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def gold_pairs_for_claim(claim):
    pairs = set()

    for evidence_set in claim.get(
        "gold_evidence",
        [],
    ):
        document_id = evidence_set[
            "document_id"
        ]

        for passage_id in evidence_set[
            "passage_ids"
        ]:
            pairs.add(
                (
                    document_id,
                    passage_id,
                )
            )

    return pairs


def candidate_is_gold(
    candidate,
    gold_pairs,
):
    return (
        candidate["document_id"],
        candidate["passage_id"],
    ) in gold_pairs


def gold_rank(
    ranked_candidates,
    gold_pairs,
):
    for rank, candidate in enumerate(
        ranked_candidates,
        start=1,
    ):
        if candidate_is_gold(
            candidate,
            gold_pairs,
        ):
            return rank

    return None


# --------------------------------------------------
# TACER feature order
#
# MUST match train.py exactly.
# --------------------------------------------------

FEATURE_NAMES = [
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


def feature_vector(features):
    return [
        features[name]
        for name in FEATURE_NAMES
    ]


# --------------------------------------------------
# Evaluation result
# --------------------------------------------------

def evaluate_ranking(
    candidates,
    gold_pairs,
):
    gold_in_candidates = any(
        candidate_is_gold(
            candidate,
            gold_pairs,
        )
        for candidate in candidates
    )

    alignment_ranked = sorted(
        candidates,
        key=lambda candidate:
            candidate["alignment_score"],
        reverse=True,
    )

    rank = gold_rank(
        alignment_ranked,
        gold_pairs,
    )

    return {
        "candidate_success":
            int(gold_in_candidates),

        "top1":
            int(
                rank is not None
                and rank <= 1
            ),

        "top3":
            int(
                rank is not None
                and rank <= 3
            ),

        "top5":
            int(
                rank is not None
                and rank <= 5
            ),

        "gold_rank":
            rank,
    }

# --------------------------------------------------
# Selective expansion merge
# --------------------------------------------------

def selective_merge(
    initial_candidates,
    expanded_candidates,
    expansion_keep,
):
    """
    Preserve the strongest initial evidence and admit
    only the strongest novel passages discovered during
    retrieval expansion.
    """

    # Rank the original depth-5 evidence.
    initial_ranked = sorted(
        initial_candidates,
        key=lambda candidate:
            candidate["alignment_score"],
        reverse=True,
    )

    # Keep the five strongest pieces of initial evidence.
    initial_selected = initial_ranked[:5]

    # Passage identity is used to determine novelty.
    initial_ids = {
        candidate["passage_id"]
        for candidate in initial_candidates
    }

    # Only passages that did not exist in the depth-5
    # candidate set count as expansion evidence.
    novel_candidates = [
        candidate
        for candidate in expanded_candidates
        if candidate["passage_id"]
        not in initial_ids
    ]

    novel_ranked = sorted(
        novel_candidates,
        key=lambda candidate:
            candidate["alignment_score"],
        reverse=True,
    )

    expansion_selected = novel_ranked[
        :expansion_keep
    ]

    # Final evidence pool is intentionally small.
    return (
        initial_selected
        + expansion_selected
    )

# --------------------------------------------------
# Load data
# --------------------------------------------------

print("Loading SciFact dev...")

claims = load_jsonl(
    CLAIMS_PATH
)

documents_list = load_jsonl(
    DOCUMENTS_PATH
)

documents = {
    document["document_id"]:
        document
    for document in documents_list
}

document_ids = list(
    documents.keys()
)

supervised_claims = [
    claim
    for claim in claims
    if gold_pairs_for_claim(claim)
]

print(
    f"Supervised claims: "
    f"{len(supervised_claims)}"
)


# --------------------------------------------------
# BM25 document index
# --------------------------------------------------

print(
    "Building document BM25 index..."
)

document_texts = []

for document_id in document_ids:
    document = documents[
        document_id
    ]

    text = (
        document["title"]
        + " "
        + " ".join(
            passage["text"]
            for passage
            in document["passages"]
        )
    )

    document_texts.append(
        text
    )


document_bm25 = BM25Okapi([
    tokenize(text)
    for text in document_texts
])


# --------------------------------------------------
# Load neural models
# --------------------------------------------------

print("Loading MS-MARCO...")

msmarco = CrossEncoder(
    MSMARCO_MODEL
)

print("Loading Alignment V1...")

alignment = CrossEncoder(
    str(ALIGNMENT_MODEL)
)

print("Loading frozen TACER...")

with TACER_MODEL.open("rb") as f:
    tacer_bundle = pickle.load(f)

tacer_model = tacer_bundle["model"]
tacer_features = tacer_bundle["features"]

if tacer_features != FEATURE_NAMES:
    raise RuntimeError(
        "TACER feature order does not match evaluator."
    )

# --------------------------------------------------
# Storage
# --------------------------------------------------

claim_results = []

fixed_results = {
    depth: []
    for depth in DEPTHS
}


# --------------------------------------------------
# Evaluate each claim
# --------------------------------------------------

for claim_index, claim in enumerate(
    supervised_claims,
    start=1,
):
    claim_text = claim["text"]

    gold_pairs = gold_pairs_for_claim(
        claim
    )

    query = tokenize(
        claim_text
    )

    # Retrieve top 20 once.
    document_scores = (
        document_bm25.get_scores(
            query
        )
    )

    ranked_document_indices = sorted(
        range(len(document_scores)),
        key=lambda index:
            float(
                document_scores[index]
            ),
        reverse=True,
    )[:max(DEPTHS)]

    # --------------------------------------------------
    # Build and score candidates at maximum depth once.
    # --------------------------------------------------

    all_candidates = []

    for document_rank, document_index in enumerate(
        ranked_document_indices,
        start=1,
    ):
        document_id = document_ids[
            document_index
        ]

        document = documents[
            document_id
        ]

        document_score = float(
            document_scores[
                document_index
            ]
        )

        for passage in document[
            "passages"
        ]:
            all_candidates.append({
                "document_id":
                    document_id,

                "passage_id":
                    passage["passage_id"],

                "text":
                    passage["text"],

                "document_score":
                    document_score,

                "document_rank":
                    document_rank,
            })

    if not all_candidates:
        continue

    pairs = [
        [
            claim_text,
            candidate["text"],
        ]
        for candidate in all_candidates
    ]

    msmarco_scores = msmarco.predict(
        pairs,
        show_progress_bar=False,
    )

    alignment_scores = alignment.predict(
        pairs,
        show_progress_bar=False,
    )

    for index, candidate in enumerate(
        all_candidates
    ):
        candidate[
            "msmarco_score"
        ] = float(
            msmarco_scores[index]
        )

        candidate[
            "alignment_score"
        ] = float(
            alignment_scores[index]
        )

    # --------------------------------------------------
    # Construct depth-specific states.
    # --------------------------------------------------

    depth_data = {}

    for depth in DEPTHS:

        candidates = [
            candidate
            for candidate
            in all_candidates
            if candidate[
                "document_rank"
            ] <= depth
        ]

        retrieved_document_scores = [
            float(
                document_scores[index]
            )
            for index
            in ranked_document_indices[
                :depth
            ]
        ]

        state = compute_tacer_state(
            claim_text=claim_text,
            retrieved_document_scores=(
                retrieved_document_scores
            ),
            candidates=candidates,
        )

        evaluation = evaluate_ranking(
            candidates,
            gold_pairs,
        )

        depth_data[depth] = {
            "candidates":
                candidates,

            "state":
                state,

            "evaluation":
                evaluation,
        }

        fixed_results[
            depth
        ].append(
            evaluation
        )

    # --------------------------------------------------
    # Frozen TACER decision at depth 5 ONLY.
    # --------------------------------------------------

    state_5 = depth_data[
        5
    ]["state"]

    x = np.asarray(
        [
            feature_vector(
                state_5["features"]
            )
        ],
        dtype=np.float64,
    )

    probability_sufficient = float(
        tacer_model.predict_proba(
            x
        )[0][1]
    )

    expand = (
        probability_sufficient
        < TACER_THRESHOLD
    )

    final_depth = (
        20
        if expand
        else 5
    )

        # --------------------------------------------------
    # Selective expansion strategies
    # --------------------------------------------------

    if expand:
        selective_3_candidates = selective_merge(
            initial_candidates=
                depth_data[5]["candidates"],
            expanded_candidates=
                depth_data[20]["candidates"],
            expansion_keep=3,
        )

        selective_5_candidates = selective_merge(
            initial_candidates=
                depth_data[5]["candidates"],
            expanded_candidates=
                depth_data[20]["candidates"],
            expansion_keep=5,
        )

    else:
        # No expansion:
        # preserve the ordinary depth-5 candidate set.
        selective_3_candidates = (
            depth_data[5]["candidates"]
        )

        selective_5_candidates = (
            depth_data[5]["candidates"]
        )

    selective_3_evaluation = evaluate_ranking(
        selective_3_candidates,
        gold_pairs,
    )

    selective_5_evaluation = evaluate_ranking(
        selective_5_candidates,
        gold_pairs,
    )

    claim_results.append({
        "claim_id":
            claim["claim_id"],

        "claim":
            claim_text,

        "probability_sufficient":
            probability_sufficient,

        "expanded":
            expand,

        "final_depth":
            final_depth,

        "passages_5":
            len(
                depth_data[
                    5
                ]["candidates"]
            ),

        "passages_20":
            len(
                depth_data[
                    20
                ]["candidates"]
            ),

        "fixed_5":
            depth_data[
                5
            ]["evaluation"],

        "fixed_10":
            depth_data[
                10
            ]["evaluation"],

        "fixed_20":
            depth_data[
                20
            ]["evaluation"],

        "adaptive":
            depth_data[
                final_depth
            ]["evaluation"],

                "selective_3":
            selective_3_evaluation,

        "selective_5":
            selective_5_evaluation,

        "selective_3_passages":
            len(
                selective_3_candidates
            ),

        "selective_5_passages":
            len(
                selective_5_candidates
            ),

        "top_alignment_5": [
            {
                "document_id":
                    candidate["document_id"],

                "passage_id":
                    candidate["passage_id"],

                "text":
                    candidate["text"],

                "alignment_score":
                    candidate["alignment_score"],

                "is_gold":
                    candidate_is_gold(
                        candidate,
                        gold_pairs,
                    ),
            }
            for candidate in sorted(
                depth_data[5]["candidates"],
                key=lambda candidate:
                    candidate["alignment_score"],
                reverse=True,
            )[:10]
        ],

        "top_alignment_20": [
            {
                "document_id":
                    candidate["document_id"],

                "passage_id":
                    candidate["passage_id"],

                "text":
                    candidate["text"],

                "alignment_score":
                    candidate["alignment_score"],

                "is_gold":
                    candidate_is_gold(
                        candidate,
                        gold_pairs,
                    ),
            }
            for candidate in sorted(
                depth_data[20]["candidates"],
                key=lambda candidate:
                    candidate["alignment_score"],
                reverse=True,
            )[:20]
        ],

    })
    if claim_index % 20 == 0:
        print(
            f"Evaluated "
            f"{claim_index} claims..."
        )


# --------------------------------------------------
# Metric helpers
# --------------------------------------------------

def summarize(
    evaluations,
):
    n = len(evaluations)

    return {
        "candidate_recall":
            sum(
                row["candidate_success"]
                for row in evaluations
            ) / n,

        "recall_1":
            sum(
                row["top1"]
                for row in evaluations
            ) / n,

        "recall_3":
            sum(
                row["top3"]
                for row in evaluations
            ) / n,

        "recall_5":
            sum(
                row["top5"]
                for row in evaluations
            ) / n,
    }


# --------------------------------------------------
# Fixed strategies
# --------------------------------------------------

summaries = {}

for depth in DEPTHS:
    summaries[
        f"fixed_{depth}"
    ] = summarize(
        fixed_results[
            depth
        ]
    )


# --------------------------------------------------
# TACER adaptive strategy
# --------------------------------------------------

adaptive_evaluations = [
    row["adaptive"]
    for row in claim_results
]

summaries[
    "tacer"
] = summarize(
    adaptive_evaluations
)

# --------------------------------------------------
# Selective merge strategies
# --------------------------------------------------

selective_3_evaluations = [
    row["selective_3"]
    for row in claim_results
]

selective_5_evaluations = [
    row["selective_5"]
    for row in claim_results
]

summaries[
    "selective_3"
] = summarize(
    selective_3_evaluations
)

summaries[
    "selective_5"
] = summarize(
    selective_5_evaluations
)

average_selective_3_passages = (
    sum(
        row["selective_3_passages"]
        for row in claim_results
    )
    / len(claim_results)
)

average_selective_5_passages = (
    sum(
        row["selective_5_passages"]
        for row in claim_results
    )
    / len(claim_results)
)


expanded_indices = [
    index
    for index, row
    in enumerate(claim_results)
    if row["expanded"]
]

expansion_rate = (
    len(expanded_indices)
    / len(claim_results)
)

average_documents = (
    sum(
        row["final_depth"]
        for row in claim_results
    )
    / len(claim_results)
)

average_passages = (
    sum(
        (
            row["passages_20"]
            if row["expanded"]
            else row["passages_5"]
        )
        for row in claim_results
    )
    / len(claim_results)
)


# --------------------------------------------------
# Random equal-budget baseline
# --------------------------------------------------

rng = random.Random(
    RANDOM_SEED
)

random_indices = set(
    rng.sample(
        range(
            len(claim_results)
        ),
        len(expanded_indices),
    )
)

random_evaluations = []

for index, row in enumerate(
    claim_results
):
    if index in random_indices:
        random_evaluations.append(
            row["fixed_20"]
        )
    else:
        random_evaluations.append(
            row["fixed_5"]
        )

summaries[
    "random"
] = summarize(
    random_evaluations
)


# --------------------------------------------------
# Routing diagnostics
# --------------------------------------------------

top5_failures = [
    row
    for row in claim_results
    if not row[
        "fixed_5"
    ]["candidate_success"]
]

top5_successes = [
    row
    for row in claim_results
    if row[
        "fixed_5"
    ]["candidate_success"]
]

detected_failures = sum(
    row["expanded"]
    for row in top5_failures
)

unnecessary_expansions = sum(
    row["expanded"]
    for row in top5_successes
)

rescuable_failures = [
    row
    for row in top5_failures
    if row[
        "fixed_20"
    ]["candidate_success"]
]

rescued_by_tacer = sum(
    row["expanded"]
    for row in rescuable_failures
)


# --------------------------------------------------
# Report
# --------------------------------------------------

print()
print("=" * 90)
print(
    "TACER ADAPTIVE RETRIEVAL — "
    "SCIFACT DEV"
)
print("=" * 90)

print()
print(
    f"Claims evaluated: "
    f"{len(claim_results)}"
)

print(
    f"Frozen threshold: "
    f"{TACER_THRESHOLD:.2f}"
)

print()
print(
    f"{'STRATEGY':<18}"
    f"{'CAND.R':>10}"
    f"{'R@1':>10}"
    f"{'R@3':>10}"
    f"{'R@5':>10}"
    f"{'AVG DOCS':>12}"
)

print("-" * 70)


def print_strategy(
    name,
    summary,
    avg_docs,
):
    print(
        f"{name:<18}"
        f"{summary['candidate_recall']:>10.3f}"
        f"{summary['recall_1']:>10.3f}"
        f"{summary['recall_3']:>10.3f}"
        f"{summary['recall_5']:>10.3f}"
        f"{avg_docs:>12.2f}"
    )


print_strategy(
    "Fixed-5",
    summaries["fixed_5"],
    5.0,
)

print_strategy(
    "Fixed-10",
    summaries["fixed_10"],
    10.0,
)

print_strategy(
    "Fixed-20",
    summaries["fixed_20"],
    20.0,
)

print_strategy(
    "Random-budget",
    summaries["random"],
    average_documents,
)

print_strategy(
    "TACER 5→20",
    summaries["tacer"],
    average_documents,
)

print_strategy(
    "Selective-3",
    summaries["selective_3"],
    average_documents,
)

print_strategy(
    "Selective-5",
    summaries["selective_5"],
    average_documents,
)

print()
print("=" * 90)
print("TACER ROUTING")
print("=" * 90)

print(
    f"Expanded claims:             "
    f"{len(expanded_indices)}"
    f"/{len(claim_results)} "
    f"({expansion_rate:.1%})"
)

print(
    f"Average documents:           "
    f"{average_documents:.2f}"
)

print(
    f"Average candidate passages:  "
    f"{average_passages:.2f}"
)

print()
print("SELECTIVE MERGE")
print("-" * 40)

print(
    f"Selective-3 final passages:  "
    f"{average_selective_3_passages:.2f}"
)

print(
    f"Selective-5 final passages:  "
    f"{average_selective_5_passages:.2f}"
)

print()
print(
    f"Top-5 retrieval failures:    "
    f"{len(top5_failures)}"
)

if top5_failures:
    print(
        f"Failures detected:           "
        f"{detected_failures}"
        f"/{len(top5_failures)} "
        f"("
        f"{detected_failures / len(top5_failures):.1%}"
        f")"
    )

print(
    f"Top-5 retrieval successes:   "
    f"{len(top5_successes)}"
)

if top5_successes:
    print(
        f"Unnecessary expansions:      "
        f"{unnecessary_expansions}"
        f"/{len(top5_successes)} "
        f"("
        f"{unnecessary_expansions / len(top5_successes):.1%}"
        f")"
    )

print()
print(
    f"Failures rescuable @20:      "
    f"{len(rescuable_failures)}"
)

if rescuable_failures:
    print(
        f"Rescuable failures expanded: "
        f"{rescued_by_tacer}"
        f"/{len(rescuable_failures)} "
        f"("
        f"{rescued_by_tacer / len(rescuable_failures):.1%}"
        f")"
    )


# --------------------------------------------------
# Save detailed results
# --------------------------------------------------

RESULTS_DIR = Path(
    "results/tacer"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULTS_PATH = (
    RESULTS_DIR
    / "adaptive_v1_scifact_dev.json"
)

with RESULTS_PATH.open(
    "w"
) as f:
    json.dump(
        {
            "threshold":
                TACER_THRESHOLD,

            "expansion_rate":
                expansion_rate,

            "average_documents":
                average_documents,

            "average_passages":
                average_passages,

            "summaries":
                summaries,

            "claims":
                claim_results,
        },
        f,
        indent=2,
    )

print()
print(
    f"Detailed results saved to: "
    f"{RESULTS_PATH}"
)