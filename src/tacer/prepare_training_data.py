import argparse
import json
from pathlib import Path

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from state import (
    compute_tacer_state,
    tokenize,
)

# --------------------------------------------------
# Configuration
# --------------------------------------------------

# --------------------------------------------------
# Command-line configuration
# --------------------------------------------------

parser = argparse.ArgumentParser(
    description=(
        "Generate TACER retrieval-state data "
        "from a SciFact split."
    )
)

parser.add_argument(
    "--split",
    choices=[
        "train",
        "dev",
    ],
    default="train",
)

args = parser.parse_args()


CLAIMS_PATH = Path(
    f"data/normalized/scifact/"
    f"claims_{args.split}.jsonl"
)

DOCUMENTS_PATH = Path(
    "data/normalized/scifact/documents.jsonl"
)

OUTPUT_PATH = Path(
    f"data/tacer/{args.split}.jsonl"
)
MSMARCO_MODEL = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

ALIGNMENT_MODEL = Path(
    "models/alignment"
)

# Initial local retrieval stage.
TOP_DOCUMENTS = 5

# These cutoffs are labels / diagnostics only.
TOP_K_VALUES = (1, 3, 5)


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


# --------------------------------------------------
# Gold utilities
# --------------------------------------------------

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
# Load data
# --------------------------------------------------

print("Loading SciFact...")

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

print(
    f"Claims: {len(claims)}"
)

print(
    f"Documents: {len(documents)}"
)


# --------------------------------------------------
# Build document BM25
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
# Load rerankers
# --------------------------------------------------

print(
    "Loading MS-MARCO..."
)

msmarco = CrossEncoder(
    MSMARCO_MODEL
)

if not ALIGNMENT_MODEL.exists():
    raise FileNotFoundError(
        "Could not find learned alignment model at "
        f"{ALIGNMENT_MODEL}"
    )

print(
    "Loading Alignment V1..."
)

alignment = CrossEncoder(
    str(ALIGNMENT_MODEL)
)


# --------------------------------------------------
# Generate TACER states
# --------------------------------------------------

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

rows = []

claims_used = 0
claims_skipped_no_gold = 0

candidate_successes = 0
candidate_failures = 0

alignment_top1_successes = 0
alignment_top3_successes = 0
alignment_top5_successes = 0


for claim in claims:

    gold_pairs = gold_pairs_for_claim(
        claim
    )

    # --------------------------------------------------
    # Important:
    #
    # SciFact insufficient-evidence claims have no
    # annotated target passage.
    #
    # They cannot cleanly tell us whether retrieval
    # succeeded or failed, so V1 excludes them.
    # --------------------------------------------------

    if not gold_pairs:
        claims_skipped_no_gold += 1
        continue

    claim_text = claim["text"]

    query = tokenize(
        claim_text
    )

    # --------------------------------------------------
    # Stage 1: document retrieval
    # --------------------------------------------------

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
    )[:TOP_DOCUMENTS]

    retrieved_document_scores = [
        float(
            document_scores[index]
        )
        for index
        in ranked_document_indices
    ]

    # --------------------------------------------------
    # Stage 2: candidate passages
    # --------------------------------------------------

    candidates = []

    for document_index in (
        ranked_document_indices
    ):
        document_id = (
            document_ids[
                document_index
            ]
        )

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
            candidates.append({
                "document_id":
                    document_id,

                "passage_id":
                    passage[
                        "passage_id"
                    ],

                "text":
                    passage["text"],

                "document_score":
                    document_score,
            })

    if not candidates:
        continue

    # --------------------------------------------------
    # Stage 3: model scores
    # --------------------------------------------------

    pairs = [
        [
            claim_text,
            candidate["text"],
        ]
        for candidate in candidates
    ]

    msmarco_scores = (
        msmarco.predict(
            pairs,
            show_progress_bar=False,
        )
    )

    alignment_scores = (
        alignment.predict(
            pairs,
            show_progress_bar=False,
        )
    )

    for index, candidate in enumerate(
        candidates
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
    # Observable TACER state
    # --------------------------------------------------

    tacer_state = compute_tacer_state(
        claim_text=claim_text,
        retrieved_document_scores=(
            retrieved_document_scores
        ),
        candidates=candidates,
    )

    features = tacer_state[
        "features"
    ]

    msmarco_ranked = tacer_state[
        "msmarco_ranked"
    ]

    alignment_ranked = tacer_state[
        "alignment_ranked"
    ]

    # --------------------------------------------------
    # Gold labels -- ONLY NOW
    # --------------------------------------------------

    gold_in_candidates = int(
        any(
            candidate_is_gold(
                candidate,
                gold_pairs,
            )
            for candidate
            in candidates
        )
    )

    alignment_gold_rank = gold_rank(
        alignment_ranked,
        gold_pairs,
    )

    msmarco_gold_rank = gold_rank(
        msmarco_ranked,
        gold_pairs,
    )

    gold_top1 = int(
        alignment_gold_rank is not None
        and alignment_gold_rank <= 1
    )

    gold_top3 = int(
        alignment_gold_rank is not None
        and alignment_gold_rank <= 3
    )

    gold_top5 = int(
        alignment_gold_rank is not None
        and alignment_gold_rank <= 5
    )

    # --------------------------------------------------
    # Store one retrieval state per claim
    # --------------------------------------------------

    row = {
        "claim_id":
            claim["claim_id"],

        "claim":
            claim_text,

        "gold_label":
            claim["gold_label"],

        "candidate_count":
            len(candidates),

        **features,

        # Targets / diagnostics.
        #
        # These fields must NEVER become runtime
        # features.
        "gold_in_candidates":
            gold_in_candidates,

        "gold_top1":
            gold_top1,

        "gold_top3":
            gold_top3,

        "gold_top5":
            gold_top5,

        "alignment_gold_rank":
            alignment_gold_rank,

        "msmarco_gold_rank":
            msmarco_gold_rank,
    }

    rows.append(
        row
    )

    claims_used += 1

    if gold_in_candidates:
        candidate_successes += 1
    else:
        candidate_failures += 1

    alignment_top1_successes += (
        gold_top1
    )

    alignment_top3_successes += (
        gold_top3
    )

    alignment_top5_successes += (
        gold_top5
    )

    if claims_used % 50 == 0:
        print(
            f"Processed {claims_used} "
            "supervised claims..."
        )


# --------------------------------------------------
# Save
# --------------------------------------------------

with OUTPUT_PATH.open(
    "w"
) as f:

    for row in rows:
        f.write(
            json.dumps(
                row,
                ensure_ascii=False,
            )
            + "\n"
        )


# --------------------------------------------------
# Summary
# --------------------------------------------------

print()
print("=" * 60)
print("TACER SUFFICIENCY TRAINING DATA")
print("=" * 60)

print(
    f"Claims available:          {len(claims)}"
)

print(
    f"Claims used:               {claims_used}"
)

print(
    "Skipped without "
    f"gold evidence:             "
    f"{claims_skipped_no_gold}"
)

print()
print("INITIAL LOCAL RETRIEVAL")
print("-" * 60)

print(
    f"Gold in candidate set:     "
    f"{candidate_successes}"
)

print(
    f"Gold absent from candidates: "
    f"{candidate_failures}"
)

if claims_used:

    print(
        "Candidate recall:          "
        f"{candidate_successes / claims_used:.3f}"
    )

    print()
    print("ALIGNMENT V1")
    print("-" * 60)

    print(
        "Gold Recall@1:             "
        f"{alignment_top1_successes / claims_used:.3f}"
    )

    print(
        "Gold Recall@3:             "
        f"{alignment_top3_successes / claims_used:.3f}"
    )

    print(
        "Gold Recall@5:             "
        f"{alignment_top5_successes / claims_used:.3f}"
    )

print()
print(
    f"Saved to: {OUTPUT_PATH}"
)