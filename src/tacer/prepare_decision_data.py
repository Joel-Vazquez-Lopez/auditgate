import argparse
import json
from pathlib import Path

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

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
    f"data/tacer/decision_{args.split}.jsonl"
)

MSMARCO_MODEL = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

ALIGNMENT_MODEL = Path(
    "models/alignment"
)

VERIFIER_MODEL = Path(
    "models/scifact-verifier"
)

VERIFIER_THRESHOLD = 0.90

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

if not VERIFIER_MODEL.exists():
    raise FileNotFoundError(
        "Could not find SciFact verifier at "
        f"{VERIFIER_MODEL}"
    )

print(
    "Loading SciFact verifier..."
)

verifier_tokenizer = (
    AutoTokenizer.from_pretrained(
        str(VERIFIER_MODEL)
    )
)

verifier = (
    AutoModelForSequenceClassification
    .from_pretrained(
        str(VERIFIER_MODEL)
    )
)

verifier.eval()

# --------------------------------------------------
# Generate TACER states
# --------------------------------------------------

def verify_passages(
    claim_text,
    candidates,
):
    evidence_texts = [
        candidate["text"]
        for candidate in candidates
    ]

    if not evidence_texts:
        return []

    claim_texts = [
        claim_text
    ] * len(evidence_texts)

    inputs = verifier_tokenizer(
        evidence_texts,
        claim_texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    )

    with torch.no_grad():
        logits = verifier(
            **inputs
        ).logits

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

    results = []

    for probs in probabilities:
        support = float(
            probs[0].item()
        )

        contradiction = float(
            probs[1].item()
        )

        results.append({
            "supported": support,
            "contradicted": contradiction,
        })

    return results

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

    verification_candidates = (
    alignment_ranked[:5]
    )

    verification_results = verify_passages(
        claim_text,
        verification_candidates,
    )


    support_scores = [
    result["supported"]
    for result in verification_results
    ]

    contradiction_scores = [
        result["contradicted"]
        for result in verification_results
    ]

    max_support = max(
        support_scores,
        default=0.0,
    )

    max_contradiction = max(
        contradiction_scores,
        default=0.0,
    )

    mean_support = (
        sum(support_scores)
        / len(support_scores)
        if support_scores
        else 0.0
    )

    mean_contradiction = (
        sum(contradiction_scores)
        / len(contradiction_scores)
        if contradiction_scores
        else 0.0
    )

    strong_support_count = sum(
        score >= VERIFIER_THRESHOLD
        for score in support_scores
    )

    strong_contradiction_count = sum(
        score >= VERIFIER_THRESHOLD
        for score in contradiction_scores
    )

    verifier_margin = abs(
        max_support
        - max_contradiction
    )

    # --------------------------------------------------
    # Current AuditGate decision from verifier state
    # --------------------------------------------------

    strong_support = (
        max_support
        >= VERIFIER_THRESHOLD
    )

    strong_contradiction = (
        max_contradiction
        >= VERIFIER_THRESHOLD
    )

    if (
        strong_support
        and not strong_contradiction
    ):
        system_decision = "ALLOW"

    elif (
        strong_contradiction
        and not strong_support
    ):
        system_decision = "BLOCK"

    else:
        system_decision = "REVIEW"

# --------------------------------------------------
# TACER-B supervision target
# --------------------------------------------------
#
# Gold information is used ONLY to determine whether
# the runtime decision state was actually sufficient.
# It must never become a TACER-B input feature.
# --------------------------------------------------

    gold_label = claim[
        "gold_label"
    ]

    decision_sufficient = int(
        (
            system_decision == "ALLOW"
            and gold_label == "supported"
        )
        or
        (
            system_decision == "BLOCK"
            and gold_label == "contradicted"
        )
    )


    # --------------------------------------------------
    # Gold diagnostics
    # --------------------------------------------------
    #
    # Used only for evaluation / analysis.
    # Never use these as runtime TACER-B features.
    # --------------------------------------------------

    gold_in_candidates = int(
        any(
            candidate_is_gold(
                candidate,
                gold_pairs,
            )
            for candidate in candidates
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

                # --------------------------------------------------
        # TACER-B verifier-state features
        # --------------------------------------------------

        "verifier_max_support":
            max_support,

        "verifier_max_contradiction":
            max_contradiction,

        "verifier_mean_support":
            mean_support,

        "verifier_mean_contradiction":
            mean_contradiction,

        "verifier_margin":
            verifier_margin,

        "verifier_strong_support_count":
            strong_support_count,

        "verifier_strong_contradiction_count":
            strong_contradiction_count,

        # Current AuditGate decision.
        # Diagnostic only.
        "system_decision":
            system_decision,

        # TACER-B supervision target.
        # NEVER use this as an input feature.
        "decision_sufficient":
            decision_sufficient,

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