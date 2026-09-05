import json
import random
from pathlib import Path

from rank_bm25 import BM25Okapi


# --------------------------------------------------
# Configuration
# --------------------------------------------------

CLAIMS_PATH = Path(
    "data/normalized/scifact/claims_train.jsonl"
)

DOCUMENTS_PATH = Path(
    "data/normalized/scifact/documents.jsonl"
)

OUTPUT_PATH = Path(
    "data/eval/evidence_selector_train.jsonl"
)

RANDOM_SEED = 42

TOP_DOCUMENTS = 10

# Number of each negative type to keep per claim.
SAME_DOCUMENT_NEGATIVES = 2
RETRIEVAL_HARD_NEGATIVES = 4


random.seed(RANDOM_SEED)


# --------------------------------------------------
# Utilities
# --------------------------------------------------

def load_jsonl(path):
    with open(path) as f:
        return [
            json.loads(line)
            for line in f
        ]


def tokenize(text):
    return (
        text.lower()
        .replace(".", " ")
        .replace(",", " ")
        .replace(":", " ")
        .replace(";", " ")
        .split()
    )


# --------------------------------------------------
# Load corpus
# --------------------------------------------------

print("Loading SciFact...")

claims = load_jsonl(CLAIMS_PATH)
documents_list = load_jsonl(DOCUMENTS_PATH)

documents = {
    doc["document_id"]: doc
    for doc in documents_list
}

document_ids = list(documents.keys())

print(f"Training claims: {len(claims)}")
print(f"Documents: {len(documents)}")


# --------------------------------------------------
# Build corpus-level BM25
# --------------------------------------------------

print("Building BM25 document index...")

document_texts = []

for document_id in document_ids:
    document = documents[document_id]

    text = (
        document["title"]
        + " "
        + " ".join(
            passage["text"]
            for passage in document["passages"]
        )
    )

    document_texts.append(text)


bm25 = BM25Okapi([
    tokenize(text)
    for text in document_texts
])


# --------------------------------------------------
# Build examples
# --------------------------------------------------

examples = []

positive_count = 0
same_document_negative_count = 0
retrieval_negative_count = 0

claims_used = 0


for claim_number, claim in enumerate(claims):

    gold_evidence = claim.get("gold_evidence", [])

    if not gold_evidence:
        continue

    claim_text = claim["text"]

    # --------------------------------------------------
    # Collect exact gold evidence pairs
    #
    # (document_id, passage_id)
    # --------------------------------------------------

    gold_pairs = set()

    for evidence_set in gold_evidence:
        document_id = evidence_set["document_id"]

        for passage_id in evidence_set["passage_ids"]:
            gold_pairs.add(
                (document_id, passage_id)
            )

    if not gold_pairs:
        continue

    claims_used += 1

    gold_document_ids = {
        document_id
        for document_id, _ in gold_pairs
    }

    # --------------------------------------------------
    # POSITIVES
    # --------------------------------------------------

    for document_id, passage_id in gold_pairs:

        document = documents[document_id]

        matching_passage = None

        for passage in document["passages"]:
            if passage["passage_id"] == passage_id:
                matching_passage = passage
                break

        if matching_passage is None:
            continue

        examples.append({
            "claim": claim_text,
            "passage": matching_passage["text"],
            "label": 1,
            "negative_type": None,
            "document_id": document_id,
            "passage_id": passage_id,
        })

        positive_count += 1

    # --------------------------------------------------
    # NEGATIVE TYPE A:
    # non-evidence passages from gold documents
    #
    # These are topically very similar but aren't
    # annotated evidence for the claim.
    # --------------------------------------------------

    same_document_candidates = []

    for document_id in gold_document_ids:

        document = documents[document_id]

        for passage in document["passages"]:

            pair = (
                document_id,
                passage["passage_id"],
            )

            if pair in gold_pairs:
                continue

            same_document_candidates.append({
                "document_id": document_id,
                "passage_id": passage["passage_id"],
                "text": passage["text"],
            })

    random.shuffle(same_document_candidates)

    for candidate in same_document_candidates[
        :SAME_DOCUMENT_NEGATIVES
    ]:

        examples.append({
            "claim": claim_text,
            "passage": candidate["text"],
            "label": 0,
            "negative_type": "same_document",
            "document_id": candidate["document_id"],
            "passage_id": candidate["passage_id"],
        })

        same_document_negative_count += 1

    # --------------------------------------------------
    # NEGATIVE TYPE B:
    # retrieval hard negatives
    #
    # Search the entire corpus using ONLY the claim.
    #
    # Gold information is used only afterward to
    # exclude actual evidence from the negative pool.
    # --------------------------------------------------

    query = tokenize(claim_text)

    document_scores = bm25.get_scores(query)

    ranked_document_indices = sorted(
        range(len(document_scores)),
        key=lambda i: document_scores[i],
        reverse=True,
    )[:TOP_DOCUMENTS]

    retrieval_candidates = []

    for document_index in ranked_document_indices:

        document_id = document_ids[document_index]
        document = documents[document_id]

        # Don't use passages from gold documents here.
        # Type A already handles those.
        if document_id in gold_document_ids:
            continue

        for passage in document["passages"]:

            pair = (
                document_id,
                passage["passage_id"],
            )

            if pair in gold_pairs:
                continue

            retrieval_candidates.append({
                "document_id": document_id,
                "passage_id": passage["passage_id"],
                "text": passage["text"],
            })

    # --------------------------------------------------
    # Rank the negative passages themselves with BM25.
    #
    # This deliberately chooses the passages that look
    # MOST relevant to the claim.
    # --------------------------------------------------

    if retrieval_candidates:

        passage_bm25 = BM25Okapi([
            tokenize(candidate["text"])
            for candidate in retrieval_candidates
        ])

        passage_scores = passage_bm25.get_scores(query)

        ranked_retrieval_candidates = sorted(
            zip(
                retrieval_candidates,
                passage_scores,
            ),
            key=lambda x: x[1],
            reverse=True,
        )

        selected = ranked_retrieval_candidates[
            :RETRIEVAL_HARD_NEGATIVES
        ]

        for candidate, retrieval_score in selected:

            examples.append({
                "claim": claim_text,
                "passage": candidate["text"],
                "label": 0,
                "negative_type": "retrieval_hard",
                "document_id": candidate["document_id"],
                "passage_id": candidate["passage_id"],
                "retrieval_score": float(
                    retrieval_score
                ),
            })

            retrieval_negative_count += 1

    if claims_used % 100 == 0:
        print(
            f"Processed {claims_used} evidence-bearing claims..."
        )


# --------------------------------------------------
# Shuffle and save
# --------------------------------------------------

random.shuffle(examples)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with open(OUTPUT_PATH, "w") as f:
    for example in examples:
        f.write(
            json.dumps(example) + "\n"
        )


# --------------------------------------------------
# Report
# --------------------------------------------------

negative_count = (
    same_document_negative_count
    + retrieval_negative_count
)

print()
print("=" * 60)
print("EVIDENCE SELECTOR TRAINING DATA")
print("=" * 60)

print(f"Claims used:              {claims_used}")
print(f"Total examples:           {len(examples)}")
print(f"Positive evidence:        {positive_count}")
print(
    f"Same-document negatives:  "
    f"{same_document_negative_count}"
)
print(
    f"Retrieval hard negatives: "
    f"{retrieval_negative_count}"
)
print(f"Total negatives:          {negative_count}")

if positive_count:
    print(
        f"Negative / positive:      "
        f"{negative_count / positive_count:.2f}"
    )

print()
print(f"Saved to: {OUTPUT_PATH}")