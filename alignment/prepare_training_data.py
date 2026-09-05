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

OUTPUT_DIR = Path(
    "data/alignment"
)

OUTPUT_PATH = OUTPUT_DIR / "train.jsonl"

RANDOM_SEED = 42

# Number of negative examples we try to generate
# for every positive evidence example.
SAME_DOCUMENT_NEGATIVES = 1
RETRIEVAL_HARD_NEGATIVES = 2

# Retrieve more documents than we actually need so
# that we have a useful pool of hard negatives.
TOP_RETRIEVED_DOCUMENTS = 10


# --------------------------------------------------
# Utilities
# --------------------------------------------------

def load_jsonl(path):
    with open(path) as f:
        return [
            json.loads(line)
            for line in f
        ]


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for row in rows:
            f.write(
                json.dumps(row) + "\n"
            )


def tokenize(text):
    return (
        text.lower()
        .replace(".", " ")
        .replace(",", " ")
        .replace(":", " ")
        .replace(";", " ")
        .replace("(", " ")
        .replace(")", " ")
        .split()
    )


# --------------------------------------------------
# Load SciFact
# --------------------------------------------------

print("Loading SciFact...")

claims = load_jsonl(CLAIMS_PATH)
documents_list = load_jsonl(DOCUMENTS_PATH)

documents = {
    document["document_id"]: document
    for document in documents_list
}

print(f"Claims: {len(claims)}")
print(f"Documents: {len(documents)}")


# --------------------------------------------------
# Build corpus-level BM25
# --------------------------------------------------

document_ids = list(documents.keys())

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


print("Building BM25 index...")

bm25 = BM25Okapi([
    tokenize(text)
    for text in document_texts
])


# --------------------------------------------------
# Build alignment examples
# --------------------------------------------------

random.seed(RANDOM_SEED)

examples = []

positive_count = 0
same_document_negative_count = 0
retrieval_negative_count = 0

claims_used = 0


for claim_index, claim in enumerate(claims):

    gold_evidence = claim.get(
        "gold_evidence",
        []
    )

    if not gold_evidence:
        continue

    claim_text = claim["text"]

    # --------------------------------------------------
    # Collect gold passage identities
    # --------------------------------------------------

    gold_pairs = set()

    for evidence_set in gold_evidence:

        document_id = evidence_set[
            "document_id"
        ]

        for passage_id in evidence_set[
            "passage_ids"
        ]:
            gold_pairs.add(
                (
                    document_id,
                    passage_id,
                )
            )

    if not gold_pairs:
        continue

    claims_used += 1

    # --------------------------------------------------
    # POSITIVES
    #
    # Every gold evidence passage is proposition-aligned.
    #
    # Importantly, this does NOT mean it supports the
    # claim. Contradiction evidence is still aligned.
    # --------------------------------------------------

    claim_positive_count = 0

    for document_id, passage_id in gold_pairs:

        document = documents.get(
            document_id
        )

        if document is None:
            continue

        passage_lookup = {
            passage["passage_id"]: passage
            for passage in document["passages"]
        }

        passage = passage_lookup.get(
            passage_id
        )

        if passage is None:
            continue

        examples.append({
            "claim": claim_text,
            "evidence": passage["text"],
            "label": 1,
            "negative_type": None,
            "document_id": document_id,
            "passage_id": passage_id,
        })

        positive_count += 1
        claim_positive_count += 1

    if claim_positive_count == 0:
        continue

    # --------------------------------------------------
    # SAME-DOCUMENT NEGATIVES
    #
    # These are especially valuable because they come
    # from the correct scientific paper but are not the
    # annotated evidence proposition.
    # --------------------------------------------------

    same_document_candidates = []

    gold_document_ids = {
        document_id
        for document_id, _ in gold_pairs
    }

    for document_id in gold_document_ids:

        document = documents.get(
            document_id
        )

        if document is None:
            continue

        for passage in document["passages"]:

            pair = (
                document_id,
                passage["passage_id"],
            )

            if pair in gold_pairs:
                continue

            same_document_candidates.append({
                "document_id": document_id,
                "passage_id": passage[
                    "passage_id"
                ],
                "text": passage["text"],
            })

    random.shuffle(
        same_document_candidates
    )

    same_document_limit = (
        claim_positive_count
        * SAME_DOCUMENT_NEGATIVES
    )

    for candidate in (
        same_document_candidates[
            :same_document_limit
        ]
    ):

        examples.append({
            "claim": claim_text,
            "evidence": candidate["text"],
            "label": 0,
            "negative_type": "same_document",
            "document_id": candidate[
                "document_id"
            ],
            "passage_id": candidate[
                "passage_id"
            ],
        })

        same_document_negative_count += 1

    # --------------------------------------------------
    # RETRIEVAL HARD NEGATIVES
    #
    # Retrieve documents using ONLY the claim.
    #
    # Gold evidence is NOT used to perform retrieval.
    # It is used only afterwards to prevent accidentally
    # labelling a gold passage as a negative.
    # --------------------------------------------------

    query = tokenize(claim_text)

    scores = bm25.get_scores(query)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )[:TOP_RETRIEVED_DOCUMENTS]

    hard_negative_candidates = []

    for document_index in ranked_indices:

        document_id = document_ids[
        document_index
    ]

    # --------------------------------------------------
    # Do not mine retrieval hard negatives from a
    # document known to contain gold evidence.
    #
    # Non-gold passages inside a gold document may still
    # express the same proposition in different wording,
    # making their negative label ambiguous.
    # --------------------------------------------------

        if document_id in gold_document_ids:
            continue

        document = documents[
            document_id
        ]

        for passage in document["passages"]:

            pair = (
                document_id,
                passage["passage_id"],
            )

            # Never create a false negative.
            if pair in gold_pairs:
                continue

            hard_negative_candidates.append({
                "document_id": document_id,
                "passage_id": passage[
                    "passage_id"
                ],
                "text": passage["text"],
                "bm25_document_score":
                    float(
                        scores[
                            document_index
                        ]
                    ),
            })

    # --------------------------------------------------
    # Rank hard negatives at passage level
    #
    # We want negatives that resemble the claim,
    # not random irrelevant sentences.
    # --------------------------------------------------

    if hard_negative_candidates:

        passage_bm25 = BM25Okapi([
            tokenize(candidate["text"])
            for candidate
            in hard_negative_candidates
        ])

        passage_scores = (
            passage_bm25.get_scores(
                query
            )
        )

        ranked_hard_negatives = sorted(
            zip(
                hard_negative_candidates,
                passage_scores,
            ),
            key=lambda item: item[1],
            reverse=True,
        )

        retrieval_limit = (
            claim_positive_count
            * RETRIEVAL_HARD_NEGATIVES
        )

        selected = 0
        seen_pairs = set()

        for candidate, passage_score in (
            ranked_hard_negatives
        ):

            pair = (
                candidate["document_id"],
                candidate["passage_id"],
            )

            if pair in seen_pairs:
                continue

            seen_pairs.add(pair)

            examples.append({
                "claim": claim_text,
                "evidence": candidate[
                    "text"
                ],
                "label": 0,
                "negative_type":
                    "retrieval_hard",
                "document_id": candidate[
                    "document_id"
                ],
                "passage_id": candidate[
                    "passage_id"
                ],
                "bm25_document_score":
                    candidate[
                        "bm25_document_score"
                    ],
                "bm25_passage_score":
                    float(passage_score),
            })

            retrieval_negative_count += 1
            selected += 1

            if selected >= retrieval_limit:
                break


# --------------------------------------------------
# Shuffle and save
# --------------------------------------------------

random.shuffle(examples)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

write_jsonl(
    OUTPUT_PATH,
    examples,
)


# --------------------------------------------------
# Summary
# --------------------------------------------------

negative_count = (
    same_document_negative_count
    + retrieval_negative_count
)

print()
print("=" * 60)
print("PROPOSITION ALIGNMENT TRAINING DATA")
print("=" * 60)

print(
    f"Claims used:              "
    f"{claims_used}"
)

print(
    f"Total examples:           "
    f"{len(examples)}"
)

print(
    f"Positive aligned pairs:   "
    f"{positive_count}"
)

print(
    f"Same-document negatives:  "
    f"{same_document_negative_count}"
)

print(
    f"Retrieval hard negatives: "
    f"{retrieval_negative_count}"
)

print(
    f"Total negatives:          "
    f"{negative_count}"
)

if positive_count:
    print(
        f"Negative / positive:      "
        f"{negative_count / positive_count:.2f}"
    )

print()
print(
    f"Saved to: {OUTPUT_PATH}"
)