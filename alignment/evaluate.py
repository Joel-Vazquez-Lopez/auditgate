import json
from pathlib import Path

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


# --------------------------------------------------
# Configuration
# --------------------------------------------------

CLAIMS_PATH = Path(
    "data/normalized/scifact/claims_dev.jsonl"
)

DOCUMENTS_PATH = Path(
    "data/normalized/scifact/documents.jsonl"
)

MSMARCO_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
ALIGNMENT_MODEL = "models/alignment"

TOP_DOCUMENTS = 5


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
        .replace("(", " ")
        .replace(")", " ")
        .split()
    )


def first_gold_rank(ranked, gold_pairs):
    for rank, (candidate, _) in enumerate(
        ranked,
        start=1,
    ):
        pair = (
            candidate["document_id"],
            candidate["passage_id"],
        )

        if pair in gold_pairs:
            return rank

    return None


# --------------------------------------------------
# Metric tracker
# --------------------------------------------------

class Metrics:
    def __init__(self):
        self.hits1 = 0
        self.hits3 = 0
        self.hits5 = 0
        self.rr = 0.0

    def update(self, rank):
        if rank is None:
            return

        self.rr += 1.0 / rank

        if rank <= 1:
            self.hits1 += 1

        if rank <= 3:
            self.hits3 += 1

        if rank <= 5:
            self.hits5 += 1

    def report(self, name, evaluated):
        print()
        print(name)
        print("-" * len(name))

        print(
            f"Recall@1: {self.hits1 / evaluated:.3f}"
        )
        print(
            f"Recall@3: {self.hits3 / evaluated:.3f}"
        )
        print(
            f"Recall@5: {self.hits5 / evaluated:.3f}"
        )
        print(
            f"MRR:      {self.rr / evaluated:.3f}"
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

document_ids = list(documents.keys())

print(f"Claims: {len(claims)}")
print(f"Documents: {len(documents)}")


# --------------------------------------------------
# Build document BM25
# --------------------------------------------------

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
# Load models
# --------------------------------------------------

print("Loading MS-MARCO reranker...")

msmarco = CrossEncoder(
    MSMARCO_MODEL
)

print("Loading AuditGate alignment model...")

alignment = CrossEncoder(
    str(ALIGNMENT_MODEL)
)


# --------------------------------------------------
# Metrics
# --------------------------------------------------

bm25_metrics = Metrics()
msmarco_metrics = Metrics()
alignment_metrics = Metrics()

# Pairwise comparison:
# does learned alignment rank the first gold
# evidence above or below MS-MARCO?
alignment_wins = 0
msmarco_wins = 0
ties = 0

comparison_examples = []

evaluated = 0


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

for claim in claims:

    gold_evidence = claim.get(
        "gold_evidence",
        [],
    )

    if not gold_evidence:
        continue

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

    claim_text = claim["text"]

    query = tokenize(claim_text)

    # --------------------------------------------------
    # Stage 1:
    # Retrieve documents WITHOUT gold information
    # --------------------------------------------------

    document_scores = bm25.get_scores(
        query
    )

    ranked_document_indices = sorted(
        range(len(document_scores)),
        key=lambda i: document_scores[i],
        reverse=True,
    )[:TOP_DOCUMENTS]

    # --------------------------------------------------
    # Stage 2:
    # Collect every passage from retrieved documents
    # --------------------------------------------------

    candidates = []

    for document_index in ranked_document_indices:

        document_id = document_ids[
            document_index
        ]

        document = documents[
            document_id
        ]

        for passage in document["passages"]:

            candidates.append({
                "document_id": document_id,
                "passage_id": passage[
                    "passage_id"
                ],
                "text": passage["text"],
            })

    if not candidates:
        continue

    # --------------------------------------------------
    # BM25 passage baseline
    # --------------------------------------------------

    passage_bm25 = BM25Okapi([
        tokenize(candidate["text"])
        for candidate in candidates
    ])

    passage_scores = passage_bm25.get_scores(
        query
    )

    bm25_ranked = sorted(
        zip(
            candidates,
            passage_scores,
        ),
        key=lambda item: float(item[1]),
        reverse=True,
    )

    # --------------------------------------------------
    # General MS-MARCO reranker
    # --------------------------------------------------

    pairs = [
        [
            claim_text,
            candidate["text"],
        ]
        for candidate in candidates
    ]

    msmarco_scores = msmarco.predict(
        pairs,
        show_progress_bar=False,
    )

    msmarco_ranked = sorted(
        zip(
            candidates,
            msmarco_scores,
        ),
        key=lambda item: float(item[1]),
        reverse=True,
    )

    # --------------------------------------------------
    # Learned proposition alignment
    # --------------------------------------------------

    alignment_scores = alignment.predict(
        pairs,
        show_progress_bar=False,
    )

    alignment_ranked = sorted(
        zip(
            candidates,
            alignment_scores,
        ),
        key=lambda item: float(item[1]),
        reverse=True,
    )

    # --------------------------------------------------
    # Gold is used ONLY here:
    # measuring where the correct evidence ranked.
    # --------------------------------------------------

    bm25_rank = first_gold_rank(
        bm25_ranked,
        gold_pairs,
    )

    msmarco_rank = first_gold_rank(
        msmarco_ranked,
        gold_pairs,
    )

    alignment_rank = first_gold_rank(
        alignment_ranked,
        gold_pairs,
    )

    # --------------------------------------------------
    # Pairwise comparison:
    # MS-MARCO vs learned proposition alignment
    # --------------------------------------------------

    # If a method failed to retrieve any gold evidence,
    # treat its rank as infinity for comparison purposes.
    msmarco_compare = (
        msmarco_rank
        if msmarco_rank is not None
        else float("inf")
    )

    alignment_compare = (
        alignment_rank
        if alignment_rank is not None
        else float("inf")
    )

    if alignment_compare < msmarco_compare:
        alignment_wins += 1
        winner = "alignment"

    elif msmarco_compare < alignment_compare:
        msmarco_wins += 1
        winner = "msmarco"

    else:
        ties += 1
        winner = "tie"

    # Save non-tie examples so we can inspect
    # where the two models behave differently.
    if winner != "tie":
        comparison_examples.append({
            "claim": claim_text,
            "winner": winner,
            "msmarco_rank": msmarco_rank,
            "alignment_rank": alignment_rank,
        })

    bm25_metrics.update(
        bm25_rank
    )

    msmarco_metrics.update(
        msmarco_rank
    )

    alignment_metrics.update(
        alignment_rank
    )

    evaluated += 1

    if evaluated % 20 == 0:
        print(
            f"Evaluated {evaluated} claims..."
        )


# --------------------------------------------------
# Results
# --------------------------------------------------

print()
print("=" * 60)
print("PROPOSITION ALIGNMENT BENCHMARK")
print("=" * 60)

print(
    f"\nClaims evaluated: {evaluated}"
)

if evaluated == 0:
    raise RuntimeError(
        "No claims with gold evidence found."
    )


bm25_metrics.report(
    "BM25 passage ranking",
    evaluated,
)

msmarco_metrics.report(
    "MS-MARCO cross-encoder",
    evaluated,
)

alignment_metrics.report(
    "AuditGate learned alignment",
    evaluated,
)



# --------------------------------------------------
# Pairwise comparison results
# --------------------------------------------------

print()

print("=" * 60)
print("PAIRWISE COMPARISON")
print("=" * 60)

print(
    f"Alignment wins: {alignment_wins}"
)

print(
    f"MS-MARCO wins:  {msmarco_wins}"
)

print(
    f"Ties:           {ties}"
)

print()
print("EXAMPLE ALIGNMENT WINS")
print("-" * 60)

shown = 0

for example in comparison_examples:

    if example["winner"] != "alignment":
        continue

    print()
    print(example["claim"])

    print(
        "MS-MARCO rank:",
        example["msmarco_rank"],
    )

    print(
        "Alignment rank:",
        example["alignment_rank"],
    )

    shown += 1

    if shown >= 5:
        break


print()
print("EXAMPLE MS-MARCO WINS")
print("-" * 60)

shown = 0

for example in comparison_examples:

    if example["winner"] != "msmarco":
        continue

    print()
    print(example["claim"])

    print(
        "MS-MARCO rank:",
        example["msmarco_rank"],
    )

    print(
        "Alignment rank:",
        example["alignment_rank"],
    )

    shown += 1

    if shown >= 5:
        break