import json
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from pathlib import Path

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
SELECTOR_MODEL = "models/evidence-selector"

TOP_DOCUMENTS = 5
TOP_PASSAGES = 5


# --------------------------------------------------
# Utilities
# --------------------------------------------------

def load_jsonl(path):
    with open(path) as f:
        return [
            json.loads(line)
            for line in f
        ]


# --------------------------------------------------
# Load data
# --------------------------------------------------

print("Loading SciFact...")

claims = load_jsonl(CLAIMS_PATH)
documents_list = load_jsonl(DOCUMENTS_PATH)

documents = {
    doc["document_id"]: doc
    for doc in documents_list
}

print(f"Claims: {len(claims)}")
print(f"Documents: {len(documents)}")


# --------------------------------------------------
# Simple BM25 implementation
#
# This mirrors the purpose of the Rust retriever.
# For benchmarking the reranker, we need candidate
# passages without using gold evidence.
# --------------------------------------------------

from rank_bm25 import BM25Okapi


def tokenize(text):
    return (
        text.lower()
        .replace(".", " ")
        .replace(",", " ")
        .replace(":", " ")
        .replace(";", " ")
        .split()
    )


document_ids = list(documents.keys())

document_texts = []

for document_id in document_ids:
    doc = documents[document_id]

    text = (
        doc["title"]
        + " "
        + " ".join(
            passage["text"]
            for passage in doc["passages"]
        )
    )

    document_texts.append(text)


print("Building BM25 index...")

bm25 = BM25Okapi([
    tokenize(text)
    for text in document_texts
])


# --------------------------------------------------
# Load semantic reranker
# --------------------------------------------------

print("Loading MS-MARCO reranker...")

reranker = CrossEncoder(MSMARCO_MODEL)

print("Loading AuditGate evidence selector...")

selector_tokenizer = AutoTokenizer.from_pretrained(
    SELECTOR_MODEL
)

selector_model = (
    AutoModelForSequenceClassification
    .from_pretrained(SELECTOR_MODEL)
)

selector_model.eval()

if torch.backends.mps.is_available():
    selector_device = torch.device("mps")
else:
    selector_device = torch.device("cpu")

selector_model.to(selector_device)


# --------------------------------------------------
# Metrics
# --------------------------------------------------


selector_hits_at_1 = 0
selector_hits_at_3 = 0
selector_hits_at_5 = 0
selector_rr = 0.0

bm25_hits_at_1 = 0
bm25_hits_at_3 = 0
bm25_hits_at_5 = 0

reranker_hits_at_1 = 0
reranker_hits_at_3 = 0
reranker_hits_at_5 = 0

bm25_rr = 0.0
reranker_rr = 0.0

evaluated = 0


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

for claim_index, claim in enumerate(claims):

    gold_evidence = claim.get("gold_evidence", [])

    if not gold_evidence:
        continue

    # --------------------------------------------------
    # IMPORTANT:
    #
    # Gold evidence is used ONLY to determine whether
    # retrieval succeeded.
    #
    # It is NEVER passed to BM25 or the reranker.
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

    claim_text = claim["text"]

    # --------------------------------------------------
    # Stage 1: retrieve documents with BM25
    # --------------------------------------------------

    query = tokenize(claim_text)

    scores = bm25.get_scores(query)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )[:TOP_DOCUMENTS]

    # --------------------------------------------------
    # Stage 2:
    # collect ALL passages from retrieved documents.
    #
    # No gold information is used here.
    # --------------------------------------------------

    candidates = []

    for document_index in ranked_indices:

        document_id = document_ids[document_index]
        document = documents[document_id]

        for passage in document["passages"]:

            candidates.append({
                "document_id": document_id,
                "passage_id": passage["passage_id"],
                "text": passage["text"],
            })

    if not candidates:
        continue

    # --------------------------------------------------
    # BM25 passage ranking
    #
    # Build a tiny BM25 index over candidate passages.
    # --------------------------------------------------

    passage_bm25 = BM25Okapi([
        tokenize(candidate["text"])
        for candidate in candidates
    ])

    passage_scores = passage_bm25.get_scores(query)

    bm25_ranked = sorted(
        zip(candidates, passage_scores),
        key=lambda x: x[1],
        reverse=True,
    )

    # --------------------------------------------------
    # Semantic reranking
    # --------------------------------------------------

    pairs = [
        [claim_text, candidate["text"]]
        for candidate in candidates
    ]

    semantic_scores = reranker.predict(
        pairs,
        show_progress_bar=False,
    )

    semantic_ranked = sorted(
        zip(candidates, semantic_scores),
        key=lambda x: float(x[1]),
        reverse=True,
    )

    # --------------------------------------------------
    # AuditGate evidence selector
    # --------------------------------------------------

    selector_scores = []

    batch_size = 32

    for start in range(0, len(candidates), batch_size):

        batch = candidates[
            start:start + batch_size
        ]

        batch_claims = [
            claim_text
            for _ in batch
        ]

        batch_passages = [
            candidate["text"]
            for candidate in batch
        ]

        inputs = selector_tokenizer(
            batch_claims,
            batch_passages,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(selector_device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            logits = selector_model(
                **inputs
            ).logits

            probabilities = torch.softmax(
                logits,
                dim=-1,
            )

        # label 1 = relevant
        selector_scores.extend(
            probabilities[:, 1]
            .detach()
            .cpu()
            .tolist()
        )


    selector_ranked = sorted(
        zip(candidates, selector_scores),
        key=lambda x: x[1],
        reverse=True,
    )

    # --------------------------------------------------
    # Find first gold evidence rank
    # --------------------------------------------------

    def first_gold_rank(ranked):

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


    bm25_rank = first_gold_rank(bm25_ranked)
    reranker_rank = first_gold_rank(semantic_ranked)
    selector_rank = first_gold_rank(
    selector_ranked
)

    # --------------------------------------------------
    # Update metrics
    # --------------------------------------------------

    # AuditGate selector
    if selector_rank is not None:
        selector_rr += 1.0 / selector_rank

        if selector_rank <= 1:
            selector_hits_at_1 += 1

        if selector_rank <= 3:
            selector_hits_at_3 += 1

        if selector_rank <= 5:
            selector_hits_at_5 += 1


    # BM25
    if bm25_rank is not None:
        bm25_rr += 1.0 / bm25_rank

        if bm25_rank <= 1:
            bm25_hits_at_1 += 1

        if bm25_rank <= 3:
            bm25_hits_at_3 += 1

        if bm25_rank <= 5:
            bm25_hits_at_5 += 1


    # MS-MARCO reranker
    if reranker_rank is not None:
        reranker_rr += 1.0 / reranker_rank

        if reranker_rank <= 1:
            reranker_hits_at_1 += 1

        if reranker_rank <= 3:
            reranker_hits_at_3 += 1

        if reranker_rank <= 5:
            reranker_hits_at_5 += 1


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
print("EVIDENCE RETRIEVAL BENCHMARK")
print("=" * 60)

print(f"\nClaims evaluated: {evaluated}")

if evaluated == 0:
    raise RuntimeError(
        "No claims with gold evidence were found."
    )


def report(
    name,
    hits1,
    hits3,
    hits5,
    reciprocal_rank,
):
    print(f"\n{name}")
    print("-" * len(name))

    print(
        f"Recall@1: {hits1 / evaluated:.3f}"
    )

    print(
        f"Recall@3: {hits3 / evaluated:.3f}"
    )

    print(
        f"Recall@5: {hits5 / evaluated:.3f}"
    )

    print(
        f"MRR:      {reciprocal_rank / evaluated:.3f}"
    )


report(
    "BM25 passage ranking",
    bm25_hits_at_1,
    bm25_hits_at_3,
    bm25_hits_at_5,
    bm25_rr,
)

report(
    "MS-MARCO cross-encoder",
    reranker_hits_at_1,
    reranker_hits_at_3,
    reranker_hits_at_5,
    reranker_rr,
)

report(
    "AuditGate evidence selector",
    selector_hits_at_1,
    selector_hits_at_3,
    selector_hits_at_5,
    selector_rr,
)