import json
from pathlib import Path

RAW = Path("data/raw/scifact")
OUT = Path("data/normalized/scifact")

OUT.mkdir(parents=True, exist_ok=True)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def save_jsonl(path, rows):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


# --------------------------------------------------
# Documents
# --------------------------------------------------

documents = []

for doc in load_jsonl(RAW / "corpus.jsonl"):
    doc_id = doc["doc_id"]

    passages = []

    for i, sentence in enumerate(doc["abstract"]):
        passages.append({
            "passage_id": f"scifact:{doc_id}:s{i}",
            "sentence_index": i,
            "text": sentence,
        })

    documents.append({
        "document_id": f"scifact:{doc_id}",
        "title": doc["title"],
        "passages": passages,
        "metadata": {
            "structured": doc["structured"],
        },
    })

save_jsonl(OUT / "documents.jsonl", documents)

print(f"Saved {len(documents)} documents")


# --------------------------------------------------
# Claims
# --------------------------------------------------

for split in ["train", "dev", "test"]:

    claims = []

    for claim in load_jsonl(RAW / f"claims_{split}.jsonl"):

        evidence = claim.get("evidence", {})

        # Test claims do not have gold labels.
        if "evidence" not in claim:
            label = None

        elif not evidence:
            label = "insufficient_evidence"

        else:
            labels = {
                item["label"]
                for evidence_sets in evidence.values()
                for item in evidence_sets
            }

            if labels == {"SUPPORT"}:
                label = "supported"

            elif labels == {"CONTRADICT"}:
                label = "contradicted"

            else:
                raise ValueError(
                    f"Unexpected labels for claim {claim['id']}: {labels}"
                )

        evidence_sets = []

        for doc_id, sets in evidence.items():
            for item in sets:
                evidence_sets.append({
                    "document_id": f"scifact:{doc_id}",
                    "passage_ids": [
                        f"scifact:{doc_id}:s{i}"
                        for i in item["sentences"]
                    ],
                })

        claims.append({
            "claim_id": f"scifact:{claim['id']}",
            "text": claim["claim"],
            "gold_label": label,
            "gold_evidence": evidence_sets,
        })

    save_jsonl(
        OUT / f"claims_{split}.jsonl",
        claims,
    )

    print(f"Saved {len(claims)} {split} claims")
