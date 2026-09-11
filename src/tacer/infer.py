import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import CrossEncoder

from state import compute_tacer_state


MSMARCO_MODEL = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

ALIGNMENT_MODEL = Path(
    "models/alignment"
)

TACER_MODEL = Path(
    "models/tacer/candidate.pkl"
)


# --------------------------------------------------
# Load learned models
# --------------------------------------------------

msmarco = CrossEncoder(
    MSMARCO_MODEL
)

alignment = CrossEncoder(
    str(ALIGNMENT_MODEL)
)

with TACER_MODEL.open("rb") as f:
    artifact = pickle.load(f)

model = artifact["model"]
feature_names = artifact["features"]


# --------------------------------------------------
# Input
# --------------------------------------------------

payload = json.load(sys.stdin)

claim = payload["claim"]

retrieved_document_scores = [
    float(score)
    for score in payload[
        "retrieved_document_scores"
    ]
]

candidates = payload["candidates"]

if not candidates:
    raise ValueError(
        "TACER inference requires candidates."
    )


# --------------------------------------------------
# Learned candidate scores
# --------------------------------------------------

pairs = [
    [
        claim,
        candidate["text"],
    ]
    for candidate in candidates
]

msmarco_scores = msmarco.predict(
    pairs,
    show_progress_bar=False,
)

alignment_scores = alignment.predict(
    pairs,
    show_progress_bar=False,
)

for index, candidate in enumerate(candidates):
    candidate["msmarco_score"] = float(
        msmarco_scores[index]
    )

    candidate["alignment_score"] = float(
        alignment_scores[index]
    )


# --------------------------------------------------
# Exact TACER state
# --------------------------------------------------

state = compute_tacer_state(
    claim_text=claim,
    retrieved_document_scores=(
        retrieved_document_scores
    ),
    candidates=candidates,
)

feature_values = state["features"]

x = np.asarray(
    [[
        float(feature_values[name])
        for name in feature_names
    ]],
    dtype=np.float64,
)


# --------------------------------------------------
# Frozen TACER-A
# --------------------------------------------------

probability_sufficient = float(
    model.predict_proba(x)[0, 1]
)


# --------------------------------------------------
# Return learned scores too.
#
# This will let Rust reuse the same ranking information
# rather than recomputing an incompatible approximation.
# --------------------------------------------------

scored_candidates = []

for candidate in candidates:
    scored_candidates.append({
        "document_id":
            candidate["document_id"],

        "passage_id":
            candidate["passage_id"],

        "msmarco_score":
            candidate["msmarco_score"],

        "alignment_score":
            candidate["alignment_score"],
    })


print(
    json.dumps({
        "probability_sufficient":
            probability_sufficient,

        "features":
            feature_values,

        "candidates":
            scored_candidates,
    })
)