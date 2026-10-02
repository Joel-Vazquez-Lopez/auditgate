from pathlib import Path

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import CrossEncoder
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


VERIFIABILITY_MODEL_DIR = (
    Path(__file__).parent.parent
    / "verifiability"
    / "models"
    / "distilbert"
    / "best"
)

NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-base"
EXTRACTION_MODEL_NAME = (
    "Babelscape/t5-base-summarization-claim-extractor"
)
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)


# --------------------------------------------------
# Verifiability model
# --------------------------------------------------

verifiability_tokenizer = AutoTokenizer.from_pretrained(
    VERIFIABILITY_MODEL_DIR
)

verifiability_model = (
    AutoModelForSequenceClassification.from_pretrained(
        VERIFIABILITY_MODEL_DIR
    ).to(device)
)

verifiability_model.eval()


# --------------------------------------------------
# Shared NLI model
# --------------------------------------------------

nli_tokenizer = AutoTokenizer.from_pretrained(
    NLI_MODEL_NAME
)

nli_model = (
    AutoModelForSequenceClassification.from_pretrained(
        NLI_MODEL_NAME
    ).to(device)
)

nli_model.eval()

# --------------------------------------------------
# Claim extraction model
# --------------------------------------------------

extraction_tokenizer = AutoTokenizer.from_pretrained(
    EXTRACTION_MODEL_NAME,
    use_fast=False,
)

extraction_model = AutoModelForSeq2SeqLM.from_pretrained(
    EXTRACTION_MODEL_NAME
).to(device)

extraction_model.eval()

# --------------------------------------------------
# Reranker model
# --------------------------------------------------

reranker_model = CrossEncoder(
    RERANKER_MODEL_NAME,
    device=str(device),
)


# --------------------------------------------------
# API
# --------------------------------------------------

app = FastAPI(title="AuditGate Model Service")


class VerifiabilityRequest(BaseModel):
    text: str


class FaithfulnessRequest(BaseModel):
    source: str
    candidate: str

class VerificationRequest(BaseModel):
    claim: str
    evidence: list[str]


class ExtractionSource(BaseModel):
    id: int
    text: str


class ExtractionRequest(BaseModel):
    sources: list[ExtractionSource]


class RerankerRequest(BaseModel):
    claim: str
    passages: list[str]


@app.get("/health")

@app.get("/health")

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/verifiability")
def classify_verifiability(
    request: VerifiabilityRequest,
) -> dict[str, str | float]:
    inputs = verifiability_tokenizer(
        request.text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    ).to(device)

    with torch.no_grad():
        logits = verifiability_model(**inputs).logits

    probabilities = torch.softmax(logits, dim=-1)[0]
    predicted_id = int(torch.argmax(probabilities).item())

    return {
        "kind": verifiability_model.config.id2label[
            predicted_id
        ],
        "confidence": float(
            probabilities[predicted_id].item()
        ),
    }


def loses_conditional_structure(
    source: str,
    candidate: str,
) -> bool:
    return (
        source.lower().startswith("if ")
        and not candidate.lower().startswith("if ")
    )


@app.post("/faithfulness")
def evaluate_faithfulness(
    request: FaithfulnessRequest,
) -> dict:
    inputs = nli_tokenizer(
        request.source,
        request.candidate,
        return_tensors="pt",
        truncation=True,
    ).to(device)

    with torch.no_grad():
        logits = nli_model(**inputs).logits

    probabilities = torch.softmax(logits, dim=1)[0]

    scores = {
        "contradiction": float(probabilities[0].item()),
        "entailment": float(probabilities[1].item()),
        "neutral": float(probabilities[2].item()),
    }

    decision = "faithful"

    if (
        loses_conditional_structure(
            request.source,
            request.candidate,
        )
        or scores["entailment"] < 0.90
    ):
        decision = "unsafe"

    return {
        "decision": decision,
        "scores": scores,
    }

@app.post("/verify")
def verify(
    request: VerificationRequest,
) -> dict:
    if not request.evidence:
        return {"results": []}

    claims = [request.claim] * len(request.evidence)

    inputs = nli_tokenizer(
        request.evidence,
        claims,
        return_tensors="pt",
        truncation=True,
        padding=True,
    ).to(device)

    with torch.no_grad():
        logits = nli_model(**inputs).logits

    probabilities = torch.softmax(logits, dim=1)

    results = []

    for probs in probabilities:
        scores = {
            "contradicted": float(probs[0].item()),
            "supported": float(probs[1].item()),
            "insufficient_evidence": float(
                probs[2].item()
            ),
        }

        label = max(scores, key=scores.get)

        results.append(
            {
                "label": label,
                "confidence": scores[label],
                "scores": scores,
            }
        )

    return {"results": results}

@app.post("/extract")
def extract(
    request: ExtractionRequest,
) -> dict:
    claims = []

    for source in request.sources:
        inputs = extraction_tokenizer(
            source.text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(device)

        with torch.no_grad():
            outputs = extraction_model.generate(
                **inputs,
                max_new_tokens=128,
            )

        generated = extraction_tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        )

        extracted_claims = [
            claim.strip()
            for claim in generated.split(".")
            if claim.strip()
        ]

        for claim in extracted_claims:
            claims.append(
                {
                    "text": claim + ".",
                    "source_id": source.id,
                    "kind": "verifiable",
                }
            )

    return {"claims": claims}

@app.post("/rerank")
def rerank(
    request: RerankerRequest,
) -> dict:
    if not request.passages:
        return {"scores": []}

    pairs = [
        [request.claim, passage]
        for passage in request.passages
    ]

    scores = reranker_model.predict(pairs)

    return {
        "scores": [
            float(score)
            for score in scores
        ]
    }