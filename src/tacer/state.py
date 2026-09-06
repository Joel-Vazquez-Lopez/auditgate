import math
import re
from collections import Counter

import numpy as np


# --------------------------------------------------
# Basic utilities
# --------------------------------------------------

def tokenize(text):
    return re.findall(
        r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*",
        text.lower(),
    )


def mean(values):
    if not values:
        return 0.0

    return sum(values) / len(values)


def std(values):
    if not values:
        return 0.0

    mu = mean(values)

    variance = mean([
        (value - mu) ** 2
        for value in values
    ])

    return math.sqrt(variance)


# --------------------------------------------------
# Ranking-score features
# --------------------------------------------------

def softmax(values):
    if not values:
        return []

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    values = values - np.max(values)

    exp_values = np.exp(values)

    denominator = exp_values.sum()

    if denominator <= 0:
        return [
            1.0 / len(values)
            for _ in values
        ]

    return (
        exp_values / denominator
    ).tolist()


def normalized_entropy_from_scores(scores):
    """
    Convert arbitrary ranking scores to a probability
    distribution with softmax, then calculate normalized
    entropy.

    0 -> highly concentrated
    1 -> highly diffuse
    """

    if len(scores) <= 1:
        return 0.0

    probabilities = softmax(scores)

    entropy = -sum(
        probability * math.log(
            probability + 1e-12
        )
        for probability in probabilities
    )

    max_entropy = math.log(
        len(probabilities)
    )

    if max_entropy <= 0:
        return 0.0

    return entropy / max_entropy


def score_features(scores, prefix):
    """
    Scores MUST already be ordered from highest to lowest.
    """

    scores = [
        float(score)
        for score in scores
    ]

    if not scores:
        return {
            f"{prefix}_top": 0.0,
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_gap_1_2": 0.0,
            f"{prefix}_gap_1_3": 0.0,
            f"{prefix}_entropy": 0.0,
        }

    top = scores[0]

    gap_1_2 = (
        top - scores[1]
        if len(scores) >= 2
        else 0.0
    )

    gap_1_3 = (
        top - scores[2]
        if len(scores) >= 3
        else 0.0
    )

    return {
        f"{prefix}_top": top,
        f"{prefix}_mean": mean(scores),
        f"{prefix}_std": std(scores),
        f"{prefix}_gap_1_2": gap_1_2,
        f"{prefix}_gap_1_3": gap_1_3,
        f"{prefix}_entropy":
            normalized_entropy_from_scores(
                scores
            ),
    }


# --------------------------------------------------
# Interpretable coverage features
# --------------------------------------------------

def token_overlap_features(claim, evidence):
    """
    Produce raw interpretable overlap features.

    No gold information is used here.
    """

    claim_tokens = tokenize(claim)
    evidence_tokens = tokenize(evidence)

    if not claim_tokens or not evidence_tokens:
        return {
            "exact_overlap": 0.0,
            "rare_overlap": 0.0,
            "phrase_overlap": 0.0,
            "density": 0.0,
        }

    claim_counts = Counter(
        claim_tokens
    )

    evidence_set = set(
        evidence_tokens
    )

    unique_claim = set(
        claim_tokens
    )

    matched_unique = sum(
        1
        for token in unique_claim
        if token in evidence_set
    )

    exact_overlap = (
        matched_unique
        / len(unique_claim)
    )

    rare_tokens = {
        token
        for token, count
        in claim_counts.items()
        if count == 1
        and len(token) >= 4
    }

    if rare_tokens:
        rare_overlap = (
            sum(
                1
                for token in rare_tokens
                if token in evidence_set
            )
            / len(rare_tokens)
        )
    else:
        rare_overlap = 0.0

    claim_bigrams = {
        (
            claim_tokens[index],
            claim_tokens[index + 1],
        )
        for index
        in range(len(claim_tokens) - 1)
    }

    evidence_bigrams = {
        (
            evidence_tokens[index],
            evidence_tokens[index + 1],
        )
        for index
        in range(len(evidence_tokens) - 1)
    }

    if claim_bigrams:
        phrase_overlap = (
            len(
                claim_bigrams
                & evidence_bigrams
            )
            / len(claim_bigrams)
        )
    else:
        phrase_overlap = 0.0

    matched_evidence_tokens = sum(
        1
        for token in evidence_tokens
        if token in unique_claim
    )

    density = (
        matched_evidence_tokens
        / len(evidence_tokens)
    )

    return {
        "exact_overlap": exact_overlap,
        "rare_overlap": rare_overlap,
        "phrase_overlap": phrase_overlap,
        "density": density,
    }


# --------------------------------------------------
# Complete observable TACER state
# --------------------------------------------------

def compute_tacer_state(
    claim_text,
    retrieved_document_scores,
    candidates,
):
    """
    Compute the complete runtime-observable TACER state.

    candidates must already contain:
        msmarco_score
        alignment_score
        document_id
        passage_id
        text

    This function deliberately has no access to gold
    evidence or SciFact labels.
    """

    if not candidates:
        raise ValueError(
            "Cannot compute TACER state "
            "without candidate passages."
        )

    msmarco_ranked = sorted(
        candidates,
        key=lambda candidate:
            candidate["msmarco_score"],
        reverse=True,
    )

    alignment_ranked = sorted(
        candidates,
        key=lambda candidate:
            candidate["alignment_score"],
        reverse=True,
    )

    document_features = score_features(
        retrieved_document_scores,
        "bm25_document",
    )

    msmarco_ranked_scores = [
        candidate["msmarco_score"]
        for candidate in msmarco_ranked
    ]

    alignment_ranked_scores = [
        candidate["alignment_score"]
        for candidate in alignment_ranked
    ]

    msmarco_features = score_features(
        msmarco_ranked_scores,
        "msmarco",
    )

    alignment_features = score_features(
        alignment_ranked_scores,
        "alignment",
    )

    msmarco_top = msmarco_ranked[0]
    alignment_top = alignment_ranked[0]

    top1_agreement = int(
        (
            msmarco_top["document_id"],
            msmarco_top["passage_id"],
        )
        ==
        (
            alignment_top["document_id"],
            alignment_top["passage_id"],
        )
    )

    coverage = token_overlap_features(
        claim_text,
        alignment_top["text"],
    )

    features = {
        **document_features,
        **msmarco_features,
        **alignment_features,
        **coverage,
        "ranker_top1_agreement":
            top1_agreement,
    }

    return {
        "features": features,
        "msmarco_ranked": msmarco_ranked,
        "alignment_ranked": alignment_ranked,
    }