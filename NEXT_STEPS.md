# AuditGate — Next Steps

## Current State

AuditGate is an evidence-grounded decision engine for determining whether an AI-generated claim has sufficient trustworthy evidence to proceed.

The current single-claim pipeline includes:

- online evidence retrieval
- document fetching and passage extraction
- reranking
- evidence-state construction
- TACER adaptive evidence acquisition
- evidence quality estimation
- NLI verification
- ALLOW / REVIEW / BLOCK decisions
- structured audit output and TACER traces

The semantic extraction layer currently provides:

1. T5-based atomic claim extraction
2. provenance validation through `source_id`
3. an NLI-based faithfulness gate
4. structural protection against conditional meaning loss
5. exact-source fallback when generated decomposition is unsafe

The extraction regression suite currently passes all 14 tests.

---

## Next Milestone: Verifiability Classifier V1

### Goal

Determine whether a faithful extracted unit contains an externally checkable proposition and therefore should enter evidence acquisition.

This is separate from truth assessment.

Examples:

- `The Moon is made primarily of cheese.` → Verifiable
- `The experiment included 240 participants.` → Verifiable
- `Researchers suggest that the treatment may reduce mortality.` → Verifiable
- `The WHO reports that global life expectancy increased.` → Verifiable
- `I think the Eiffel Tower is beautiful.` → NonVerifiable
- `Is the Eiffel Tower in Paris?` → NonVerifiable
- `Visit the Eiffel Tower when you go to Paris.` → NonVerifiable

### Architecture

Target semantic pipeline:

SourceUnit
→ T5 atomic extraction
→ FaithfulnessGate
→ VerifiabilityClassifier
→ evidence acquisition only for Verifiable claims

A `VerifiabilityClassifier` Rust trait has been introduced as the intended interface.

---

## Experimental Result

Two generic approaches were tested before implementing a production classifier.

### Generic NLI meta-hypothesis

`cross-encoder/nli-deberta-v3-base` was asked whether statements entailed a meta-hypothesis describing external verifiability.

This failed to separate the classes.

Notably, questions and commands received very high entailment scores while ordinary factual claims received low scores.

Conclusion:

Generic NLI entailment should not be used as the verifiability classifier.

### Zero-shot classification

`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` was tested with competing Verifiable and NonVerifiable labels.

The nine-case probe achieved only 5/9 correct top-label classifications and misclassified ordinary facts, false-but-verifiable claims, hedged claims, and predictions.

Conclusion:

Do not solve the problem by threshold-tuning or prompt/template tuning against the small regression set.

---

## Planned Solution

Train a small dedicated AuditGate verifiability classifier.

### Dataset

Prefer public datasets as the primary source.

Research datasets involving:

- propositional claim detection
- factual statement detection
- subjectivity/objectivity
- claim detection
- check-worthiness

Do not directly equate check-worthiness with verifiability.

Construct an AuditGate-specific mapping:

- externally checkable proposition → Verifiable
- opinion / preference → NonVerifiable
- question → NonVerifiable
- command → NonVerifiable
- subjective judgement without externally testable proposition → NonVerifiable

Use targeted AuditGate examples only to cover boundary cases not represented by public datasets.

Keep the existing extraction examples as frozen regression/evaluation cases rather than training examples.

Prevent train/test leakage and near-duplicate/template leakage.

### Model

Benchmark lightweight encoder classifiers, initially considering:

- DistilBERT
- DeBERTa-v3-base

Choose based on classification quality, calibration, latency, and memory rather than model size alone.

### Evaluation

Measure:

- accuracy
- macro F1
- per-class precision / recall / F1
- confusion matrix
- calibration
- threshold behaviour
- performance by linguistic category
- out-of-domain performance
- inference latency
- memory/model size

Pay particular attention to false negatives because they cause AuditGate to skip claims that should have been audited.

---

## Later Engineering Work

### Persistent model loading

Current Python subprocess architecture repeatedly loads models and is too slow.

The full extraction regression suite took approximately 160 seconds after faithfulness integration.

Target architecture:

- load T5 once
- load reranker once
- load verifiability classifier once
- load DeBERTa NLI once and share it between faithfulness and verification
- reuse loaded models across requests
- add batching where appropriate

### Multi-claim and document auditing

After the semantic front-end is stable:

- integrate semantic extraction into long-text auditing
- preserve mappings from atomic claims to original spans
- support PDF / DOCX / TXT / MD inputs
- preserve page / section / paragraph provenance where possible
- support hybrid uploaded-document + web evidence
- produce document-level audit reports

### Product Layer

Eventually expose AuditGate through an interactive web application where users can:

- paste a single claim
- paste a larger AI response/report
- upload a document
- inspect extracted claims
- inspect evidence and provenance
- inspect TACER acquisition traces
- see Supported / Contradicted / Unresolved / No-check-needed outcomes

---

## Restart Point

When development resumes:

1. Research suitable public verifiability/claim/subjectivity datasets.
2. Define exact dataset-to-AuditGate label mappings.
3. Build a reproducible dataset preparation pipeline.
4. Establish simple baselines.
5. Fine-tune the first lightweight classifier.
6. Evaluate on frozen and out-of-domain test sets.
7. Integrate the selected classifier behind `VerifiabilityClassifier`.
8. Optimize model lifecycle/persistent loading.
9. Continue toward long-text and document auditing.
