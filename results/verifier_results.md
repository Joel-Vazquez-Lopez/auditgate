# Verifier Experiments

All results below use the SciFact development set with gold evidence passages.

## Results

| Verifier | Accuracy |
|---|---:|
| Generic NLI (`cross-encoder/nli-deberta-v3-base`) | 0.415 |
| Fact-oriented NLI (`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`) | 0.436 |
| SciFact fine-tuned verifier | **0.872** |

## SciFact Fine-Tuned Verifier

Evaluation claims: 188

Accuracy: **0.872**

Confusion matrix:

| Gold | Predicted Supported | Predicted Contradicted |
|---|---:|---:|
| Supported | 114 | 10 |
| Contradicted | 14 | 50 |

Supported recall: 114 / 124 = 0.919

Contradicted recall: 50 / 64 = 0.781

## Observation

Task-specific fine-tuning substantially improved verification performance on
SciFact compared with the zero-shot NLI baselines.

However, the fine-tuned model remains capable of high-confidence errors,
particularly on claims requiring numerical reasoning.

Example failure:

Claim:
"1/2000 in UK have abnormal PrP positivity."

Evidence:
"Of the 32,441 appendix samples 16 were positive for abnormal PrP,
indicating an overall prevalence of 493 per million population."

Prediction:
contradicted (confidence 0.996)

Gold:
supported

This model is treated as a domain-specialized experimental verifier rather
than the default general-purpose AuditGate verifier.