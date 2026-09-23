import json
import sys

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


MODEL_NAME = "Babelscape/t5-base-summarization-claim-extractor"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=False,
)

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)


def extract_atomic_claims(text: str) -> list[str]:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
    )

    generated = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
    )

    return [
        claim.strip()
        for claim in generated.split(".")
        if claim.strip()
    ]


def main() -> None:
    request = json.load(sys.stdin)

    claims = []

    for source in request["sources"]:
        source_id = source["id"]
        text = source["text"]

        for claim in extract_atomic_claims(text):
            claims.append(
                {
                    "text": claim + ".",
                    "source_id": source_id,
                    # Temporary until the separate verifiability stage exists.
                    "kind": "verifiable",
                }
            )

    json.dump({"claims": claims}, sys.stdout)


if __name__ == "__main__":
    main()
