from pathlib import Path

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


MODEL_DIR = Path(__file__).parent / "models" / "distilbert" / "best"

CASES = [
    ("simple factual", "The Eiffel Tower was completed in 1889.", "VERIFIABLE"),
    ("compound factual 1", "The Eiffel Tower is in Paris.", "VERIFIABLE"),
    ("compound factual 2", "The Eiffel Tower was completed in 1889.", "VERIFIABLE"),
    ("pure opinion", "I think the Eiffel Tower is beautiful.", "NON_VERIFIABLE"),
    ("mixed fact", "The Eiffel Tower is in Paris.", "VERIFIABLE"),
    ("mixed opinion", "I think the Eiffel Tower is beautiful.", "NON_VERIFIABLE"),
    ("numerical", "The experiment included 240 participants.", "VERIFIABLE"),
    ("false but verifiable", "The Moon is made primarily of cheese.", "VERIFIABLE"),
    ("hedged", "Researchers suggest that the treatment may reduce mortality.", "VERIFIABLE"),
    ("causal", "Smoking increases the risk of lung cancer.", "VERIFIABLE"),
    ("question", "Is the Eiffel Tower in Paris?", "NON_VERIFIABLE"),
    ("command", "Visit the Eiffel Tower when you go to Paris.", "NON_VERIFIABLE"),
    (
        "attributed",
        "The WHO reports that global life expectancy increased between 2000 and 2019.",
        "VERIFIABLE",
    ),
    ("uncertain prediction", "The new policy could reduce emissions by 2030.", "VERIFIABLE"),
    ("fact + judgement fact", "The study included 500 participants.", "VERIFIABLE"),
    ("fact + judgement opinion", "The design was excellent.", "NON_VERIFIABLE"),
    ("two claims 1", "The drug reduced blood pressure.", "VERIFIABLE"),
    ("two claims 2", "The drug increased heart rate.", "VERIFIABLE"),
]


def main() -> None:
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR
    ).to(device)
    model.eval()

    correct = 0

    for name, text, expected in CASES:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits

        probabilities = torch.softmax(logits, dim=-1)[0]
        predicted_id = int(torch.argmax(probabilities).item())
        predicted = model.config.id2label[predicted_id]
        confidence = float(probabilities[predicted_id].item())

        passed = predicted == expected
        correct += int(passed)

        status = "PASS" if passed else "FAIL"

        print(f"\n[{status}] {name}")
        print(f"Text:       {text}")
        print(f"Expected:   {expected}")
        print(f"Predicted:  {predicted}")
        print(f"Confidence: {confidence:.3f}")

    print("\n==============================")
    print(f"Challenge result: {correct}/{len(CASES)}")
    print("==============================")


if __name__ == "__main__":
    main()