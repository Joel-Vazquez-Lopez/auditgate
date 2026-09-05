import json
import sys

from sentence_transformers import CrossEncoder


MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

model = CrossEncoder(MODEL)


def main():
    data = json.load(sys.stdin)

    claim = data["claim"]
    passages = data["passages"]

    if not passages:
        print(json.dumps({"scores": []}))
        return

    pairs = [
        [claim, passage]
        for passage in passages
    ]

    scores = model.predict(pairs)

    print(
        json.dumps({
            "scores": [
                float(score)
                for score in scores
            ]
        })
    )


if __name__ == "__main__":
    main()