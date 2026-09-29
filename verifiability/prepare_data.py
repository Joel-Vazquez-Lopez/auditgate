from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


DATA_DIR = Path(__file__).parent / "data"

CLAIMBUSTER_CROWDSOURCED = DATA_DIR / "raw" / "claimbuster" / "crowdsourced.csv"
CLAIMBUSTER_GROUNDTRUTH = DATA_DIR / "raw" / "claimbuster" / "groundtruth.csv"
CLAIMIFY = DATA_DIR / "raw" / "claimify" / "data.csv"

PROCESSED_DIR = DATA_DIR / "processed"


def normalize_text(text: str) -> str:
    return " ".join(str(text).split()).strip()


def main() -> None:
    raw_files = [
        CLAIMBUSTER_CROWDSOURCED,
        CLAIMBUSTER_GROUNDTRUTH,
        CLAIMIFY,
    ]

    for path in raw_files:
        if not path.exists():
            raise FileNotFoundError(f"Missing raw dataset: {path}")

        print(f"FOUND {path}")

        claimbuster_crowd = pd.read_csv(CLAIMBUSTER_CROWDSOURCED)
    claimbuster_gold = pd.read_csv(CLAIMBUSTER_GROUNDTRUTH)
    claimify = pd.read_csv(CLAIMIFY)
    claimbuster_columns = {
        "Sentence_id",
        "Text",
        "Speaker",
        "Speaker_title",
        "Speaker_party",
        "File_id",
        "Length",
        "Line_number",
        "Sentiment",
        "Verdict",
    }

    claimify_columns = {
        "answer_id",
        "question",
        "sentence_id",
        "sentence",
        "contains_factual_claim",
    }

    if set(claimbuster_crowd.columns) != claimbuster_columns:
        raise ValueError("Unexpected ClaimBuster crowdsourced schema")

    if set(claimbuster_gold.columns) != claimbuster_columns:
        raise ValueError("Unexpected ClaimBuster groundtruth schema")

    if set(claimify.columns) != claimify_columns:
        raise ValueError("Unexpected Claimify schema")

    claimbuster_label_map = {
        -1: "NON_VERIFIABLE",
        0: "VERIFIABLE",
        1: "VERIFIABLE",
    }

    claimbuster_crowd["label"] = claimbuster_crowd["Verdict"].map(
        claimbuster_label_map
    )
    claimbuster_gold["label"] = claimbuster_gold["Verdict"].map(
        claimbuster_label_map
    )

    claimify["label"] = claimify["contains_factual_claim"].map(
        {
            False: "NON_VERIFIABLE",
            True: "VERIFIABLE",
        }
    )

    for name, dataset in [
        ("ClaimBuster crowdsourced", claimbuster_crowd),
        ("ClaimBuster groundtruth", claimbuster_gold),
        ("Claimify", claimify),
    ]:
        if dataset["label"].isna().any():
            raise ValueError(f"Unmapped labels found in {name}")

        print(f"\n{name} mapped labels:")
        print(dataset["label"].value_counts())
    
    claimbuster_crowd_standard = pd.DataFrame({
        "text": claimbuster_crowd["Text"],
        "label": claimbuster_crowd["label"],
        "source_dataset": "claimbuster",
        "annotation_source": "crowdsourced",
        "source_id": claimbuster_crowd["Sentence_id"],
        "group_id": claimbuster_crowd["File_id"],
        "original_label": claimbuster_crowd["Verdict"],
    })

    claimbuster_gold_standard = pd.DataFrame({
        "text": claimbuster_gold["Text"],
        "label": claimbuster_gold["label"],
        "source_dataset": "claimbuster",
        "annotation_source": "groundtruth",
        "source_id": claimbuster_gold["Sentence_id"],
        "group_id": claimbuster_gold["File_id"],
        "original_label": claimbuster_gold["Verdict"],
    })

    claimify_standard = pd.DataFrame({
        "text": claimify["sentence"],
        "label": claimify["label"],
        "source_dataset": "claimify",
        "annotation_source": "manual",
        "source_id": claimify["sentence_id"],
        "group_id": claimify["answer_id"],
        "original_label": claimify["contains_factual_claim"],
    })

    for dataset in [
        claimbuster_crowd_standard,
        claimbuster_gold_standard,
        claimify_standard,
    ]:
        dataset["normalized_text"] = dataset["text"].map(normalize_text)

    print("\nNormalized-text duplicates:")
    print(
        "ClaimBuster crowdsourced:",
        claimbuster_crowd_standard["normalized_text"].duplicated().sum(),
    )
    print(
        "ClaimBuster groundtruth:",
        claimbuster_gold_standard["normalized_text"].duplicated().sum(),
    )
    print(
        "Claimify:",
        claimify_standard["normalized_text"].duplicated().sum(),
    )

    crowd_texts = set(claimbuster_crowd_standard["normalized_text"])
    gold_texts = set(claimbuster_gold_standard["normalized_text"])
    claimify_texts = set(claimify_standard["normalized_text"])

    print("\nCross-dataset normalized-text overlap:")
    print("Crowdsourced ↔ groundtruth:", len(crowd_texts & gold_texts))
    print("Crowdsourced ↔ Claimify:", len(crowd_texts & claimify_texts))
    print("Groundtruth ↔ Claimify:", len(gold_texts & claimify_texts))

    claimbuster_crowd_clean = claimbuster_crowd_standard[
        ~claimbuster_crowd_standard["normalized_text"].isin(gold_texts)
    ].copy()

    claimbuster_crowd_clean = claimbuster_crowd_clean.drop_duplicates(
        subset="normalized_text"
    )

    claimbuster_gold_clean = claimbuster_gold_standard.drop_duplicates(
        subset="normalized_text"
    ).copy()

    claimify_clean = claimify_standard.drop_duplicates(
        subset="normalized_text"
    ).copy()

    print("\nRows after leakage protection and deduplication:")
    print("ClaimBuster crowdsourced:", len(claimbuster_crowd_clean))
    print("ClaimBuster groundtruth:  ", len(claimbuster_gold_clean))
    print("Claimify:                 ", len(claimify_clean))

    assert not (
        set(claimbuster_crowd_clean["normalized_text"])
        & set(claimbuster_gold_clean["normalized_text"])
    )
    
    def split_grouped(dataset: pd.DataFrame):
        splitter = StratifiedGroupKFold(
            n_splits=5,
            shuffle=True,
            random_state=42,
        )

        train_idx, validation_idx = next(
            splitter.split(
                dataset,
                y=dataset["label"],
                groups=dataset["group_id"],
            )
        )

        return (
            dataset.iloc[train_idx].copy(),
            dataset.iloc[validation_idx].copy(),
        )

    claimbuster_train, claimbuster_validation = split_grouped(
        claimbuster_crowd_clean
    )
    claimify_train, claimify_validation = split_grouped(
        claimify_clean
    )

    print("\nGroup-safe 80/20 splits:")

    for name, train, validation in [
        ("ClaimBuster", claimbuster_train, claimbuster_validation),
        ("Claimify", claimify_train, claimify_validation),
    ]:
        print(f"\n{name}")
        print("Train rows:", len(train))
        print("Validation rows:", len(validation))
        print(
            "Train VERIFIABLE:",
            round((train["label"] == "VERIFIABLE").mean(), 3),
        )
        print(
            "Validation VERIFIABLE:",
            round((validation["label"] == "VERIFIABLE").mean(), 3),
        )

        assert not (
            set(train["group_id"])
            & set(validation["group_id"])
        )

        assert not (
            set(train["normalized_text"])
            & set(validation["normalized_text"])
        )
        train = pd.concat(
        [claimbuster_train, claimify_train],
        ignore_index=True,
    )

    validation = pd.concat(
        [claimbuster_validation, claimify_validation],
        ignore_index=True,
    )

    expert_test = claimbuster_gold_clean.reset_index(drop=True)

    print("\nFinal datasets:")
    for name, dataset in [
        ("train", train),
        ("validation", validation),
        ("expert_test", expert_test),
    ]:
        print(
            f"{name}: {len(dataset)} rows, "
            f"{(dataset['label'] == 'VERIFIABLE').mean():.3f} VERIFIABLE"
        )
        train_texts = set(train["normalized_text"])
    validation_texts = set(validation["normalized_text"])
    expert_test_texts = set(expert_test["normalized_text"])

    assert not (train_texts & validation_texts)
    assert not (train_texts & expert_test_texts)
    assert not (validation_texts & expert_test_texts)

    for name, dataset in [
        ("train", train),
        ("validation", validation),
        ("expert_test", expert_test),
    ]:
        assert dataset["text"].notna().all(), f"Missing text in {name}"
        assert dataset["label"].notna().all(), f"Missing label in {name}"
        assert set(dataset["label"]) <= {
            "VERIFIABLE",
            "NON_VERIFIABLE",
        }

    print("\nFinal integrity checks: PASS")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    output_datasets = {
        "train.jsonl": train,
        "validation.jsonl": validation,
        "expert_test.jsonl": expert_test,
    }

    for filename, dataset in output_datasets.items():
        output_path = PROCESSED_DIR / filename

        dataset.to_json(
            output_path,
            orient="records",
            lines=True,
            force_ascii=False,
        )

        print(f"WROTE {output_path} ({len(dataset):,} rows)")
    
    print("\nStandardized schemas:")
    print(list(claimbuster_crowd_standard.columns))
    print(list(claimbuster_gold_standard.columns))
    print(list(claimify_standard.columns))
    
    print("\nLoaded datasets:")
    print(f"ClaimBuster crowdsourced: {len(claimbuster_crowd):,}")
    print(f"ClaimBuster groundtruth:   {len(claimbuster_gold):,}")
    print(f"Claimify:                  {len(claimify):,}")

    print("\nSchemas:")
    print("ClaimBuster crowdsourced:", list(claimbuster_crowd.columns))
    print("ClaimBuster groundtruth:  ", list(claimbuster_gold.columns))
    print("Claimify:                 ", list(claimify.columns))


if __name__ == "__main__":
    main()