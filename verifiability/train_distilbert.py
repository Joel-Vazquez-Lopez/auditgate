import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


MODEL_NAME = "distilbert-base-uncased"
DATA_DIR = Path(__file__).parent / "data" / "processed"

LABEL2ID = {
    "NON_VERIFIABLE": 0,
    "VERIFIABLE": 1,
}

MAX_LENGTH = 256


def main() -> None:
    dataset = load_dataset(
        "json",
        data_files={
            "train": str(DATA_DIR / "train.jsonl"),
            "validation": str(DATA_DIR / "validation.jsonl"),
            "expert_test": str(DATA_DIR / "expert_test.jsonl"),
        },
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    def tokenize_batch(batch):
        tokenized = tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
        )
        tokenized["labels"] = [
            LABEL2ID[label] for label in batch["label"]
        ]
        return tokenized

    tokenized_dataset = dataset.map(
        tokenize_batch,
        batched=True,
    )

    print(dataset)
    print(f"\nTokenizer: {MODEL_NAME}")
        
    example = tokenized_dataset["train"][0]

    print("\nTokenization check:")
    print("Text:", example["text"])
    print("Label:", example["label"])
    print("Encoded label:", example["labels"])
    print("Token count:", len(example["input_ids"]))


    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={
            0: "NON_VERIFIABLE",
            1: "VERIFIABLE",
        },
        label2id=LABEL2ID,
    )
    
    print("\nModel check:")
    print("Model:", MODEL_NAME)
    print("Number of labels:", model.config.num_labels)
    print("Label mapping:", model.config.id2label)

    training_args = TrainingArguments(
        output_dir=str(Path(__file__).parent / "models" / "distilbert"),
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=42,
        report_to="none",
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
    )
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels,
            predictions,
            labels=[1],
            average=None,
            zero_division=0,
        )

        return {
            "accuracy": accuracy_score(labels, predictions),
            "macro_f1": f1_score(labels, predictions, average="macro"),
            "verifiable_precision": precision[0],
            "verifiable_recall": recall[0],
            "verifiable_f1": f1[0],
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("\nTrainer check:")
    print("Epochs:", training_args.num_train_epochs)
    print("Train batch size:", training_args.per_device_train_batch_size)
    print("Eval batch size:", training_args.per_device_eval_batch_size)
    print("Learning rate:", training_args.learning_rate)

if __name__ == "__main__":
    main()