from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer


LABELS = {0: "TASK_1", 1: "TASK_2", 2: "TASK_3"}

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def norm_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append({"line": line_no, "text": norm_text(obj["text"]), "label": int(obj["label"])})
    return rows


def stratified_split(items: list[dict], val_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    buckets: dict[int, list[dict]] = defaultdict(list)
    for item in items:
        buckets[item["label"]].append(item)

    rng = random.Random(seed)
    train: list[dict] = []
    val: list[dict] = []
    for label in sorted(buckets):
        bucket = list(buckets[label])
        rng.shuffle(bucket)
        n_val = max(1, int(len(bucket) * val_ratio))
        val.extend(bucket[:n_val])
        train.extend(bucket[n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def read_eval_json(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for i, item in enumerate(data["accuracy_test"], 1):
        expected = item["expected_label"]
        label = {value: key for key, value in LABELS.items()}[expected]
        rows.append({"line": i, "text": norm_text(item["text"]), "label": label})
    return rows


def predict(model_dir: Path, rows: list[dict], batch_size: int, max_len: int) -> tuple[np.ndarray, np.ndarray]:
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()

    y_pred: list[int] = []
    confidence: list[float] = []
    texts = [row["text"] for row in rows]

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded = tokenizer(
                batch,
                truncation=True,
                max_length=max_len,
                padding=True,
                return_tensors="pt",
            )
            probs = torch.softmax(model(**encoded).logits, dim=-1)
            y_pred.extend(probs.argmax(dim=-1).cpu().numpy().tolist())
            confidence.extend(probs.max(dim=-1).values.cpu().numpy().tolist())

    return np.asarray(y_pred, dtype=int), np.asarray(confidence, dtype=float)


def print_report(rows: list[dict], y_pred: np.ndarray, confidence: np.ndarray) -> None:
    y_true = np.asarray([row["label"] for row in rows], dtype=int)
    label_ids = [0, 1, 2]
    names = [LABELS[label] for label in label_ids]

    print(f"Rows: {len(rows)}")
    print(f"Class counts: {dict(Counter(y_true))}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print("Confusion matrix rows=true cols=pred labels=TASK_1,TASK_2,TASK_3")
    print(confusion_matrix(y_true, y_pred, labels=label_ids))
    print()
    print(classification_report(y_true, y_pred, labels=label_ids, target_names=names, digits=4, zero_division=0))

    misses = [(row, int(pred), float(conf)) for row, pred, conf in zip(rows, y_pred, confidence) if row["label"] != pred]
    if misses:
        print("Misclassified:")
        for row, pred, conf in misses:
            print(
                f"  L{row['line']} true={LABELS[row['label']]} pred={LABELS[pred]} "
                f"conf={conf:.4f} text={row['text']}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained guardrail classifier.")
    parser.add_argument("--model", default="guardrail_model_retrained/best")
    parser.add_argument("--data", default="data/guardrail_dataset_clean.jsonl")
    parser.add_argument("--eval-json", default="")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--max-len", type=int, default=256)
    args = parser.parse_args()

    if args.eval_json:
        rows = read_eval_json(Path(args.eval_json))
        print(f"Evaluation source: {args.eval_json} accuracy_test")
    else:
        all_rows = read_jsonl(Path(args.data))
        _, rows = stratified_split(all_rows, val_ratio=args.val_ratio, seed=args.seed)
        print(f"Evaluation source: validation split from {args.data}")

    y_pred, confidence = predict(Path(args.model), rows, batch_size=args.batch, max_len=args.max_len)
    print_report(rows, y_pred, confidence)


if __name__ == "__main__":
    main()
