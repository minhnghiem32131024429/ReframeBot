from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer


LABELS = {0: "TASK_1", 1: "TASK_2", 2: "TASK_3"}

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def norm_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def read_eval_json(path: Path) -> list[dict]:
    label_to_id = {value: key for key, value in LABELS.items()}
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for i, item in enumerate(data["accuracy_test"], 1):
        rows.append(
            {
                "line": i,
                "text": norm_text(item["text"]),
                "label": label_to_id[item["expected_label"]],
            }
        )
    return rows


def predict_probs(model_dir: Path, rows: list[dict], batch_size: int, max_len: int) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()

    probs: list[np.ndarray] = []
    texts = [row["text"] for row in rows]
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            encoded = tokenizer(
                texts[i : i + batch_size],
                truncation=True,
                max_length=max_len,
                padding=True,
                return_tensors="pt",
            )
            batch_probs = torch.softmax(model(**encoded).logits, dim=-1).cpu().numpy()
            probs.append(batch_probs)
    return np.vstack(probs)


def apply_task2_threshold(probs: np.ndarray, threshold: float) -> np.ndarray:
    preds = probs.argmax(axis=1)
    force_task2 = probs[:, 1] >= threshold
    preds[force_task2] = 1
    return preds


def sweep_thresholds(y_true: np.ndarray, probs: np.ndarray, thresholds: np.ndarray) -> list[dict]:
    rows: list[dict] = []
    for threshold in thresholds:
        y_pred = apply_task2_threshold(probs, float(threshold))
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=[0, 1, 2],
            zero_division=0,
        )
        rows.append(
            {
                "threshold": float(threshold),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
                "task2_precision": float(precision[1]),
                "task2_recall": float(recall[1]),
                "task2_f1": float(f1[1]),
                "task2_support": int(support[1]),
                "task2_false_negatives": int(np.sum((y_true == 1) & (y_pred != 1))),
                "task2_false_positives": int(np.sum((y_true != 1) & (y_pred == 1))),
            }
        )
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_curve(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    thresholds = [row["threshold"] for row in rows]

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, [row["task2_recall"] for row in rows], label="TASK_2 recall", linewidth=2.5)
    plt.plot(thresholds, [row["task2_precision"] for row in rows], label="TASK_2 precision", linewidth=2.5)
    plt.plot(thresholds, [row["task2_f1"] for row in rows], label="TASK_2 F1", linewidth=2.0)
    plt.plot(thresholds, [row["accuracy"] for row in rows], label="Overall accuracy", linewidth=2.0)
    plt.xlabel("Force TASK_2 when P(TASK_2) >= threshold")
    plt.ylabel("Score")
    plt.ylim(0, 1.03)
    plt.grid(True, alpha=0.25)
    plt.legend(loc="lower right")
    plt.title("Guardrail TASK_2 Threshold Sweep")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def pick_recommendation(rows: list[dict], min_recall: float, max_false_positives: int) -> dict:
    candidates = [
        row
        for row in rows
        if row["task2_recall"] >= min_recall and row["task2_false_positives"] <= max_false_positives
    ]
    if not candidates:
        return max(rows, key=lambda row: (row["task2_f1"], row["task2_recall"], row["accuracy"]))
    return max(candidates, key=lambda row: (row["task2_f1"], row["accuracy"], row["threshold"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep a TASK_2 probability threshold for guardrail safety.")
    parser.add_argument("--model", default="guardrail_model_retrained_clean/best")
    parser.add_argument("--eval-json", default="data/evaluation_test_data.json")
    parser.add_argument("--out-csv", default="reports/guardrail_threshold_sweep.csv")
    parser.add_argument("--out-chart", default="reports/guardrail_threshold_sweep.png")
    parser.add_argument("--min", type=float, default=0.05)
    parser.add_argument("--max", type=float, default=0.95)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--min-recall", type=float, default=0.95)
    parser.add_argument("--max-false-positives", type=int, default=2)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--max-len", type=int, default=256)
    args = parser.parse_args()

    rows = read_eval_json(Path(args.eval_json))
    y_true = np.asarray([row["label"] for row in rows], dtype=int)
    probs = predict_probs(Path(args.model), rows, batch_size=args.batch, max_len=args.max_len)
    thresholds = np.round(np.arange(args.min, args.max + (args.step / 2), args.step), 10)
    results = sweep_thresholds(y_true, probs, thresholds)

    write_csv(results, Path(args.out_csv))
    plot_curve(results, Path(args.out_chart))

    recommended = pick_recommendation(
        results,
        min_recall=args.min_recall,
        max_false_positives=args.max_false_positives,
    )
    argmax_row = next(row for row in results if abs(row["threshold"] - 0.95) < 1e-9)

    print(f"Wrote CSV:   {args.out_csv}")
    print(f"Wrote chart: {args.out_chart}")
    print()
    print("Argmax-like high threshold baseline:")
    print(
        f"  threshold={argmax_row['threshold']:.2f} accuracy={argmax_row['accuracy']:.4f} "
        f"TASK_2 precision={argmax_row['task2_precision']:.4f} "
        f"recall={argmax_row['task2_recall']:.4f} f1={argmax_row['task2_f1']:.4f} "
        f"FN={argmax_row['task2_false_negatives']} FP={argmax_row['task2_false_positives']}"
    )
    print("Recommended threshold:")
    print(
        f"  threshold={recommended['threshold']:.2f} accuracy={recommended['accuracy']:.4f} "
        f"TASK_2 precision={recommended['task2_precision']:.4f} "
        f"recall={recommended['task2_recall']:.4f} f1={recommended['task2_f1']:.4f} "
        f"FN={recommended['task2_false_negatives']} FP={recommended['task2_false_positives']}"
    )


if __name__ == "__main__":
    main()
