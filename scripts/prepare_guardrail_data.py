from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


LABELS = {0: "TASK_1", 1: "TASK_2", 2: "TASK_3"}


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
            text = norm_text(obj.get("text", ""))
            label = obj.get("label")
            if not text or label is None:
                continue
            label = int(label)
            if label not in LABELS:
                raise ValueError(f"{path}:{line_no} has invalid label: {label!r}")
            rows.append({"text": text, "label": label, "_source": str(path), "_line": line_no})
    return rows


def dedupe_rows(rows: list[dict]) -> tuple[list[dict], list[tuple[str, list[int]]]]:
    seen: dict[str, dict] = {}
    conflicts: list[tuple[str, list[int]]] = []

    for row in rows:
        key = row["text"].casefold()
        existing = seen.get(key)
        if existing is None:
            seen[key] = row
            continue
        if existing["label"] != row["label"]:
            conflicts.append((row["text"], [existing["label"], row["label"]]))
            continue

    return list(seen.values()), conflicts


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps({"text": row["text"], "label": row["label"]}, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a deduplicated guardrail training dataset.")
    parser.add_argument("--base", default="data/guardrail_dataset.jsonl")
    parser.add_argument("--augment", default="data/guardrail_hard_cases.jsonl")
    parser.add_argument("--out", default="data/guardrail_dataset_clean.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-duplicates", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(Path(args.base))
    augment_path = Path(args.augment)
    if augment_path.exists():
        rows.extend(read_jsonl(augment_path))

    before = len(rows)
    conflicts: list[tuple[str, list[int]]] = []
    if not args.keep_duplicates:
        rows, conflicts = dedupe_rows(rows)

    if conflicts:
        examples = "\n".join(f"- {text!r}: labels={labels}" for text, labels in conflicts[:10])
        raise RuntimeError(f"Found conflicting labels for identical text:\n{examples}")

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    write_jsonl(rows, Path(args.out))

    counts = Counter(row["label"] for row in rows)
    print(f"Wrote {len(rows)} rows to {args.out} ({before - len(rows)} duplicates removed).")
    for label in sorted(LABELS):
        count = counts[label]
        pct = count / len(rows) if rows else 0.0
        print(f"  {label} {LABELS[label]}: {count} ({pct:.2%})")


if __name__ == "__main__":
    main()
