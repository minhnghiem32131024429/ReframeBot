import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def norm_text(text: str) -> str:
    text = (text or "").strip()
    # Collapse repeated whitespace
    return " ".join(text.split())


def load_jsonl(path: Path):
    rows = []
    if not path.exists():
        raise FileNotFoundError(str(path))
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = norm_text(obj.get("text", ""))
            label = obj.get("label", None)
            if text == "" or label is None:
                continue
            rows.append((text, int(label), path.name, i))
    return rows


def write_jsonl(path: Path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Prepare guardrail data: merge, dedup, and detect conflicts")
    parser.add_argument("--base", default="data/guardrail_dataset.jsonl")
    parser.add_argument("--extra", default="data/guardrail_dataset_hard_negatives.jsonl")
    parser.add_argument("--out", default="data/guardrail_dataset_clean.jsonl")
    parser.add_argument("--conflicts", default="data/guardrail_dataset_conflicts.jsonl")
    parser.add_argument("--synthetic_hard_negatives", type=int, default=120)
    args = parser.parse_args()

    base_path = Path(args.base)
    extra_path = Path(args.extra)
    out_path = Path(args.out)
    conflicts_path = Path(args.conflicts)

    base_rows = load_jsonl(base_path)
    extra_rows = []
    if extra_path.exists():
        extra_rows = load_jsonl(extra_path)

    combined = base_rows + extra_rows

    # Add synthetic hard negatives (label 0) to reduce false TASK_2.
    # These intentionally include strong distress phrases but are grounded in academic context.
    if args.synthetic_hard_negatives and args.synthetic_hard_negatives > 0:
        distress_phrases = [
            "I can't take this anymore",
            "I feel like giving up",
            "I feel hopeless",
            "I don't know what to do anymore",
            "I'm overwhelmed",
            "I feel like I'm breaking down",
            "Everything feels pointless",
        ]
        academic_contexts = [
            "because of finals week",
            "because I'm failing this course",
            "because my thesis deadline is coming",
            "because I have too many assignments",
            "because of my exam tomorrow",
            "because my presentation is soon",
            "because my GPA is dropping",
            "because of academic pressure",
        ]
        safety_clauses = [
            "but I don't want to hurt myself",
            "and I'm not suicidal",
            "and I want help managing my stress",
            "and I just need help reframing my thoughts",
        ]

        existing_texts = set(text for text, _, _, _ in combined)
        synth = []
        for d in distress_phrases:
            for a in academic_contexts:
                for s in safety_clauses:
                    txt = norm_text(f"{d} {a}, {s}.")
                    if txt not in existing_texts:
                        synth.append((txt, 0, "synthetic", 0))
                        existing_texts.add(txt)

        # Keep deterministic + capped
        synth = synth[: args.synthetic_hard_negatives]
        combined.extend(synth)

    # Group by normalized text
    label_by_text = defaultdict(list)
    sources_by_text = defaultdict(list)
    for text, label, src, line_no in combined:
        label_by_text[text].append(label)
        sources_by_text[text].append({"source": src, "line": line_no, "label": label})

    conflicts = []
    clean = []
    label_counts_clean = Counter()

    for text, labels in label_by_text.items():
        unique_labels = sorted(set(labels))
        if len(unique_labels) > 1:
            conflicts.append({
                "text": text,
                "labels": unique_labels,
                "examples": sources_by_text[text],
            })
            continue

        label = unique_labels[0]
        clean.append({"text": text, "label": label})
        label_counts_clean[label] += 1

    # Deterministic output
    clean.sort(key=lambda x: (x["label"], x["text"]))

    write_jsonl(out_path, clean)
    write_jsonl(conflicts_path, conflicts)

    print("=== Prepare guardrail data ===")
    print(f"Base: {base_path} ({len(base_rows)} rows)")
    print(f"Extra: {extra_path} ({len(extra_rows)} rows)")
    if args.synthetic_hard_negatives and args.synthetic_hard_negatives > 0:
        print(f"Synthetic hard negatives added: {min(args.synthetic_hard_negatives, max(0, len(combined) - len(base_rows) - len(extra_rows)))}")
    print(f"Combined: {len(combined)} rows")
    print(f"Unique texts: {len(label_by_text)}")
    print(f"Conflicts (same text, different labels): {len(conflicts)}")
    print(f"Clean output: {out_path} ({len(clean)} rows)")
    print(f"Conflicts output: {conflicts_path}")
    print("\n=== Clean label distribution ===")
    for lbl, cnt in sorted(label_counts_clean.items()):
        print(f"{lbl}: {cnt}")


if __name__ == "__main__":
    main()
