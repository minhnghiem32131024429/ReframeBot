from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


LABELS = {0: "TASK_1_CBT", 1: "TASK_2_CRISIS", 2: "TASK_3_OOS"}

CRISIS_PHRASES = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\bkill myself\b",
        r"\bsuicid(?:e|al)\b",
        r"\bend (?:my life|it all|everything)\b",
        r"\bself[- ]harm\b",
        r"\bharm myself\b",
        r"\boverdose\b",
        r"\bno reason to live\b",
        r"\bnot be here anymore\b",
        r"\bbetter off without me\b",
        r"\bnever wake up\b",
        r"\bwant to die\b",
        r"\bdon'?t want to live\b",
        r"\bstop existing\b",
    ]
]

ACADEMIC_PHRASES = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\bexam\b",
        r"\bfinals?\b",
        r"\bmidterm\b",
        r"\bgrades?\b",
        r"\bgpa\b",
        r"\bclass(?:es)?\b",
        r"\bcourse\b",
        r"\bprofessor\b",
        r"\bthesis\b",
        r"\bdissertation\b",
        r"\bassignments?\b",
        r"\bhomework\b",
        r"\bstud(?:y|ying|ied|ies)\b",
        r"\bschool\b",
        r"\buniversity\b",
        r"\bcollege\b",
        r"\bcampus\b",
        r"\blectures?\b",
        r"\bpapers?\b",
        r"\bresearch\b",
        r"\bscholarship\b",
        r"\bsemester\b",
        r"\bacademic\b",
    ]
]

OOS_TAXONOMY = {
    "questions/how-to/recommendation": re.compile(
        r"\?|\bhow do i\b|\bcan you\b|\bwhat(?:'s| is)\b|\btell me\b|\brecommend\b",
        re.IGNORECASE,
    ),
    "food/cooking/restaurants": re.compile(
        r"\b(recipe|cook|bake|restaurant|pizza|dinner|coffee|cuisine|cake|cookies|food)\b",
        re.IGNORECASE,
    ),
    "mental-health-info": re.compile(
        r"\b(cbt|depression|anxiety disorder|suicide prevention|self-harm prevalence|clinical depression)\b",
        re.IGNORECASE,
    ),
    "work/career": re.compile(r"\b(job|promotion|interview|work|manager|career)\b", re.IGNORECASE),
    "travel/outdoors/weather": re.compile(
        r"\b(weather|forecast|vacation|flight|travel|hiking|beach|japan|park|lake|sunset|hanoi|da nang)\b",
        re.IGNORECASE,
    ),
    "entertainment/media/arts": re.compile(
        r"\b(movie|joke|poem|album|band|music|concert|guitar|video game|book|comic|programmers)\b",
        re.IGNORECASE,
    ),
    "tech/devices": re.compile(
        r"\b(computer|smartphone|phone|app|tv|programming|earbuds|laptop|headphones)\b",
        re.IGNORECASE,
    ),
    "home/errands/life-admin": re.compile(
        r"\b(laundry|garage|furniture|kitchen|office|glasses|stamps|bills|wallet|license|landlord|faucet)\b",
        re.IGNORECASE,
    ),
    "health/fitness/wellness": re.compile(
        r"\b(fitness|marathon|yoga|meditat|ankle|running|stretches|wellness)\b",
        re.IGNORECASE,
    ),
}


def read_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append({"line": line_no, "text": " ".join(obj["text"].split()), "label": int(obj["label"])})
    return rows


def print_distribution(rows: list[dict]) -> None:
    counts = Counter(row["label"] for row in rows)
    print(f"Total rows: {len(rows)}")
    for label in sorted(LABELS):
        count = counts[label]
        pct = count / len(rows) if rows else 0.0
        print(f"  {label} {LABELS[label]}: {count} ({pct:.2%})")
    if counts:
        print(f"Imbalance max/min: {max(counts.values()) / min(counts.values()):.2f}x")


def print_duplicates(rows: list[dict]) -> None:
    by_text: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_text[row["text"].casefold()].append(row)

    duplicates = {text: hits for text, hits in by_text.items() if len(hits) > 1}
    conflicts = {
        text: hits
        for text, hits in duplicates.items()
        if len({hit["label"] for hit in hits}) > 1
    }
    print(f"Exact duplicate texts: {len(duplicates)}")
    print(f"Conflicting exact labels: {len(conflicts)}")
    for text, hits in list(conflicts.items())[:10]:
        labels = [hit["label"] for hit in hits]
        lines = [hit["line"] for hit in hits]
        print(f"  conflict labels={labels} lines={lines}: {text}")


def print_boundary_checks(rows: list[dict], samples: int) -> None:
    for label in sorted(LABELS):
        subset = [row for row in rows if row["label"] == label]
        crisis_hits = [row for row in subset if any(pattern.search(row["text"]) for pattern in CRISIS_PHRASES)]
        academic_hits = [row for row in subset if any(pattern.search(row["text"]) for pattern in ACADEMIC_PHRASES)]
        print(
            f"{LABELS[label]} phrase rates: "
            f"crisis={len(crisis_hits)}/{len(subset)} ({len(crisis_hits) / len(subset):.1%}), "
            f"academic={len(academic_hits)}/{len(subset)} ({len(academic_hits) / len(subset):.1%})"
        )
        if label == 0 and crisis_hits:
            print("  CBT rows with crisis phrases:")
            for row in crisis_hits[:samples]:
                print(f"    L{row['line']}: {row['text']}")
        if label == 1 and academic_hits:
            print("  Crisis rows with academic phrases:")
            for row in academic_hits[:samples]:
                print(f"    L{row['line']}: {row['text']}")


def print_oos_taxonomy(rows: list[dict], samples: int) -> None:
    oos = [row for row in rows if row["label"] == 2]
    counts: Counter[str] = Counter()
    examples: dict[str, list[str]] = defaultdict(list)
    for row in oos:
        matched = [name for name, pattern in OOS_TAXONOMY.items() if pattern.search(row["text"])]
        if not matched:
            matched = ["other"]
        for name in matched:
            counts[name] += 1
            if len(examples[name]) < samples:
                examples[name].append(f"L{row['line']}: {row['text']}")

    print("OOS taxonomy (multi-label heuristic):")
    for name, count in counts.most_common():
        print(f"  {name}: {count}")
        for example in examples[name]:
            print(f"    {example}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit guardrail dataset quality.")
    parser.add_argument("--data", default="data/guardrail_dataset_clean.jsonl")
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()

    rows = read_rows(Path(args.data))
    print_distribution(rows)
    print()
    print_duplicates(rows)
    print()
    print_boundary_checks(rows, samples=args.samples)
    print()
    print_oos_taxonomy(rows, samples=args.samples)


if __name__ == "__main__":
    main()
