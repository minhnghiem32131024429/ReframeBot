import json
import re
from collections import Counter, defaultdict
from pathlib import Path

DATASET_PATH = Path('data/guardrail_dataset.jsonl')

# Split into strong vs weak hints to avoid false flags like "yoga class" or "creative project".
STRONG_ACADEMIC_HINTS = [
    'exam','final','midterm','quiz','assignment','homework','deadline','thesis','dissertation',
    'gpa','grade','grades','lecture','professor','presentation','presenting','speech',
    'study','studying','semester','term paper','lab report','coursework'
]

WEAK_ACADEMIC_HINTS = [
    'course',
    # 'class' is intentionally excluded because it matches many non-academic uses (e.g., yoga class)
    # 'project' is intentionally excluded because it matches creative/home projects
]


def _compile_patterns(hints):
    patterns = []
    for h in hints:
        h = (h or "").strip().lower()
        if not h:
            continue
        patterns.append(re.compile(r"(?<!\w)" + re.escape(h) + r"(?!\w)", flags=re.IGNORECASE))
    return patterns


STRONG_ACADEMIC_PATTERNS = _compile_patterns(STRONG_ACADEMIC_HINTS)
WEAK_ACADEMIC_PATTERNS = _compile_patterns(WEAK_ACADEMIC_HINTS)

CRISIS_HINTS = [
    'suicide','suicidal','kill myself','end my life','take my life','harm myself','hurt myself','overdose',
    'jump','hang myself','shoot myself','razor','bridge','rope'
]

AMBIGUOUS_DISTRESS = [
    "i feel like giving up",
    "i can't take this anymore",
    "no point",
    "meaningless",
    "i don't want to wake up",
    "i wish i could vanish",
    "i want to disappear",
]


def load_jsonl(path: Path):
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                rows.append((i, obj.get('text',''), obj.get('label', None)))
            except Exception:
                # Non-JSON lines are considered invalid.
                rows.append((i, line, 'INVALID'))
    return rows


def main():
    rows = load_jsonl(DATASET_PATH)
    labels = [lbl for _, _, lbl in rows]
    counts = Counter(labels)

    print('=== Label counts ===')
    for k, v in sorted(counts.items(), key=lambda x: str(x[0])):
        print(f'{k}: {v}')

    # Duplicates
    text_to_lines = defaultdict(list)
    for line_no, text, lbl in rows:
        if lbl == 'INVALID':
            continue
        text_to_lines[(text.strip(), lbl)].append(line_no)

    dups = [(k, v) for k, v in text_to_lines.items() if len(v) > 1]
    print('\n=== Duplicates (same text + label) ===')
    print(f'Total duplicates groups: {len(dups)}')
    for (text, lbl), line_nos in dups[:15]:
        short = (text[:90] + '...') if len(text) > 90 else text
        print(f'Label {lbl} repeated {len(line_nos)}x at lines {line_nos}: {short}')

    # Suspects
    label2_academic = []
    label0_crisis = []
    label1_ambiguous = []
    invalid_lines = []

    for line_no, text, lbl in rows:
        lower = text.lower()
        if lbl == 'INVALID':
            invalid_lines.append((line_no, text))
            continue

        strong_hits = sum(1 for p in STRONG_ACADEMIC_PATTERNS if p.search(lower))
        weak_hits = sum(1 for p in WEAK_ACADEMIC_PATTERNS if p.search(lower))
        is_academicish = (strong_hits >= 1) or (weak_hits >= 2)

        if lbl == 2 and is_academicish:
            label2_academic.append((line_no, text))

        if lbl == 0 and any(h in lower for h in CRISIS_HINTS):
            label0_crisis.append((line_no, text))

        if lbl == 1 and any(p in lower for p in AMBIGUOUS_DISTRESS) and not any(h in lower for h in CRISIS_HINTS):
            label1_ambiguous.append((line_no, text))

    print('\n=== Invalid (non-JSON) lines ===')
    print(f'Total invalid: {len(invalid_lines)}')
    for ln, t in invalid_lines[:20]:
        print(f'Line {ln}: {t[:120]}')

    print('\n=== Suspects: label=2 but academic-ish ===')
    print(f'Total suspects: {len(label2_academic)}')
    for ln, t in label2_academic[:30]:
        print(f'Line {ln}: {t}')

    print('\n=== Suspects: label=0 but crisis-ish ===')
    print(f'Total suspects: {len(label0_crisis)}')
    for ln, t in label0_crisis[:30]:
        print(f'Line {ln}: {t}')

    print('\n=== Suspects: label=1 but ambiguous distress (no explicit self-harm) ===')
    print(f'Total suspects: {len(label1_ambiguous)}')
    for ln, t in label1_ambiguous[:30]:
        print(f'Line {ln}: {t}')


if __name__ == '__main__':
    main()
