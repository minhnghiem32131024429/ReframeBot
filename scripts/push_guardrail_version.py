"""Upload a guardrail checkpoint to Hugging Face with branch/tag versioning.

Usage:
    python scripts/push_guardrail_version.py --version v2-guardrail-clean

By default this uploads guardrail_model_retrained_clean/best to:
    Nhatminh1234/ReframeBot-Guardrail-DistilBERT

Versioning strategy:
    - optionally archive current main as a tag with --archive-main-as
    - upload to a branch named by --version
    - create/update a lightweight tag with the same --version
    - optionally also update main with --update-main
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi


_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_REPO_ROOT / ".env")


DEFAULT_REPO_ID = "Nhatminh1234/ReframeBot-Guardrail-DistilBERT"
DEFAULT_MODEL_PATH = _REPO_ROOT / "guardrail_model_retrained_clean" / "best"


MODEL_CARD = """\
---
language:
- en
license: apache-2.0
base_model: distilbert/distilbert-base-uncased
tags:
- text-classification
- guardrail
- safety
- cbt
- mental-health
- academic-stress
pipeline_tag: text-classification
---

# ReframeBot-Guardrail-DistilBERT

A 3-class DistilBERT classifier for routing ReframeBot user turns:

| Label | Meaning |
|---|---|
| `TASK_1` | CBT / academic stress |
| `TASK_2` | Crisis / self-harm signal |
| `TASK_3` | Out-of-scope |

This version was retrained on `data/guardrail_dataset_clean.jsonl`, which
merges the original guardrail data with curated hard cases for CBT/Crisis
boundaries, Vietnamese text, pills/overdose language, and OOS work/mental
health informational prompts.

## Current System Threshold

The ReframeBot runtime uses the classifier's full probability vector and
routes to `TASK_2` when:

```text
P(TASK_2) >= 0.10
```

after academic-context/follow-up overrides and after the regex + semantic
crisis detector has already run.

## Evaluation

Hard out-of-domain eval set (`data/evaluation_test_data.json`, 60 samples):

| Mode | Accuracy | TASK_2 Precision | TASK_2 Recall | TASK_2 F1 |
|---|---:|---:|---:|---:|
| Argmax only | 0.9667 | 1.0000 | 0.9048 | 0.9500 |
| Tuned `P(TASK_2) >= 0.10` | 0.9833 | 0.9545 | 1.0000 | 0.9767 |

Threshold sweep artifact in the project repo:

- `reports/guardrail_threshold_sweep.csv`
- `reports/guardrail_threshold_sweep.png`

## Usage

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="Nhatminh1234/ReframeBot-Guardrail-DistilBERT",
    revision="v2-guardrail-clean",
)

classifier("I'm stressed about my final exam")
```

For full class probabilities:

```python
classifier("I bought pills to overdose", top_k=None)
```

## Safety Note

This classifier is a routing component, not a standalone crisis intervention
system. ReframeBot also uses regex + semantic crisis detection and crisis
response handling around this model.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Push a versioned guardrail model to Hugging Face Hub.")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--version", required=True, help="Branch/tag name, e.g. v2-guardrail-clean")
    parser.add_argument("--archive-main-as", default="", help="Create a tag from current main before uploading.")
    parser.add_argument("--update-main", action="store_true", help="Also upload this checkpoint to main.")
    return parser.parse_args()


def ensure_token() -> str:
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        print("ERROR: HF_TOKEN is not set in .env")
        sys.exit(1)
    return token


def main() -> None:
    args = parse_args()
    token = ensure_token()
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"ERROR: model path does not exist: {model_path}")
        sys.exit(1)

    api = HfApi(token=token)
    repo_id = args.repo_id
    version = args.version

    print(f"Target repo: https://huggingface.co/{repo_id}")
    print(f"Source:      {model_path}")
    print(f"Version:     {version}")

    if args.archive_main_as:
        print(f"Archiving current main as tag: {args.archive_main_as}")
        try:
            api.create_tag(
                repo_id=repo_id,
                tag=args.archive_main_as,
                revision="main",
                repo_type="model",
                exist_ok=True,
            )
        except TypeError:
            try:
                api.create_tag(
                    repo_id=repo_id,
                    tag=args.archive_main_as,
                    revision="main",
                    repo_type="model",
                )
            except Exception as exc:
                if "already exists" not in str(exc).lower():
                    raise

    try:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        api.create_branch(repo_id=repo_id, branch=version, repo_type="model", exist_ok=True)
    except TypeError:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        try:
            api.create_branch(repo_id=repo_id, branch=version, repo_type="model")
        except Exception as exc:
            if "already exists" not in str(exc).lower():
                raise

    commit_message = f"Upload guardrail {version}"
    api.upload_folder(
        folder_path=str(model_path),
        repo_id=repo_id,
        repo_type="model",
        revision=version,
        commit_message=commit_message,
        ignore_patterns=["optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json"],
    )
    api.upload_file(
        path_or_fileobj=MODEL_CARD.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        revision=version,
        commit_message=f"Update model card for {version}",
    )

    try:
        api.create_tag(repo_id=repo_id, tag=version, revision=version, repo_type="model", exist_ok=True)
    except TypeError:
        try:
            api.create_tag(repo_id=repo_id, tag=version, revision=version, repo_type="model")
        except Exception as exc:
            if "already exists" not in str(exc).lower():
                raise

    if args.update_main:
        api.upload_folder(
            folder_path=str(model_path),
            repo_id=repo_id,
            repo_type="model",
            revision="main",
            commit_message=f"Promote guardrail {version} to main",
            ignore_patterns=["optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json"],
        )
        api.upload_file(
            path_or_fileobj=MODEL_CARD.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            revision="main",
            commit_message=f"Update model card for {version}",
        )

    print("Done.")
    print(f"Versioned model: https://huggingface.co/{repo_id}/tree/{version}")
    if args.update_main:
        print(f"Main updated:     https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
