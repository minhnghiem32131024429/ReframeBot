from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_REPO_ROOT / ".env")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def read_eval_items(path: Path) -> list[dict[str, str]]:
    items = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise ValueError(f"{path} must contain a JSON list.")
    for item in items:
        if "question" not in item or "ground_truth" not in item:
            raise ValueError("Each RAG eval item must contain question and ground_truth.")
    return items


def retrieve_contexts(
    *,
    questions: list[str],
    rag_db: Path,
    embed_model: str,
    top_k: int,
) -> list[list[str]]:
    import chromadb
    from sentence_transformers import SentenceTransformer

    if not rag_db.exists():
        raise FileNotFoundError(f"RAG DB not found: {rag_db}. Run scripts/build_rag_db.py first.")

    embedder = SentenceTransformer(embed_model)
    client = chromadb.PersistentClient(path=str(rag_db))
    collection = client.get_collection(name="cbt_knowledge")

    contexts: list[list[str]] = []
    for question in questions:
        emb = embedder.encode([question]).tolist()
        results = collection.query(query_embeddings=emb, n_results=top_k)
        contexts.append(results.get("documents", [[]])[0])
    return contexts


def generate_answer(api_url: str, question: str, username: str, session_id: str, timeout: int) -> str:
    response = requests.post(
        api_url,
        json={
            "username": username,
            "session_id": session_id,
            "history": [{"role": "user", "content": question}],
        },
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    return data["response"]


def build_samples(args: argparse.Namespace) -> list[dict[str, Any]]:
    items = read_eval_items(Path(args.data))
    questions = [item["question"] for item in items]
    contexts = retrieve_contexts(
        questions=questions,
        rag_db=Path(args.rag_db),
        embed_model=args.embed_model,
        top_k=args.top_k,
    )

    samples: list[dict[str, Any]] = []
    for i, (item, item_contexts) in enumerate(zip(items, contexts), 1):
        answer = item.get("answer")
        if not answer or args.regenerate_answers:
            answer = generate_answer(
                args.api_url,
                item["question"],
                username=args.username,
                session_id=f"{args.session_prefix}-{i}",
                timeout=args.timeout,
            )
        samples.append(
            {
                "question": item["question"],
                "answer": answer,
                "contexts": item_contexts,
                "ground_truth": item["ground_truth"],
            }
        )
    return samples


def run_ragas(samples: list[dict[str, Any]], args: argparse.Namespace):
    try:
        from datasets import Dataset
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_openai import ChatOpenAI
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
        from ragas.run_config import RunConfig
    except ImportError as exc:
        raise RuntimeError(
            "Missing RAGAS dependencies. Install/update with: uv sync --extra scripts"
        ) from exc

    if args.judge_provider == "groq":
        api_key = os.environ.get(args.groq_api_key_env, "")
        if not api_key:
            raise RuntimeError(
                f"{args.groq_api_key_env} is required for --judge-provider groq. "
                "Set it in .env or the shell before running this script."
            )
        evaluator_llm = ChatOpenAI(
            model=args.judge_model,
            temperature=0,
            base_url=args.groq_base_url,
            api_key=api_key,
        )
    elif args.judge_provider == "openai":
        api_key = os.environ.get(args.openai_api_key_env, "")
        if not api_key:
            raise RuntimeError(
                f"{args.openai_api_key_env} is required for --judge-provider openai. "
                "Set it in .env or the shell before running this script."
            )
        evaluator_llm = ChatOpenAI(
            model=args.judge_model,
            temperature=0,
            api_key=api_key,
        )
    else:
        raise RuntimeError(
            f"Unsupported judge provider: {args.judge_provider}. Use 'groq' or 'openai'."
        )

    dataset = Dataset.from_list(samples)
    evaluator_embeddings = HuggingFaceEmbeddings(
        model_name=args.eval_embedding_model,
        model_kwargs={"device": args.eval_embedding_device},
        encode_kwargs={"normalize_embeddings": True},
    )

    metrics = [faithfulness, context_precision, context_recall]
    if args.answer_relevancy:
        metrics.append(answer_relevancy)

    ragas_timeout = None if args.ragas_timeout <= 0 else args.ragas_timeout
    return evaluate(
        dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=RunConfig(timeout=ragas_timeout, max_workers=args.ragas_workers),
        raise_exceptions=args.raise_exceptions,
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ReframeBot RAG with RAGAS.")
    parser.add_argument("--data", default="data/rag_eval_dataset.json")
    parser.add_argument("--api-url", default="http://localhost:8000/chat")
    parser.add_argument("--rag-db", default="rag_db")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--embed-model", default="all-MiniLM-L6-v2", help="Retriever embedding model.")
    parser.add_argument("--judge-provider", choices=["groq", "openai"], default="groq")
    parser.add_argument("--judge-model", default="llama-3.3-70b-versatile", help="LLM used by RAGAS as evaluator.")
    parser.add_argument("--groq-base-url", default="https://api.groq.com/openai/v1")
    parser.add_argument("--groq-api-key-env", default="GROQ_API_KEY")
    parser.add_argument("--openai-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--eval-embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--eval-embedding-device", default="cpu")
    parser.add_argument("--answer-relevancy", action="store_true", help="Also run answer_relevancy.")
    parser.add_argument(
        "--ragas-timeout",
        type=int,
        default=0,
        help="RAGAS metric timeout in seconds. 0 disables RAGAS asyncio timeout.",
    )
    parser.add_argument("--ragas-workers", type=int, default=2)
    parser.add_argument("--raise-exceptions", action="store_true")
    parser.add_argument("--regenerate-answers", action="store_true", help="Ignore answer fields in dataset, if present.")
    parser.add_argument("--username", default="ragas-eval")
    parser.add_argument("--session-prefix", default="ragas")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--out-json", default="reports/ragas_eval.json")
    parser.add_argument("--out-csv", default="reports/ragas_eval.csv")
    args = parser.parse_args()

    samples = build_samples(args)
    result = run_ragas(samples, args)

    try:
        rows = result.to_pandas().to_dict(orient="records")
    except Exception:
        rows = samples

    summary = {}
    try:
        for key in rows[0].keys():
            values = [row.get(key) for row in rows if isinstance(row.get(key), (int, float))]
            if values:
                summary[key] = sum(values) / len(values)
    except Exception:
        summary = {}
    write_json(Path(args.out_json), {"summary": summary, "samples": rows})
    write_csv(Path(args.out_csv), rows)

    print("RAGAS summary:")
    for key, value in summary.items():
        try:
            print(f"  {key}: {float(value):.4f}")
        except Exception:
            print(f"  {key}: {value}")
    print(f"JSON: {args.out_json}")
    print(f"CSV:  {args.out_csv}")


if __name__ == "__main__":
    main()
