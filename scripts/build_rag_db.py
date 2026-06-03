"""Build the ChromaDB RAG database from data/knowledge.txt.

Usage:
    uv run --extra scripts python scripts/build_rag_db.py
    uv run --extra scripts python scripts/build_rag_db.py --text data/knowledge.txt --db rag_db --max-chars 900
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TOPIC_RE = re.compile(r"^===\s*TOPIC:\s*(.*?)\s*===$", re.MULTILINE)


def _clean_text(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text.strip())


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def _split_long_block(block: str, max_chars: int, overlap_sentences: int) -> list[str]:
    """Split a long paragraph/list into sentence-aware chunks."""
    sentences = _split_sentences(block)
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        sentence_len = len(sentence) + 1
        if current and current_len + sentence_len > max_chars:
            chunks.append(" ".join(current).strip())
            current = current[-overlap_sentences:] if overlap_sentences > 0 else []
            current_len = sum(len(item) + 1 for item in current)
        current.append(sentence)
        current_len += sentence_len

    if current:
        chunks.append(" ".join(current).strip())
    return chunks


def _topic_sections(text: str) -> list[tuple[str, str]]:
    matches = list(_TOPIC_RE.finditer(text))
    if not matches:
        return [("GENERAL", _clean_text(text))]

    sections: list[tuple[str, str]] = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        topic = match.group(1).strip()
        body = _clean_text(text[start:end])
        if body:
            sections.append((topic, body))
    return sections


def build_chunks(text: str, max_chars: int = 900, overlap_sentences: int = 1) -> list[dict[str, object]]:
    chunks: list[dict[str, object]] = []

    for topic, body in _topic_sections(text):
        heading = f"=== TOPIC: {topic} ==="
        blocks = [block.strip() for block in re.split(r"\n\s*\n", body) if block.strip()]
        for block_index, block in enumerate(blocks):
            numbered_items = re.findall(r"(?ms)^\d+\.\s+.*?(?=^\d+\.\s+|\Z)", block)
            pieces = numbered_items if len(numbered_items) > 1 else [block]
            if topic.upper() != "COGNITIVE DISTORTIONS":
                pieces = [block]

            for piece_index, piece in enumerate(pieces):
                piece = _clean_text(piece)
                text_with_heading = f"{heading}\n{piece}"
                if len(text_with_heading) <= max_chars:
                    split_pieces = [piece]
                else:
                    budget = max(200, max_chars - len(heading) - 1)
                    split_pieces = _split_long_block(piece, budget, overlap_sentences)

                for split_index, split_piece in enumerate(split_pieces):
                    content = f"{heading}\n{split_piece.strip()}"
                    chunks.append(
                        {
                            "text": content,
                            "metadata": {
                                "topic": topic,
                                "block_index": block_index,
                                "piece_index": piece_index,
                                "split_index": split_index,
                            },
                        }
                    )

    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ChromaDB RAG database from a text file")
    parser.add_argument("--text", default=str(_REPO_ROOT / "data" / "knowledge.txt"))
    parser.add_argument("--db", default=str(_REPO_ROOT / "rag_db"))
    parser.add_argument("--max-chars", type=int, default=900, help="Maximum characters per chunk")
    parser.add_argument("--overlap-sentences", type=int, default=1, help="Sentence overlap for long chunks")
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    args = parser.parse_args()

    import chromadb
    from sentence_transformers import SentenceTransformer

    text_path = Path(args.text)
    if not text_path.exists():
        raise FileNotFoundError(f"Text file not found: {text_path}")

    print(f"Loading embedding model: {args.model}")
    embedder = SentenceTransformer(args.model)

    print(f"Reading: {text_path}")
    text = text_path.read_text(encoding="utf-8")
    chunk_records = build_chunks(
        text,
        max_chars=args.max_chars,
        overlap_sentences=args.overlap_sentences,
    )
    chunks = [str(record["text"]) for record in chunk_records]
    metadatas = [dict(record["metadata"]) for record in chunk_records]
    print(f"Created {len(chunks)} topic-aware chunks (max_chars={args.max_chars})")

    client = chromadb.PersistentClient(path=args.db)
    try:
        client.delete_collection(name="cbt_knowledge")
    except Exception:
        pass
    collection = client.get_or_create_collection(name="cbt_knowledge")

    embeddings = embedder.encode(chunks).tolist()
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=[str(i) for i in range(len(chunks))],
    )

    print(f"RAG database saved to: {args.db}")


if __name__ == "__main__":
    main()
