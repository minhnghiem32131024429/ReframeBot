from scripts.build_rag_db import build_chunks


def test_build_chunks_preserves_topic_heading():
    text = """=== TOPIC: COGNITIVE DISTORTIONS ===
1. Catastrophizing: You predict the future negatively. Example: "I failed, so everything is over."
2. Mind Reading: You assume others are judging you.
"""

    chunks = build_chunks(text, max_chars=500)

    assert len(chunks) == 2
    assert all(str(chunk["text"]).startswith("=== TOPIC: COGNITIVE DISTORTIONS ===") for chunk in chunks)
    assert "Catastrophizing" in str(chunks[0]["text"])
    assert chunks[0]["metadata"]["topic"] == "COGNITIVE DISTORTIONS"


def test_build_chunks_splits_long_blocks_on_sentence_boundaries():
    text = """=== TOPIC: TEST ===
First sentence is intentionally short. Second sentence has enough text to force a split when the maximum chunk size is low. Third sentence should remain readable and not start from the middle of a word.
"""

    chunks = build_chunks(text, max_chars=120, overlap_sentences=0)

    assert len(chunks) > 1
    assert all(str(chunk["text"]).startswith("=== TOPIC: TEST ===") for chunk in chunks)
    assert not any(str(chunk["text"]).startswith("=== TOPIC: TEST ===\nird") for chunk in chunks)
