"""Shared loader for the persisted Getnet corpus, used by every retriever adapter."""

import json
from pathlib import Path

from getnet_support.domain.models import KnowledgeChunk, Locale, Market

CORPUS_FILES = {
    Market.BR: "getnet_br.json",
    Market.GLOBAL: "getnet_global.json",
}

DEFAULT_CORPUS_DIR = Path(__file__).parent / "corpus"


def load_corpus_chunks(corpus_dir: Path | None = None) -> tuple[KnowledgeChunk, ...]:
    """Load every persisted corpus file into typed, immutable chunks."""
    directory = corpus_dir or DEFAULT_CORPUS_DIR
    chunks: list[KnowledgeChunk] = []
    for filename in CORPUS_FILES.values():
        raw_items = json.loads((directory / filename).read_text(encoding="utf-8"))
        chunks.extend(
            KnowledgeChunk(
                id=item["id"],
                text=item["text"],
                title=item["title"],
                source=item["source"],
                market=Market(item["market"]),
                language=Locale(item["language"]),
                topic=item["topic"],
                retrieved_at=item["retrieved_at"],
                volatility=item["volatility"],
            )
            for item in raw_items
        )
    return tuple(chunks)
