"""Load the committed corpus into typed chunks at process startup.

REQ-14: the corpus is committed in the repository; startup never crawls the
web. This module only reads local JSON files.
"""

import json
from functools import lru_cache
from importlib import resources

from getnet_support.domain.chat import Language, Market, Volatility
from getnet_support.domain.knowledge import CorpusChunk

_CORPUS_FILES = ("getnet_br.json", "getnet_global.json")


def _parse_chunk(raw: dict[str, str]) -> CorpusChunk:
    """Translate one raw corpus record into a domain :class:`CorpusChunk`."""
    return CorpusChunk(
        id=raw["id"],
        text=raw["text"],
        title=raw["title"],
        source=raw["source"],
        market=Market(raw["market"]),
        language=Language(raw["language"]),
        topic=raw["topic"],
        retrieved_at=raw["retrieved_at"],
        volatility=Volatility(raw["volatility"]),
    )


@lru_cache(maxsize=1)
def load_corpus() -> tuple[CorpusChunk, ...]:
    """Load and merge every committed corpus file, once per process."""
    chunks: list[CorpusChunk] = []
    package = "getnet_support.adapters.corpus"
    for filename in _CORPUS_FILES:
        raw_text = resources.files(package).joinpath(filename).read_text(encoding="utf-8")
        chunks.extend(_parse_chunk(record) for record in json.loads(raw_text))
    return tuple(chunks)
