"""Lookups from a readable name to a DocSection or Release id.

The LLM extracts prose surfaces — "Session Objects", "2.34.2" — while the ids
are `doc:<chunk_id>` and `<repo>:<version>`. Without this bridge every
DOCUMENTED_IN / CHANGED_IN / EXPLAINS fact pointing at a doc section or a
release is unresolvable, which is what blocked 927 facts (D31's shortcut).

Ambiguous headings are dropped rather than guessed: "Timeouts" is a section
in both requests and urllib3, and attaching a fact to the wrong library is
worse than not attaching it.
"""

from graphrag.ingest.chunk import Chunk
from graphrag.ingest.node_ids import doc_section_id, release_id

_AMBIGUOUS = object()


def _normalize(name: str) -> str:
    return " ".join(name.lower().split())


def build_doc_release_index(
    chunks: list[Chunk],
) -> tuple[dict[str, str], dict[str, str]]:
    """Return (doc section lookup, release lookup), both keyed by normalized
    readable name. Doc sections are findable by their leaf heading ("Session
    Objects") and by their full breadcrumb ("Advanced Usage > Session
    Objects") — prose uses either."""
    docs: dict[str, object] = {}
    releases: dict[str, object] = {}

    def offer(table: dict, key: str, value: str) -> None:
        existing = table.get(key)
        if existing is None:
            table[key] = value
        elif existing != value:
            table[key] = _AMBIGUOUS

    for chunk in chunks:
        if not chunk.section:
            continue

        if chunk.kind == "doc":
            target = doc_section_id(chunk.chunk_id)
            offer(docs, _normalize(chunk.section), target)
            leaf = chunk.section.split(">")[-1]
            offer(docs, _normalize(leaf), target)

        elif chunk.kind == "changelog":
            offer(
                releases,
                _normalize(chunk.section),
                release_id(chunk.repo, chunk.section),
            )

    return (
        {k: v for k, v in docs.items() if v is not _AMBIGUOUS},
        {k: v for k, v in releases.items() if v is not _AMBIGUOUS},
    )


def lookup(table: dict[str, str], surface: str) -> str | None:
    """Find a surface in one of the tables, or None."""
    return table.get(_normalize(surface))
