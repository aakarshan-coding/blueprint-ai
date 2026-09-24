"""Citation validation (design §9). Checks that every citation an answer
makes actually resolves to a chunk_id that was retrieved this turn — this is
what eliminates invented sources, and it costs nothing beyond a regex.
"""

import re
from dataclasses import dataclass

_CITATION_PATTERN = re.compile(r"\[([^\[\]]+)\]")


@dataclass(frozen=True)
class CitationCheck:
    valid: bool
    invalid_citations: list[str]


def extract_citations(answer: str) -> list[str]:
    """Every bracketed id the answer cites, in order, duplicates included."""
    return _CITATION_PATTERN.findall(answer)


def validate_citations(answer: str, *, retrieved_ids: set[str]) -> CitationCheck:
    """An answer is valid only if it cites at least one chunk, and every
    citation it makes is one this turn actually retrieved. An answer with
    claims and zero citations is exactly the failure mode this exists to
    catch, not a pass by default.
    """
    citations = extract_citations(answer)
    invalid = [c for c in citations if c not in retrieved_ids]
    return CitationCheck(valid=bool(citations) and not invalid, invalid_citations=invalid)
