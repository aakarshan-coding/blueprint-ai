"""LLM extraction pass: relationships that exist only as prose, not as code
structure — the eight relationship types ast_extract.py cannot see.

Same discipline as the AST pass: the model returns surface names ("Session",
"PoolManager"), never canonical ids. Resolution is resolve.py's job, using
one alias table for both passes.

chunk_id is never asked of the model. The caller already knows which chunk
it sent — the model retyping that id back is a chance to get it wrong, not a
fact worth extracting. We attach it ourselves after the call returns.
"""

from typing import Literal

from pydantic import BaseModel

MODEL = "gpt-4o"  # placeholder — swap for whichever OpenAI model you want to pay for

EntityType = Literal[
    "Module", "Class", "Function", "Exception", "Parameter", "Concept",
    "DocSection", "Release",
]

RelationshipType = Literal[
    "DOCUMENTED_IN", "EXPLAINS", "IMPLEMENTS", "CONTROLS",
    "DELEGATES_TO", "WRAPS_EXCEPTION", "CHANGED_IN", "CONSTRAINS",
]


class Relationship(BaseModel):
    """One relationship as the model read it — surface names, not resolved ids."""

    source_surface: str
    source_type: EntityType
    relationship: RelationshipType
    target_surface: str
    target_type: EntityType
    confidence: float


class ExtractionResult(BaseModel):
    """The model's full answer for one chunk: every relationship it found."""

    relationships: list[Relationship]


class ExtractedEdge(BaseModel):
    """A Relationship joined with the chunk_id the caller supplied."""

    source_surface: str
    source_type: EntityType
    relationship: RelationshipType
    target_surface: str
    target_type: EntityType
    confidence: float
    chunk_id: str


_SYSTEM_PROMPT = """\
You extract relationships between software entities from documentation, \
docstrings, and changelog text.

Entity types: Module, Class, Function, Exception, Parameter, Concept, \
DocSection, Release.

Relationship types you may use:
- DOCUMENTED_IN: a symbol is described by a doc section
- EXPLAINS: a doc section explains a concept
- IMPLEMENTS: a symbol implements a concept
- CONTROLS: a parameter controls a behavior or concept
- DELEGATES_TO: one component hands off work to another
- WRAPS_EXCEPTION: one exception is raised in response to another
- CHANGED_IN: a symbol changed in a release
- CONSTRAINS: a concept limits or restricts another

Only extract relationships the text actually states. Do not infer a \
relationship from general knowledge of these libraries — if the chunk \
doesn't say it, leave it out. Use the exact surface name as written in the \
text (e.g. "Session", not "requests.Session") — do not resolve or guess a \
fully-qualified name.
"""


def extract_relationships(
    chunk_text: str, *, chunk_id: str, client, model: str = MODEL
) -> list[ExtractedEdge]:
    """Extract every LLM-derived relationship stated in one prose chunk."""
    response = client.responses.parse(
        model=model,
        instructions=_SYSTEM_PROMPT,
        input=chunk_text,
        text_format=ExtractionResult,
    )

    return [
        ExtractedEdge(**r.model_dump(), chunk_id=chunk_id)
        for r in response.output_parsed.relationships
    ]
