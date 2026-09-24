"""Canonical id schemes for the four entity types that aren't Python symbols
(Module/Class/Function already have one for free: their dotted path).

Exception is deliberately not given its own id scheme here — see D31 in
DECISIONS.md for why classes are typed uniformly as "Class" rather than
distinguishing Exception by inheritance at this stage.
"""


def parameter_id(function_id: str, param_name: str) -> str:
    """A parameter's id is scoped under the function it belongs to — the
    same name in two different functions is two different parameters."""
    return f"{function_id}.{param_name}"


def release_id(repo: str, version: str) -> str:
    """A release is scoped by repo — requests 2.34.2 and urllib3 2.34.2 are
    unrelated releases that happen to share a version number."""
    return f"{repo}:{version}"


def doc_section_id(chunk_id: str) -> str:
    """A DocSection *is* the chunk that represents it, not a separate thing
    with its own identity — reusing the chunk_id guarantees the id already
    exists in Postgres, with no second id scheme to keep in sync."""
    return f"doc:{chunk_id}"


def concept_id(surface: str) -> str:
    """Concepts have no ground truth elsewhere — the LLM's surface text is
    the only source of identity they have. Normalized (case/space-folded,
    same rule as resolve.py) so "Connection Pooling" and "connection
    pooling" collapse to one node instead of two near-duplicates.
    """
    normalized = "".join(ch for ch in surface.lower() if ch.isalnum())
    return f"concept:{normalized}"


def parameter_concept_id(param_name: str) -> str:
    """The id for a parameter *as documentation discusses it* — the shared
    notion, not any one function's signature slot.

    "the url parameter" appears on 21 public documented functions
    (`requests.get`, `requests.post`, `Session.request`, ...). Those are
    peers, not a call chain, so unlike `verify` there is no canonical owner
    to resolve to and never will be. Forcing prose to pick one was the
    mistake; the shared notion gets its own node, and the 21 real parameters
    link to it with IMPLEMENTS.

    Namespaced separately from concept_id so "the url parameter" cannot
    collide with a prose concept that happens to be called "url".
    """
    normalized = "".join(
        ch for ch in param_name.lower() if ch.isalnum() or ch == "_"
    )
    return f"concept:parameter:{normalized}"
