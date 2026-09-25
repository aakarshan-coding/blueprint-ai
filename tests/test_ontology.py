from typing import get_args

from graphrag.ingest.llm_extract import RelationshipType
from graphrag.ontology import LLM_RELATIONSHIPS, RELATIONSHIP_TYPES


def test_llm_extract_relationship_literal_matches_the_ontology():
    # llm_extract.py's Literal can't derive from a runtime set, so this test
    # is what catches the two definitions drifting apart instead of a type
    # checker.
    assert set(get_args(RelationshipType)) == LLM_RELATIONSHIPS


def test_ontology_has_fourteen_relationship_types():
    assert len(RELATIONSHIP_TYPES) == 14


def test_every_relationship_type_has_a_meaning():
    """The meanings are what the answer model is shown next to graph lines
    (D63). "verify is defined in Session.request" was read as "Session.request
    applies verify" because nothing said DEFINED_IN means location, not
    behaviour. A type without a meaning would reach the model undefined."""
    from graphrag.ontology import RELATIONSHIP_MEANINGS

    assert set(RELATIONSHIP_MEANINGS) == RELATIONSHIP_TYPES
    assert all(meaning.strip() for meaning in RELATIONSHIP_MEANINGS.values())
