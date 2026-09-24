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
