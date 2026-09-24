"""The frozen ontology from the design (§4): every entity type and
relationship type the graph is allowed to contain. Single source of truth —
load_graph.py validates every write against this before it touches Neo4j, so
a typo or an out-of-ontology label fails loudly instead of silently creating
a node type nobody can query.
"""

ENTITY_TYPES = frozenset({
    "Module", "Class", "Function", "Exception", "Parameter", "Concept",
    "DocSection", "Release",
})

AST_RELATIONSHIPS = frozenset({
    "DEFINED_IN", "INHERITS_FROM", "CALLS", "IMPORTS", "RAISES", "HAS_PARAMETER",
})

LLM_RELATIONSHIPS = frozenset({
    "DOCUMENTED_IN", "EXPLAINS", "IMPLEMENTS", "CONTROLS", "DELEGATES_TO",
    "WRAPS_EXCEPTION", "CHANGED_IN", "CONSTRAINS",
})

RELATIONSHIP_TYPES = AST_RELATIONSHIPS | LLM_RELATIONSHIPS
