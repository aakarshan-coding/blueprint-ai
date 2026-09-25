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

# What each relationship means, in the words the answer model is shown next
# to the graph lines that use it (D63). A graph line states a relationship;
# the model has to know what kind of claim that is. "verify is defined in
# Session.request" was read as "Session.request is the function that applies
# verify" five runs out of five (D60) because nothing said DEFINED_IN is about
# location, not behaviour. These are the single source for that legend.
RELATIONSHIP_MEANINGS = {
    "DEFINED_IN": (
        "where the thing's source code lives: a parameter in its function, a "
        "method in its class, a class in its module. Location only. It does "
        "not say what uses, applies, or implements the thing."
    ),
    "INHERITS_FROM": "this class is a subclass of that one.",
    "CALLS": (
        "this function invokes that one. It says what runs, not which one is "
        "responsible for the behaviour the question asks about."
    ),
    "IMPORTS": "this module imports that name.",
    "RAISES": "this function can throw that exception.",
    "HAS_PARAMETER": "this function accepts that parameter.",
    "DOCUMENTED_IN": "that documentation section describes this thing.",
    "EXPLAINS": "that documentation section explains this concept.",
    "IMPLEMENTS": "this code realises that concept or behaviour.",
    "CONTROLS": "this parameter or setting governs that behaviour.",
    "DELEGATES_TO": "this component hands its work off to that one.",
    "WRAPS_EXCEPTION": (
        "this exception is raised in place of that one, inside an except "
        "handler that caught it."
    ),
    "CHANGED_IN": "this thing was changed in that release.",
    "CONSTRAINS": "this thing limits or restricts that one.",
}
