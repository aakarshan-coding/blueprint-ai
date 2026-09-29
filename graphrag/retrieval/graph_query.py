"""Turns a GRAPH-routed question into an actual run_template call.

Resolve first, then plan (D59). The earlier planner did three things in one
blind model call -- pick a template, name an entity by its surface, name a
relationship -- and resolution happened afterwards. An audit of every
graph-routed benchmark question found each failure was a consequence of that
order: surfaces the resolver couldn't map (four of them ambiguous across
requests/urllib3 when the question said which package it meant), hub
entities that returned 30-96 facts because the model couldn't see what kind
of node it was choosing, and templates paired with parameters they don't
take.

Now it is two stages with our own code in between:

  1. a cheap call lists the entities the question mentions, each with the
     package the wording assigns it ("urllib3's ReadTimeoutError");
  2. the resolver turns those into real canonical ids -- using the package
     qualifier to break a tie it would otherwise refuse to guess at -- and
     the graph reports each id's kind and degree;
  3. a second call picks a template and an entity FROM THAT LIST, through a
     schema built per call in which entity_id is an enum of exactly those
     ids and each template carries only the parameters it takes.

The model still never sees or invents a canonical id it wasn't handed, and
never writes Cypher. Resolution stays a controlled step we own; what changed
is that it happens before the model chooses, not after.
"""

import re
from dataclasses import dataclass
from typing import Annotated, Callable, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, create_model

from graphrag.ingest.resolve import Resolver
from graphrag.ontology import RELATIONSHIP_TYPES
from graphrag.retrieval.consistency import VOTES, majority_vote
from graphrag.ontology import PACKAGES
from graphrag.retrieval.cypher_templates import MAX_HOPS, TEMPLATES

MODEL = "gpt-4o-mini"

# A template is offered to the planner exactly when it carries a description.
# Deriving both the offered set and the prompt from this one dict is what
# stops them drifting (D54).
PLANNABLE = {name: t for name, t in sorted(TEMPLATES.items()) if t.description}

Package = Literal["requests", "urllib3", "unknown"]


# --- stage 1: what does the question mention? --------------------------------

class Mention(BaseModel):
    """One entity the question names, as written, with the package the
    question's wording assigns it -- "which requests exception" is a
    qualifier the resolver needs and the old planner threw away."""

    model_config = ConfigDict(extra="forbid")

    surface: str
    package: Package


class Mentions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mentions: list[Mention]


MENTION_PROMPT = """\
List every code entity the question names: a class, function, method, \
exception, parameter, or module from the requests or urllib3 libraries.

Write each surface exactly as it appears in the question ("Session", \
"HTTPAdapter.send", "ReadTimeoutError") -- never expand it to a dotted path \
you were not given. Do not list generic words ("exception", "library", \
"request") or values ("False", "30").

For each, record which package the question says it belongs to. "urllib3's \
ReadTimeoutError" and "which requests exception" are explicit; if the \
question does not say, use "unknown". Never infer the package from your own \
knowledge of where a name is defined -- only from the question's wording.
"""


# The mention list is the anchor of everything downstream, and it is not
# stable: the same question gave "Session.request" on one call and "Session"
# plus "request" on the next (D73, live), which changed the candidates and so
# the plan. Three cheap calls, majority on the surface set. Planner voting
# was retired as not worth 3x (D60); this is the one place the vote sits
# upstream of every other decision.
MENTION_VOTES = 3


def extract_mentions(
    question: str, *, client, model: str = MODEL, votes: int = MENTION_VOTES
) -> list[Mention]:
    """The entities a question names, with their package qualifiers."""
    def ask() -> Mentions:
        response = client.responses.parse(
            model=model,
            instructions=MENTION_PROMPT,
            input=question,
            text_format=Mentions,
            temperature=0,
        )
        return response.output_parsed

    result = majority_vote(
        ask, trials=votes,
        key=lambda m: tuple(sorted((x.surface, x.package) for x in m.mentions)),
    )
    mentions = [m for m in result.mentions if mentioned_in(m.surface, question)]
    have = {m.surface for m in mentions}
    for surface in dotted_names_in(question):
        if surface not in have:
            mentions.append(Mention(surface=surface, package="unknown"))
            have.add(surface)
    # "HTTPAdapter's constructor" names HTTPAdapter.__init__ (D90): the
    # question about its parameters anchored on the class and listed its
    # methods. A constructor is a method with a fixed name; read it off.
    if re.search(r"\bconstructor\b", question, re.IGNORECASE):
        for m in list(mentions):
            if m.surface[:1].isupper() and "." not in m.surface and f"{m.surface}.__init__" not in have:
                mentions.append(Mention(surface=f"{m.surface}.__init__", package=m.package))
                have.add(f"{m.surface}.__init__")
    return mentions


_DOTTED = re.compile(r"(?<![\w.])([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+)(?:\(\))?(?![\w.])")


def dotted_names_in(question: str) -> list[str]:
    """Every dotted code name written in the question, as written.

    The model was asked to copy surfaces verbatim and, with three votes,
    still returned "Session" and "request" for "Session.request" on some
    calls and "request" alone for "requests.api" on others (D81, live). A
    dotted name in the question is not a judgement call; it is read off the
    text. The model's mentions still carry the package qualifier; these
    carry none.
    """
    names = []
    for m in _DOTTED.finditer(question):
        surface = m.group(1)
        if len(surface) < 5 or surface.lower() in ("e.g", "i.e"):
            continue
        if surface not in names:
            names.append(surface)
    return names


def mentioned_in(surface: str, question: str) -> bool:
    """Whether the question actually contains the surface.

    The prompt says "write each surface exactly as it appears in the
    question", and the model does not always obey: "a streamed body cut
    short" came back as `ReadTimeoutError`, "rejects a header value" as
    `HeaderParsingError`, and "which requests class sends the request" as
    `Session` and `HTTPAdapter` (D76, live). Each invented name resolved
    and became the anchor, and the chain walked from the wrong place. A
    name the question does not contain is not a mention of it. Plurals and
    call parentheses are forgiven.
    """
    raw = surface.strip().removesuffix("()")
    if not raw:
        return False
    # A code name that starts with a capital ("Exception", "Session") must
    # appear with that capital: "that exception's base class" names no
    # class, but the extractor returned `Exception`, which resolved to
    # builtins.Exception and anchored a plan that walked nothing (3h-16).
    if raw[0].isupper():
        return any(form in question for form in (raw, raw.removesuffix("s"), raw.removesuffix("es")))
    s = raw.lower()
    if s in GENERIC_WORDS:
        return False
    q = question.lower()
    return s in q or s.removesuffix("s") in q or s.removesuffix("es") in q


# Lowercase words the extractor returns as "mentions" that name no entity;
# "connection" resolved to the module urllib3.connection four times over
# and anchored a call chain from it (3h-12, D76). A real lowercase entity
# is a function or parameter ("iter_content", "verify") and is kept.
GENERIC_WORDS = frozenset({
    "exception", "exceptions", "error", "errors", "function", "functions", "method",
    "methods", "class", "classes", "module", "modules", "library", "libraries",
    "parameter", "parameters", "adapter", "adapters", "connection", "connections",
    "response", "body", "header", "headers", "url", "urls",
    "session", "sessions", "socket", "redirect", "redirects", "retries", "retry",
})


# --- between the stages: mentions -> real candidates -------------------------

@dataclass(frozen=True)
class Candidate:
    """A canonical id the question could mean, with what the graph knows
    about it. `kind` and `degree` are shown to the planner so it can tell a
    Module with 65 edges from an Exception with 6 before choosing."""

    canonical_id: str
    kind: str
    degree: int


Describe = Callable[[list[str]], dict[str, tuple[str, int]]]


def describe_entities(session, ids: list[str]) -> dict[str, tuple[str, int]]:
    """Each id's labels and neighbourhood size, from the live graph."""
    rows = session.run(
        "MATCH (n) WHERE n.id IN $ids "
        "OPTIONAL MATCH (n)-[r]-() "
        "RETURN n.id AS id, labels(n) AS labels, count(r) AS degree",
        ids=list(ids),
    ).data()
    out: dict[str, tuple[str, int]] = {}
    for row in rows:
        if "id" not in row:
            continue
        labels = sorted(label for label in (row.get("labels") or []) if label)
        out[row["id"]] = ("/".join(labels) or "unknown", int(row.get("degree") or 0))
    return out


def resolve_mentions(
    mentions: list[Mention], *, resolver: Resolver, describe: Describe
) -> list[Candidate]:
    """Turn the question's mentions into the canonical ids it could mean.

    A surface that resolves cleanly gives one candidate. A surface the
    resolver finds ambiguous gives all of its candidates -- narrowed to the
    package the question named, when it named one, which is exactly the tie
    the resolver alone refuses to break. A surface nothing matches gives
    nothing: the planner is never offered an id that doesn't exist.
    """
    ids: list[str] = []
    groups: list[list[str]] = []
    for mention in mentions:
        resolution = resolver.resolve(mention.surface)
        if resolution.canonical_id is not None:
            found = [resolution.canonical_id]
        elif resolution.candidates:
            found = list(resolution.candidates)
            if mention.package != "unknown":
                narrowed = [c for c in found if c.startswith(f"{mention.package}.")]
                if narrowed:
                    found = narrowed
        else:
            found = []
        groups.append(found)
        for canonical_id in found:
            if canonical_id not in ids:
                ids.append(canonical_id)

    if not ids:
        return []
    info = describe(ids)
    kind_of = lambda i: info.get(i, ("unknown", 0))[0]  # noqa: E731
    # "Timeout" is ambiguous between a class and thirty parameters named
    # timeout, and the resolver offers them all. Handed to the planner, that
    # list buried the two ids the question was about (3h-24 showed 33
    # candidates, D74). When one mention's candidates mix a class or
    # function with parameters, the parameters are the noise.
    keep: set[str] = set()
    for found in groups:
        non_params = [i for i in found if "Parameter" not in kind_of(i)]
        keep.update(non_params if len(found) > 1 and non_params else found)
    return [
        Candidate(canonical_id, kind=kind_of(canonical_id),
                  degree=info.get(canonical_id, ("unknown", 0))[1])
        for canonical_id in ids if canonical_id in keep
    ]


# --- stage 2: a plan, from a schema built for these candidates --------------

_RELATIONSHIPS = tuple(sorted(RELATIONSHIP_TYPES))
_HOPS = tuple(range(1, MAX_HOPS + 1))


def _plan_model_for(template_id: str, entity_ids: list[str]) -> type[BaseModel] | None:
    """One pydantic model per template, with exactly that template's
    parameters. entity_id is an enum of the resolved candidates, so an id
    the model wasn't handed fails validation instead of reaching Cypher.
    Hop limits are an int enum rather than a range: OpenAI's strict schema
    mode rejects minimum/maximum. Returns None when the template needs an
    entity and there is none to offer."""
    fields: dict = {"template_id": (Literal[template_id], ...)}
    for spec in TEMPLATES[template_id].params:
        if spec.kind == "entity_id":
            if not entity_ids:
                return None
            fields["entity_id"] = (Literal[tuple(entity_ids)], ...)
        elif spec.kind == "relationship_type":
            fields["relationship"] = (Literal[_RELATIONSHIPS], ...)
        elif spec.kind == "hop_limit":
            fields["max_hops"] = (Literal[_HOPS], ...)
        elif spec.kind == "package":
            fields["package"] = (Literal[PACKAGES], ...)
        elif spec.kind == "identifier":
            fields["name"] = (str, ...)
    return create_model(
        f"Plan_{template_id}", __config__=ConfigDict(extra="forbid"), **fields
    )


def build_plan_model(candidates: list[Candidate]) -> type[BaseModel]:
    """The schema the planner answers in, for this question's candidates:
    a discriminated union over every template that can be filled."""
    entity_ids = [c.canonical_id for c in candidates]
    members = [
        m for m in (_plan_model_for(name, entity_ids) for name in PLANNABLE)
        if m is not None
    ]
    # A plain Union, not Field(discriminator=...): the discriminated form
    # serialises as `oneOf`, which OpenAI's strict schema mode rejects
    # ("'oneOf' is not permitted"). A plain Union is `anyOf`, which it
    # accepts, and each member's Literal template_id still lets pydantic
    # validate a returned plan against exactly one template's fields.
    plan_type = members[0] if len(members) == 1 else Union[tuple(members)]
    return create_model(
        "GraphQueryPlan", __config__=ConfigDict(extra="forbid"), plan=(plan_type, ...)
    )


_PREAMBLE = """\
You turn a question already routed to the knowledge graph into a plan for \
running one pre-approved query template.

Templates:
"""

_CLOSING = """
The message lists the entities that were resolved from the question, each \
with its kind and how many edges it has. entity_id must be one of those \
ids, exactly as listed. Prefer the most specific entity the question is \
about: a Module or a package root ("requests", "urllib3") has dozens of \
edges and returns everything at once, which answers nothing -- if the \
question is about a specific class, function or exception, choose that. \
When the question asks which things relate to an entity by one \
relationship ("which classes inherit from X", "what does X raise", "how \
many ..."), choose T8_RELATED_BY with that relationship rather than the \
unfiltered neighbourhood. The one time a Module is the right choice: when \
the question asks what a module (or class, or function) defines or \
contains -- its functions, classes, submodules or parameters -- choose \
T8_RELATED_BY with DEFINED_IN on that module, class or function; the \
members point at it. When the question asks which classes are also a \
builtin type ("which exceptions are also ValueErrors", "which classes are \
warnings"), choose T8_RELATED_BY with INHERITS_FROM on that builtin \
("builtins.ValueError", "builtins.Warning"). A Parameter has no call, \
delegation or wrapping edges -- for a Parameter, T1_NEIGHBORS is the only \
template that returns anything.
"""

SYSTEM_PROMPT = (
    _PREAMBLE
    + "\n".join(f"- {name}: {t.description}" for name, t in PLANNABLE.items())
    + "\n"
    + _CLOSING
)


def plan_graph_query(
    question: str,
    *,
    candidates: list[Candidate],
    client,
    model: str = MODEL,
    votes: int = VOTES,
):
    """Pick a template and fill it from the resolved candidates, by majority
    of cheap calls. Returns None when nothing resolved: with no entity to
    anchor a query there is no graph question to ask, and the passages the
    pipeline always fetches carry the answer instead. (Offering only the
    corpus-wide count in that case is how "how many exceptions inherit from
    RequestException" became a total over every INHERITS_FROM edge.)"""
    if not candidates:
        return None

    plan_model = build_plan_model(candidates)
    listing = "\n".join(
        f"- {c.canonical_id} ({c.kind}, {c.degree} edges)" for c in candidates
    )
    prompt_input = f"Question: {question}\n\nEntities resolved from the question:\n{listing}"

    def ask():
        response = client.responses.parse(
            model=model,
            instructions=SYSTEM_PROMPT,
            input=prompt_input,
            text_format=plan_model,
            temperature=0,
        )
        return response.output_parsed.plan

    return majority_vote(ask, trials=votes)


# A Module node has exactly two kinds of edge: its members' DEFINED_IN and
# its IMPORTS. Every other relationship on a Module returns nothing.
MODULE_RELATIONSHIPS = frozenset({"DEFINED_IN", "IMPORTS"})


def repair_plan(
    template_id: str, values: dict, candidates: list[Candidate]
) -> tuple[str, dict, str | None]:
    """Correct the two plan shapes the model gets structurally wrong, from
    what the graph knows about the anchor rather than from the question.

    Told in its prompt that a module's contents are DEFINED_IN, gpt-4o-mini
    still planned INHERITS_FROM on `requests.exceptions` for "how many warning
    classes does requests.exceptions define" and a corpus-wide count for
    "how many modules make up requests" (D73, live check). Both are
    detectable without the question: a Module has no INHERITS_FROM edges,
    and a corpus-wide count is never what a question naming one module
    wants. The repair is recorded on the result so the benchmark can see
    how often it fired. `candidates` are the question's own, not the ones
    seeded from passages.
    """
    kinds = {c.canonical_id: c.kind for c in candidates}
    entity = values.get("entity_id")
    relationship = values.get("relationship")
    kind = kinds.get(entity, "") if entity else ""

    # A wrap chain starts from an exception. Anchored on a function
    # ("which urllib3 error is caught in iter_content, which requests
    # exception replaces it") it returns nothing: WRAPS_EXCEPTION joins two
    # exceptions. What the function raises is the start of that answer;
    # retrieve() then fetches what each raised exception wraps (D74).
    # A Parameter has no call, wrap or raise edges; T1_NEIGHBORS is the one
    # template that returns anything for it (the prompt says so, and the
    # model still planned a wrap chain from `urlopen.retries`, 3h-11).
    if "Parameter" in kind and template_id != "T1_NEIGHBORS":
        return (
            "T1_NEIGHBORS", {"entity_id": entity},
            "a Parameter has only DEFINED_IN and CONTROLS edges; its neighbourhood instead",
        )
    # RAISES edges hang off methods. A raises plan anchored on a class
    # ("which Session method raises it") walks nothing; what its methods
    # raise is the answer (D76). Only RAISES is repaired here: a wrap chain
    # on a class is left to run, because the labels cannot say whether the
    # class is an exception. urllib3's exceptions carry the label Class
    # alone, and a pre-run rule keyed on "Class but not Exception" replaced
    # the wrap chain from ProtocolError with an empty query on five of the
    # fifteen two-hop questions (D77). retrieve() falls back to T10 after
    # the chain runs and returns nothing.
    if "Class" in kind and template_id == "T8_RELATED_BY" and relationship == "RAISES":
        return (
            "T10_RAISED_BY_METHODS_OF", {"entity_id": entity},
            "RAISES edges hang off methods; what this class's methods raise",
        )
    if template_id == "T3_EXCEPTION_WRAP_CHAIN" and "Function" in kind:
        return (
            "T8_RELATED_BY", {"entity_id": entity, "relationship": "RAISES"},
            "a wrap chain cannot start from a function; what it raises instead",
        )
    if template_id == "T8_RELATED_BY" and "Function" in kind and relationship == "WRAPS_EXCEPTION":
        return (
            template_id, {**values, "relationship": "RAISES"},
            "a function has no WRAPS_EXCEPTION edges; RAISES instead",
        )
    # A wrap chain from a package root ("urllib3") walks nothing. If the
    # question also named something specific, start there (3h-25, D74).
    if template_id == "T3_EXCEPTION_WRAP_CHAIN" and "Module" in kind:
        specific = [c for c in candidates if "Module" not in c.kind and "Parameter" not in c.kind]
        if specific:
            other = specific[0]
            if "Function" in other.kind:
                return (
                    "T8_RELATED_BY", {"entity_id": other.canonical_id, "relationship": "RAISES"},
                    f"a wrap chain from a module walks nothing; what {other.canonical_id} raises instead",
                )
            return (
                template_id, {**values, "entity_id": other.canonical_id},
                f"a wrap chain from a module walks nothing; {other.canonical_id} instead",
            )
    if (
        template_id == "T8_RELATED_BY"
        and entity is not None
        and "Module" in kinds.get(entity, "")
        and relationship not in MODULE_RELATIONSHIPS
    ):
        return (
            template_id, {**values, "relationship": "DEFINED_IN"},
            f"a Module has no {relationship} edges; DEFINED_IN instead",
        )
    if template_id == "T6_COUNT_BY_REL":
        modules = [c for c in candidates if "Module" in c.kind]
        others = [c for c in candidates if "Module" not in c.kind]
        # "How many urllib3 exception classes inherit directly from HTTPError"
        # planned a corpus-wide INHERITS_FROM count (D90). One class named,
        # one relationship: the count is over that class's edges.
        if len(others) == 1 and not modules and "Class" in others[0].kind and relationship:
            return (
                "T8_RELATED_BY", {"entity_id": others[0].canonical_id, "relationship": relationship},
                f"the question names one class; its {relationship} edges, not a corpus-wide count",
            )
        # "How many classes in requests.exceptions have more than one base"
        # resolves both `requests` and `requests.exceptions`; the most
        # specific module is the one the question is about (D82).
        if modules and (len(modules) == 1 or not others):
            module = max(modules, key=lambda c: len(c.canonical_id))
            return (
                "T8_RELATED_BY",
                {"entity_id": module.canonical_id, "relationship": "DEFINED_IN"},
                f"the question names a module; its members, not a corpus-wide {relationship} count",
            )
    return template_id, values, None


def build_template_values(plan) -> tuple[dict, set[str]]:
    """The values run_template needs, straight from the plan: every field is
    already validated by its template's schema, and entity_id is already a
    canonical id the resolver produced."""
    values = {k: v for k, v in plan.model_dump().items() if k != "template_id"}
    known_ids = {values["entity_id"]} if "entity_id" in values else set()
    return values, known_ids
