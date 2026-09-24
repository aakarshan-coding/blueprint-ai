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

from dataclasses import dataclass
from typing import Annotated, Callable, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, create_model

from graphrag.ingest.resolve import Resolver
from graphrag.ontology import RELATIONSHIP_TYPES
from graphrag.retrieval.consistency import VOTES, majority_vote
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


def extract_mentions(
    question: str, *, client, model: str = MODEL, votes: int = VOTES
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
    return list(result.mentions)


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
        for canonical_id in found:
            if canonical_id not in ids:
                ids.append(canonical_id)

    if not ids:
        return []
    info = describe(ids)
    return [
        Candidate(canonical_id, kind=info.get(canonical_id, ("unknown", 0))[0],
                  degree=info.get(canonical_id, ("unknown", 0))[1])
        for canonical_id in ids
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
unfiltered neighbourhood. A Parameter has no call, delegation or wrapping \
edges -- for a Parameter, T1_NEIGHBORS is the only template that returns \
anything.
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


def build_template_values(plan) -> tuple[dict, set[str]]:
    """The values run_template needs, straight from the plan: every field is
    already validated by its template's schema, and entity_id is already a
    canonical id the resolver produced."""
    values = {k: v for k, v in plan.model_dump().items() if k != "template_id"}
    known_ids = {values["entity_id"]} if "entity_id" in values else set()
    return values, known_ids
