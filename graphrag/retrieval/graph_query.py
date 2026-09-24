"""Turns a GRAPH-routed question into an actual run_template call.

Two steps, kept separate on purpose: the model picks WHAT to look up, by
plain-English name — it never sees or invents a canonical graph id. Our own
code (the Resolver, same one used in ingestion) turns that name into a real
id. This is the same discipline as ingestion's surface-name extraction: the
model proposes a name, resolution is a controlled step we own, never the
model's to skip.
"""

from typing import Literal

from pydantic import BaseModel

from graphrag.ingest.resolve import Resolver
from graphrag.ontology import RELATIONSHIP_TYPES
from graphrag.retrieval.consistency import VOTES, majority_vote
from graphrag.retrieval.cypher_templates import TEMPLATES

MODEL = "gpt-4o-mini"

# A template is offered to the planner exactly when it carries a description.
# Deriving both the allowed ids and the prompt from this one dict is what stops
# them drifting: previously each was hand-written, and T8_RELATED_BY ended up
# returnable but undescribed — so the planner could not knowingly choose the
# template built for the aggregation questions.
PLANNABLE = {name: t for name, t in sorted(TEMPLATES.items()) if t.description}

TemplateId = Literal[tuple(PLANNABLE)]


class GraphQueryPlan(BaseModel):
    template_id: TemplateId
    entity_surface: str | None = None
    relationship: Literal[tuple(sorted(RELATIONSHIP_TYPES))] | None = None
    max_hops: int | None = None


_PREAMBLE = """\
You turn a question already routed to the knowledge graph into a plan for \
running one pre-approved query template.

Templates:
"""

_CLOSING = """
entity_surface is the name exactly as written in the question ("Session", \
"ConnectTimeout") — never a resolved or fully-qualified id, you don't have \
access to those.
"""

SYSTEM_PROMPT = (
    _PREAMBLE
    + "\n".join(f"- {name}: {t.description}" for name, t in PLANNABLE.items())
    + "\n"
    + _CLOSING
)


def plan_graph_query(
    question: str, *, client, model: str = MODEL, votes: int = VOTES
) -> GraphQueryPlan:
    """Pick a template and name what it needs, by majority of cheap calls.

    Voted for the same reason as the router (D53): asked once, the planner
    returned a different plan for the same question on 11.7% of benchmark
    questions — a different template, a different entity, or a different hop
    count. Every field here is discrete, so the vote covers the whole plan
    rather than a chosen key.
    """
    def ask() -> GraphQueryPlan:
        response = client.responses.parse(
            model=model,
            instructions=SYSTEM_PROMPT,
            input=question,
            text_format=GraphQueryPlan,
            temperature=0,
        )
        return response.output_parsed

    return majority_vote(ask, trials=votes)


def build_template_values(
    plan: GraphQueryPlan, *, resolver: Resolver
) -> tuple[dict, set[str]]:
    """Resolve the plan's entity surface (if any) and assemble run_template's
    values + known_entity_ids, keeping only the params its template needs.
    """
    values: dict = {}
    known_ids: set[str] = set()

    if plan.entity_surface is not None:
        resolution = resolver.resolve(plan.entity_surface)
        if resolution.canonical_id is None:
            raise ValueError(
                f"could not resolve {plan.entity_surface!r} to a known entity "
                f"(candidates: {resolution.candidates})"
            )
        values["entity_id"] = resolution.canonical_id
        known_ids.add(resolution.canonical_id)

    if plan.relationship is not None:
        values["relationship"] = plan.relationship

    if plan.max_hops is not None:
        values["max_hops"] = plan.max_hops

    template = TEMPLATES[plan.template_id]
    needed = {spec.name for spec in template.params}
    values = {k: v for k, v in values.items() if k in needed}

    return values, known_ids
