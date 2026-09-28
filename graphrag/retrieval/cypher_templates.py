"""The Cypher template library (design §8). The model never writes Cypher —
it picks a template id and fills typed parameters, and every parameter is
validated against what kind of thing it is before the query runs at all.
This is both the security control and what makes results reproducible: the
same (template, params) pair always produces the same query.

Adding a template only means adding a query and declaring its ParamSpecs —
validation itself is generic, so a new template can't accidentally skip a
check that an older one remembered to do.
"""

from dataclasses import dataclass
from typing import Literal

from graphrag.ontology import PACKAGES, RELATIONSHIP_TYPES

ParamKind = Literal["entity_id", "entity_ids", "relationship_type", "hop_limit", "package"]

MAX_HOPS = 5


@dataclass(frozen=True)
class ParamSpec:
    name: str
    kind: ParamKind


@dataclass(frozen=True)
class Template:
    cypher: str
    params: tuple[ParamSpec, ...]
    # What the planner is told this template does. This is the ONLY place a
    # template is described: graph_query builds both its prompt and its list of
    # allowed template ids from these, so a template cannot be offered to the
    # model without an explanation or described without being offered. Those
    # two lists used to be hand-written separately and drifted — T8_RELATED_BY
    # was returnable but undescribed, leaving the planner unable to knowingly
    # pick the one template built for the aggregation questions.
    # None means "not offered to the planner", which needs a reason in a comment.
    description: str | None = None


TEMPLATES: dict[str, Template] = {
    "T1_NEIGHBORS": Template(
        # Matches both directions on purpose — "what is connected to X" should
        # find edges pointing at X too. But it MUST report which way each edge
        # runs: rendering an incoming edge as though X were the subject
        # produces a false statement ("BaseAdapter inherits from HTTPAdapter").
        # Labels ride along so the verbalizer can phrase by kind: a Parameter's
        # DEFINED_IN edge reads "is a parameter of", a method's "is a method
        # of" (D63). The id alone can't tell them apart.
        cypher=(
            "MATCH (a {id: $entity_id})-[r]-(b) "
            "RETURN type(r) AS relationship, b.id AS neighbor, "
            "r.chunk_id AS chunk_id, startNode(r).id = $entity_id AS outgoing, "
            "labels(a) AS entity_labels, labels(b) AS neighbor_labels, "
            "coalesce(r.source, 'llm') AS source"
        ),
        params=(ParamSpec("entity_id", "entity_id"),),
        description=(
            "Everything directly connected to one entity, in either direction "
            "and of any relationship type. Use when the question asks broadly "
            "what relates to X, or when no more specific template fits. "
            "Needs entity_surface."
        ),
    ),
    # No description on purpose: this one needs two entities, and a plan carries
    # only one entity_surface, so the planner could never fill it in. Offering
    # it would just produce plans that fail validation.
    "T2_PATH_BETWEEN": Template(
        cypher=(
            "MATCH p = shortestPath((a {id: $source_id})-[*..4]-(b {id: $target_id})) "
            "RETURN [n IN nodes(p) | n.id] AS path_ids, "
            "[r IN relationships(p) | type(r)] AS path_rels"
        ),
        params=(
            ParamSpec("source_id", "entity_id"),
            ParamSpec("target_id", "entity_id"),
        ),
    ),
    "T3_EXCEPTION_WRAP_CHAIN": Template(
        # Undirected on purpose. Edges read "requests exception WRAPS
        # urllib3 exception", so "what does requests raise when urllib3
        # raises X" — the flagship question — asks the INCOMING side. An
        # outgoing-only walk answers the rarer direction and silently
        # misses the common one. verbalize() reports direction per hop.
        cypher=(
            "MATCH p = (a {id: $entity_id})-[:WRAPS_EXCEPTION*1..__max_hops__]-(b) "
            "RETURN [n IN nodes(p) | n.id] AS chain, "
            "[r IN relationships(p) | r.chunk_id] AS chunk_ids, "
            "[r IN relationships(p) | startNode(r).id] AS starts, "
            "[r IN relationships(p) | coalesce(r.source, 'llm')] AS sources"
        ),
        params=(
            ParamSpec("entity_id", "entity_id"),
            ParamSpec("max_hops", "hop_limit"),
        ),
        description=(
            "What an exception ultimately wraps, following WRAPS_EXCEPTION "
            "transitively in either direction. Use for 'what does requests "
            "raise when urllib3 raises X' and for tracing an exception back to "
            "its underlying cause. Needs entity_surface and max_hops (1-5; use "
            "2 or 3 unless the question asks for something shallower or deeper)."
        ),
    ),
    "T5_DELEGATION_CHAIN": Template(
        # Walks CALLS alongside DELEGATES_TO. DELEGATES_TO is the LLM's
        # prose-derived view of hand-off and is sparse: from requests.get it
        # returned nothing. CALLS is now inferred by jedi (D58) and carries the
        # real chain get -> request -> Session.request -> Session.send ->
        # adapter.send -> conn.urlopen. `rels` names each hop's type so the
        # verbalizer can say "calls" or "delegates to" rather than guess.
        # Outgoing only (D76): "what does X hand off to" is a forward walk.
        # Undirected, the walk from requests.api.get at 5 hops returned 213
        # paths against 84 forward, and the cap kept callers-of-callers over
        # the chain to urlopen. A Class anchor starts from each of its
        # methods, since CALLS edges hang off methods, not the class: "what
        # does Session's send() ultimately call" planned T5 on Session and
        # walked nothing. LIMIT bounds the fan-out at depth 5.
        cypher=(
            "MATCH (a {id: $entity_id}) "
            "OPTIONAL MATCH (m)-[:DEFINED_IN]->(a) WHERE a:Class "
            "WITH a, collect(m) AS members "
            "UNWIND members + [a] AS s "
            "MATCH p = (s)-[:DELEGATES_TO|CALLS*1..__max_hops__]->(b) "
            "RETURN [n IN nodes(p) | n.id] AS chain, "
            "[r IN relationships(p) | r.chunk_id] AS chunk_ids, "
            "[r IN relationships(p) | startNode(r).id] AS starts, "
            "[r IN relationships(p) | type(r)] AS rels, "
            "[r IN relationships(p) | coalesce(r.source, 'llm')] AS sources "
            "LIMIT 500"
        ),
        params=(
            ParamSpec("entity_id", "entity_id"),
            ParamSpec("max_hops", "hop_limit"),
        ),
        description=(
            "What one function or class ultimately calls or delegates work "
            "to, following CALLS and DELEGATES_TO transitively. Use for "
            "'what does requests.get hand off to' or 'what does Session.send "
            "call underneath'. Needs entity_surface and max_hops (1-5; use 4 "
            "or 5 to trace a hand-off all the way down, since each hop is "
            "one call: requests.get reaches urllib3's urlopen at hop 5)."
        ),
    ),
    "T6_COUNT_BY_REL": Template(
        cypher="MATCH ()-[r:__relationship__]->() RETURN count(r) AS count",
        params=(ParamSpec("relationship", "relationship_type"),),
        description=(
            "How many edges of one relationship type exist across the WHOLE "
            "corpus. It cannot be scoped to an entity, so use it only for "
            "corpus-wide totals. If the question names an entity, use "
            "T8_RELATED_BY instead. Needs relationship, no entity."
        ),
    ),
    "T8_RELATED_BY": Template(
        # Everything connected to an entity by ONE relationship type. Answers
        # both "which classes inherit from BaseAdapter" (the list) and "how
        # many do" (count the list) — which is what the aggregation questions
        # actually need. T6_COUNT_BY_REL counts that relationship across the
        # whole graph and can't be scoped to an entity, so it answered
        # "how many inherit from RequestException?" with a global total.
        cypher=(
            "MATCH (a {id: $entity_id})-[r:__relationship__]-(b) "
            "RETURN b.id AS neighbor, r.chunk_id AS chunk_id, "
            "startNode(r).id = $entity_id AS outgoing, "
            "labels(a) AS entity_labels, labels(b) AS neighbor_labels, "
            "coalesce(r.source, 'llm') AS source"
        ),
        params=(
            ParamSpec("entity_id", "entity_id"),
            ParamSpec("relationship", "relationship_type"),
        ),
        description=(
            "Everything connected to one entity by a single named relationship "
            "type. Use for 'which classes inherit from X', 'what does X call', "
            "'what does module X define' (DEFINED_IN), 'what parameters does X "
            "accept' (DEFINED_IN), and for COUNTING those — 'how many exceptions inherit from "
            "RequestException' is answered by counting what this returns. "
            "Prefer this over T6_COUNT_BY_REL whenever the question names an "
            "entity. Needs entity_surface and relationship."
        ),
    ),
    # No description on purpose: not a planner choice. retrieve() runs it
    # itself after a wrap chain, on the nodes the chain kept (D71). Eleven of
    # the 27 three-hop questions end "...and what does that inherit from?";
    # the chain template walks WRAPS_EXCEPTION only, so the last hop, an
    # INHERITS_FROM edge the graph already holds, was never fetched.
    # Generalised in D74: the same one-hop expansion serves INHERITS_FROM
    # (parents), WRAPS_EXCEPTION (what a raised exception wraps) and RAISES
    # (what a called function raises), each chosen by retrieve() from the
    # plan's shape. Outgoing only: the kept nodes are the subjects.
    "T9_EDGES_FROM": Template(
        cypher=(
            "MATCH (a)-[r:__relationship__]->(b) WHERE a.id IN $entity_ids "
            "RETURN a.id AS entity, b.id AS neighbor, r.chunk_id AS chunk_id, "
            "coalesce(r.source, 'llm') AS source, "
            "labels(a) AS entity_labels, labels(b) AS neighbor_labels"
        ),
        params=(
            ParamSpec("entity_ids", "entity_ids"),
            ParamSpec("relationship", "relationship_type"),
        ),
    ),
    # No description on purpose: reached by repair_plan when a raises or wrap
    # plan is anchored on a plain class (D76). RAISES edges hang off methods;
    # "which Session method raises InvalidSchema" planned RAISES on the class
    # Session and got nothing. Rows are the answer, like T8: not capped.
    "T10_RAISED_BY_METHODS_OF": Template(
        cypher=(
            "MATCH (m)-[:DEFINED_IN]->(c {id: $entity_id}) "
            "MATCH (m)-[r:RAISES]->(e) "
            "RETURN m.id AS entity, e.id AS neighbor, r.chunk_id AS chunk_id, "
            "coalesce(r.source, 'llm') AS source, "
            "labels(m) AS entity_labels, labels(e) AS neighbor_labels"
        ),
        params=(ParamSpec("entity_id", "entity_id"),),
    ),
    "T11_EDGES_IN_PACKAGE": Template(
        # Every edge of one type whose source lives in one package. "List all
        # requests exceptions that wrap a urllib3 exception" has no anchor
        # entity: it is a listing over the package (D80). Rows are the
        # answer, like T8: not capped. LIMIT bounds a CALLS-sized listing.
        cypher=(
            "MATCH (a)-[r:__relationship__]->(b) WHERE a.id STARTS WITH $package + '.' "
            "RETURN a.id AS entity, b.id AS neighbor, r.chunk_id AS chunk_id, "
            "coalesce(r.source, 'llm') AS source, "
            "labels(a) AS entity_labels, labels(b) AS neighbor_labels "
            "ORDER BY a.id LIMIT 100"
        ),
        params=(
            ParamSpec("package", "package"),
            ParamSpec("relationship", "relationship_type"),
        ),
        description=(
            "Every edge of one relationship type whose source is anywhere in "
            "one package (requests or urllib3), listed. Use for 'list all "
            "requests exceptions that wrap a urllib3 exception' "
            "(WRAPS_EXCEPTION, requests) or 'which requests classes inherit "
            "from a builtin' (INHERITS_FROM, requests), when the question "
            "names no single entity to start from. Needs package and "
            "relationship, no entity."
        ),
    ),
    "T7_DOCS_FOR_SYMBOL": Template(
        cypher=(
            "MATCH (a {id: $entity_id})-[:DOCUMENTED_IN]->(d) "
            "RETURN d.id AS doc_section, d.chunk_id AS chunk_id"
        ),
        params=(ParamSpec("entity_id", "entity_id"),),
        description=(
            "Which documentation sections describe one entity. Use when the "
            "question asks where something is documented or explained. "
            "Needs entity_surface."
        ),
    ),
}


def _validate_param(spec: ParamSpec, value, *, known_entity_ids: set[str]) -> None:
    if spec.kind == "entity_id":
        if value not in known_entity_ids:
            raise ValueError(
                f"{value!r} is not an entity id resolved from the question — "
                "the model may not reference an id it invented"
            )
    elif spec.kind == "entity_ids":
        if not isinstance(value, list) or not value:
            raise ValueError("entity_ids must be a non-empty list of entity ids")
        for item in value:
            if item not in known_entity_ids:
                raise ValueError(f"{item!r} is not a known entity id")
    elif spec.kind == "package":
        if value not in PACKAGES:
            raise ValueError(f"{value!r} is not a corpus package ({', '.join(PACKAGES)})")
    elif spec.kind == "relationship_type":
        if value not in RELATIONSHIP_TYPES:
            raise ValueError(f"{value!r} is not in the ontology's relationship types")
    elif spec.kind == "hop_limit":
        if not isinstance(value, int) or not (1 <= value <= MAX_HOPS):
            raise ValueError(f"max_hops must be an integer from 1 to {MAX_HOPS}")


def run_template(
    session, template_id: str, values: dict, *, known_entity_ids: set[str]
) -> list[dict]:
    """Validate every parameter, then run the named template. Never accepts
    raw Cypher — only a template id this library already wrote."""
    template = TEMPLATES.get(template_id)
    if template is None:
        raise ValueError(f"{template_id!r} is not a known template")

    for spec in template.params:
        if spec.name not in values:
            raise ValueError(f"missing required parameter {spec.name!r}")
        _validate_param(spec, values[spec.name], known_entity_ids=known_entity_ids)

    # Cypher can't bind a relationship type or a variable-length hop bound as
    # a runtime parameter — confirmed against a live instance, a $max_hops
    # bound is a hard CypherSyntaxError. Those two kinds get interpolated
    # into the query text directly instead, which is safe only because both
    # were just validated above: relationship_type against the fixed
    # ontology, hop_limit against a bounded integer range. entity_id values
    # are real property values, so they stay real bound parameters.
    cypher = template.cypher
    bound_params: dict = {}
    for spec in template.params:
        value = values[spec.name]
        if spec.kind in ("relationship_type", "hop_limit"):
            cypher = cypher.replace(f"__{spec.name}__", str(value))
        else:
            bound_params[spec.name] = value

    result = session.run(cypher, **bound_params)
    return result.data()
