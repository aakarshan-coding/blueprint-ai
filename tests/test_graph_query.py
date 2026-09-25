"""The planner, after D59: resolve first, then plan.

The old planner picked a template, named an entity by surface, and named a
relationship in one blind call; resolution happened afterwards. The audit of
all 29 graph-routed benchmark questions found every failure was a
consequence of that order: 8 surfaces the resolver couldn't map (four of
them ambiguous across requests/urllib3 when the question said which), 6 hub
entities returning 30-96 facts, and templates chosen without knowing what
kind of thing the entity was. Now our code resolves the question's mentions
into real candidates -- with node kind and degree -- and the model chooses
from that list, through a schema that only has the fields its template takes.
"""

import typing

import pytest
from pydantic import ValidationError

from graphrag.ingest.resolve import Resolver
from graphrag.retrieval.cypher_templates import TEMPLATES
from graphrag.retrieval.graph_query import (
    PLANNABLE,
    SYSTEM_PROMPT,
    Candidate,
    Mention,
    build_plan_model,
    build_template_values,
    extract_mentions,
    plan_graph_query,
    resolve_mentions,
)


# --- single source for what the planner is offered ------------------------

def test_every_offered_template_is_described_to_the_model():
    described = {name for name in TEMPLATES if f"- {name}:" in SYSTEM_PROMPT}
    assert set(PLANNABLE) == described


def test_every_offered_template_has_parameters_a_plan_can_fill():
    fillable = {"entity_id", "relationship", "max_hops"}
    unfillable = {
        name for name, t in PLANNABLE.items()
        if {spec.name for spec in t.params} - fillable
    }
    assert unfillable == set()


def test_path_between_is_not_offered_to_the_planner():
    assert "T2_PATH_BETWEEN" in TEMPLATES
    assert "T2_PATH_BETWEEN" not in PLANNABLE


# --- per-template plan schemas (fix 2) --------------------------------------

CANDIDATES = [
    Candidate("requests.exceptions.ProxyError", kind="Exception", degree=6),
    Candidate("urllib3.exceptions.ProxyError", kind="Exception", degree=4),
]


def _plan(**fields):
    """Construct a plan through the same dynamic model the planner uses."""
    model = build_plan_model(CANDIDATES)
    return model(plan=fields).plan


def test_a_plan_only_carries_the_parameters_its_template_takes():
    """T3 takes an entity and a hop count. The old shared schema let the
    model attach relationship=RAISES to it -- ignored at runtime, but the
    same looseness produced T6 with a named entity and T8 with the wrong
    relationship (D54). Per-template schemas make those unrepresentable."""
    plan = _plan(template_id="T3_EXCEPTION_WRAP_CHAIN",
                 entity_id="requests.exceptions.ProxyError", max_hops=2)
    assert not hasattr(plan, "relationship")

    with pytest.raises(ValidationError):
        _plan(template_id="T3_EXCEPTION_WRAP_CHAIN",
              entity_id="requests.exceptions.ProxyError", max_hops=2,
              relationship="RAISES")


def test_a_count_plan_cannot_name_an_entity():
    with pytest.raises(ValidationError):
        _plan(template_id="T6_COUNT_BY_REL", relationship="INHERITS_FROM",
              entity_id="requests.exceptions.ProxyError")


def test_a_plans_entity_must_be_one_of_the_resolved_candidates():
    """The model never invents an id: the schema's entity_id is an enum of
    exactly the candidates our code resolved from the question."""
    with pytest.raises(ValidationError):
        _plan(template_id="T1_NEIGHBORS", entity_id="requests.made.Up")


def test_templates_needing_an_entity_are_not_offered_without_candidates():
    model = build_plan_model([])
    annotation = model.model_fields["plan"].annotation
    # A union of several members, or a single class when only one is left.
    offered = set(typing.get_args(annotation)) or {annotation}
    names = {typing.get_args(m.model_fields["template_id"].annotation)[0] for m in offered}

    assert "T1_NEIGHBORS" not in names
    assert "T6_COUNT_BY_REL" in names


# --- stage 1: mentions -> candidates (fix 1) --------------------------------

class _FakeResponse:
    def __init__(self, output_parsed):
        self.output_parsed = output_parsed


class _FakeResponses:
    """Constructs whatever model the caller passed, from canned dicts, so the
    same fake serves the mention call and the (dynamically built) plan call."""

    def __init__(self, payloads):
        self._payloads = list(payloads)
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        payload = self._payloads[(len(self.calls) - 1) % len(self._payloads)]
        return _FakeResponse(kwargs["text_format"](**payload))


class _FakeClient:
    def __init__(self, payloads):
        self.responses = _FakeResponses(payloads)


def _describe(ids):
    table = {
        "requests.exceptions.ProxyError": ("Exception", 6),
        "urllib3.exceptions.ProxyError": ("Exception", 4),
        "requests": ("Module", 65),
        "requests.adapters.HTTPAdapter.send": ("Function", 16),
    }
    return {i: table.get(i, ("unknown", 0)) for i in ids}


RESOLVER = Resolver(
    node_universe={
        "requests.exceptions.ProxyError", "urllib3.exceptions.ProxyError",
        "requests", "requests.adapters.HTTPAdapter.send",
        "requests.sessions.Session.send", "requests.adapters.BaseAdapter.send",
    },
    import_aliases={},
)


def test_extract_mentions_returns_surfaces_with_their_package_qualifier():
    client = _FakeClient([{"mentions": [
        {"surface": "ProxyError", "package": "requests"},
    ]}])

    mentions = extract_mentions("Which requests exception is ProxyError?", client=client, votes=1)

    assert mentions == [Mention(surface="ProxyError", package="requests")]


def test_resolve_mentions_uses_the_package_qualifier_to_break_a_tie():
    """'ProxyError' exists in both packages and the resolver refuses to guess.
    The question said 'which requests exception' -- the qualifier the old
    planner threw away. Four of the eight resolution failures in the audit
    were exactly this."""
    candidates = resolve_mentions(
        [Mention(surface="ProxyError", package="requests")],
        resolver=RESOLVER, describe=_describe,
    )

    assert [c.canonical_id for c in candidates] == ["requests.exceptions.ProxyError"]


def test_resolve_mentions_keeps_every_candidate_when_there_is_no_qualifier():
    # With nothing to break the tie, both are offered and the model picks --
    # it sees the full ids, so the choice is informed rather than a guess.
    candidates = resolve_mentions(
        [Mention(surface="ProxyError", package="unknown")],
        resolver=RESOLVER, describe=_describe,
    )

    assert {c.canonical_id for c in candidates} == {
        "requests.exceptions.ProxyError", "urllib3.exceptions.ProxyError",
    }


def test_resolve_mentions_attaches_kind_and_degree():
    candidates = resolve_mentions(
        [Mention(surface="requests", package="unknown")],
        resolver=RESOLVER, describe=_describe,
    )

    assert candidates == [Candidate("requests", kind="Module", degree=65)]


def test_resolve_mentions_drops_what_nothing_resolves():
    candidates = resolve_mentions(
        [Mention(surface="TotallyMadeUpThing", package="unknown")],
        resolver=RESOLVER, describe=_describe,
    )

    assert candidates == []


def test_resolve_mentions_dedupes_a_candidate_reached_twice():
    candidates = resolve_mentions(
        [Mention(surface="ProxyError", package="requests"),
         Mention(surface="requests.exceptions.ProxyError", package="unknown")],
        resolver=RESOLVER, describe=_describe,
    )

    assert len(candidates) == 1


# --- stage 2: plan from candidates ------------------------------------------

def test_plan_graph_query_chooses_from_the_candidates_it_is_given():
    client = _FakeClient([{"plan": {
        "template_id": "T3_EXCEPTION_WRAP_CHAIN",
        "entity_id": "requests.exceptions.ProxyError", "max_hops": 2,
    }}])

    plan = plan_graph_query("q", candidates=CANDIDATES, client=client, votes=1)

    assert plan.template_id == "T3_EXCEPTION_WRAP_CHAIN"
    assert plan.entity_id == "requests.exceptions.ProxyError"
    assert plan.max_hops == 2


def test_plan_graph_query_shows_the_model_each_candidates_kind_and_degree():
    client = _FakeClient([{"plan": {
        "template_id": "T1_NEIGHBORS", "entity_id": "requests.exceptions.ProxyError",
    }}])

    plan_graph_query("q", candidates=CANDIDATES, client=client, votes=1)

    sent = client.responses.calls[0]["input"]
    assert "requests.exceptions.ProxyError" in sent
    assert "Exception" in sent and "6" in sent


def test_plan_graph_query_returns_none_when_nothing_resolved():
    """No candidates means no graph query. The alternative -- offering only
    the corpus-wide count -- is how 'how many exceptions inherit from
    RequestException' became a total over every INHERITS_FROM edge."""
    client = _FakeClient([{"plan": {"template_id": "T6_COUNT_BY_REL", "relationship": "RAISES"}}])

    assert plan_graph_query("q", candidates=[], client=client, votes=1) is None
    assert client.responses.calls == []


def test_plan_graph_query_takes_the_majority_plan():
    client = _FakeClient([
        {"plan": {"template_id": "T3_EXCEPTION_WRAP_CHAIN", "entity_id": "requests.exceptions.ProxyError", "max_hops": 2}},
        {"plan": {"template_id": "T3_EXCEPTION_WRAP_CHAIN", "entity_id": "requests.exceptions.ProxyError", "max_hops": 3}},
        {"plan": {"template_id": "T3_EXCEPTION_WRAP_CHAIN", "entity_id": "requests.exceptions.ProxyError", "max_hops": 2}},
    ])

    plan = plan_graph_query("q", candidates=CANDIDATES, client=client, votes=3)

    assert plan.max_hops == 2
    assert len(client.responses.calls) == 3


# --- values for run_template --------------------------------------------------

def test_build_template_values_passes_the_plans_fields_through():
    plan = _plan(template_id="T8_RELATED_BY",
                 entity_id="requests.exceptions.ProxyError", relationship="INHERITS_FROM")

    values, known_ids = build_template_values(plan)

    assert values == {"entity_id": "requests.exceptions.ProxyError", "relationship": "INHERITS_FROM"}
    assert known_ids == {"requests.exceptions.ProxyError"}


def test_build_template_values_for_a_count_plan_has_no_entity():
    plan = _plan(template_id="T6_COUNT_BY_REL", relationship="WRAPS_EXCEPTION")

    values, known_ids = build_template_values(plan)

    assert values == {"relationship": "WRAPS_EXCEPTION"}
    assert known_ids == set()
