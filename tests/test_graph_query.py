import typing

from graphrag.retrieval.cypher_templates import TEMPLATES
from graphrag.retrieval.graph_query import (
    PLANNABLE,
    SYSTEM_PROMPT,
    GraphQueryPlan,
    build_template_values,
    plan_graph_query,
)
from graphrag.ingest.resolve import Resolver


def test_every_allowed_template_is_described_to_the_model():
    """The planner may only return a template it was told about.

    These two lists were hand-written separately and silently drifted:
    T8_RELATED_BY was returnable but absent from the prompt, so the planner
    could not knowingly pick the one template built for the aggregation
    questions. Both now derive from TEMPLATES, and this test is what keeps
    them from separating again.
    """
    allowed = set(typing.get_args(GraphQueryPlan.model_fields["template_id"].annotation))
    described = {name for name in TEMPLATES if f"- {name}:" in SYSTEM_PROMPT}

    assert allowed == described


def test_every_offered_template_has_parameters_a_plan_can_fill():
    """Offering a template the plan can't fill just produces failing plans.

    A GraphQueryPlan carries one entity_surface, one relationship and one
    max_hops, so it can fill exactly those three parameters. T2_PATH_BETWEEN
    needs source_id AND target_id, which is why it has no description and is
    not offered — this test is what makes that reasoning enforced rather than
    remembered.
    """
    fillable = {"entity_id", "relationship", "max_hops"}

    unfillable = {
        name: sorted({spec.name for spec in t.params} - fillable)
        for name, t in PLANNABLE.items()
        if {spec.name for spec in t.params} - fillable
    }

    assert unfillable == {}


def test_the_aggregation_template_is_offered_to_the_planner():
    """T8_RELATED_BY answers 'how many X relate to Y', the aggregation
    category's shape. It was allowed but undescribed, so the planner never
    knowingly chose it and aggregation questions fell to a corpus-wide count."""
    assert "T8_RELATED_BY" in PLANNABLE
    assert "T8_RELATED_BY" in SYSTEM_PROMPT


def test_path_between_is_not_offered_to_the_planner():
    assert "T2_PATH_BETWEEN" in TEMPLATES
    assert "T2_PATH_BETWEEN" not in PLANNABLE
    assert "T2_PATH_BETWEEN" not in SYSTEM_PROMPT


class _FakeResponse:
    def __init__(self, output_parsed):
        self.output_parsed = output_parsed


class _FakeResponses:
    def __init__(self, result):
        self._result = result
        self.last_call = None

    def parse(self, **kwargs):
        self.last_call = kwargs
        return _FakeResponse(self._result)


class _FakeClient:
    def __init__(self, result):
        self.responses = _FakeResponses(result)


def test_plan_graph_query_returns_the_models_plan():
    canned = GraphQueryPlan(
        template_id="T3_EXCEPTION_WRAP_CHAIN",
        entity_surface="ConnectTimeout",
        relationship=None,
        max_hops=2,
    )
    client = _FakeClient(canned)

    plan = plan_graph_query("What does ConnectTimeout ultimately wrap?", client=client)

    assert plan.template_id == "T3_EXCEPTION_WRAP_CHAIN"
    assert plan.entity_surface == "ConnectTimeout"
    assert plan.max_hops == 2


def test_plan_graph_query_uses_structured_output():
    canned = GraphQueryPlan(template_id="T1_NEIGHBORS", entity_surface="Session")
    client = _FakeClient(canned)

    plan_graph_query("What connects to Session?", client=client)

    assert client.responses.last_call["text_format"] is GraphQueryPlan


class _SequenceResponses:
    def __init__(self, results):
        self._results = list(results)
        self.calls = 0
        self.last_call = None

    def parse(self, **kwargs):
        self.last_call = kwargs
        result = self._results[self.calls % len(self._results)]
        self.calls += 1
        return _FakeResponse(result)


class _SequenceClient:
    def __init__(self, results):
        self.responses = _SequenceResponses(results)


def test_plan_graph_query_takes_the_majority_plan():
    """The planner returned a different plan on 11.7% of benchmark questions
    (D53) — a different template, entity or hop count each run."""
    client = _SequenceClient([
        GraphQueryPlan(template_id="T3_EXCEPTION_WRAP_CHAIN",
                       entity_surface="ReadTimeoutError", max_hops=2),
        GraphQueryPlan(template_id="T3_EXCEPTION_WRAP_CHAIN",
                       entity_surface="ReadTimeoutError", max_hops=3),
        GraphQueryPlan(template_id="T3_EXCEPTION_WRAP_CHAIN",
                       entity_surface="ReadTimeoutError", max_hops=2),
    ])

    plan = plan_graph_query("q", client=client, votes=3)

    assert plan.max_hops == 2
    assert client.responses.calls == 3


def test_plan_graph_query_votes_across_differing_templates():
    client = _SequenceClient([
        GraphQueryPlan(template_id="T1_NEIGHBORS", entity_surface="requests"),
        GraphQueryPlan(template_id="T5_DELEGATION_CHAIN",
                       entity_surface="requests", max_hops=2),
        GraphQueryPlan(template_id="T1_NEIGHBORS", entity_surface="requests"),
    ])

    plan = plan_graph_query("q", client=client, votes=3)

    assert plan.template_id == "T1_NEIGHBORS"


def test_plan_graph_query_can_be_called_without_voting():
    client = _SequenceClient([
        GraphQueryPlan(template_id="T1_NEIGHBORS", entity_surface="Session")
    ])

    plan_graph_query("q", client=client, votes=1)

    assert client.responses.calls == 1


def test_build_template_values_resolves_the_entity_surface():
    resolver = Resolver(
        node_universe={"requests.exceptions.ConnectTimeout"}, import_aliases={},
    )
    plan = GraphQueryPlan(
        template_id="T3_EXCEPTION_WRAP_CHAIN", entity_surface="ConnectTimeout", max_hops=2,
    )

    values, known_ids = build_template_values(plan, resolver=resolver)

    assert values["entity_id"] == "requests.exceptions.ConnectTimeout"
    assert values["max_hops"] == 2
    assert known_ids == {"requests.exceptions.ConnectTimeout"}


def test_build_template_values_raises_when_entity_surface_cannot_be_resolved():
    import pytest

    resolver = Resolver(node_universe=set(), import_aliases={})
    plan = GraphQueryPlan(template_id="T1_NEIGHBORS", entity_surface="TotallyMadeUpThing")

    with pytest.raises(ValueError):
        build_template_values(plan, resolver=resolver)


def test_build_template_values_passes_through_relationship_with_no_entity():
    resolver = Resolver(node_universe=set(), import_aliases={})
    plan = GraphQueryPlan(template_id="T6_COUNT_BY_REL", relationship="WRAPS_EXCEPTION")

    values, known_ids = build_template_values(plan, resolver=resolver)

    assert values == {"relationship": "WRAPS_EXCEPTION"}
    assert known_ids == set()
