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
    # Every kind _plan_model_for knows how to turn into a schema field.
    fillable = {"entity_id", "relationship", "max_hops", "package", "name"}
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


def test_planner_is_told_when_a_module_is_the_right_anchor():
    """The closing guidance warns off Module anchors in general; a question
    about a module's contents is the exception, and the planner needs to be
    told which template and relationship express it (D73)."""
    from graphrag.retrieval.graph_query import SYSTEM_PROMPT
    assert "DEFINED_IN" in SYSTEM_PROMPT
    assert "builtins.ValueError" in SYSTEM_PROMPT


def test_repair_plan_turns_a_modules_wrong_relationship_into_defined_in():
    """A Module has only DEFINED_IN and IMPORTS edges. Planned INHERITS_FROM
    on requests.exceptions returns nothing; its members are what the
    question wanted (D73)."""
    from graphrag.retrieval.graph_query import Candidate, repair_plan
    candidates = [Candidate("requests.exceptions", kind="Module", degree=40)]

    template_id, values, note = repair_plan(
        "T8_RELATED_BY",
        {"entity_id": "requests.exceptions", "relationship": "INHERITS_FROM"},
        candidates,
    )

    assert template_id == "T8_RELATED_BY"
    assert values == {"entity_id": "requests.exceptions", "relationship": "DEFINED_IN"}
    assert note


def test_repair_plan_scopes_a_corpus_wide_count_to_the_one_module_named():
    from graphrag.retrieval.graph_query import Candidate, repair_plan
    candidates = [Candidate("requests", kind="Module", degree=65)]

    template_id, values, note = repair_plan(
        "T6_COUNT_BY_REL", {"relationship": "DEFINED_IN"}, candidates
    )

    assert template_id == "T8_RELATED_BY"
    assert values == {"entity_id": "requests", "relationship": "DEFINED_IN"}
    assert note


def test_repair_plan_leaves_a_sound_plan_alone():
    from graphrag.retrieval.graph_query import Candidate, repair_plan
    candidates = [
        Candidate("requests.exceptions.Timeout", kind="Class/Exception", degree=6),
        Candidate("requests", kind="Module", degree=65),
        Candidate("urllib3", kind="Module", degree=70),
    ]
    for template_id, values in (
        ("T8_RELATED_BY", {"entity_id": "requests.exceptions.Timeout", "relationship": "INHERITS_FROM"}),
        ("T8_RELATED_BY", {"entity_id": "requests", "relationship": "IMPORTS"}),
        ("T6_COUNT_BY_REL", {"relationship": "WRAPS_EXCEPTION"}),  # two modules named: no repair
    ):
        out_id, out_values, note = repair_plan(template_id, values, candidates)
        assert (out_id, out_values, note) == (template_id, values, None)


def test_repair_plan_starts_a_wrap_chain_from_what_a_function_raises():
    """A wrap chain anchored on iter_content returned nothing (3h-17, ag-20):
    WRAPS_EXCEPTION joins exceptions. What the function raises is where
    that answer starts (D74)."""
    from graphrag.retrieval.graph_query import Candidate, repair_plan
    candidates = [Candidate("requests.models.Response.iter_content", kind="Function", degree=12)]

    template_id, values, note = repair_plan(
        "T3_EXCEPTION_WRAP_CHAIN",
        {"entity_id": "requests.models.Response.iter_content", "max_hops": 2}, candidates,
    )
    assert (template_id, values) == (
        "T8_RELATED_BY", {"entity_id": "requests.models.Response.iter_content", "relationship": "RAISES"})
    assert note

    template_id, values, note = repair_plan(
        "T8_RELATED_BY",
        {"entity_id": "requests.models.Response.iter_content", "relationship": "WRAPS_EXCEPTION"}, candidates,
    )
    assert values["relationship"] == "RAISES" and note


def test_repair_plan_moves_a_wrap_chain_off_a_package_root():
    """Planned T3 on "urllib3" with HTTPAdapter.send also resolved (3h-25)."""
    from graphrag.retrieval.graph_query import Candidate, repair_plan
    candidates = [
        Candidate("urllib3", kind="Module", degree=70),
        Candidate("requests.adapters.HTTPAdapter.send", kind="Function", degree=30),
    ]
    template_id, values, note = repair_plan(
        "T3_EXCEPTION_WRAP_CHAIN", {"entity_id": "urllib3", "max_hops": 2}, candidates
    )
    assert (template_id, values) == (
        "T8_RELATED_BY", {"entity_id": "requests.adapters.HTTPAdapter.send", "relationship": "RAISES"})

    candidates = [
        Candidate("urllib3", kind="Module", degree=70),
        Candidate("urllib3.exceptions.ClosedPoolError", kind="Class", degree=3),  # label Class alone, as urllib3's are
    ]
    template_id, values, note = repair_plan(
        "T3_EXCEPTION_WRAP_CHAIN", {"entity_id": "urllib3", "max_hops": 2}, candidates
    )
    assert (template_id, values) == (
        "T3_EXCEPTION_WRAP_CHAIN", {"entity_id": "urllib3.exceptions.ClosedPoolError", "max_hops": 2})


def test_resolve_mentions_drops_parameter_noise_when_a_class_matches_too():
    """"Timeout" resolves ambiguously to one class and thirty parameters
    named timeout; the planner was shown all of them (3h-24, D74)."""
    class R:
        def resolve(self, surface, **kw):
            class Res:
                canonical_id = None
                candidates = ("requests.exceptions.Timeout",
                              "requests.sessions.Session.request.timeout",
                              "urllib3.util.timeout.Timeout")
            return Res()

    def describe(ids):
        return {
            "requests.exceptions.Timeout": ("Class/Exception", 6),
            "requests.sessions.Session.request.timeout": ("Parameter", 1),
            "urllib3.util.timeout.Timeout": ("Class", 9),
        }

    candidates = resolve_mentions([Mention(surface="Timeout", package="unknown")], resolver=R(), describe=describe)

    assert [c.canonical_id for c in candidates] == [
        "requests.exceptions.Timeout", "urllib3.util.timeout.Timeout"]


def test_resolve_mentions_keeps_parameters_when_nothing_else_matches():
    class R:
        def resolve(self, surface, **kw):
            class Res:
                canonical_id = None
                candidates = ("a.f.verify", "a.g.verify")
            return Res()

    candidates = resolve_mentions(
        [Mention(surface="verify", package="unknown")], resolver=R(),
        describe=lambda ids: {i: ("Parameter", 1) for i in ids},
    )
    assert len(candidates) == 2


def test_mentions_are_voted_by_default():
    from graphrag.retrieval.graph_query import MENTION_VOTES, extract_mentions
    import inspect
    assert MENTION_VOTES == 3
    assert inspect.signature(extract_mentions).parameters["votes"].default == 3


def test_mentions_the_question_does_not_contain_are_dropped():
    """The extractor returned `ReadTimeoutError` for "a streamed body cut
    short" and `HeaderParsingError` for "rejects a header value" (D76). An
    invented name resolves, becomes the anchor, and the chain walks from
    the wrong place. Plurals and call parentheses are forgiven."""
    from graphrag.retrieval.graph_query import mentioned_in
    q = "If a streamed body is cut short inside iter_content(), which requests exceptions surface?"
    assert mentioned_in("iter_content", q)
    assert mentioned_in("iter_content()", q)
    assert mentioned_in("requests exception", q)
    assert not mentioned_in("ReadTimeoutError", q)
    assert not mentioned_in("", q)


def test_extract_mentions_grounds_the_voted_result_in_the_question():
    from graphrag.retrieval.graph_query import Mentions

    class _Parsed:
        def __init__(self, v): self.output_parsed = v

    class _Responses:
        def parse(self, **kw):
            return _Parsed(Mentions(mentions=[
                Mention(surface="iter_content", package="requests"),
                Mention(surface="ReadTimeoutError", package="urllib3"),
            ]))

    class _Client:
        responses = _Responses()

    mentions = extract_mentions("What does iter_content raise?", client=_Client(), votes=1)
    assert [m.surface for m in mentions] == ["iter_content"]


def test_repair_plan_sends_a_raises_or_wrap_plan_on_a_plain_class_to_its_methods():
    """"Which Session method raises it" planned RAISES on the class Session:
    zero edges, since RAISES hangs off methods (3h-22, 3h-21; D76)."""
    from graphrag.retrieval.graph_query import Candidate, repair_plan
    candidates = [Candidate("requests.sessions.Session", kind="Class", degree=40)]

    out_id, out_values, note = repair_plan(
        "T8_RELATED_BY", {"entity_id": "requests.sessions.Session", "relationship": "RAISES"}, candidates)
    assert (out_id, out_values) == ("T10_RAISED_BY_METHODS_OF", {"entity_id": "requests.sessions.Session"})
    assert note

    # A wrap chain on a class is NOT repaired before it runs: urllib3's
    # exceptions carry the label Class alone, and guessing from the label
    # emptied the flagship two-hop questions (D77). retrieve() falls back
    # only after the chain returns nothing.
    for kind in ("Class", "Class/Exception"):
        cands = [Candidate("urllib3.exceptions.ProtocolError", kind=kind, degree=6)]
        out = repair_plan(
            "T3_EXCEPTION_WRAP_CHAIN", {"entity_id": "urllib3.exceptions.ProtocolError", "max_hops": 2}, cands)
        assert out[2] is None, kind


def test_a_capitalised_surface_must_match_case_and_generic_words_are_not_mentions():
    from graphrag.retrieval.graph_query import mentioned_in
    q = "what does requests convert it to, and what is that exception's base class? The connection is reused."
    assert not mentioned_in("Exception", q), "lowercase 'exception' names no class"
    assert mentioned_in("Exception", "Does Exception have a base class?")
    assert not mentioned_in("connection", q), "a generic word is not an entity"
    assert not mentioned_in("exception", q)
    assert mentioned_in("requests", q)


def test_repair_plan_sends_any_other_template_on_a_parameter_to_its_neighbourhood():
    from graphrag.retrieval.graph_query import Candidate, repair_plan
    candidates = [Candidate("urllib3.connectionpool.HTTPConnectionPool.urlopen.retries", kind="Parameter", degree=2)]
    out_id, out_values, note = repair_plan(
        "T3_EXCEPTION_WRAP_CHAIN",
        {"entity_id": "urllib3.connectionpool.HTTPConnectionPool.urlopen.retries", "max_hops": 2}, candidates,
    )
    assert (out_id, out_values) == (
        "T1_NEIGHBORS", {"entity_id": "urllib3.connectionpool.HTTPConnectionPool.urlopen.retries"})
    assert note


def test_the_plan_model_offers_the_package_template_with_a_package_enum():
    from graphrag.retrieval.graph_query import _plan_model_for
    model = _plan_model_for("T11_EDGES_IN_PACKAGE", [])
    assert model is not None, "needs no entity, so it is offered even with no candidates"
    plan = model(template_id="T11_EDGES_IN_PACKAGE", package="requests", relationship="WRAPS_EXCEPTION")
    assert plan.package == "requests"
    import pytest
    with pytest.raises(Exception):
        model(template_id="T11_EDGES_IN_PACKAGE", package="django", relationship="WRAPS_EXCEPTION")


def test_dotted_names_in_the_question_are_mentions_regardless_of_the_model():
    """"Session.request" came back as "Session" plus "request" on some calls
    and "requests.api" as "request" on others, three votes notwithstanding
    (D82). A dotted name is read off the question text."""
    from graphrag.retrieval.graph_query import dotted_names_in
    assert dotted_names_in("What parameters does Session.request accept?") == ["Session.request"]
    assert dotted_names_in("How many helpers does requests.api define, e.g. get()?") == ["requests.api"]
    assert dotted_names_in("Which classes in requests.exceptions are warnings?") == ["requests.exceptions"]
    assert dotted_names_in("What does Response.iter_content() raise?") == ["Response.iter_content"]
    assert dotted_names_in("Does requests follow redirects?") == []


def test_extract_mentions_adds_dotted_names_the_model_missed():
    from graphrag.retrieval.graph_query import Mentions

    class _Parsed:
        def __init__(self, v): self.output_parsed = v

    class _Responses:
        def parse(self, **kw):
            return _Parsed(Mentions(mentions=[Mention(surface="Session", package="requests")]))

    class _Client:
        responses = _Responses()

    mentions = extract_mentions("What parameters does Session.request accept?", client=_Client(), votes=1)
    assert [m.surface for m in mentions] == ["Session", "Session.request"]


def test_repair_plan_scopes_a_count_to_the_most_specific_module_named():
    from graphrag.retrieval.graph_query import Candidate, repair_plan
    candidates = [Candidate("requests", kind="Module", degree=65),
                  Candidate("requests.exceptions", kind="Module", degree=40)]
    template_id, values, note = repair_plan("T6_COUNT_BY_REL", {"relationship": "INHERITS_FROM"}, candidates)
    assert (template_id, values) == ("T8_RELATED_BY", {"entity_id": "requests.exceptions", "relationship": "DEFINED_IN"})

    # A module plus a specific entity: the count is left to the model's choice.
    candidates.append(Candidate("requests.exceptions.Timeout", kind="Class", degree=6))
    assert repair_plan("T6_COUNT_BY_REL", {"relationship": "INHERITS_FROM"}, candidates)[2] is None


def test_the_plan_model_takes_a_parameter_name_as_free_text_for_t13():
    from graphrag.retrieval.graph_query import _plan_model_for
    model = _plan_model_for("T13_FUNCTIONS_WITH_PARAMETER", [])
    assert model is not None
    assert model(template_id="T13_FUNCTIONS_WITH_PARAMETER", name="data").name == "data"
