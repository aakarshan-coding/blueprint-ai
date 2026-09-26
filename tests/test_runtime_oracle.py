"""The runtime oracle: what the code actually did, as ground truth for the graph.

Run a test suite under sys.settrace and record every real caller -> callee
pair, every place an exception was first raised, and every exception raised
while another was being handled (a wrap, by Python's own chaining rule). The
graph's CALLS, RAISES and WRAPS_EXCEPTION edges are then measured against
that -- observed, not inferred. code-graph-rag does the same for calls with
an execution trace (D67); the wrap oracle is ours.
"""

from graphrag.eval.runtime_oracle import (
    Oracle,
    canonical_id,
    compare_pairs,
    exception_id,
)


def test_canonical_id_maps_a_method_to_the_graphs_id(tmp_path):
    # A real code object whose co_filename points into a corpus tree.
    root = tmp_path / "src"
    (root / "requests").mkdir(parents=True)
    src = "class HTTPAdapter:\n    def send(self):\n        pass\n"
    namespace: dict = {}
    exec(compile(src, str(root / "requests" / "adapters.py"), "exec"), namespace)
    code = namespace["HTTPAdapter"].send.__code__

    assert canonical_id(code, roots={"requests": root}) == "requests.adapters.HTTPAdapter.send"


def test_canonical_id_folds_a_nested_function_into_its_parent(tmp_path):
    """iter_symbols never yields nested defs; their raises and calls belong to
    the enclosing function. Response.iter_content.<locals>.generate is
    Response.iter_content in the graph, so it must be here too."""
    root = tmp_path / "src"
    (root / "requests").mkdir(parents=True)
    src = (
        "class Response:\n"
        "    def iter_content(self):\n"
        "        def generate():\n"
        "            pass\n"
        "        return generate\n"
    )
    namespace: dict = {}
    exec(compile(src, str(root / "requests" / "models.py"), "exec"), namespace)
    inner = namespace["Response"]().iter_content().__code__

    assert canonical_id(inner, roots={"requests": root}) == "requests.models.Response.iter_content"


def test_canonical_id_is_none_outside_the_corpus(tmp_path):
    src = "def f():\n    pass\n"
    namespace: dict = {}
    exec(compile(src, str(tmp_path / "elsewhere" / "x.py"), "exec"), namespace)

    assert canonical_id(namespace["f"].__code__, roots={"requests": tmp_path / "src"}) is None


def test_canonical_id_skips_lambdas_and_comprehensions(tmp_path):
    root = tmp_path / "src"
    (root / "requests").mkdir(parents=True)
    namespace: dict = {}
    exec(compile("g = lambda: 0\n", str(root / "requests" / "u.py"), "exec"), namespace)

    assert canonical_id(namespace["g"].__code__, roots={"requests": root}) is None


def test_exception_id_uses_the_defining_module():
    class Boom(Exception):
        pass
    Boom.__module__ = "requests.exceptions"

    assert exception_id(Boom) == "requests.exceptions.Boom"
    assert exception_id(ValueError) == "builtins.ValueError"


def test_oracle_records_a_call_a_raise_and_a_wrap(tmp_path):
    """One traced run of code that calls, raises, catches and re-raises."""
    root = tmp_path / "src"
    (root / "requests").mkdir(parents=True)
    src = (
        "class Inner(Exception):\n    pass\n"
        "class Outer(Exception):\n    pass\n"
        "def low():\n    raise Inner()\n"
        "def high():\n"
        "    try:\n        low()\n"
        "    except Inner as e:\n        raise Outer() from e\n"
    )
    namespace: dict = {}
    exec(compile(src, str(root / "requests" / "m.py"), "exec"), namespace)
    for cls in ("Inner", "Outer"):
        namespace[cls].__module__ = "requests.m"

    oracle = Oracle(roots={"requests": root})
    with oracle.tracing():
        try:
            namespace["high"]()
        except namespace["Outer"]:
            pass

    assert ("requests.m.high", "requests.m.low") in oracle.calls
    assert ("requests.m.low", "requests.m.Inner") in oracle.raises
    assert ("requests.m.high", "requests.m.Outer") in oracle.raises
    assert ("requests.m.Outer", "requests.m.Inner") in oracle.wraps


def test_oracle_attributes_a_raise_to_where_it_started_not_every_frame_it_crosses(tmp_path):
    root = tmp_path / "src"
    (root / "requests").mkdir(parents=True)
    src = "def low():\n    raise ValueError()\ndef mid():\n    low()\ndef top():\n    mid()\n"
    namespace: dict = {}
    exec(compile(src, str(root / "requests" / "m.py"), "exec"), namespace)

    oracle = Oracle(roots={"requests": root})
    with oracle.tracing():
        try:
            namespace["top"]()
        except ValueError:
            pass

    assert ("requests.m.low", "builtins.ValueError") in oracle.raises
    assert ("requests.m.mid", "builtins.ValueError") not in oracle.raises
    assert ("requests.m.top", "builtins.ValueError") not in oracle.raises


def test_a_constructor_call_confirms_the_edge_to_the_class():
    """The runtime sees `Retry(...)` as a call to `Retry.__init__`; jedi's
    edge points at the class `Retry`. One fact, two spellings -- 61 of 173
    missing calls in the first full run were this (D67)."""
    report = compare_pairs(
        observed={("m.f", "m.A.__init__")},
        graph={("m.f", "m.A")},
    )

    assert report["confirmed"] == 1
    assert report["missing_from_graph"] == []


def test_implicit_callees_are_left_out_of_the_oracle():
    """`__setitem__`, `__iter__`, `__enter__` and a @property run without any
    call syntax, so no static tool is ever asked about them. They are not
    misses; they are invisible by construction, and are excluded up front."""
    report = compare_pairs(
        observed={("m.f", "m.D.__setitem__"), ("m.f", "m.R.content"), ("m.f", "m.g")},
        graph={("m.f", "m.g")},
        implicit={"m.D.__setitem__", "m.R.content"},
    )

    assert report["observed"] == 1
    assert report["confirmed"] == 1


def test_a_static_edge_to_a_superclass_is_confirmed_by_a_runtime_subclass():
    """The AST says `NewConnectionError wraps OSError`; at runtime the OSError
    was a `ConnectionRefusedError`. The static edge is the superclass of the
    runtime fact -- coarser, not wrong. `socket.timeout` *is* `TimeoutError`
    in modern Python, the same situation with an alias."""
    hierarchy = {
        "builtins.ConnectionRefusedError": [
            "builtins.ConnectionRefusedError", "builtins.ConnectionError", "builtins.OSError",
        ],
    }
    report = compare_pairs(
        observed={("u.NewConnectionError", "builtins.ConnectionRefusedError")},
        graph={("u.NewConnectionError", "builtins.OSError")},
        hierarchy=hierarchy,
    )

    assert report["confirmed"] == 1
    assert report["unconfirmed_in_graph"] == []


def test_hierarchy_matching_never_goes_downward():
    # A graph edge to the *subclass* is not confirmed by a runtime superclass:
    # the static claim is more specific than what was observed.
    hierarchy = {"builtins.OSError": ["builtins.OSError", "builtins.Exception"]}
    report = compare_pairs(
        observed={("u.X", "builtins.OSError")},
        graph={("u.X", "builtins.ConnectionRefusedError")},
        hierarchy=hierarchy,
    )

    assert report["confirmed"] == 0


def test_compare_pairs_reports_recall_and_observed_precision():
    observed = {("a", "b"), ("a", "c"), ("d", "e")}
    graph = {("a", "b"), ("a", "c"), ("x", "y")}

    report = compare_pairs(observed=observed, graph=graph)

    assert report["observed"] == 3
    assert report["in_graph"] == 3
    assert report["confirmed"] == 2
    assert report["recall"] == round(2 / 3, 3)
    assert report["observed_precision"] == round(2 / 3, 3)
    assert report["missing_from_graph"] == [("d", "e")]
    assert report["unconfirmed_in_graph"] == [("x", "y")]
