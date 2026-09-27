"""jedi-backed call resolution.

These tests build a tiny two-module package on disk because jedi resolves
against real files: it follows an import to the module it names, an
assignment to the class it constructs, and `self`/`super()` to the class
hierarchy. None of that is visible from one file's AST, which is why the
project's own resolver could never do it (D58).
"""

import textwrap

from graphrag.ingest.chunk import make_chunk_id
from graphrag.ingest.jedi_calls import build_project, resolve_calls


def _package(tmp_path):
    pkg = tmp_path / "src" / "pkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "a.py").write_text(textwrap.dedent('''\
        class Base:
            def __init__(self):
                pass

        class A(Base):
            def __init__(self):
                super().__init__()
                self.setup()

            def setup(self):
                pass

            def m(self):
                return 1
    '''))
    (pkg / "b.py").write_text(textwrap.dedent('''\
        import os
        from pkg.a import A

        def f():
            x = A()
            x.m()
            len([])
            os.path.join("a", "b")

        def g(flag):
            if flag:
                target = f
            else:
                target = A
            target()
    '''))
    return tmp_path / "src"


def _edges(tmp_path, filename):
    src = _package(tmp_path)
    project = build_project(src, others=[])
    path = src / "pkg" / filename
    return resolve_calls(
        path.read_text(), repo="pkg", path=str(path),
        dotted_module=f"pkg.{filename[:-3]}", project=project, packages=("pkg",),
    )


def test_resolves_a_method_call_through_an_assigned_variable_across_modules(tmp_path):
    """`x = A(); x.m()` — the receiver's type comes from the assignment, and
    A comes from another module. This is `conn.urlopen` in miniature."""
    pairs = {(e.source_id, e.surface, e.target_id) for e in _edges(tmp_path, "b.py")}

    assert ("pkg.b.f", "x.m", "pkg.a.A.m") in pairs
    assert ("pkg.b.f", "A", "pkg.a.A") in pairs


def test_ignores_calls_outside_the_named_packages(tmp_path):
    # Builtins and the stdlib are real calls but not corpus nodes; recording
    # them would anchor edges to ids nothing else in the graph knows.
    surfaces = {e.surface for e in _edges(tmp_path, "b.py")}

    assert "len" not in surfaces
    assert "os.path.join" not in surfaces


def test_skips_a_call_jedi_cannot_pin_to_one_definition(tmp_path):
    """`target` is f on one branch and A on the other. jedi reports both;
    choosing one would be a guess, and a wrong CALLS edge is worse than a
    missing one."""
    from_g = {e.surface for e in _edges(tmp_path, "b.py") if e.source_id == "pkg.b.g"}

    assert "target" not in from_g


def test_resolves_self_and_super_through_the_class_hierarchy(tmp_path):
    pairs = {(e.source_id, e.surface, e.target_id) for e in _edges(tmp_path, "a.py")}

    assert ("pkg.a.A.__init__", "self.setup", "pkg.a.A.setup") in pairs
    assert ("pkg.a.A.__init__", "super().__init__", "pkg.a.Base.__init__") in pairs


def test_edge_carries_the_calling_functions_chunk_id(tmp_path):
    src = _package(tmp_path)
    path = src / "pkg" / "b.py"
    edges = resolve_calls(
        path.read_text(), repo="pkg", path=str(path), dotted_module="pkg.b",
        project=build_project(src, others=[]), packages=("pkg",),
    )
    f_edge = next(e for e in edges if e.source_id == "pkg.b.f")

    # f spans lines 4-8 of b.py; the chunk id is what citations resolve to.
    assert f_edge.chunk_id == make_chunk_id("pkg", str(path), 4, 8)
    assert f_edge.method == "jedi"


def test_one_edge_per_distinct_target_per_function(tmp_path):
    # `A` is called once in f and once in g, but each function gets its own
    # edge; within one function a repeated call is one fact, not two.
    edges = [e for e in _edges(tmp_path, "b.py") if e.source_id == "pkg.b.f"]
    targets = [e.target_id for e in edges]

    assert len(targets) == len(set(targets))


# --- override-following (D68) -------------------------------------------------

from graphrag.ingest.jedi_calls import CallEdge, expand_overrides

UNIVERSE = {
    "requests.adapters.BaseAdapter", "requests.adapters.BaseAdapter.send",
    "requests.adapters.HTTPAdapter", "requests.adapters.HTTPAdapter.send",
    "requests.adapters.HTTPAdapter.close",
    "app.CachingAdapter", "app.CachingAdapter.send",
    "app.LoggingAdapter",  # inherits HTTPAdapter but does not override send
    "requests.sessions.Session.send",
}
SUBCLASSES = {
    "requests.adapters.BaseAdapter": ["requests.adapters.HTTPAdapter"],
    "requests.adapters.HTTPAdapter": ["app.CachingAdapter", "app.LoggingAdapter"],
}
CALL = CallEdge(source_id="requests.sessions.Session.send", surface="adapter.send",
                target_id="requests.adapters.BaseAdapter.send", chunk_id="c1")


def test_a_call_to_a_base_method_also_reaches_each_override():
    """jedi correctly resolves `adapter.send` to `BaseAdapter.send` -- that
    is the static type. At runtime it is `HTTPAdapter.send`, and the oracle
    counted that as a miss (D67). The graph knows the inheritance; a call to
    a base method is a call to every override that could receive it."""
    extra = expand_overrides([CALL], subclasses=SUBCLASSES, node_universe=UNIVERSE)
    targets = {e.target_id for e in extra}

    assert "requests.adapters.HTTPAdapter.send" in targets
    assert all(e.method == "override" for e in extra)
    assert all(e.source_id == CALL.source_id and e.chunk_id == "c1" for e in extra)


def test_overrides_are_followed_transitively():
    extra = expand_overrides([CALL], subclasses=SUBCLASSES, node_universe=UNIVERSE)

    assert "app.CachingAdapter.send" in {e.target_id for e in extra}


def test_a_subclass_that_does_not_override_gets_no_edge():
    # LoggingAdapter inherits HTTPAdapter.send unchanged; a call reaches the
    # inherited method, which the HTTPAdapter.send edge already covers.
    extra = expand_overrides([CALL], subclasses=SUBCLASSES, node_universe=UNIVERSE)

    assert "app.LoggingAdapter.send" not in {e.target_id for e in extra}


def test_a_call_to_a_function_or_class_is_left_alone():
    edges = [
        CallEdge(source_id="m.f", surface="parse_url", target_id="urllib3.util.url.parse_url", chunk_id="c2"),
        CallEdge(source_id="m.f", surface="HTTPAdapter", target_id="requests.adapters.HTTPAdapter", chunk_id="c2"),
    ]

    assert expand_overrides(edges, subclasses=SUBCLASSES, node_universe=UNIVERSE) == []


def test_the_original_edge_is_not_repeated():
    extra = expand_overrides([CALL], subclasses=SUBCLASSES, node_universe=UNIVERSE)

    assert CALL.target_id not in {e.target_id for e in extra}
