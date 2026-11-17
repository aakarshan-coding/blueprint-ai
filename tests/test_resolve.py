from graphrag.ingest.resolve import Resolver


def test_exact_canonical_id_resolves_to_itself():
    resolver = Resolver(
        node_universe={"requests.sessions.Session", "requests.adapters.HTTPAdapter"},
        import_aliases={},
    )

    result = resolver.resolve("requests.sessions.Session")

    assert result.canonical_id == "requests.sessions.Session"
    assert result.method == "exact"


def test_unknown_surface_with_no_alias_and_no_match_is_unresolved():
    resolver = Resolver(
        node_universe={"requests.sessions.Session"},
        import_aliases={},
    )

    result = resolver.resolve("TotallyMadeUpThing")

    assert result.canonical_id is None
    assert result.method == "unresolved"


def test_bare_name_resolves_via_its_module_import_table():
    resolver = Resolver(
        node_universe=set(),
        import_aliases={
            "requests.adapters": {"MaxRetryError": "urllib3.exceptions.MaxRetryError"},
        },
    )

    result = resolver.resolve("MaxRetryError", module_context="requests.adapters")

    assert result.canonical_id == "urllib3.exceptions.MaxRetryError"
    assert result.method == "import_alias"


def test_a_qualified_surface_resolves_by_its_dotted_suffix():
    """"HTTPAdapter.send" names one thing unambiguously, but every rung
    looked at either the whole id or the bare leaf `send` — which matches
    Session.send, BaseAdapter.send and HTTPAdapter.send, so it fell through
    as ambiguous. The planner writes qualified names exactly when the bare
    one would be ambiguous (planner audit, ag-03), so this is the case that
    most needs to work."""
    resolver = Resolver(
        node_universe={
            "requests.adapters.HTTPAdapter.send",
            "requests.adapters.BaseAdapter.send",
            "requests.sessions.Session.send",
        },
        import_aliases={},
    )

    result = resolver.resolve("HTTPAdapter.send")

    assert result.canonical_id == "requests.adapters.HTTPAdapter.send"
    assert result.method == "qualified"


def test_a_qualified_surface_matching_several_ids_stays_unresolved_with_candidates():
    resolver = Resolver(
        node_universe={"requests.exceptions.ProxyError", "urllib3.exceptions.ProxyError"},
        import_aliases={},
    )

    result = resolver.resolve("exceptions.ProxyError")

    assert result.canonical_id is None
    assert set(result.candidates) == {
        "requests.exceptions.ProxyError", "urllib3.exceptions.ProxyError",
    }


def test_a_qualified_surface_never_matches_mid_segment():
    # "Adapter.send" must not match "HTTPAdapter.send": the suffix has to
    # start at a dot boundary or it is a different name that happens to end
    # the same way.
    resolver = Resolver(
        node_universe={"requests.adapters.HTTPAdapter.send"}, import_aliases={},
    )

    result = resolver.resolve("Adapter.send")

    assert result.canonical_id is None


def test_a_name_defined_in_the_module_being_read_wins_over_a_same_named_class_elsewhere():
    """`class ReadTimeout(Timeout)` is written inside requests/exceptions.py,
    which defines Timeout. urllib3.util.timeout defines a Timeout too. The
    ladder saw two candidates and refused, so Timeout had no subclasses in
    the graph (D66). A name defined in the current module is what that
    module means by it -- plain Python scope, and the rung that was missing."""
    resolver = Resolver(
        node_universe={"requests.exceptions.Timeout", "urllib3.util.timeout.Timeout"},
        import_aliases={},
    )

    result = resolver.resolve("Timeout", module_context="requests.exceptions")

    assert result.canonical_id == "requests.exceptions.Timeout"
    assert result.method == "same_module"


def test_an_import_into_the_module_still_beats_a_same_module_lookup():
    # `from urllib3.util.timeout import Timeout` inside a module that does
    # not itself define Timeout: the import is the binding.
    resolver = Resolver(
        node_universe={"urllib3.util.timeout.Timeout", "requests.exceptions.Timeout"},
        import_aliases={"requests.adapters": {"Timeout": "urllib3.util.timeout.Timeout"}},
    )

    result = resolver.resolve("Timeout", module_context="requests.adapters")

    assert result.canonical_id == "urllib3.util.timeout.Timeout"
    assert result.method == "import_alias"


def test_a_python_builtin_exception_resolves_to_a_builtins_id():
    """`except OSError: raise ConnectionError(e)` is a wrap the AST extractor
    finds and the graph could not hold: OSError has no node and no import,
    so 43 of 81 wrap facts dropped (D57). Builtins are a fixed, known set;
    they resolve to `builtins.<name>` and are typed by the edge they sit on."""
    resolver = Resolver(node_universe=set(), import_aliases={})

    result = resolver.resolve("OSError")

    assert result.canonical_id == "builtins.OSError"
    assert result.method == "builtin"


def test_a_corpus_name_that_shadows_a_builtin_wins_over_the_builtin():
    # requests defines its own ConnectionError; a bare mention inside the
    # corpus means that one, not Python's. Corpus rungs run first.
    resolver = Resolver(
        node_universe={"requests.exceptions.ConnectionError"}, import_aliases={},
    )

    result = resolver.resolve("ConnectionError")

    assert result.canonical_id == "requests.exceptions.ConnectionError"


def test_a_builtin_type_named_in_a_module_that_does_not_bind_it_is_the_builtin():
    """`class HTTPHeaderDictItemView(set[...])` in urllib3 inherits from Python's
    set. The cross-package rung matched the one corpus symbol whose leaf is
    `set`, a method of RequestsCookieJar, and wrote a false base (D97)."""
    resolver = Resolver(
        node_universe={"requests.cookies.RequestsCookieJar.set"}, import_aliases={},
    )

    result = resolver.resolve("set", module_context="urllib3._collections")

    assert result.canonical_id == "builtins.set"
    assert result.method == "builtin"


def test_a_module_that_imports_a_builtin_name_means_its_import():
    resolver = Resolver(
        node_universe=set(), import_aliases={"pkg.mod": {"dict": "pkg.compat.dict"}},
    )

    assert resolver.resolve("dict", module_context="pkg.mod").canonical_id == "pkg.compat.dict"


def test_a_name_that_is_not_a_builtin_exception_is_still_unresolved():
    resolver = Resolver(node_universe=set(), import_aliases={})

    assert resolver.resolve("print").canonical_id is None
    assert resolver.resolve("TotallyMadeUp").canonical_id is None


def test_import_alias_only_applies_within_its_own_module():
    resolver = Resolver(
        node_universe=set(),
        import_aliases={
            "requests.adapters": {"MaxRetryError": "urllib3.exceptions.MaxRetryError"},
        },
    )

    result = resolver.resolve("MaxRetryError", module_context="requests.sessions")

    assert result.method == "unresolved"


def test_prefixed_surface_resolves_via_package_reexport():
    resolver = Resolver(
        node_universe={"requests.sessions.Session"},
        import_aliases={
            "requests": {"Session": "requests.sessions.Session"},
        },
    )

    result = resolver.resolve("requests.Session")

    assert result.canonical_id == "requests.sessions.Session"
    assert result.method == "reexport"


def test_prefixed_surface_with_no_matching_package_is_unresolved():
    resolver = Resolver(
        node_universe={"requests.sessions.Session"},
        import_aliases={"requests": {"Session": "requests.sessions.Session"}},
    )

    result = resolver.resolve("numpy.Array")

    assert result.method == "unresolved"


def test_normalized_match_finds_unique_symbol_by_simple_name():
    resolver = Resolver(
        node_universe={"urllib3.poolmanager.PoolManager"},
        import_aliases={},
    )

    result = resolver.resolve("PoolManager")

    assert result.canonical_id == "urllib3.poolmanager.PoolManager"
    assert result.method == "normalized"


def test_normalized_match_ignores_case_and_spacing():
    resolver = Resolver(
        node_universe={"requests.exceptions.ConnectionError"},
        import_aliases={},
    )

    result = resolver.resolve("Connection Error")

    assert result.canonical_id == "requests.exceptions.ConnectionError"
    assert result.method == "normalized"


def test_ambiguous_normalized_match_stays_unresolved_with_candidates_listed():
    resolver = Resolver(
        node_universe={
            "requests.models.Response",
            "urllib3.response.Response",
        },
        import_aliases={},
    )

    result = resolver.resolve("Response")

    assert result.canonical_id is None
    assert result.method == "unresolved"
    assert set(result.candidates) == {
        "requests.models.Response",
        "urllib3.response.Response",
    }


def test_exact_case_simple_name_beats_a_case_folded_collision():
    # A class and a module/function that share a name only when case is folded
    # (PoolManager vs poolmanager) must not be treated as ambiguous — Python
    # itself distinguishes them, so surface-name resolution should too.
    resolver = Resolver(
        node_universe={
            "urllib3.poolmanager.PoolManager",
            "urllib3.poolmanager",
        },
        import_aliases={},
    )

    result = resolver.resolve("PoolManager")

    assert result.canonical_id == "urllib3.poolmanager.PoolManager"
    assert result.method == "normalized"


def test_call_graph_disambiguates_when_one_candidate_is_never_called_by_another():
    # "verify" as a parameter of both Session.request (outer, public) and
    # HTTPAdapter.send (internal, receives the forwarded value) — the outer
    # one is never a callee of the inner one, so it wins.
    resolver = Resolver(
        node_universe={
            "requests.sessions.Session.request.verify",
            "requests.adapters.HTTPAdapter.send.verify",
        },
        import_aliases={},
        call_graph={("requests.sessions.Session.request", "requests.adapters.HTTPAdapter.send")},
    )

    result = resolver.resolve("verify")

    assert result.canonical_id == "requests.sessions.Session.request.verify"
    assert result.method == "call_graph"


def test_call_graph_leaves_it_unresolved_when_it_cannot_disambiguate():
    # Neither candidate calls the other — the call graph has no opinion, so
    # this must stay unresolved with candidates listed, not guess.
    resolver = Resolver(
        node_universe={
            "requests.models.Response",
            "urllib3.response.Response",
        },
        import_aliases={},
        call_graph={("some.other.function", "another.function")},
    )

    result = resolver.resolve("Response")

    assert result.canonical_id is None
    assert result.method == "unresolved"
    assert set(result.candidates) == {"requests.models.Response", "urllib3.response.Response"}


def test_no_call_graph_behaves_exactly_as_before():
    resolver = Resolver(
        node_universe={"requests.models.Response", "urllib3.response.Response"},
        import_aliases={},
    )

    result = resolver.resolve("Response")

    assert result.method == "unresolved"


def test_public_api_and_docstring_filter_picks_the_canonical_parameter():
    # "verify" exists on a public, documented method and on an internal
    # pass-through that isn't exported. The public documented one wins.
    resolver = Resolver(
        node_universe={
            "requests.sessions.Session.request.verify",
            "requests.adapters.HTTPAdapter.send.verify",
        },
        import_aliases={},
        public_ids={"requests.sessions.Session"},
        documented_params={
            "requests.sessions.Session.request": {"verify"},
            "requests.adapters.HTTPAdapter.send": {"verify"},
        },
    )

    result = resolver.resolve("verify")

    assert result.canonical_id == "requests.sessions.Session.request.verify"
    assert result.method == "public_api"


def test_public_but_undocumented_does_not_win():
    resolver = Resolver(
        node_universe={
            "requests.sessions.Session.helper.verify",
            "requests.adapters.HTTPAdapter.send.verify",
        },
        import_aliases={},
        public_ids={"requests.sessions.Session"},
        documented_params={},  # nothing documents it
    )

    result = resolver.resolve("verify")

    assert result.method == "unresolved"


def test_filter_leaving_several_survivors_stays_unresolved():
    # timeout is a real, different parameter in both libraries — genuinely
    # ambiguous, so refusing to pick is correct behaviour, not a failure.
    resolver = Resolver(
        node_universe={
            "requests.sessions.Session.request.timeout",
            "urllib3.connectionpool.HTTPConnectionPool.urlopen.timeout",
        },
        import_aliases={},
        public_ids={
            "requests.sessions.Session",
            "urllib3.connectionpool.HTTPConnectionPool",
        },
        documented_params={
            "requests.sessions.Session.request": {"timeout"},
            "urllib3.connectionpool.HTTPConnectionPool.urlopen": {"timeout"},
        },
    )

    result = resolver.resolve("timeout")

    assert result.method == "unresolved"
    assert len(result.candidates) == 2


def test_self_call_resolves_against_the_enclosing_class():
    # "self.send" inside Session.request unambiguously means
    # Session.send — self is the one receiver whose type is always known.
    # 17% of all call surfaces in the corpus are self.* and were being
    # dropped entirely.
    resolver = Resolver(
        node_universe={
            "requests.sessions.Session",
            "requests.sessions.Session.request",
            "requests.sessions.Session.send",
        },
        import_aliases={},
    )

    result = resolver.resolve(
        "self.send", enclosing_class="requests.sessions.Session"
    )

    assert result.canonical_id == "requests.sessions.Session.send"
    assert result.method == "self_reference"


def test_self_call_without_a_known_enclosing_class_stays_unresolved():
    resolver = Resolver(
        node_universe={"requests.sessions.Session.send"}, import_aliases={}
    )

    assert resolver.resolve("self.send").canonical_id is None


def test_self_call_to_a_method_the_class_does_not_have_stays_unresolved():
    resolver = Resolver(
        node_universe={"requests.sessions.Session"}, import_aliases={}
    )

    result = resolver.resolve(
        "self.nonexistent", enclosing_class="requests.sessions.Session"
    )

    assert result.canonical_id is None


def test_a_name_jedi_pins_to_one_class_resolves_to_it():
    from graphrag.ingest.resolve import resolve_code_name
    resolver = Resolver(node_universe={"requests.cookies.RequestsCookieJar.set"}, import_aliases={})

    r = resolve_code_name(resolver, "set", module_context="m", jedi_classes={"builtins.set"})

    assert (r.canonical_id, r.method) == ("builtins.set", "jedi")


def test_a_name_jedi_finds_two_classes_for_gets_no_edge_and_no_fallback():
    """`HTTPSConnection` is DummyConnection when ssl is missing. The import
    rung would answer HTTPSConnection; choosing is a guess, so nothing is
    written and the candidates are kept for the count (D98)."""
    from graphrag.ingest.resolve import resolve_code_name
    resolver = Resolver(
        node_universe=set(),
        import_aliases={"m": {"HTTPSConnection": "urllib3.connection.HTTPSConnection"}},
    )

    r = resolve_code_name(
        resolver, "HTTPSConnection", module_context="m",
        jedi_classes={"urllib3.connection.HTTPSConnection", "urllib3.connection.DummyConnection"},
    )

    assert r.canonical_id is None
    assert r.method == "ambiguous"
    assert set(r.candidates) == {"urllib3.connection.HTTPSConnection", "urllib3.connection.DummyConnection"}


def test_when_jedi_finds_nothing_an_import_still_proves_the_name():
    """pyodide is not installed, so jedi cannot see JsException. The module's
    own import statement still names it."""
    from graphrag.ingest.resolve import resolve_code_name
    resolver = Resolver(node_universe=set(), import_aliases={"m": {"JsException": "pyodide.ffi.JsException"}})

    r = resolve_code_name(resolver, "JsException", module_context="m", jedi_classes=set())

    assert r.canonical_id == "pyodide.ffi.JsException"


def test_when_jedi_finds_nothing_a_cross_package_guess_is_refused():
    """`TimeoutError` in a module that never imported urllib3's is Python's.
    The name-matching rung said urllib3.exceptions.TimeoutError (D98)."""
    from graphrag.ingest.resolve import resolve_code_name
    resolver = Resolver(node_universe={"urllib3.exceptions.TimeoutError"}, import_aliases={})

    assert resolver.resolve("TimeoutError", module_context="m").method == "normalized"
    r = resolve_code_name(resolver, "TimeoutError", module_context="m", jedi_classes=set())

    assert r.canonical_id is None
