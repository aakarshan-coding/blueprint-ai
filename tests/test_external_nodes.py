from graphrag.ingest.load_graph import infer_external_type


def test_inherits_from_target_must_be_a_class():
    assert infer_external_type("INHERITS_FROM") == "Class"


def test_raises_target_must_be_an_exception():
    assert infer_external_type("RAISES") == "Exception"


def test_calls_target_is_typed_function():
    # Known weakness: calling a class constructor looks identical to calling
    # a function, so this can be wrong. Recorded via type_source="inferred"
    # at write time rather than pretended away.
    assert infer_external_type("CALLS") == "Function"


def test_a_relationship_with_no_clear_implication_gets_no_type():
    # Better to leave a node unlabeled than to assert a type the
    # relationship doesn't actually imply.
    assert infer_external_type("DELEGATES_TO") is None
    assert infer_external_type("CONTROLS") is None
