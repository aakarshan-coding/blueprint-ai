from graphrag.ingest.node_ids import concept_id, doc_section_id, parameter_id, release_id


def test_parameter_id_is_scoped_under_its_function():
    assert parameter_id("requests.sessions.Session.request", "verify") == (
        "requests.sessions.Session.request.verify"
    )


def test_release_id_is_scoped_by_repo():
    assert release_id("requests", "2.34.2") == "requests:2.34.2"
    assert release_id("urllib3", "2.34.2") == "urllib3:2.34.2"


def test_doc_section_id_reuses_the_chunks_own_id():
    # A DocSection *is* the chunk representing it — no separate id scheme
    # needed, and it guarantees the id already exists in Postgres to cite.
    assert doc_section_id("e1f3ec2877dc6c73") == "doc:e1f3ec2877dc6c73"


def test_concept_id_normalizes_so_variants_collapse_together():
    assert concept_id("Connection Pooling") == concept_id("connection pooling")
    assert concept_id("Connection Pooling") == "concept:connectionpooling"


from graphrag.ingest.node_ids import parameter_concept_id


def test_parameter_concept_id_is_namespaced_and_normalized():
    assert parameter_concept_id("url") == "concept:parameter:url"
    assert parameter_concept_id("allow_redirects") == "concept:parameter:allow_redirects"


def test_parameter_concept_id_folds_case_so_variants_collapse():
    assert parameter_concept_id("URL") == parameter_concept_id("url")


def test_parameter_concept_is_distinct_from_a_plain_concept_of_the_same_name():
    # "the url parameter" and a prose concept happening to be called "url"
    # are different things and must not collide into one node.
    from graphrag.ingest.node_ids import concept_id

    assert parameter_concept_id("url") != concept_id("url")
