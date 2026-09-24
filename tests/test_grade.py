from graphrag.eval.grade import grade_answer, grade_out_of_scope, looks_like_refusal


def test_refusal_is_recognised():
    assert looks_like_refusal("This question is outside the scope of the corpus.")
    assert looks_like_refusal("The context does not provide that information.")


def test_a_real_answer_is_not_mistaken_for_a_refusal():
    assert not looks_like_refusal("requests raises ConnectionError [c1].")


def test_out_of_scope_graded_correct_only_when_it_refuses():
    assert grade_out_of_scope("This is outside the scope.").verdict == "correct"


def test_out_of_scope_graded_incorrect_when_it_answers_anyway():
    graded = grade_out_of_scope("You can use requests with Django by calling get() [c1].")
    assert graded.verdict == "incorrect"


class _FakeResponse:
    def __init__(self, parsed):
        self.output_parsed = parsed


class _FakeResponses:
    def __init__(self, parsed):
        self._parsed = parsed
        self.last_call = None

    def parse(self, **kwargs):
        self.last_call = kwargs
        return _FakeResponse(self._parsed)


class _FakeClient:
    def __init__(self, parsed):
        self.responses = _FakeResponses(parsed)


def test_grade_answer_shows_the_judge_question_reference_and_answer():
    from graphrag.eval.grade import Grade

    client = _FakeClient(Grade(verdict="correct", reason="matches"))
    grade_answer("the question", "the reference", "the answer", client=client)

    sent = client.responses.last_call["input"]
    assert "the question" in sent
    assert "the reference" in sent
    assert "the answer" in sent


def test_grade_answer_does_not_tell_the_judge_which_system_produced_it():
    from graphrag.eval.grade import Grade

    client = _FakeClient(Grade(verdict="correct", reason="matches"))
    grade_answer("q", "ref", "ans", client=client)

    sent = client.responses.last_call["input"].lower()
    assert "baseline" not in sent
    assert "hybrid" not in sent
    assert "graph" not in sent


def test_judge_is_deterministic():
    from graphrag.eval.grade import Grade

    client = _FakeClient(Grade(verdict="correct", reason="matches"))
    grade_answer("q", "ref", "ans", client=client)

    assert client.responses.last_call["temperature"] == 0


from graphrag.eval.grade import grade_mechanically


def test_all_required_terms_present_is_correct():
    g = grade_mechanically(
        "requests raises ConnectTimeout [c1].", must_contain=["ConnectTimeout"]
    )
    assert g.verdict == "correct"


def test_a_missing_required_term_is_incorrect():
    g = grade_mechanically(
        "requests raises something else [c1].", must_contain=["ConnectTimeout"]
    )
    assert g.verdict == "incorrect"
    assert "ConnectTimeout" in g.reason


def test_multiple_required_terms_all_needed():
    g = grade_mechanically(
        "Session.request calls Session.send [c1].",
        must_contain=["Session.request", "Session.send", "HTTPAdapter.send"],
    )
    assert g.verdict == "partial"  # some but not all


def test_any_of_accepts_one_acceptable_answer():
    # "which exception wraps MaxRetryError" has several valid answers
    # depending on the underlying reason — any one of them is correct.
    g = grade_mechanically(
        "It raises ProxyError [c1].",
        must_contain_any=["ConnectionError", "ConnectTimeout", "ProxyError"],
    )
    assert g.verdict == "correct"


def test_term_matching_ignores_case():
    g = grade_mechanically("it raises connecttimeout", must_contain=["ConnectTimeout"])
    assert g.verdict == "correct"


def test_a_negated_claim_is_not_credited():
    # "does not raise ConnectTimeout" contains the term but asserts the
    # opposite — bare substring matching would wrongly mark this correct.
    g = grade_mechanically(
        "requests does not raise ConnectTimeout here.",
        must_contain=["ConnectTimeout"],
    )
    assert g.verdict == "incorrect"


def test_a_term_inside_a_longer_different_name_is_not_credited():
    # "ReadTimeout" and "ReadTimeoutError" are DIFFERENT exceptions. An answer
    # that only echoes the question's ReadTimeoutError must not be credited
    # for stating ReadTimeout.
    g = grade_mechanically(
        "urllib3's ReadTimeoutError maps to ConnectionError.",
        must_contain=["ReadTimeout"],
    )
    assert g.verdict == "incorrect"


def test_retryerror_is_not_credited_by_maxretryerror():
    g = grade_mechanically(
        "This happens when MaxRetryError is raised.",
        must_contain_any=["RetryError"],
    )
    assert g.verdict == "incorrect"


def test_a_standalone_term_still_matches():
    g = grade_mechanically(
        "requests raises ReadTimeout here.", must_contain=["ReadTimeout"]
    )
    assert g.verdict == "correct"


def test_punctuation_around_a_term_still_counts_as_standalone():
    g = grade_mechanically(
        "It raises `ReadTimeout`, as documented.", must_contain=["ReadTimeout"]
    )
    assert g.verdict == "correct"


def test_a_dotted_path_still_matches_its_leaf_term():
    g = grade_mechanically(
        "It raises requests.exceptions.ReadTimeout.", must_contain=["ReadTimeout"]
    )
    assert g.verdict == "correct"
