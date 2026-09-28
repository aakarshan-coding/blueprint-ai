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


def test_a_capitalised_term_must_match_case_but_a_lowercase_one_need_not():
    """Case-folded, "Timeout" was satisfied by "a read timeout" and "Retry" by
    "retry logic" (D79). A class name must appear as a class name; a
    parameter or plain word ("verify", "yes") may appear in any case."""
    assert grade_mechanically("it raises connecttimeout", must_contain=["ConnectTimeout"]).verdict == "incorrect"
    assert grade_mechanically("it raises ConnectTimeout", must_contain=["ConnectTimeout"]).verdict == "correct"
    assert grade_mechanically("after a read timeout", must_contain=["Timeout"]).verdict == "incorrect"
    assert grade_mechanically("Pass VERIFY=False", must_contain=["verify"]).verdict == "correct"


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


def test_a_term_asserted_in_one_sentence_and_hedged_in_another_is_credited():
    """"It inherits from InvalidJSONError. The base of InvalidJSONError is not
    specified." asserts the term; the hedge is about something else (D79)."""
    g = grade_mechanically(
        "It inherits from InvalidJSONError [c1]. The base class of InvalidJSONError is not specified.",
        must_contain=["InvalidJSONError"],
    )
    assert g.verdict == "correct"


def test_a_count_is_not_satisfied_by_a_version_number():
    assert grade_mechanically("This applies to Python 3 only.", must_contain_any=["3", "three"]).verdict == "incorrect"
    assert grade_mechanically("There are 3 warning classes.", must_contain_any=["3", "three"]).verdict == "correct"


def test_required_terms_and_any_of_terms_are_both_enforced():
    """th-13: RequestException is required, and one of IOError / OSError."""
    kw = dict(must_contain=["RequestException"], must_contain_any=["IOError", "OSError"])
    assert grade_mechanically("RequestException inherits from OSError.", **kw).verdict == "correct"
    assert grade_mechanically("RequestException is the base.", **kw).verdict == "partial"
    assert grade_mechanically("It inherits from OSError.", **kw).verdict == "partial"
    assert grade_mechanically("No idea.", **kw).verdict == "incorrect"


def test_a_refusal_that_answers_anyway_is_not_a_refusal():
    """The baseline wrote "the context does not provide this. However, based
    on general knowledge: ..." plus a code block, and was credited with
    declining on three out-of-scope questions (D79)."""
    assert not looks_like_refusal(
        "The context does not provide an example. However, here is one:\n```python\nimport requests\n```")
    assert looks_like_refusal("The context does not provide information about Django integration.")


def test_no_benchmark_question_can_be_passed_by_echoing_itself():
    """Seven questions could be answered correctly by repeating the question
    (D79). The term lists must discriminate; this pins that for every
    mechanically graded question, present and future."""
    import yaml
    from pathlib import Path
    questions = yaml.safe_load(Path("graphrag/eval/benchmark_questions.yaml").read_text(encoding="utf-8"))
    questions = questions["questions"] if isinstance(questions, dict) else questions
    echoable = [
        q["id"] for q in questions
        if (q.get("must_contain") or q.get("must_contain_any"))
        and grade_mechanically(q["question"], must_contain=q.get("must_contain"),
                               must_contain_any=q.get("must_contain_any")).verdict == "correct"
    ]
    assert echoable == []
