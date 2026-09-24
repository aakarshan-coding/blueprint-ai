from graphrag.answer.citations import extract_citations, validate_citations


def test_extract_citations_finds_every_bracketed_id():
    answer = "Session delegates to PoolManager [e1f3]. This also wraps errors [c1][c2]."
    assert extract_citations(answer) == ["e1f3", "c1", "c2"]


def test_extract_citations_returns_empty_list_when_none_present():
    assert extract_citations("No citations here.") == []


def test_validate_citations_passes_when_every_citation_was_retrieved():
    answer = "Session delegates to PoolManager [e1f3]."
    result = validate_citations(answer, retrieved_ids={"e1f3", "other"})

    assert result.valid is True
    assert result.invalid_citations == []


def test_validate_citations_fails_on_an_invented_chunk_id():
    answer = "This is a fact [made_up_id]."
    result = validate_citations(answer, retrieved_ids={"e1f3"})

    assert result.valid is False
    assert result.invalid_citations == ["made_up_id"]


def test_validate_citations_fails_when_answer_has_no_citations_at_all():
    # An answer with claims but zero citations is exactly the failure mode
    # the design's validation loop exists to catch.
    result = validate_citations("This is just a claim with nothing backing it.", retrieved_ids={"e1f3"})

    assert result.valid is False


def test_validate_citations_reports_every_invalid_citation_not_just_the_first():
    answer = "Claim one [bad1]. Claim two [bad2]."
    result = validate_citations(answer, retrieved_ids={"real"})

    assert result.invalid_citations == ["bad1", "bad2"]
