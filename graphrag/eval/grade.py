"""Grading for the benchmark (design §10).

Two graders, because the questions are two different kinds:

- out-of-scope questions are graded mechanically — did the system refuse?
  That's a yes/no fact about the answer, not a judgement call, so no model
  is involved and no model bias can enter.
- everything else is graded by an LLM judge against the hand-written expected
  answer, with the rubric below published rather than hidden. The judge never
  sees which system produced an answer, so it cannot favour one.
"""

import re
from typing import Literal

from pydantic import BaseModel

MODEL = "gpt-4o-mini"

Verdict = Literal["correct", "partial", "incorrect"]

JUDGE_RUBRIC = """\
You grade an answer against a known-correct reference answer.

correct   - the answer states the key fact(s) in the reference. Extra detail
            is fine. Different wording is fine.
partial   - the answer gets some of the reference right but misses a required
            part of it, or hedges so heavily the fact isn't actually asserted.
incorrect - the answer contradicts the reference, states the wrong entity, or
            says the information isn't available when the reference shows it
            is.

Grade only on factual agreement with the reference. Do not reward fluency,
length, or confidence. An answer that admits it doesn't know is "incorrect"
for a question that has a real answer — honest, but still not the answer.
"""

_REFUSAL_MARKERS = (
    "outside the scope",
    "not in the corpus",
    "does not provide",
    "no supporting information",
    "cannot answer",
    "can't answer",
    "not covered",
    "unable to answer",
    "i don't know",
    "no information",
)


class Grade(BaseModel):
    verdict: Verdict
    reason: str


# An answer that opens with a refusal marker and then answers anyway
# ("The context does not provide this. However, based on general knowledge:
# ...") is not a refusal. The baseline did this on three out-of-scope
# questions and was credited for declining (D79).
_ANSWERED_ANYWAY = ("however", "```", "general knowledge", "that said", "here is how", "here's how")


def looks_like_refusal(answer: str) -> bool:
    """Whether an answer declines to answer, rather than asserting a fact."""
    lowered = answer.lower()
    if not any(marker in lowered for marker in _REFUSAL_MARKERS):
        return False
    return not any(tail in lowered for tail in _ANSWERED_ANYWAY)


def grade_out_of_scope(answer: str) -> Grade:
    """Mechanical grading: an out-of-scope question is answered correctly
    only by declining. No model involved — this is checkable, not a judgement."""
    if looks_like_refusal(answer):
        return Grade(verdict="correct", reason="Declined to answer, as required.")
    return Grade(
        verdict="incorrect",
        reason="Answered a question that is outside the corpus instead of refusing.",
    )


def grade_answer(question: str, expected: str, answer: str, *, client, model: str = MODEL) -> Grade:
    """LLM-judge an answer against the hand-written reference."""
    response = client.responses.parse(
        model=model,
        instructions=JUDGE_RUBRIC,
        input=(
            f"Question: {question}\n\n"
            f"Reference answer: {expected}\n\n"
            f"Answer to grade: {answer}"
        ),
        text_format=Grade,
        temperature=0,
    )
    return response.output_parsed


def _states(answer: str, term: str) -> bool:
    """Whether the answer actually states this term, standing on its own.

    Plain substring matching credits an answer that merely echoes the
    question: "ReadTimeout" appears inside "ReadTimeoutError", and
    "RetryError" inside "MaxRetryError" — different exceptions in both cases.
    A term must be bounded by something other than a name character, so a
    dotted path (`requests.exceptions.ReadTimeout`) and punctuation
    (`` `ReadTimeout`, ``) still count while a longer name does not.
    """
    if term[:1].isupper():
        # A class name must appear as a class name. Case-folded, "Timeout"
        # was satisfied by "a read timeout", "Response" by
        # "urllib3.response", "Retry" by "retry logic" (D79).
        pattern = rf"(?<![A-Za-z0-9_]){re.escape(term)}(?![A-Za-z0-9_])"
        return re.search(pattern, answer) is not None
    if term.isdigit():
        # A count, not a version: "Python 3" must not satisfy "3".
        pattern = rf"(?<!python )(?<![A-Za-z0-9_./]){re.escape(term)}(?![A-Za-z0-9_./])"
        return re.search(pattern, answer.lower()) is not None
    pattern = rf"(?<![A-Za-z0-9_]){re.escape(term.lower())}(?![A-Za-z0-9_])"
    return re.search(pattern, answer.lower()) is not None


_NEGATORS = (
    "does not", "doesn't", "is not", "isn't", "no information",
    "not provide", "cannot", "can't", "never raises", "unable to",
)


def _negated_near(answer: str, term: str) -> bool:
    """Whether every sentence containing `term` also negates it.

    Bare substring matching would credit "requests does not raise
    ConnectTimeout" for containing ConnectTimeout. But an answer often
    asserts a fact and then hedges about something else in a later
    sentence that repeats the name: "it inherits from InvalidJSONError.
    The base class of InvalidJSONError is not specified." The first rule
    ("any sentence negated") discounted the assertion on four questions,
    for both systems (D79). A term counts if any sentence asserts it.
    """
    lowered = answer.lower()
    sentences = [s for s in lowered.replace("\n", ". ").split(".") if term.lower() in s]
    if not sentences:
        return False
    return all(any(neg in s for neg in _NEGATORS) for s in sentences)


def grade_mechanically(
    answer: str,
    *,
    must_contain: list[str] | None = None,
    must_contain_any: list[str] | None = None,
) -> Grade:
    """Grade by checking for required terms — no model, no variance.

    Used wherever a question's answer is checkable rather than a matter of
    judgement, which removes the LLM judge's run-to-run variance (D45: ~5% of
    verdicts flipped between identical runs) from those questions entirely.

    `must_contain` requires every term; `must_contain_any` requires at least
    one, for questions with several equally valid answers.
    """
    # Both lists apply when both are given: "RequestException, which
    # inherits from IOError (or OSError)" is must_contain plus
    # must_contain_any. The old code returned on the any-list alone and
    # never checked the required terms (D79, 3h-14).
    required = must_contain or []
    missing = [
        t for t in required
        if not _states(answer, t) or _negated_near(answer, t)
    ]
    any_hit = [
        t for t in (must_contain_any or [])
        if _states(answer, t) and not _negated_near(answer, t)
    ]
    any_missing = bool(must_contain_any) and not any_hit

    if not missing and not any_missing:
        reason = f"States {any_hit[0]}." if any_hit and not required else "States every required term."
        return Grade(verdict="correct", reason=reason)
    parts = list(missing)
    if any_missing:
        parts.append("one of " + "/".join(must_contain_any))
    stated_some = (len(missing) < len(required)) or (required and not missing) or bool(any_hit)
    if stated_some:
        return Grade(verdict="partial", reason=f"Missing: {', '.join(parts)}.")
    if must_contain_any and not required:
        return Grade(verdict="incorrect", reason=f"States none of: {', '.join(must_contain_any)}.")
    return Grade(verdict="incorrect", reason=f"Missing: {', '.join(parts)}.")
