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
You check an answer against a known-correct reference answer.

First, list the reference's key facts: the separate things a correct answer
must state. Keep each one short. Do not list anything the reference does not
say.

Then, for each key fact, decide whether the answer states it. Different
wording counts. A fact stated alongside other, additional information counts.
List only the key facts the answer does NOT state as `missing`.

Set `contradicts` only if the answer asserts something the reference says is
false, or names the wrong entity in place of the right one.
Set `declines` only if the answer says the information is unavailable or that
it cannot answer, instead of answering.

Additional detail in the answer beyond the reference -- examples, context,
related facts, code -- is never a reason to mark anything missing. Grade only
on whether the key facts are stated. Do not reward or penalise fluency,
length, or confidence.
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
    # "The context does not contain information about the capital of France"
    # was scored as answering (D89): a refusal in words the list lacked.
    "does not contain",
    "doesn't contain",
    "not contain information",
    "no relevant information",
    "does not include",
)


_CREDIT_FOR = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}


class Grade(BaseModel):
    verdict: Verdict
    reason: str
    # How much of the reference the answer states, 0 to 1 (D87). Mechanical
    # grading sets it to the share of required terms present; the judge to
    # the share of key facts stated. Left unset, it follows the verdict:
    # correct 1, partial 0.5, incorrect 0.
    credit: float | None = None

    def model_post_init(self, __context) -> None:
        if self.credit is None:
            self.credit = _CREDIT_FOR[self.verdict]


class JudgeReport(BaseModel):
    """What the judge writes: the reference's key facts and which the
    answer left out. The verdict is derived from this, not chosen by the
    judge, so "partial" can only mean a key fact is missing (D87)."""

    key_facts: list[str]
    missing: list[str]
    contradicts: bool
    declines: bool
    reason: str


def grade_from_report(report: JudgeReport) -> Grade:
    total = max(len(report.key_facts), 1)
    missing = min(len(report.missing), total)
    if report.declines or report.contradicts:
        return Grade(verdict="incorrect", reason=report.reason, credit=0.0)
    if missing == 0:
        return Grade(verdict="correct", reason=report.reason, credit=1.0)
    if missing >= total:
        return Grade(verdict="incorrect", reason=f"Missing: {', '.join(report.missing)}.", credit=0.0)
    return Grade(
        verdict="partial", reason=f"Missing: {', '.join(report.missing)}.",
        credit=round(1 - missing / total, 3),
    )


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
        text_format=JudgeReport,
        temperature=0,
    )
    parsed = response.output_parsed
    if isinstance(parsed, Grade):  # a test double answering in the old shape
        return parsed
    return grade_from_report(parsed)


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

    # Credit is the share of required slots filled: each must_contain term
    # is a slot, and the any-list is one more slot (D87).
    slots = len(required) + (1 if must_contain_any else 0)
    filled = (len(required) - len(missing)) + (1 if (must_contain_any and not any_missing) else 0)
    credit = round(filled / slots, 3) if slots else 1.0
    if not missing and not any_missing:
        reason = f"States {any_hit[0]}." if any_hit and not required else "States every required term."
        return Grade(verdict="correct", reason=reason, credit=1.0)
    parts = list(missing)
    if any_missing:
        parts.append("one of " + "/".join(must_contain_any))
    stated_some = (len(missing) < len(required)) or (required and not missing) or bool(any_hit)
    if stated_some:
        return Grade(verdict="partial", reason=f"Missing: {', '.join(parts)}.", credit=credit)
    if must_contain_any and not required:
        return Grade(verdict="incorrect", reason=f"States none of: {', '.join(must_contain_any)}.", credit=0.0)
    return Grade(verdict="incorrect", reason=f"Missing: {', '.join(parts)}.", credit=0.0)
