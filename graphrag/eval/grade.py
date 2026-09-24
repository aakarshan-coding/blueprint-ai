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


def looks_like_refusal(answer: str) -> bool:
    """Whether an answer declines to answer, rather than asserting a fact."""
    lowered = answer.lower()
    return any(marker in lowered for marker in _REFUSAL_MARKERS)


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
    pattern = rf"(?<![A-Za-z0-9_]){re.escape(term.lower())}(?![A-Za-z0-9_])"
    return re.search(pattern, answer.lower()) is not None


_NEGATORS = (
    "does not", "doesn't", "is not", "isn't", "no information",
    "not provide", "cannot", "can't", "never raises", "unable to",
)


def _negated_near(answer: str, term: str) -> bool:
    """Whether the sentence containing `term` also negates it.

    Bare substring matching would credit "requests does not raise
    ConnectTimeout" for containing ConnectTimeout. Checking only the
    sentence the term appears in keeps an unrelated negation elsewhere in
    the answer from discarding a genuine mention.
    """
    lowered = answer.lower()
    for sentence in lowered.replace("\n", ". ").split("."):
        if term.lower() in sentence:
            if any(neg in sentence for neg in _NEGATORS):
                return True
    return False


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
    if must_contain_any:
        hit = [
            t for t in must_contain_any
            if _states(answer, t) and not _negated_near(answer, t)
        ]
        if hit:
            return Grade(verdict="correct", reason=f"States {hit[0]}.")
        return Grade(
            verdict="incorrect",
            reason=f"States none of: {', '.join(must_contain_any)}.",
        )

    required = must_contain or []
    missing = [
        t for t in required
        if not _states(answer, t) or _negated_near(answer, t)
    ]
    if not missing:
        return Grade(verdict="correct", reason="States every required term.")
    if len(missing) < len(required):
        return Grade(
            verdict="partial", reason=f"Missing: {', '.join(missing)}."
        )
    return Grade(verdict="incorrect", reason=f"Missing: {', '.join(missing)}.")
