"""The graphrag command parses its two subcommands (D93)."""

import pytest

from graphrag.cli import main


def test_ingest_needs_a_repo_or_a_corpus_file():
    with pytest.raises(SystemExit):
        main(["ingest"])


def test_ask_requires_a_question():
    with pytest.raises(SystemExit):
        main(["ask"])


def test_ask_accepts_compare_and_the_answer_blocks_are_labelled():
    from graphrag.cli import build_parser, format_answer
    args = build_parser().parse_args(["ask", "q", "--compare", "--show-plan"])
    assert args.compare and args.show_plan
    block = format_answer("text only (no graph)", {"answer": "A [c1].", "graph_facts_used": 0, "vector_passages_used": 5})
    assert block.splitlines() == ["== text only (no graph): 0 graph facts, 5 passages ==", "A [c1]."]
