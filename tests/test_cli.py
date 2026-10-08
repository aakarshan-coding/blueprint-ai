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


def _stats(**over):
    base = {"nodes": 188, "edges": 477, "parser_edges": 450, "model_edges": 27,
            "by_type": [("DEFINED_IN", 179), ("HAS_PARAMETER", 141)],
            "graph_packages": ["itsdangerous"], "chunks": {"itsdangerous": 101}}
    base.update(over)
    return base


def test_status_reports_the_corpus_and_what_the_stores_hold():
    from graphrag.cli import format_status
    from graphrag.corpus import Corpus, Repo
    corpus = Corpus(name="itsdangerous", repos=(Repo(name="itsdangerous", root=".demo/itsdangerous", src="src", docs="docs"),))
    text = format_status(corpus, from_last_ingest=True, stats=_stats())
    assert "Corpus: itsdangerous" in text and "last `graphrag ingest`" in text
    assert "188 nodes, 477 relationships" in text
    assert "parser: 450" in text and "model: 27" in text
    assert "101 chunks" in text and "Warning" not in text


def test_status_warns_when_the_stores_hold_a_different_corpus():
    from graphrag.cli import format_status
    from graphrag.corpus import Corpus, Repo
    corpus = Corpus(name="demo", repos=(Repo(name="widget", root="w"),))
    text = format_status(corpus, from_last_ingest=True, stats=_stats())
    assert "Warning" in text and "itsdangerous" in text and "widget" in text


def test_status_says_so_when_the_stores_are_empty():
    from graphrag.cli import format_status
    from graphrag.corpus import Corpus, Repo
    corpus = Corpus(name="demo", repos=(Repo(name="widget", root="w"),))
    text = format_status(corpus, from_last_ingest=False, stats=_stats(
        nodes=0, edges=0, parser_edges=0, model_edges=0, by_type=[], graph_packages=[], chunks={}))
    assert "stores are empty" in text and "nothing has been ingested" in text


def test_status_explains_unreachable_databases(monkeypatch, capsys):
    import graphrag.settings
    from graphrag.corpus import Corpus, Repo, set_corpus

    def down():
        raise ConnectionRefusedError()

    monkeypatch.setattr(graphrag.settings, "neo4j_driver", down)
    set_corpus(Corpus(name="demo", repos=(Repo(name="widget", root="w"),)))
    try:
        assert main(["status"]) == 1
    finally:
        set_corpus(None)
    assert "docker compose up -d" in capsys.readouterr().out
