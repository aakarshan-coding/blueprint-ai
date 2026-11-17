"""The graphrag command parses its two subcommands (D93)."""

import json

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


@pytest.fixture
def tiny_repo(tmp_path, monkeypatch):
    """A one-package repo, ACTIVE_FILE redirected, and chunk/parse stubbed out."""
    import graphrag.cli
    import graphrag.ingest.run_ingestion as run_ingestion

    (tmp_path / "repo" / "widget").mkdir(parents=True)
    (tmp_path / "repo" / "widget" / "__init__.py").write_text("", encoding="utf-8")
    active = tmp_path / "data" / "corpus.json"
    monkeypatch.setattr(graphrag.cli, "ACTIVE_FILE", active)
    monkeypatch.setattr(run_ingestion, "chunk_corpus", lambda corpus: [])
    monkeypatch.setattr(run_ingestion, "run_ast_pass", lambda corpus: ({}, {}, [], {}))
    return tmp_path / "repo", active


def test_a_dry_run_does_not_switch_the_active_corpus(tiny_repo):
    repo, active = tiny_repo
    assert main(["ingest", str(repo), "--dry-run"]) == 0
    assert not active.exists()


def test_a_failed_store_write_does_not_switch_the_active_corpus(tiny_repo, monkeypatch):
    import graphrag.settings

    repo, active = tiny_repo

    def down():
        raise ConnectionRefusedError()

    monkeypatch.setattr(graphrag.settings, "neo4j_driver", down)
    with pytest.raises(ConnectionRefusedError):
        main(["ingest", str(repo)])
    assert not active.exists()


def test_a_completed_ingest_records_the_corpus(tiny_repo, monkeypatch):
    from unittest.mock import MagicMock

    import graphrag.ingest.embed
    import graphrag.ingest.run_ingestion as run_ingestion
    import graphrag.settings

    repo, active = tiny_repo
    monkeypatch.setattr(graphrag.settings, "neo4j_driver", MagicMock)
    monkeypatch.setattr(graphrag.settings, "postgres_conn", MagicMock)
    monkeypatch.setattr(graphrag.ingest.embed, "load_model", lambda: None)
    monkeypatch.setattr(run_ingestion, "run_full_ingestion", lambda *a, **k: {})
    monkeypatch.setattr(run_ingestion, "write_vectors", lambda **k: 0)
    assert main(["ingest", str(repo)]) == 0
    assert json.loads(active.read_text(encoding="utf-8"))["repos"][0]["name"] == "widget"


@pytest.fixture
def stub_stores(monkeypatch):
    """The stores, the embedder and the model client replaced; returns the driver mock."""
    from unittest.mock import MagicMock

    import openai

    import graphrag.ingest.embed
    import graphrag.ingest.run_ingestion as run_ingestion
    import graphrag.settings

    driver = MagicMock()
    monkeypatch.setattr(graphrag.settings, "neo4j_driver", driver)
    monkeypatch.setattr(graphrag.settings, "postgres_conn", MagicMock)
    monkeypatch.setattr(openai, "OpenAI", MagicMock)
    monkeypatch.setattr(graphrag.ingest.embed, "load_model", lambda: None)
    monkeypatch.setattr(run_ingestion, "run_full_ingestion", lambda *a, **k: {})
    monkeypatch.setattr(run_ingestion, "write_vectors", lambda **k: 0)
    return driver


def test_with_llm_refuses_to_overwrite_an_existing_model_edge_log(tiny_repo, stub_stores, tmp_path, capsys):
    repo, active = tiny_repo
    log = tmp_path / "llm_edges_log.jsonl"
    log.write_text('{"measured": true}\n', encoding="utf-8")
    assert main(["ingest", str(repo), "--with-llm", "--llm-log", str(log)]) == 2
    assert log.read_text(encoding="utf-8") == '{"measured": true}\n'
    assert not stub_stores.called
    assert not active.exists()
    assert "--overwrite-llm-log" in capsys.readouterr().err


def test_overwrite_llm_log_lets_the_model_pass_replace_the_log(tiny_repo, stub_stores, tmp_path):
    repo, active = tiny_repo
    log = tmp_path / "llm_edges_log.jsonl"
    log.write_text('{"measured": true}\n', encoding="utf-8")
    args = ["ingest", str(repo), "--with-llm", "--llm-log", str(log), "--overwrite-llm-log"]
    assert main(args) == 0
    assert stub_stores.called
    assert active.exists()


def test_an_existing_model_edge_log_does_not_matter_without_with_llm(tiny_repo, stub_stores, tmp_path):
    repo, active = tiny_repo
    log = tmp_path / "llm_edges_log.jsonl"
    log.write_text('{"measured": true}\n', encoding="utf-8")
    assert main(["ingest", str(repo), "--llm-log", str(log)]) == 0
    assert stub_stores.called
    assert log.read_text(encoding="utf-8") == '{"measured": true}\n'
