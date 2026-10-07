"""The corpus config (D93): detected from a repository's layout, or loaded
from a file; every prompt and validator reads package names from it."""

import json
import textwrap

from graphrag.corpus import Corpus, Repo, detect_repo, get_packages, set_corpus


def _repo(tmp_path, name, *, src=True, docs=True, changelog="CHANGELOG.md"):
    root = tmp_path / f"{name}-repo"
    pkg = root / ("src" if src else ".") / name
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "core.py").write_text("def f():\n    return 1\n")
    if docs:
        (root / "docs").mkdir()
        (root / "docs" / "guide.md").write_text("# Guide\n\nUse f().\n")
    if changelog:
        (root / changelog).write_text("# 1.0\n- first\n")
    (root / "tests").mkdir()
    (root / "tests" / "__init__.py").write_text("")
    return root


def test_detect_repo_reads_the_layout_src_docs_and_changelog(tmp_path):
    root = _repo(tmp_path, "widget")
    repo = detect_repo(root)
    assert (repo.name, repo.src, repo.docs, repo.changelog) == ("widget", "src", "docs", "CHANGELOG.md")
    assert repo.src_root == root / "src"


def test_detect_repo_handles_a_flat_layout_and_ignores_the_tests_package(tmp_path):
    root = _repo(tmp_path, "flat", src=False, docs=False, changelog=None)
    repo = detect_repo(root)
    assert (repo.name, repo.src, repo.docs, repo.changelog) == ("flat", ".", None, None)


def test_detect_repo_needs_a_package_name_when_several_are_present(tmp_path):
    root = _repo(tmp_path, "one")
    (root / "src" / "two").mkdir()
    (root / "src" / "two" / "__init__.py").write_text("")
    import pytest
    with pytest.raises(ValueError):
        detect_repo(root)
    assert detect_repo(root, name="two").name == "two"


def test_corpus_round_trips_through_a_file_and_exposes_packages(tmp_path):
    corpus = Corpus(name="demo", repos=(Repo(name="a", root="a-repo", src="src"), Repo(name="b", root="b-repo")))
    path = tmp_path / "c.json"
    path.write_text(json.dumps(corpus.to_dict()))
    assert Corpus.from_file(path) == corpus
    assert corpus.packages == ("a", "b")


def test_prompts_and_validation_follow_the_active_corpus(tmp_path):
    """The package enum, the router prompt and the mention prompt all read
    the active corpus; nothing names requests or urllib3 (D93)."""
    from graphrag.retrieval.cypher_templates import run_template
    from graphrag.retrieval.graph_query import mention_prompt
    from graphrag.retrieval.router import system_prompt
    set_corpus(Corpus(name="demo", repos=(Repo(name="widget", root=str(tmp_path), src="src"),)))
    try:
        assert get_packages() == ("widget",)
        assert "widget" in system_prompt(get_packages()) and "requests" not in system_prompt(get_packages())
        assert "widget" in mention_prompt(get_packages())

        class S:
            calls = []
            def run(self, q, **p):
                self.calls.append((q, p))
                class R:
                    def data(self_inner): return []
                return R()
        run_template(S(), "T11_EDGES_IN_PACKAGE", {"package": "widget", "relationship": "RAISES"}, known_entity_ids=set())
        import pytest
        with pytest.raises(ValueError):
            run_template(S(), "T11_EDGES_IN_PACKAGE", {"package": "requests", "relationship": "RAISES"}, known_entity_ids=set())
    finally:
        set_corpus(None)


def test_ingestion_walks_a_detected_repo_end_to_end_without_stores(tmp_path):
    from graphrag.ingest.run_ingestion import build_public_and_doc_index, chunk_corpus, run_ast_pass
    root = _repo(tmp_path, "widget")
    (root / "src" / "widget" / "core.py").write_text(textwrap.dedent("""
        class Base:
            def run(self):
                raise NotImplementedError()

        class Impl(Base):
            def run(self):
                return helper(1)

        def helper(x):
            return x
    """))
    corpus = Corpus(name="demo", repos=(detect_repo(root),))
    chunks = chunk_corpus(corpus)
    assert {c.kind for c in chunks} >= {"code", "doc"}
    assert any(c.section == "Guide" for c in chunks), "the Markdown doc was chunked"
    nodes, aliases, edges, types = run_ast_pass(corpus)
    assert "widget.core.Impl" in nodes and types["widget.core.Impl"] == "Class"
    assert any(e.source_id == "widget.core.Impl" and "Base" in e.target_surfaces for e in edges["inherits_from"])
    assert any(e.source_id == "widget.core.Impl.run" and e.target_id == "widget.core.helper" for e in edges["jedi_calls"])
    public, documented = build_public_and_doc_index(aliases, corpus)
    assert isinstance(public, set)
