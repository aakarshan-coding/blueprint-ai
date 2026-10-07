"""The graphrag command parses its two subcommands (D93)."""

import pytest

from graphrag.cli import main


def test_ingest_needs_a_repo_or_a_corpus_file():
    with pytest.raises(SystemExit):
        main(["ingest"])


def test_ask_requires_a_question():
    with pytest.raises(SystemExit):
        main(["ask"])
