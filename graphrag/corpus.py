"""Which code the graph is over (D93).

Until D93 the two libraries were a dict in the ingestion script and their names
were typed into the prompts and the ontology. A corpus is now a small config:
one or more repositories, each with a package name, a source folder and
optionally docs and a changelog. The tool detects it from a repository's
layout; the measured two-library setup is a file under corpora/.

Resolution order for "the active corpus": set_corpus() in this process, then
data/corpus.json (written by `graphrag ingest`), then corpora/requests-urllib3.yaml
if its repositories are present. Nothing here raises at import time.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

ACTIVE_FILE = Path("data/corpus.json")
DEFAULT_FILE = Path("corpora/requests-urllib3.yaml")

CHANGELOG_NAMES = ("HISTORY.md", "HISTORY.rst", "CHANGES.rst", "CHANGES.md", "CHANGELOG.md", "CHANGELOG.rst")
SKIP_DIRS = {"tests", "test", "docs", "doc", "examples", "scripts", "build", "dist", ".git", "venv", ".venv"}


@dataclass(frozen=True)
class Repo:
    name: str          # the importable package name, e.g. "requests"
    root: str          # repository directory
    src: str = "."     # folder under root that holds the package
    docs: str | None = None
    changelog: str | None = None

    @property
    def root_path(self) -> Path:
        return Path(self.root)

    @property
    def src_root(self) -> Path:
        return self.root_path / self.src

    @property
    def docs_root(self) -> Path | None:
        return self.root_path / self.docs if self.docs else None

    @property
    def changelog_path(self) -> Path | None:
        return self.root_path / self.changelog if self.changelog else None


@dataclass(frozen=True)
class Corpus:
    name: str
    repos: tuple[Repo, ...]

    @property
    def packages(self) -> tuple[str, ...]:
        return tuple(r.name for r in self.repos)

    def to_dict(self) -> dict:
        return {"name": self.name, "repos": [asdict(r) for r in self.repos]}

    @classmethod
    def from_dict(cls, d: dict) -> "Corpus":
        repos = tuple(Repo(**{k: v for k, v in r.items() if k in Repo.__dataclass_fields__}) for r in d["repos"])
        return cls(name=d.get("name") or "+".join(r.name for r in repos), repos=repos)

    @classmethod
    def from_file(cls, path: str | Path) -> "Corpus":
        text = Path(path).read_text(encoding="utf-8")
        data = json.loads(text) if str(path).endswith(".json") else yaml.safe_load(text)
        return cls.from_dict(data)

    def exists(self) -> bool:
        return all(r.src_root.joinpath(r.name).is_dir() for r in self.repos)


def detect_repo(path: str | Path, *, name: str | None = None) -> Repo:
    """Describe a repository from its layout: the package is the directory with
    an __init__.py under src/ or the root; docs is docs/ if present; the
    changelog is the first conventional name found."""
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"{root} is not a directory")
    for src in ("src", "."):
        base = root / src
        if not base.is_dir():
            continue
        candidates = sorted(
            p.parent.name for p in base.glob("*/__init__.py") if p.parent.name not in SKIP_DIRS
        )
        if name and name in candidates:
            pkg = name
        elif candidates:
            if name:
                raise ValueError(f"package {name!r} not found under {base}; found {candidates}")
            if len(candidates) > 1:
                raise ValueError(f"several packages under {base}: {candidates}; pass --package")
            pkg = candidates[0]
        else:
            continue
        docs = "docs" if (root / "docs").is_dir() else ("doc" if (root / "doc").is_dir() else None)
        changelog = next((n for n in CHANGELOG_NAMES if (root / n).exists()), None)
        return Repo(name=pkg, root=str(root), src=src, docs=docs, changelog=changelog)
    raise ValueError(f"no Python package found in {root} (looked for <pkg>/__init__.py under src/ and the root)")


_active: Corpus | None = None


def set_corpus(corpus: Corpus | None) -> None:
    global _active
    _active = corpus


def get_corpus() -> Corpus:
    if _active is not None:
        return _active
    if ACTIVE_FILE.exists():
        return Corpus.from_file(ACTIVE_FILE)
    if DEFAULT_FILE.exists():
        return Corpus.from_file(DEFAULT_FILE)
    raise RuntimeError("no corpus: run `graphrag ingest <repo>` first, or pass --corpus")


def get_packages() -> tuple[str, ...]:
    """The package names of the active corpus, or () when none is configured.
    Prompts and validators call this; none of them may fail at import."""
    try:
        return get_corpus().packages
    except Exception:
        return ()
