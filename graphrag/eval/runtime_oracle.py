"""A runtime oracle for the graph: what the code actually did.

Every CALLS, RAISES and WRAPS_EXCEPTION edge in the graph is *inferred* --
by jedi, by the AST pass, or by a model reading prose. This runs the corpus's
own test suite under `sys.settrace` and records what really happened:

- `calls`   -- (caller, callee) for every call between two corpus functions
- `raises`  -- (function, exception) where an exception was first raised
- `wraps`   -- (raised, caught) whenever an exception was raised while another
               was being handled. That is Python's own implicit chaining rule
               (`__context__`), so it is exactly "raised in place of".

Then the graph's edges are scored against that, observed rather than
spot-checked. code-graph-rag measures its call edges the same way (D67); the
wrap oracle is this project's own, because no surveyed tool models wrapping.

Two honest limits. Recall is measured against what the tests *exercise*: a
code path no test runs is invisible here, so an edge the oracle never saw may
still be right -- "observed precision" is a lower bound. And the ids are
mapped from code objects the way iter_symbols maps source: a nested function
belongs to its enclosing one, lambdas and comprehensions are skipped.

    python -m graphrag.eval.runtime_oracle            # run requests' tests, compare, write JSON
    python -m graphrag.eval.runtime_oracle --no-run   # compare a saved trace only
"""

import argparse
import importlib
import json
import re
import sys
import threading
from contextlib import contextmanager
from pathlib import Path

REPO_ROOTS = {
    "requests": Path("requests_repo/src"),
    "urllib3": Path("urllib3_repo/src"),
}
TRACE_PATH = Path("results/oracle/trace.json")
REPORT_PATH = Path("results/oracle/report.json")


def canonical_id(code, *, roots: dict[str, Path]) -> str | None:
    """The graph id for a code object, or None if it is not corpus code.

    Module from the file path relative to the package's source root; symbol
    from `co_qualname` with any `<locals>` segments folded away, because the
    graph never has a node for a nested function -- its calls and raises are
    the enclosing function's (iter_symbols, D10). Lambdas, comprehensions and
    module bodies have no node and return None.
    """
    path = Path(code.co_filename)
    for root in roots.values():
        try:
            rel = path.resolve().relative_to(root.resolve())
        except (ValueError, OSError):
            continue
        parts = list(rel.with_suffix("").parts)
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        module = ".".join(parts)
        qualname = code.co_qualname
        if "<" in qualname:
            # "<module>", "<lambda>", "<listcomp>", or a nested def:
            # "iter_content.<locals>.generate" -> "iter_content".
            if ".<locals>." in qualname:
                qualname = qualname.split(".<locals>.")[0]
            else:
                return None
        return f"{module}.{qualname}"
    return None


def exception_id(exc_type: type) -> str:
    """The graph id for an exception class: its defining module and name.
    Builtins become `builtins.<Name>`, matching the resolver's `builtin` rung."""
    return f"{exc_type.__module__}.{exc_type.__name__}"


class Oracle:
    def __init__(self, *, roots: dict[str, Path] | None = None):
        self.roots = roots or REPO_ROOTS
        self.calls: set[tuple[str, str]] = set()
        self.raises: set[tuple[str, str]] = set()
        self.wraps: set[tuple[str, str]] = set()
        # An exception propagates through every frame it unwinds, and each
        # frame gets an "exception" event. Only the first sighting is the
        # raise site; later ones are the same exception passing through.
        self._seen: set[int] = set()
        # Each exception type's ancestors, itself first, so a static edge to a
        # superclass can be matched by the runtime subclass that occurred.
        self.hierarchy: dict[str, list[str]] = {}
        # Callees that run without call syntax (properties); filled after the
        # run by find_implicit, excluded from the oracle by compare_pairs.
        self.implicit: set[str] = set()
        # Every call in the process hits the tracer -- including the local
        # httpbin server's threads -- and canonical_id resolves a path per
        # lookup. Uncached, the first run managed 25 tests in an hour. Code
        # objects are long-lived and hashable, so the answer is memoised.
        self._ids: dict = {}

    def _id(self, code) -> str | None:
        try:
            return self._ids[code]
        except KeyError:
            result = canonical_id(code, roots=self.roots)
            self._ids[code] = result
            return result

    def trace(self, frame, event, arg):
        if event == "call":
            callee = self._id(frame.f_code)
            if callee is None:
                return None
            caller_frame = frame.f_back
            caller = self._id(caller_frame.f_code) if caller_frame else None
            if caller and caller != callee:
                self.calls.add((caller, callee))
            frame.f_trace_lines = False
            return self.trace
        if event == "exception":
            exc_type, value, _tb = arg
            key = id(value)
            if key in self._seen:
                return self.trace
            self._seen.add(key)
            where = self._id(frame.f_code)
            raised = exception_id(exc_type)
            self._record_hierarchy(exc_type)
            if where:
                self.raises.add((where, raised))
            context = getattr(value, "__context__", None)
            if context is not None and context is not value:
                self._record_hierarchy(type(context))
                self.wraps.add((raised, exception_id(type(context))))
        return self.trace

    def _record_hierarchy(self, exc_type: type) -> None:
        key = exception_id(exc_type)
        if key not in self.hierarchy:
            self.hierarchy[key] = [exception_id(t) for t in exc_type.__mro__ if t is not object]

    @contextmanager
    def tracing(self):
        previous = sys.gettrace()
        sys.settrace(self.trace)
        threading.settrace(self.trace)
        try:
            yield self
        finally:
            sys.settrace(previous)
            threading.settrace(previous)

    def to_json(self) -> dict:
        return {
            "calls": sorted(self.calls),
            "raises": sorted(self.raises),
            "wraps": sorted(self.wraps),
            "implicit": sorted(self.implicit),
            "hierarchy": self.hierarchy,
        }


_IMPLICIT_DUNDER = re.compile(r"\.__(?!init__$)\w+__$")


def _fold_constructor(pair: tuple[str, str]) -> tuple[str, str]:
    """`Retry(...)` is seen at runtime as a call to `Retry.__init__`; jedi's
    edge points at the class. One fact, two spellings."""
    a, b = pair
    return (a, b[: -len(".__init__")]) if b.endswith(".__init__") else pair


def compare_pairs(
    *,
    observed: set[tuple[str, str]],
    graph: set[tuple[str, str]],
    implicit: frozenset[str] | set[str] = frozenset(),
    hierarchy: dict[str, list[str]] | None = None,
) -> dict:
    """Score a set of graph edges against the observed pairs.

    recall: of what happened, how much the graph has.
    observed_precision: of what the graph has, how much was seen to happen --
    a lower bound, since an edge on an untested path can still be true.

    Three rules make the comparison fair without making it lenient, each a
    statement about Python rather than a fudge: a call to a class is a call
    to its `__init__`; dunders other than `__init__` and the ids in `implicit`
    (properties) run without call syntax, so no static tool is ever asked
    about them and they are excluded up front; and a static edge to a
    superclass is confirmed by a runtime subclass -- `hierarchy` maps an id
    to its ancestors, itself first -- but never the other way round.
    """
    hierarchy = hierarchy or {}

    def ancestors(x: str) -> list[str]:
        return hierarchy.get(x, [x])

    seen = {
        _fold_constructor(p) for p in observed
        if p[1] not in implicit and not _IMPLICIT_DUNDER.search(p[1])
    }
    generalised: dict[tuple[str, str], set[tuple[str, str]]] = {
        p: {(a, b) for a in ancestors(p[0]) for b in ancestors(p[1])} for p in seen
    }
    confirmed_seen = {p for p, forms in generalised.items() if forms & graph}
    confirmed_graph = set().union(*(forms & graph for forms in generalised.values())) if generalised else set()

    return {
        "observed": len(seen),
        "in_graph": len(graph),
        "confirmed": len(confirmed_seen),
        "recall": round(len(confirmed_seen) / len(seen), 3) if seen else None,
        "observed_precision": round(len(confirmed_graph) / len(graph), 3) if graph else None,
        "missing_from_graph": sorted(seen - confirmed_seen),
        "unconfirmed_in_graph": sorted(graph - confirmed_graph),
    }


def _in_corpus(pair: tuple[str, str]) -> bool:
    return all(p.split(".")[0] in ("requests", "urllib3", "builtins") for p in pair)


def find_implicit(ids: set[str]) -> set[str]:
    """The ids among `ids` that are properties: attribute reads that run a
    function with no call syntax, so no static call extractor is ever asked
    about them. Resolved by importing the module and walking the attribute
    chain on the class -- only meaningful inside run_suite, where the corpus
    trees are on sys.path."""
    implicit: set[str] = set()
    for canonical in ids:
        parts = canonical.split(".")
        # Longest importable prefix is the module; the rest is Class.attr.
        for split in range(len(parts) - 1, 0, -1):
            try:
                obj = importlib.import_module(".".join(parts[:split]))
            except Exception:
                continue
            try:
                for attr in parts[split:]:
                    obj = getattr(obj, attr)
            except AttributeError:
                break
            if isinstance(obj, property) or type(obj).__name__ == "cached_property":
                implicit.add(canonical)
            break
    return implicit


def run_suite(pytest_args: list[str]) -> Oracle:
    """Run pytest under the tracer with the corpus source trees first on
    sys.path, so `import requests` is the tree the graph was built from and
    not a copy in site-packages."""
    import pytest
    import types

    for root in REPO_ROOTS.values():
        sys.path.insert(0, str(root.resolve()))
    for name in [m for m in sys.modules if m.split(".")[0] in ("requests", "urllib3")]:
        del sys.modules[name]
    # urllib3's checkout imports `urllib3._version`, a file its build step
    # generates and the clone does not contain. Stub it so the corpus tree
    # imports; the version string plays no part in anything traced.
    if not (REPO_ROOTS["urllib3"] / "urllib3" / "_version.py").exists():
        stub = types.ModuleType("urllib3._version")
        stub.__version__ = "2.2.0"
        stub.__version_tuple__ = (2, 2, 0)
        sys.modules["urllib3._version"] = stub

    oracle = Oracle()
    with oracle.tracing():
        pytest.main(pytest_args)
    oracle.implicit = find_implicit({callee for _caller, callee in oracle.calls})
    return oracle


def graph_edges(session, relationship: str) -> dict[str, set[tuple[str, str]]]:
    """Edges of one type from the live graph, split by their `source` tag."""
    rows = session.run(
        f"MATCH (a)-[r:{relationship}]->(b) RETURN a.id AS a, b.id AS b, coalesce(r.source, 'llm') AS src"
    ).data()
    out: dict[str, set] = {}
    for row in rows:
        out.setdefault(row["src"], set()).add((row["a"], row["b"]))
    out["all"] = set().union(*out.values()) if out else set()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-run", action="store_true", help="Compare the saved trace only.")
    parser.add_argument("pytest_args", nargs="*", default=None)
    args = parser.parse_args()

    if args.no_run:
        trace = json.loads(TRACE_PATH.read_text(encoding="utf-8"))
        observed = {k: {tuple(p) for p in trace[k]} for k in ("calls", "raises", "wraps")}
        implicit = set(trace.get("implicit", []))
        hierarchy = trace.get("hierarchy", {})
    else:
        pytest_args = args.pytest_args or ["requests_repo/tests", "-q", "-p", "no:cacheprovider", "-p", "no:cov"]
        oracle = run_suite(pytest_args)
        observed = {"calls": oracle.calls, "raises": oracle.raises, "wraps": oracle.wraps}
        implicit, hierarchy = oracle.implicit, oracle.hierarchy
        TRACE_PATH.write_text(json.dumps(oracle.to_json(), indent=1), encoding="utf-8")
        print(f"\ntrace written to {TRACE_PATH}: {len(oracle.calls)} calls, "
              f"{len(oracle.raises)} raises, {len(oracle.wraps)} wraps, "
              f"{len(oracle.implicit)} implicit callees")

    from neo4j import GraphDatabase

    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "graphragpassword"))
    report: dict = {}
    with driver.session() as s:
        for key, rel in (("calls", "CALLS"), ("raises", "RAISES"), ("wraps", "WRAPS_EXCEPTION")):
            seen = {p for p in observed[key] if _in_corpus(p)}
            by_source = graph_edges(s, rel)
            report[key] = {
                src: compare_pairs(
                    observed=seen, graph=edges,
                    implicit=implicit if key == "calls" else frozenset(),
                    hierarchy=hierarchy if key in ("raises", "wraps") else None,
                )
                for src, edges in by_source.items()
            }
    driver.close()
    REPORT_PATH.write_text(json.dumps(report, indent=1), encoding="utf-8")

    for key in ("calls", "raises", "wraps"):
        print(f"\n{key.upper()}")
        for src, r in report[key].items():
            print(f"  {src:<8} observed {r['observed']:>5}  graph {r['in_graph']:>5}  confirmed {r['confirmed']:>5}  "
                  f"recall {r['recall']}  observed-precision {r['observed_precision']}")
    print(f"\nreport written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
