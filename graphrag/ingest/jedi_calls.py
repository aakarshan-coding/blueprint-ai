"""CALLS edges resolved by jedi, a real static-analysis engine, instead of by
this project's own surface-name resolver (D58).

The resolver in resolve.py matches names. It can turn `self.send` into
`Session.send` because `self` is the one receiver whose type is always
known, and that is as far as name matching goes: `conn.urlopen` in
`HTTPAdapter.send` -- the single most important call in the corpus -- needs
to know that `conn` came from `self.get_connection_with_tls_context(...)`,
whose return value came from `self.poolmanager.connection_from_host(...)`,
which is a urllib3 `HTTPConnectionPool`. That is type inference, and every
tool surveyed in docs/research/ that produces a usable call graph does it.
jedi does it too, and it is the one such tool that is maintained, pure
Python, pip-installable, and runs on this interpreter -- PyCG (archived,
Python 3.6-era) and scip-python (a Windows path bug at startup) were both
tried first and could not be made to run.

Measured head-to-head on all 2,646 call sites before adopting it: where the
two disagreed, jedi added 229 correct edges (attribute chains, inherited
`self.*` methods, `super()`, local closures) and the resolver's 169 edges
that jedi lacked were mostly false -- `getattr` matched to
`LookupDict.__getattr__`, `bool` and `cast` to unrelated dunders. Precision
over recall is the standard for a graph an answer model will trust (D56,
D57), so the resolver-based CALLS write is retired rather than merged.

What jedi is NOT used for: RAISES, INHERITS_FROM and WRAPS_EXCEPTION still
go through the resolver, because their surfaces are class names written in
the same module or imported by name, which name matching handles, and
because each of those edges also carries an LLM-side counterpart resolved
the same way. Changing one path at a time keeps the benchmark attributable.
"""

import ast
from dataclasses import dataclass
from pathlib import Path

import jedi

from graphrag.ingest.chunk import iter_symbols, make_chunk_id


@dataclass(frozen=True)
class CallEdge:
    """One resolved call: a function, the surface it wrote, and the canonical
    id jedi pinned it to. `chunk_id` is the calling function's, so the edge is
    citable -- the resolver-era CALLS edges carried the placeholder "ast"."""

    source_id: str
    surface: str
    target_id: str
    chunk_id: str
    method: str = "jedi"
    # Which of the calling function's own parameters are handed to this
    # call, as (parameter name, keyword name or None, positional index or
    # None). `self.cert_verify(conn, request.url, verify, cert)` passes
    # `verify` at position 2 and `cert` at position 3 (D83).
    passes: tuple[tuple[str, str | None, int | None], ...] = ()


def build_project(src_root: Path | str, *, others: list[Path | str]) -> jedi.Project:
    """A jedi project rooted at one package's source tree, with the other
    corpus packages importable.

    One project per repo, not one for both: jedi derives a definition's
    `full_name` from the project root, so analysing a urllib3 file under a
    project rooted at requests_repo/src reports `urllib3_repo.src.urllib3...`
    and nothing in the graph matches it.
    """
    return jedi.Project(path=str(src_root), added_sys_path=[str(o) for o in others])


_CALLABLE_KINDS = ("function", "class")


def _definitions(script: jedi.Script, func: ast.expr) -> set[str]:
    """The callables a call's target names: by `goto` first, `infer` second.

    The cursor goes on the last character of the target -- the `m` of `x.m`,
    the `A` of a bare `A`. goto answers "where is this name defined"; for
    `Retry.from_int(n)` that is the classmethod, which is the edge wanted.
    infer answers "what does it evaluate to", and for that call it is
    `builtins.classmethod.__get__`, so the graph had no edge to `from_int`
    at all (D80). infer alone was chosen in D58 because goto, for
    `target = f if flag else A; target()`, lands on the `target = ...`
    statement -- one name, ambiguity gone, a local variable as a target.
    Both are right about something: goto's answer is used only when it is a
    function or class; when it is a statement, a parameter or nothing,
    infer's answer is used, which for that branch-bound variable is f and
    A, two callables, correctly ambiguous and skipped by the caller.
    """
    line, col = func.end_lineno, func.end_col_offset - 1
    try:
        defined = script.goto(line, col)
        names = {d.full_name for d in defined if d.full_name and d.type in _CALLABLE_KINDS}
        if names:
            return names
        values = script.infer(line, col)
    except Exception:
        # jedi raises on a few syntactic corners it can't place a cursor in;
        # an unresolvable call is a missing edge, never a crash.
        return set()
    return {v.full_name for v in values if v.full_name and v.type in _CALLABLE_KINDS}


def resolve_calls(
    text: str,
    *,
    repo: str,
    path: str,
    dotted_module: str,
    project: jedi.Project,
    packages: tuple[str, ...],
) -> list[CallEdge]:
    """Every call in every function of one module that jedi pins to exactly
    one definition inside `packages`.

    Exactly one: a target with two possible definitions (a variable bound to
    different callables on different branches) is skipped, because choosing
    would be a guess and a wrong CALLS edge is worse than a missing one.
    Inside `packages`: builtins and the stdlib resolve fine but are not
    corpus nodes, and an edge to `builtins.len` anchors nothing.
    """
    tree = ast.parse(text)
    script = jedi.Script(text, path=path, project=project)
    prefixes = tuple(f"{p}." for p in packages)
    edges: list[CallEdge] = []

    for name, node, start, end in iter_symbols(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        source_id = f"{dotted_module}.{name}"
        chunk_id = make_chunk_id(repo, path, start, end)
        seen: set[str] = set()
        own_params = {a.arg for a in node.args.posonlyargs + node.args.args + node.args.kwonlyargs}
        own_params.discard("self"); own_params.discard("cls")

        for call in ast.walk(node):
            if not isinstance(call, ast.Call):
                continue
            if not isinstance(call.func, (ast.Name, ast.Attribute)):
                continue

            targets = {t for t in _definitions(script, call.func) if t.startswith(prefixes)}
            if len(targets) != 1:
                continue
            target = targets.pop()
            # A function calling the same thing twice is one fact about it,
            # matching how RAISES and the resolver-era CALLS were shaped.
            if target in seen:
                continue
            seen.add(target)

            passes = []
            for i, arg in enumerate(call.args):
                if isinstance(arg, ast.Name) and arg.id in own_params:
                    passes.append((arg.id, None, i))
            for kw in call.keywords:
                if kw.arg and isinstance(kw.value, ast.Name) and kw.value.id in own_params:
                    passes.append((kw.value.id, kw.arg, None))
            edges.append(
                CallEdge(
                    source_id=source_id,
                    surface=ast.unparse(call.func),
                    target_id=target,
                    chunk_id=chunk_id,
                    passes=tuple(passes),
                )
            )

    return edges


def expand_overrides(
    edges: list[CallEdge],
    *,
    subclasses: dict[str, list[str]],
    node_universe: set[str],
) -> list[CallEdge]:
    """Follow a call to a base-class method down to every override of it.

    jedi resolves `adapter.send` in `Session.send` to `BaseAdapter.send`,
    which is the receiver's static type and the right static answer. At
    runtime it is `HTTPAdapter.send`, and the runtime oracle counted that
    as a miss -- a third of the remaining CALLS misses were this shape
    (D67). The graph already knows the inheritance, so a call to a base
    method is also a call to each override a subclass defines, followed
    transitively. A subclass that inherits the method unchanged gets no
    edge: the base edge already covers it. Emitted with method="override"
    so the tier stays visible on the edge (D68).
    """
    extra: list[CallEdge] = []
    seen: set[tuple[str, str]] = set()
    for edge in edges:
        owner, _, method = edge.target_id.rpartition(".")
        if not method or owner not in subclasses:
            continue
        frontier = list(subclasses.get(owner, []))
        visited: set[str] = set()
        while frontier:
            cls = frontier.pop()
            if cls in visited:
                continue
            visited.add(cls)
            frontier.extend(subclasses.get(cls, []))
            override = f"{cls}.{method}"
            if override in node_universe and override != edge.target_id:
                key = (edge.source_id, override)
                if key in seen:
                    continue
                seen.add(key)
                extra.append(CallEdge(
                    source_id=edge.source_id, surface=edge.surface,
                    target_id=override, chunk_id=edge.chunk_id, method="override",
                ))
    return extra
