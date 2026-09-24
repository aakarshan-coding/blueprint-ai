"""Deterministic structural extraction: nodes and edges Python's own parser
can prove, at zero cost and with no model in the loop.

This pass runs first. It establishes the node universe (every module, class,
function, dotted-path canonical id) that the LLM pass in llm_extract.py must
resolve its prose-derived surface forms against.

An extracted edge's target is a *surface name as written* (e.g. "ConnectionError"
in a bases list, "Timeout" in another), not yet a canonical id. Some surface
names refer to a class in this same module; others to an external builtin
(IOError) or an imported name this pass hasn't resolved. Turning surface names
into canonical ids — or deciding one has no node to resolve to — is
resolve.py's job (Phase 1 step 4), using the alias table this pass also
produces. Keeping that resolution out of this module is deliberate: this pass
only asserts what ast can prove outright.
"""

import ast
from dataclasses import dataclass
from pathlib import Path

from graphrag.ingest.chunk import iter_symbols, make_chunk_id


def module_dotted_name(path: str, *, src_root: str, package_root: str) -> str:
    """Return a module's dotted import path from its file path.

    "src/requests/adapters.py" -> "requests.adapters". This is the canonical
    id every Module/Class/Function node is keyed by — the same id an `import`
    statement would use, which is what lets IMPORTS edges resolve later.
    """
    rel = Path(path).relative_to(src_root).with_suffix("")
    parts = rel.parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) if parts else package_root


@dataclass(frozen=True)
class InheritsFromEdge:
    """A class's declared bases, as written — not yet resolved to canonical ids."""

    source_id: str
    target_surfaces: list[str]
    chunk_id: str


def _base_surface(expr: ast.expr) -> str:
    """Render a base-class expression back to its written form.

    Usually a bare Name ("ConnectionError"). Occasionally an Attribute
    ("compat.CompatJSONDecodeError") — ast.unparse handles both.
    """
    return ast.unparse(expr)


def extract_inherits_from(
    text: str, *, repo: str, path: str, dotted_module: str
) -> list[InheritsFromEdge]:
    """Extract every class's base list as an INHERITS_FROM edge.

    One edge per class, carrying every base — not one edge per base — so a
    class's full bases list is visible atomically rather than reconstructed by
    joining scattered edges.
    """
    tree = ast.parse(text)
    edges = []

    for name, node, start, end in iter_symbols(tree):
        if not isinstance(node, ast.ClassDef) or not node.bases:
            continue
        edges.append(
            InheritsFromEdge(
                source_id=f"{dotted_module}.{name}",
                target_surfaces=[_base_surface(b) for b in node.bases],
                chunk_id=make_chunk_id(repo, path, start, end),
            )
        )

    return edges


@dataclass(frozen=True)
class ImportEdge:
    """One name brought into a module's namespace by an import statement."""

    imported: str
    local_name: str


def _resolve_relative(
    dotted_module: str, level: int, module: str | None, *, is_package: bool = False
) -> str:
    """Resolve a relative import ("from . import x") to a dotted module path.

    level=1 means "this package". For a regular module that means dropping
    its own last segment (`requests.adapters` -> `requests`). For a package's
    `__init__.py` it means the module's own dotted name, because
    `module_dotted_name` already stripped the `__init__` — `requests/__init__.py`
    *is* `requests`, so there is no segment to drop. Dropping one anyway
    yields an empty package and ids like ".exceptions.ConnectionError".
    """
    segments_to_drop = level - 1 if is_package else level
    package = (
        ".".join(dotted_module.split(".")[:-segments_to_drop])
        if segments_to_drop
        else dotted_module
    )
    return f"{package}.{module}" if module else package


def extract_imports(
    text: str, *, dotted_module: str, is_package: bool = False
) -> list[ImportEdge]:
    """Extract every name an import statement brings into scope.

    Each imported name becomes its own edge — "from x import a, b" is two
    facts, not one — because code elsewhere refers to `a` and `b`
    independently, and resolution needs a name-by-name lookup.
    """
    tree = ast.parse(text)
    edges = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                local = alias.asname or alias.name
                edges.append(ImportEdge(imported=alias.name, local_name=local))

        elif isinstance(node, ast.ImportFrom):
            if node.level:
                module_base = _resolve_relative(
                    dotted_module, node.level, node.module, is_package=is_package
                )
            else:
                module_base = node.module

            for alias in node.names:
                local = alias.asname or alias.name
                imported = f"{module_base}.{alias.name}" if module_base else alias.name
                edges.append(ImportEdge(imported=imported, local_name=local))

    return edges


@dataclass(frozen=True)
class RaisesEdge:
    """Every exception a function raises, by surface name — not yet resolved."""

    source_id: str
    exceptions_raised: list[str]
    chunk_id: str


def _is_variable_name(surface: str) -> bool:
    """A lowercase bare identifier names a variable, not an exception class.

    Exception classes are CapWords by convention and in both corpora without
    exception; `e`, `err`, `new_e`, `reraise` are the bound names and locals
    that `raise` statements re-throw. Recording those as exceptions put nodes
    literally called `e` into the graph.
    """
    return surface.isidentifier() and surface[0].islower()


def _assigned_class(name: str, scope: ast.AST) -> str | None:
    """The class a local variable was constructed from, if visible in scope.

    urllib3 writes `new_e = ProtocolError("aborted", e); raise new_e from e`
    six times. The variable is not the fact; the class it holds is. Only a
    direct `name = Class(...)` assignment counts — anything less certain is
    dropped rather than guessed.
    """
    for node in ast.walk(scope):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, (ast.Name, ast.Attribute))
        ):
            return ast.unparse(node.value.func)
    return None


def _raised_name(raise_node: ast.Raise, *, scope: ast.AST | None = None) -> str | None:
    """Return the exception class a `raise` statement names, if any.

    Covers `raise Foo(...)` (a Call) and bare `raise Foo` (a Name). A bare
    `raise` re-raising the current exception has no name and is skipped. A
    raised *variable* is followed to the class it was assigned in `scope`, or
    skipped when that can't be established.
    """
    exc = raise_node.exc
    if exc is None:
        return None
    if isinstance(exc, ast.Call):
        exc = exc.func
    surface = ast.unparse(exc)
    if _is_variable_name(surface):
        return _assigned_class(surface, scope) if scope is not None else None
    return surface


def extract_raises(
    text: str, *, repo: str, path: str, dotted_module: str
) -> list[RaisesEdge]:
    """Extract every exception a function raises, in source order.

    One edge per function, listing every raise inside it — including inside
    nested try/except — because a function's exception surface is one fact
    about the function, not one fact per statement.
    """
    tree = ast.parse(text)
    edges = []

    for name, node, start, end in iter_symbols(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        raised = [
            _raised_name(child, scope=node)
            for child in ast.walk(node)
            if isinstance(child, ast.Raise)
        ]
        raised = [r for r in raised if r is not None]
        if not raised:
            continue

        edges.append(
            RaisesEdge(
                source_id=f"{dotted_module}.{name}",
                exceptions_raised=raised,
                chunk_id=make_chunk_id(repo, path, start, end),
            )
        )

    return edges


@dataclass(frozen=True)
class Parameter:
    name: str
    has_default: bool
    default: str | None


@dataclass(frozen=True)
class ParametersEdge:
    source_id: str
    parameters: list[Parameter]


_SKIP_FIRST_ARG = ("self", "cls")


def _params_from_arguments(args: ast.arguments) -> list[Parameter]:
    """Flatten an ast.arguments into an ordered parameter list.

    self/cls is dropped — it names the method's own object, not something a
    caller chooses, so it isn't a fact worth a HAS_PARAMETER edge.
    """
    positional = list(args.posonlyargs) + list(args.args)
    defaults = list(args.defaults)
    # ast.arguments right-aligns defaults against the tail of `positional`.
    pad = len(positional) - len(defaults)
    defaults = [None] * pad + defaults

    params = []
    for i, arg in enumerate(positional):
        if i == 0 and arg.arg in _SKIP_FIRST_ARG:
            continue
        default = defaults[i]
        params.append(
            Parameter(
                name=arg.arg,
                has_default=default is not None,
                default=ast.unparse(default) if default is not None else None,
            )
        )

    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        params.append(
            Parameter(
                name=arg.arg,
                has_default=default is not None,
                default=ast.unparse(default) if default is not None else None,
            )
        )

    if args.vararg:
        params.append(Parameter(name=args.vararg.arg, has_default=False, default=None))
    if args.kwarg:
        params.append(Parameter(name=args.kwarg.arg, has_default=False, default=None))

    return params


def extract_parameters(text: str, *, dotted_module: str) -> list[ParametersEdge]:
    """Extract every function's parameter list, in declaration order.

    One edge per function. A parameter's default is recorded as the source
    text of the expression ("None", "30"), not evaluated — some defaults
    (a class, a sentinel object) aren't safely eval-able out of context.
    """
    tree = ast.parse(text)
    edges = []

    for name, node, _start, _end in iter_symbols(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        edges.append(
            ParametersEdge(
                source_id=f"{dotted_module}.{name}",
                parameters=_params_from_arguments(node.args),
            )
        )

    return edges


@dataclass(frozen=True)
class CallsEdge:
    """Every call a function makes, by surface expression — not yet resolved.

    "conn.urlopen" means a call was made on whatever "conn" is; this pass
    can't know its type without inference resolve.py doesn't attempt either.
    A call like "self.get_connection" is the resolvable case: resolve.py can
    settle it immediately because "self" always means the enclosing class.
    """

    source_id: str
    calls: list[str]


def _call_target(call: ast.Call) -> str | None:
    """Return a call's surface expression ("conn.urlopen", "helper"), if any.

    A call whose target isn't a Name or an Attribute chain (e.g. calling the
    result of another call, "get_handler()()") has no fixed surface name to
    record and is skipped.
    """
    func = call.func
    if isinstance(func, (ast.Name, ast.Attribute)):
        return ast.unparse(func)
    return None


def extract_calls(text: str, *, dotted_module: str) -> list[CallsEdge]:
    """Extract every call a function makes, in source order.

    One edge per function, matching the shape of RAISES and HAS_PARAMETER — a
    function's call surface is one fact about the function.
    """
    tree = ast.parse(text)
    edges = []

    for name, node, _start, _end in iter_symbols(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        calls = [
            _call_target(child)
            for child in ast.walk(node)
            if isinstance(child, ast.Call)
        ]
        calls = [c for c in calls if c is not None]
        if not calls:
            continue

        edges.append(CallsEdge(source_id=f"{dotted_module}.{name}", calls=calls))

    return edges


@dataclass(frozen=True)
class ExceptionWrapEdge:
    """A caught exception and the one raised in its place — a WRAPS_EXCEPTION
    fact, extracted deterministically from an except handler.

    This is the relationship the requests+urllib3 corpus was chosen to
    showcase. The LLM pass does find it in docstrings — every WRAPS edge in
    the first full graph came from there — but prose describes the common
    path and misses the rest: the graph had `ConnectionError wraps
    ReadTimeoutError` (true, in `iter_content`) and not `ReadTimeout wraps
    ReadTimeoutError` (true, in `send`), and the benchmark asked about `send`.
    An except/raise pair is plainly visible in the syntax tree; this pass
    reads it with the control flow between catch and raise intact.
    """

    source_id: str
    caught_surface: str
    raised_surface: str
    chunk_id: str


def _isinstance_narrowing(test: ast.expr, bound: str | None) -> list[str] | None:
    """The types an `if isinstance(e, ...)` guard narrows the caught exception
    to, or None when the test says nothing certain about it.

    `isinstance(e.reason, X)` counts too: a MaxRetryError's `.reason` is the
    urllib3 error it carries, and `raise ConnectTimeout(e)` under that guard
    wraps the reason, which is the fact worth having. A negated or compound
    test (`not isinstance`, `isinstance(...) and retry`) does not pin the type
    on the raise's path, so it does not narrow.
    """
    if bound is None or not isinstance(test, ast.Call):
        return None
    if not (isinstance(test.func, ast.Name) and test.func.id == "isinstance"):
        return None
    if len(test.args) != 2:
        return None

    subject = test.args[0]
    while isinstance(subject, ast.Attribute):
        subject = subject.value
    if not (isinstance(subject, ast.Name) and subject.id == bound):
        return None

    types = test.args[1]
    if isinstance(types, ast.Tuple):
        return [ast.unparse(t) for t in types.elts]
    return [ast.unparse(types)]


def _wrap_pairs(
    stmts: list[ast.stmt], caught: list[str], *, bound: str | None, scope: ast.AST
):
    """Yield (caught_surface, raised_surface) for every raise under `stmts`,
    carrying the isinstance-narrowed type down the branch it guards.

    The handler's tuple says what *could* arrive; an isinstance guard says
    what *did*. Pairing every caught type with every raise regardless turned
    one handler in `HTTPAdapter.send` into six edges, four of them false —
    "ReadTimeout wraps _SSLError" among them — and a false graph fact overrides
    a correct passage (D56).

    Nested try statements are descended for their body/else/finally but not
    their handlers, which the caller visits as handlers of their own.
    """
    for stmt in stmts:
        if isinstance(stmt, ast.Raise):
            raised = _raised_name(stmt, scope=scope)
            if raised is not None:
                for caught_name in caught:
                    if raised != caught_name:
                        yield caught_name, raised

        elif isinstance(stmt, ast.If):
            narrowed = _isinstance_narrowing(stmt.test, bound)
            yield from _wrap_pairs(stmt.body, narrowed or caught, bound=bound, scope=scope)
            yield from _wrap_pairs(stmt.orelse, caught, bound=bound, scope=scope)

        elif isinstance(stmt, ast.Try):
            for block in (stmt.body, stmt.orelse, stmt.finalbody):
                yield from _wrap_pairs(block, caught, bound=bound, scope=scope)

        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue

        else:
            for field in ("body", "orelse", "cases"):
                block = getattr(stmt, field, None)
                if isinstance(block, list):
                    if field == "cases":
                        for case in block:
                            yield from _wrap_pairs(case.body, caught, bound=bound, scope=scope)
                    else:
                        yield from _wrap_pairs(block, caught, bound=bound, scope=scope)


def extract_exception_wrapping(
    text: str, *, repo: str, path: str, dotted_module: str
) -> list[ExceptionWrapEdge]:
    """Pair each exception a handler catches with the exception raised in its
    place, following isinstance guards so the pairing reflects the branch the
    raise actually sits on. A handler catching a tuple with no guard yields
    one edge per caught type; a raise inside nested control flow still counts."""
    tree = ast.parse(text)
    edges: list[ExceptionWrapEdge] = []

    for name, node, start, end in iter_symbols(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        chunk_id = make_chunk_id(repo, path, start, end)

        for handler in [n for n in ast.walk(node) if isinstance(n, ast.ExceptHandler)]:
            caught = _caught_surfaces(handler)
            if not caught:
                continue

            seen: set[tuple[str, str]] = set()
            for caught_name, raised_name in _wrap_pairs(
                handler.body, caught, bound=handler.name, scope=handler
            ):
                if (caught_name, raised_name) in seen:
                    continue
                seen.add((caught_name, raised_name))
                edges.append(
                    ExceptionWrapEdge(
                        source_id=f"{dotted_module}.{name}",
                        caught_surface=caught_name,
                        raised_surface=raised_name,
                        chunk_id=chunk_id,
                    )
                )

    return edges


def _caught_surfaces(handler: ast.ExceptHandler) -> list[str]:
    """Every exception type one handler catches. `except (A, B)` catches two;
    a bare `except:` catches everything and names nothing."""
    if handler.type is None:
        return []
    if isinstance(handler.type, ast.Tuple):
        return [ast.unparse(e) for e in handler.type.elts]
    return [ast.unparse(handler.type)]
