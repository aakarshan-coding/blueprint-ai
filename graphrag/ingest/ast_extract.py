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


def _raised_name(raise_node: ast.Raise) -> str | None:
    """Return the exception name a `raise` statement names, if any.

    Covers `raise Foo(...)` (a Call) and bare `raise Foo` (a Name). A bare
    `raise` re-raising the current exception has no name and is skipped.
    """
    exc = raise_node.exc
    if exc is None:
        return None
    if isinstance(exc, ast.Call):
        exc = exc.func
    return ast.unparse(exc)


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
            _raised_name(child)
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
    showcase, and neither the structural pass nor the prose pass captured it:
    `RAISES` records only what a function throws, and the LLM pass never reads
    code bodies. An except/raise pair is plainly visible in the syntax tree
    but expresses a *relationship*, not a structure — the category that fell
    between the two passes.
    """

    source_id: str
    caught_surface: str
    raised_surface: str
    chunk_id: str


def extract_exception_wrapping(
    text: str, *, repo: str, path: str, dotted_module: str
) -> list[ExceptionWrapEdge]:
    """Pair every caught exception type with every exception raised in its
    handler. A handler catching a tuple produces one edge per caught type; a
    handler raising from inside nested control flow still counts, since the
    raise is a consequence of that catch either way."""
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

            raised = [
                _raised_name(child)
                for child in ast.walk(handler)
                if isinstance(child, ast.Raise)
            ]
            # A bare `raise` re-raises the same exception rather than wrapping
            # it in a different one — no wrapping fact to record.
            raised = [r for r in raised if r is not None]

            for caught_name in caught:
                for raised_name in raised:
                    if raised_name == caught_name:
                        continue
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
