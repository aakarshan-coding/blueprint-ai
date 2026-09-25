"""Entity resolution: turn a surface name ("Session", "PoolManager") into the
canonical id (dotted path) it refers to, or admit it can't be resolved yet.

This is the ladder from the design (§6). Every rung is tried in order; the
first that succeeds wins. Nothing is ever silently dropped — a surface that
resolves to nothing still gets a Resolution recording that fact, so the
resolution rate is always a real, reportable number.
"""

import builtins
from dataclasses import dataclass
from typing import Literal

Method = Literal[
    "exact", "self_reference", "import_alias", "reexport", "qualified",
    "normalized", "public_api", "call_graph", "builtin", "unresolved",
]

# Python's own exception classes, by name. `except OSError: raise
# ConnectionError(e)` is a wrap the AST extractor finds and the graph could
# not hold: OSError has no node and no import to anchor it, so 43 of 81 wrap
# facts dropped (D57). The set is fixed and known, so a bare mention that no
# corpus rung claims resolves to `builtins.<name>`; the edge it sits on types
# the node as Exception on write, the same way socket.timeout is handled.
BUILTIN_EXCEPTIONS = frozenset(
    name for name, obj in vars(builtins).items()
    if isinstance(obj, type) and issubclass(obj, BaseException)
)


@dataclass(frozen=True)
class Resolution:
    surface: str
    canonical_id: str | None
    method: Method
    candidates: tuple[str, ...] = ()
    """Other canonical ids the surface could plausibly mean.

    Populated only when normalized matching found more than one — never
    picked between silently. These are the resolve.py stretch: they're the
    input step 4 (embedding similarity) needs, and they're what turns an
    ambiguous name into a reportable case instead of a silent guess.
    """


def _normalize(name: str) -> str:
    """Fold a name down to bare lowercase letters/digits for loose matching."""
    return "".join(ch for ch in name.lower() if ch.isalnum())


class Resolver:
    """Resolves surface names against a corpus's known nodes and imports.

    node_universe: every canonical id the AST pass proved exists (every
    Module/Class/Function dotted path).

    import_aliases: per-module import tables from extract_imports, keyed by
    the importing module's dotted name — {"requests.adapters": {"MaxRetryError":
    "urllib3.exceptions.MaxRetryError", ...}}.
    """

    def __init__(
        self,
        node_universe: set[str],
        import_aliases: dict[str, dict[str, str]],
        call_graph: set[tuple[str, str]] | frozenset[tuple[str, str]] = frozenset(),
        public_ids: set[str] | frozenset[str] = frozenset(),
        documented_params: dict[str, set[str]] | None = None,
    ) -> None:
        """call_graph: (caller_function_id, callee_function_id) pairs, from
        already-resolved CALLS edges. Used only to break ties among
        ambiguous candidates that share a name — e.g. a "verify" parameter
        that exists on 9 functions because the same flag is forwarded down
        a call chain. The one none of the others calls into is preferred as
        the "outermost", public-facing owner of that name.
        """
        self.node_universe = node_universe
        self.import_aliases = import_aliases
        self.call_graph = call_graph
        self.public_ids = public_ids
        self.documented_params = documented_params or {}

        self._by_exact_name: dict[str, list[str]] = {}
        self._by_normalized_name: dict[str, list[str]] = {}
        # Every proper dotted suffix of an id ("HTTPAdapter.send",
        # "adapters.HTTPAdapter.send") -> the ids it ends. A qualified surface
        # is looked up here whole, so "HTTPAdapter.send" finds one id where
        # the leaf "send" alone finds three.
        self._by_suffix: dict[str, list[str]] = {}
        for canonical_id in node_universe:
            parts = canonical_id.split(".")
            simple_name = parts[-1]
            self._by_exact_name.setdefault(simple_name, []).append(canonical_id)
            self._by_normalized_name.setdefault(_normalize(simple_name), []).append(
                canonical_id
            )
            for k in range(2, len(parts)):
                self._by_suffix.setdefault(".".join(parts[-k:]), []).append(canonical_id)

    def resolve(
        self,
        surface: str,
        *,
        module_context: str | None = None,
        enclosing_class: str | None = None,
    ) -> Resolution:
        if surface in self.node_universe:
            return Resolution(surface, surface, "exact")

        # "self.send" written inside Session is Session.send — self is the one
        # receiver whose type is always known, needing no inference. Without
        # this rule 17% of the corpus's call surfaces resolve to nothing, and
        # they are the most reliable calls there are.
        if enclosing_class and surface.startswith("self."):
            candidate = f"{enclosing_class}.{surface.removeprefix('self.')}"
            if candidate in self.node_universe:
                return Resolution(surface, candidate, "self_reference")
            return Resolution(surface, None, "unresolved")

        if module_context is not None:
            aliases = self.import_aliases.get(module_context, {})
            if surface in aliases:
                return Resolution(surface, aliases[surface], "import_alias")

        if "." in surface:
            package, _, leaf = surface.rpartition(".")
            aliases = self.import_aliases.get(package, {})
            if leaf in aliases:
                return Resolution(surface, aliases[leaf], "reexport")

            # A qualified surface ("HTTPAdapter.send") is matched as a whole
            # dotted suffix, on a segment boundary. Before this rung every
            # remaining rung looked at the bare leaf, so "HTTPAdapter.send"
            # collapsed to `send` and tied with Session.send and
            # BaseAdapter.send — the planner qualifies a name precisely when
            # the bare one would be ambiguous, which made this the case that
            # failed most often at query time.
            suffix_matches = self._by_suffix.get(surface, [])
            if len(suffix_matches) == 1:
                return Resolution(surface, suffix_matches[0], "qualified")
            if len(suffix_matches) > 1:
                return Resolution(
                    surface, None, "unresolved", candidates=tuple(sorted(suffix_matches))
                )

        # Exact case first: PoolManager (a class) and poolmanager (the module
        # it lives in) are different things to Python, and folding case would
        # collide them into a false ambiguity.
        exact_matches = self._by_exact_name.get(surface, [])
        if len(exact_matches) == 1:
            return Resolution(surface, exact_matches[0], "normalized")

        matches = self._by_normalized_name.get(_normalize(surface), [])
        if len(matches) == 1:
            return Resolution(surface, matches[0], "normalized")
        if len(matches) > 1:
            narrowed = self._narrow_to_public_documented(surface, matches)
            if len(narrowed) == 1:
                return Resolution(surface, narrowed[0], "public_api")

            remaining = narrowed if narrowed else matches
            winner = self._pick_by_call_graph(remaining)
            if winner is not None:
                return Resolution(surface, winner, "call_graph")
            return Resolution(surface, None, "unresolved", candidates=tuple(remaining))

        # Last, after every corpus rung: a name the corpus defines itself
        # (requests' own ConnectionError) must win over Python's.
        if surface in BUILTIN_EXCEPTIONS:
            return Resolution(surface, f"builtins.{surface}", "builtin")

        return Resolution(surface, None, "unresolved")

    def _narrow_to_public_documented(
        self, surface: str, candidates: list[str]
    ) -> list[str]:
        """Keep only candidates that are both publicly exported and documented.

        The same parameter name appears on many functions because the value is
        forwarded down a call chain, but documentation talks about it as users
        meet it: on the public, documented entry point. A candidate qualifies
        only if its owning function (or that function's class) is exported from
        a package's __init__, AND that function's docstring actually documents
        this parameter with a `:param <name>:` line.

        Returns [] when nothing qualifies, so the caller falls back to the full
        candidate list rather than resolving to nothing.
        """
        if not self.public_ids or not self.documented_params:
            return []

        survivors = []
        for candidate in candidates:
            owner = candidate.rsplit(".", 1)[0]
            holder = owner.rsplit(".", 1)[0]
            is_public = owner in self.public_ids or holder in self.public_ids
            is_documented = surface in self.documented_params.get(owner, set())
            if is_public and is_documented:
                survivors.append(candidate)
        return survivors

    def _pick_by_call_graph(self, candidates: list[str]) -> str | None:
        """Among ambiguous candidates, prefer the one no other candidate's
        owning function calls into — the "outermost" one. Returns None
        (stay unresolved) unless exactly one candidate qualifies; this never
        guesses among several equally-plausible outer candidates.
        """
        if not self.call_graph:
            return None

        owners = {c: c.rsplit(".", 1)[0] for c in candidates}
        called_by_another = {
            c
            for c, owner in owners.items()
            if any(
                (other_owner, owner) in self.call_graph
                for other_c, other_owner in owners.items()
                if other_c != c
            )
        }
        winners = [c for c in candidates if c not in called_by_another]
        return winners[0] if len(winners) == 1 else None
