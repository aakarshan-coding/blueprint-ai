"""Fake adapters at the three external seams -- OpenAI, Postgres, Neo4j -- plus
a fake resolver and embedder. One copy, shared by the tests that cross those
seams (architecture review, #5: these were hand-rolled in ten files).

Each fake is a *configuration*, not a script: it answers from what it was
constructed with and records what it was asked, so a test reads as the setup
it needs rather than a class definition.
"""

from graphrag.retrieval.router import RouterDecision


class FakeEmbed:
    """Every text gets the same vector; ranking then keeps original order."""

    def encode(self, texts, **kwargs):
        return [[0.1] * 384 for _ in texts]


class _Cursor:
    def __init__(self, rows):
        self._rows = rows

    def execute(self, q, p=None):
        pass

    def executemany(self, q, rows):
        pass

    def fetchall(self):
        return self._rows

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeConn:
    """Postgres: every query returns the rows it was built with (the
    vector-search column order: chunk_id, repo, path, section, kind, text, dist)."""

    def __init__(self, rows=()):
        self._rows = list(rows)

    def cursor(self):
        return _Cursor(self._rows)

    def commit(self):
        pass


class FakeSession:
    """Neo4j: a template query returns `rows`; the describe query (which asks
    for labels) returns `descriptions` -- {id: (labels, degree)}."""

    def __init__(self, rows=(), descriptions=None):
        self._rows = list(rows)
        self._descriptions = descriptions or {}
        self.queries = []

    def run(self, query, **params):
        self.queries.append((query, params))
        rows = self._rows
        if "labels(n)" in query:
            rows = [
                {"id": i, "labels": labels, "degree": degree}
                for i, (labels, degree) in self._descriptions.items()
                if i in params.get("ids", [])
            ]

        class R:
            def data(self_inner):
                return rows

        return R()


class _Parsed:
    def __init__(self, v):
        self.output_parsed = v


class _Text:
    def __init__(self, t):
        self.output_text = t


class _Responses:
    def __init__(self, *, route, mentions, plan, answer):
        self._route, self._mentions, self._plan, self._answer = route, mentions, plan, answer
        self.parse_calls = 0
        self.last_input = None

    def parse(self, **kw):
        self.parse_calls += 1
        text_format = kw["text_format"]
        if text_format is RouterDecision:
            return _Parsed(RouterDecision(route=self._route, confidence=0.95))
        if text_format.__name__ == "Mentions":
            return _Parsed(text_format(mentions=self._mentions))
        # The plan model is built per question; the fake constructs its
        # canned plan through whatever class the pipeline passed, so an
        # entity the fake names that isn't a candidate fails validation
        # exactly as it would in production.
        return _Parsed(text_format(plan=self._plan))

    def create(self, **kw):
        self.last_input = kw["input"]
        return _Text(self._answer)


class FakeOpenAI:
    def __init__(self, *, route="GRAPH", mentions=None, plan=None, answer="An answer [c1]."):
        self.responses = _Responses(
            route=route,
            mentions=mentions if mentions is not None else [{"surface": "Session", "package": "unknown"}],
            plan=plan or {"template_id": "T1_NEIGHBORS", "entity_id": "requests.sessions.Session"},
            answer=answer,
        )


class FakeResolver:
    """Resolves exactly the surfaces in `table`; everything else is unresolved."""

    def __init__(self, table=None):
        self.table = table if table is not None else {"Session": "requests.sessions.Session"}
        self.asked = []

    def resolve(self, surface, **kw):
        self.asked.append(surface)

        class R:
            canonical_id = self.table.get(surface)
            candidates = ()
            method = "exact" if surface in self.table else "unresolved"

        return R()


PASSAGE_ROW = ("c1", "requests", "src/requests/models.py", "Response.iter_content", "code",
               "A useful passage.", 0.1)
DOC_ROW = ("d1", "requests", "docs/user/advanced.rst", "SSL Cert Verification", "doc",
           "Some prose.", 0.2)
T1_ROW = {"relationship": "CALLS", "neighbor": "requests.adapters.HTTPAdapter.send",
          "chunk_id": "g1", "outgoing": True}
