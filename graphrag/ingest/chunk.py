"""Split corpus files into citable chunks.

Every chunk carries an id derived from its location, so re-running ingestion
over unchanged files produces identical ids. That stability is what lets the
graph store and the vector store stay joined.

Chunk size is a tuning knob, not a constant. It trades retrieval precision
(smaller is sharper) against answerability (larger carries more context), and
the right value is settled by the recall@k measurement in Phase 2 — not here.
"""

import ast
import hashlib
import re
from dataclasses import dataclass

# An reStructuredText section heading is a title line followed by a line of
# repeated punctuation at least as long as the title.
_ADORNMENT = re.compile(r"""^([=\-`:.'"~^_*+#])\1+\s*$""")

DEFAULT_MAX_CHARS = 2000


@dataclass(frozen=True)
class Chunk:
    """One citable piece of the corpus.

    `text` and `embed_text` are two views of the same chunk, because searching
    and citing are different jobs. A citation must show the whole symbol; a
    vector wants a short, on-topic summary of it. For prose the two coincide.
    """

    chunk_id: str
    repo: str
    path: str
    kind: str
    section: str | None
    start_line: int
    end_line: int
    text: str
    embed_text: str


def make_chunk_id(repo: str, path: str, start_line: int, end_line: int) -> str:
    """Return a stable id for a span of lines in a file."""
    raw = f"{repo}:{path}:{start_line}:{end_line}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _build_chunk(
    *,
    repo: str,
    path: str,
    kind: str,
    section: str | None,
    start_line: int,
    end_line: int,
    text: str,
    embed_text: str | None = None,
) -> Chunk:
    """Assemble a Chunk, deriving its id from its location.

    Chunks embed their own text unless given something better to embed.
    """
    return Chunk(
        chunk_id=make_chunk_id(repo, path, start_line, end_line),
        repo=repo,
        path=path,
        kind=kind,
        section=section,
        start_line=start_line,
        end_line=end_line,
        text=text,
        embed_text=embed_text if embed_text is not None else text,
    )


def _find_headings(lines: list[str]) -> list[tuple[int, str, str]]:
    """Return (line_index, title, adornment_char) for each section heading."""
    found = []
    for i in range(len(lines) - 1):
        title = lines[i].strip()
        if not title:
            continue
        match = _ADORNMENT.match(lines[i + 1])
        if match and len(lines[i + 1].strip()) >= len(title):
            found.append((i, title, match.group(1)))
    return found


def _paragraph_blocks(
    lines: list[str], start: int, end: int
) -> list[tuple[int, int]]:
    """Return (start, end) line ranges for blank-line-separated blocks."""
    blocks: list[tuple[int, int]] = []
    open_at: int | None = None

    for i in range(start, end + 1):
        if lines[i].strip():
            if open_at is None:
                open_at = i
        elif open_at is not None:
            blocks.append((open_at, i - 1))
            open_at = None

    if open_at is not None:
        blocks.append((open_at, end))
    return blocks


def _pack_blocks(
    lines: list[str], blocks: list[tuple[int, int]], max_chars: int
) -> list[tuple[int, int]]:
    """Greedily group blocks into spans of at most max_chars.

    An indented block is a literal block continuation (code samples, output).
    Those are never split off from the paragraph that introduces them, so a
    span containing one may exceed max_chars. Splitting mid-example would
    produce a chunk that cites code without the sentence explaining it.
    """
    groups: list[tuple[int, int]] = []
    span: list[int] | None = None

    for block_start, block_end in blocks:
        if span is None:
            span = [block_start, block_end]
            continue

        joined = "\n".join(lines[span[0] : block_end + 1]).strip()
        is_literal_continuation = lines[block_start].startswith((" ", "\t"))

        if len(joined) <= max_chars or is_literal_continuation:
            span[1] = block_end
        else:
            groups.append((span[0], span[1]))
            span = [block_start, block_end]

    if span is not None:
        groups.append((span[0], span[1]))
    return groups


def chunk_rst(
    text: str,
    *,
    repo: str,
    path: str,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> list[Chunk]:
    """Split an .rst document into chunks, one or more per section.

    Section nesting is inferred from adornment characters in order of first
    appearance, which is how reStructuredText itself assigns heading levels.
    Sections longer than max_chars are split on paragraph boundaries; every
    piece keeps the full section breadcrumb so a citation stays locatable.
    """
    lines = text.splitlines()
    headings = _find_headings(lines)

    chunks: list[Chunk] = []
    adornment_order: list[str] = []
    breadcrumb: list[str] = []

    for position, (line_index, title, char) in enumerate(headings):
        if char not in adornment_order:
            adornment_order.append(char)
        level = adornment_order.index(char)
        breadcrumb = breadcrumb[:level] + [title]
        section = " > ".join(breadcrumb)

        is_last = position + 1 == len(headings)
        section_end = len(lines) - 1 if is_last else headings[position + 1][0] - 1

        blocks = _paragraph_blocks(lines, line_index, section_end)
        for span_start, span_end in _pack_blocks(lines, blocks, max_chars):
            chunks.append(
                _build_chunk(
                    repo=repo,
                    path=path,
                    kind="doc",
                    section=section,
                    start_line=span_start + 1,
                    end_line=span_end + 1,
                    text="\n".join(lines[span_start : span_end + 1]).strip(),
                )
            )

    return chunks


_FUNCTION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)


def _definition_start(node: ast.AST) -> int:
    """Line where a definition begins, counting its decorators."""
    decorators = getattr(node, "decorator_list", [])
    return min([node.lineno] + [d.lineno for d in decorators])


def iter_symbols(tree: ast.Module):
    """Yield (dotted_name, node, start_line, end_line) for every symbol.

    Shared by the code and docstring passes so both agree on what a symbol is
    and where it begins — a disagreement there would silently orphan docstrings
    from the code they describe.
    """
    for node in tree.body:
        if isinstance(node, _FUNCTION_NODES):
            yield node.name, node, _definition_start(node), node.end_lineno

        elif isinstance(node, ast.ClassDef):
            methods = [n for n in node.body if isinstance(n, _FUNCTION_NODES)]
            # The class chunk covers its header, docstring, and class-level
            # attributes — everything up to where the first method begins.
            class_end = (
                _definition_start(methods[0]) - 1 if methods else node.end_lineno
            )
            yield node.name, node, _definition_start(node), class_end

            for method in methods:
                yield (
                    f"{node.name}.{method.name}",
                    method,
                    _definition_start(method),
                    method.end_lineno,
                )


def _embeddable_summary(
    lines: list[str], node: ast.AST, start_line: int
) -> str | None:
    """Return signature + docstring, or None when there is no docstring.

    A long function body is mostly control flow and local names, which dilute
    its embedding until it matches nothing well. Its signature and docstring
    say what it is in a few lines. Undocumented symbols have no such summary,
    so they fall back to embedding their full text.
    """
    docstring = ast.get_docstring(node)
    if not docstring:
        return None

    literal = node.body[0]
    signature = "\n".join(lines[start_line - 1 : literal.lineno - 1]).strip()
    if not signature:
        return docstring.strip()
    return f"{signature}\n{docstring.strip()}"


@dataclass(frozen=True)
class DocstringSource:
    """A symbol's raw docstring, paired with its code chunk's own chunk_id.

    Not a Chunk — never stored, embedded, or written to Postgres. D6 removed
    docstrings as their own chunk to kill duplicate storage; this exists
    purely to hand the LLM pass prose to read, without re-introducing that
    duplication. The chunk_id matches the code chunk's, because that's the
    chunk a citation drawn from this docstring should point at.
    """

    symbol: str
    chunk_id: str
    docstring: str


def iter_docstrings_for_llm(
    text: str, *, repo: str, path: str
) -> list[DocstringSource]:
    """Yield every documented symbol's docstring, for the LLM pass only."""
    sources = []

    for name, node, start, end in iter_symbols(ast.parse(text)):
        docstring = ast.get_docstring(node)
        if not docstring:
            continue
        sources.append(
            DocstringSource(
                symbol=name,
                chunk_id=make_chunk_id(repo, path, start, end),
                docstring=docstring.strip(),
            )
        )

    return sources


def chunk_python(text: str, *, repo: str, path: str) -> list[Chunk]:
    """Split a Python module into one chunk per symbol.

    Top-level functions, classes, and methods each become their own chunk,
    named by dotted path ("HTTPAdapter.send"). That granularity is chosen so a
    graph edge can cite the exact symbol that justified it rather than the file
    it happened to live in.

    Code chunks are never split by size. A function is an atomic unit — half a
    function is neither readable as evidence nor valid as a citation. Size is
    handled by embedding a summary instead, not by cutting the symbol up.
    """
    lines = text.splitlines()
    return [
        _build_chunk(
            repo=repo,
            path=path,
            kind="code",
            section=name,
            start_line=start,
            end_line=end,
            text="\n".join(lines[start - 1 : end]).rstrip(),
            embed_text=_embeddable_summary(lines, node, start),
        )
        for name, node, start, end in iter_symbols(ast.parse(text))
    ]


# A release heading looks like "2.34.2 (2026-05-14)" or plain "2.8.0".
_RELEASE_TITLE = re.compile(r"^(\d+(?:\.\d+)+)(?:\s*\(.*\))?$")


def chunk_changelog(
    text: str,
    *,
    repo: str,
    path: str,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> list[Chunk]:
    """Split a changelog into one chunk per release.

    Both corpora mark releases as headings with an underline, so heading
    detection is shared with chunk_rst. Only headings that parse as a version
    number become releases — prose headings like "Release History" are not
    releases and must not become Release nodes in the graph.
    """
    lines = text.splitlines()
    releases = [
        (line_index, match.group(1))
        for line_index, title, _char in _find_headings(lines)
        if (match := _RELEASE_TITLE.match(title))
    ]

    chunks: list[Chunk] = []
    for position, (line_index, version) in enumerate(releases):
        is_last = position + 1 == len(releases)
        release_end = len(lines) - 1 if is_last else releases[position + 1][0] - 1

        blocks = _paragraph_blocks(lines, line_index, release_end)
        for span_start, span_end in _pack_blocks(lines, blocks, max_chars):
            chunks.append(
                _build_chunk(
                    repo=repo,
                    path=path,
                    kind="changelog",
                    section=version,
                    start_line=span_start + 1,
                    end_line=span_end + 1,
                    text="\n".join(lines[span_start : span_end + 1]).strip(),
                )
            )

    return chunks
