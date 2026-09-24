from graphrag.ingest.ast_extract import (
    extract_calls,
    extract_exception_wrapping,
    extract_imports,
    extract_inherits_from,
    extract_parameters,
    extract_raises,
    module_dotted_name,
)
from graphrag.ingest.chunk import chunk_python


def test_module_dotted_name_from_src_layout():
    dotted = module_dotted_name(
        "src/requests/adapters.py", src_root="src", package_root="requests"
    )
    assert dotted == "requests.adapters"


def test_module_dotted_name_for_package_init():
    dotted = module_dotted_name(
        "src/requests/__init__.py", src_root="src", package_root="requests"
    )
    assert dotted == "requests"


EXC_SOURCE = '''class RequestException(IOError):
    """There was an ambiguous exception."""


class ConnectionError(RequestException):
    """A connection error occurred."""


class ConnectTimeout(ConnectionError, Timeout):
    """The request timed out while trying to connect."""
'''


def test_inherits_from_records_every_base_by_surface_name():
    edges = extract_inherits_from(
        EXC_SOURCE,
        repo="requests",
        path="src/requests/exceptions.py",
        dotted_module="requests.exceptions",
    )

    by_source = {e.source_id: e.target_surfaces for e in edges}
    assert by_source["requests.exceptions.RequestException"] == ["IOError"]
    assert by_source["requests.exceptions.ConnectionError"] == ["RequestException"]
    assert by_source["requests.exceptions.ConnectTimeout"] == [
        "ConnectionError",
        "Timeout",
    ]


def test_inherits_from_edge_cites_the_same_chunk_id_as_its_class_chunk():
    edges = extract_inherits_from(
        EXC_SOURCE,
        repo="requests",
        path="src/requests/exceptions.py",
        dotted_module="requests.exceptions",
    )
    chunks = chunk_python(EXC_SOURCE, repo="requests", path="src/requests/exceptions.py")

    chunk_id_by_section = {c.section: c.chunk_id for c in chunks}
    for edge in edges:
        class_name = edge.source_id.rsplit(".", 1)[-1]
        assert edge.chunk_id == chunk_id_by_section[class_name]


IMPORT_SOURCE = '''import os.path
from urllib3.exceptions import MaxRetryError, ProtocolError
from .models import Response
from . import utils
import json as simplejson
'''


def test_extract_imports_records_plain_import():
    edges = extract_imports(IMPORT_SOURCE, dotted_module="requests.adapters")
    assert ("os.path", "os.path") in [(e.imported, e.local_name) for e in edges]


def test_extract_imports_records_each_from_import_separately():
    edges = extract_imports(IMPORT_SOURCE, dotted_module="requests.adapters")
    names = {(e.imported, e.local_name) for e in edges}
    assert ("urllib3.exceptions.MaxRetryError", "MaxRetryError") in names
    assert ("urllib3.exceptions.ProtocolError", "ProtocolError") in names


def test_extract_imports_resolves_relative_import_against_dotted_module():
    edges = extract_imports(IMPORT_SOURCE, dotted_module="requests.adapters")
    names = {(e.imported, e.local_name) for e in edges}
    # adapters.py lives in the requests package, so ".models" is requests.models
    assert ("requests.models.Response", "Response") in names
    assert ("requests.utils", "utils") in names


def test_extract_imports_records_alias():
    edges = extract_imports(IMPORT_SOURCE, dotted_module="requests.adapters")
    names = {(e.imported, e.local_name) for e in edges}
    assert ("json", "simplejson") in names


RAISE_SOURCE = '''def send(self, request):
    try:
        conn = self.get_connection(request.url)
    except LocationValueError as e:
        raise InvalidURL(e, request=request)
    except MaxRetryError as e:
        raise ConnectTimeout(e, request=request)
    return conn


def close(self):
    raise NotImplementedError
'''


def test_extract_raises_finds_every_raise_in_a_function():
    edges = extract_raises(
        RAISE_SOURCE, repo="requests", path="src/requests/adapters.py", dotted_module="requests.adapters"
    )

    by_source = {e.source_id: e.exceptions_raised for e in edges}
    assert by_source["requests.adapters.send"] == ["InvalidURL", "ConnectTimeout"]
    assert by_source["requests.adapters.close"] == ["NotImplementedError"]


def test_extract_raises_edge_cites_its_function_chunk_id():
    edges = extract_raises(
        RAISE_SOURCE, repo="requests", path="src/requests/adapters.py", dotted_module="requests.adapters"
    )
    chunks = chunk_python(RAISE_SOURCE, repo="requests", path="src/requests/adapters.py")

    chunk_id_by_section = {c.section: c.chunk_id for c in chunks}
    for edge in edges:
        name = edge.source_id.rsplit(".", 1)[-1]
        assert edge.chunk_id == chunk_id_by_section[name]


PARAM_SOURCE = '''class Session:
    def request(self, method, url, verify=None, timeout: int = 30, *args, **kwargs):
        pass


def helper(x):
    pass
'''


def test_extract_parameters_lists_each_parameter_with_its_default():
    edges = extract_parameters(PARAM_SOURCE, dotted_module="requests.sessions")

    by_source = {e.source_id: e.parameters for e in edges}
    request_params = by_source["requests.sessions.Session.request"]
    names = [p.name for p in request_params]

    assert names == ["method", "url", "verify", "timeout", "args", "kwargs"]


def test_extract_parameters_excludes_self():
    edges = extract_parameters(PARAM_SOURCE, dotted_module="requests.sessions")
    by_source = {e.source_id: e.parameters for e in edges}

    assert "self" not in [p.name for p in by_source["requests.sessions.Session.request"]]


def test_extract_parameters_records_default_and_has_default_flag():
    edges = extract_parameters(PARAM_SOURCE, dotted_module="requests.sessions")
    by_source = {e.source_id: e.parameters for e in edges}
    by_name = {p.name: p for p in by_source["requests.sessions.Session.request"]}

    assert by_name["verify"].has_default is True
    assert by_name["verify"].default == "None"
    assert by_name["method"].has_default is False
    assert by_name["method"].default is None


CALL_SOURCE = '''class HTTPAdapter:
    def send(self, request):
        conn = self.get_connection(request.url)
        resp = conn.urlopen(method=request.method, url=request.url)
        os.path.join("a", "b")
        return resp

    def get_connection(self, url):
        pass
'''


def test_extract_calls_records_each_call_by_surface_expression():
    edges = extract_calls(CALL_SOURCE, dotted_module="requests.adapters")
    by_source = {e.source_id: e.calls for e in edges}

    assert by_source["requests.adapters.HTTPAdapter.send"] == [
        "self.get_connection",
        "conn.urlopen",
        "os.path.join",
    ]


def test_extract_calls_skips_functions_with_no_calls():
    edges = extract_calls(CALL_SOURCE, dotted_module="requests.adapters")
    by_source = {e.source_id: e.calls for e in edges}

    assert "requests.adapters.HTTPAdapter.get_connection" not in by_source


def test_relative_import_inside_a_package_init_resolves_to_the_package_itself():
    # requests/__init__.py has dotted name "requests" (the __init__ is
    # stripped). Inside it, "from .exceptions import X" means
    # requests.exceptions.X — NOT ".exceptions.X". Dropping a segment the way
    # a regular module does leaves an empty package and a malformed id.
    source = "from .exceptions import ConnectionError\n"

    edges = extract_imports(source, dotted_module="requests", is_package=True)

    assert edges[0].imported == "requests.exceptions.ConnectionError"


def test_relative_import_inside_a_regular_module_still_drops_one_segment():
    source = "from .exceptions import ConnectionError\n"

    edges = extract_imports(source, dotted_module="requests.adapters", is_package=False)

    assert edges[0].imported == "requests.exceptions.ConnectionError"


WRAP_SOURCE = '''def send(self, request):
    try:
        conn.urlopen()
    except (ProtocolError, OSError) as err:
        raise ConnectionError(err, request=request)
    except MaxRetryError as e:
        if isinstance(e.reason, ConnectTimeoutError):
            raise ConnectTimeout(e, request=request)
        raise ConnectionError(e, request=request)
    except ClosedPoolError as e:
        raise ConnectionError(e, request=request)
'''


def test_wraps_exception_pairs_each_caught_type_with_what_is_raised():
    edges = extract_exception_wrapping(
        WRAP_SOURCE, repo="requests", path="src/requests/adapters.py",
        dotted_module="requests.adapters",
    )
    pairs = {(e.caught_surface, e.raised_surface) for e in edges}

    assert ("ProtocolError", "ConnectionError") in pairs
    assert ("OSError", "ConnectionError") in pairs
    assert ("ClosedPoolError", "ConnectionError") in pairs


def test_wraps_exception_finds_raises_nested_inside_the_handler():
    # The MaxRetryError handler raises ConnectTimeout from inside an if —
    # a handler's raises are not always its direct children.
    edges = extract_exception_wrapping(
        WRAP_SOURCE, repo="requests", path="src/requests/adapters.py",
        dotted_module="requests.adapters",
    )
    pairs = {(e.caught_surface, e.raised_surface) for e in edges}

    assert ("MaxRetryError", "ConnectTimeout") in pairs
    assert ("MaxRetryError", "ConnectionError") in pairs


def test_wraps_exception_records_the_function_and_chunk_it_came_from():
    edges = extract_exception_wrapping(
        WRAP_SOURCE, repo="requests", path="src/requests/adapters.py",
        dotted_module="requests.adapters",
    )

    assert all(e.source_id == "requests.adapters.send" for e in edges)
    assert all(e.chunk_id for e in edges)


def test_a_bare_reraise_produces_no_wrapping_edge():
    # "except X: raise" re-raises the same exception — it doesn't wrap it in
    # a different one, so there is no WRAPS_EXCEPTION fact to record.
    source = "def f():\n    try:\n        g()\n    except ValueError:\n        raise\n"
    edges = extract_exception_wrapping(
        source, repo="requests", path="p.py", dotted_module="m"
    )

    assert edges == []
