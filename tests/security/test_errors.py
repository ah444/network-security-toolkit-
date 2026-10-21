"""S8: error responses must not leak exception text, tracebacks or server paths."""

import ast
import io

WEB = "toolkit/web"


def _leaks(tree):
    """(line, variable) for each jsonify(...) inside `except ... as name` that uses name."""
    found = []
    for handler in ast.walk(tree):
        if not (isinstance(handler, ast.ExceptHandler) and handler.name):
            continue
        for node in ast.walk(handler):
            is_jsonify = (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                          and node.func.id == "jsonify")
            if is_jsonify and any(isinstance(n, ast.Name) and n.id == handler.name
                                  for n in ast.walk(node)):
                found.append((node.lineno, handler.name))
    return found


def test_no_exception_text_in_json_responses(repo_root):
    modules = sorted((repo_root / WEB).glob("*.py"))
    assert len(modules) > 5, "the web package moved?"
    for path in modules:
        assert _leaks(ast.parse(path.read_text(encoding="utf-8"))) == [], path.name


def test_scanner_detects_a_leak():
    sample = "try:\n    x()\nexcept Exception as e:\n    jsonify({'error': f'bad {e}'})\n"
    assert _leaks(ast.parse(sample)) == [(4, "e")]


def test_no_paths_or_tracebacks_in_errors(client):
    garbage = b"\x00\xff\xfe" * 2000
    response = client.post("/upload", data={"dataset": (io.BytesIO(garbage), "broken.csv")},
                           content_type="multipart/form-data")
    assert response.status_code in (400, 500)
    message = response.get_json()["error"]
    for leak in ("Traceback", "Error tokenizing", "codec", ":\\", "/tools/", "uploads"):
        assert leak not in message, f"error message leaks internals: {message!r}"
    assert "broken.csv" in message  # still tells the user which file failed
