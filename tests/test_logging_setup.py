"""Log files, levels and redaction (plan D17, v5.1 step 1.1b)."""

import ast
import logging
import warnings

import pytest

from toolkit.core.data import logs as data_logs
from toolkit.core.data.settings import log_level

TOOL_FILES = ("main.py", "toolkit/cli/controllers/scan.py", "toolkit/cli/controllers/ids.py",
              "toolkit/controllers/cli/web.py", "toolkit/controllers/web/server.py")


@pytest.fixture
def clean_logging():
    """Undo setup_logging after the test, so other tests see pytest's logging only."""
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    for handler in root.handlers:
        if handler not in handlers:
            handler.close()
    root.handlers[:] = handlers
    root.setLevel(level)
    logging.captureWarnings(False)
    data_logs.forget_secrets()


def _read_logs(path):
    for handler in logging.getLogger().handlers:
        handler.flush()
    return path.read_text(encoding="utf-8")


def test_log_file_in_instance_logs_utf8_and_rotates(tmp_path, monkeypatch, clean_logging):
    monkeypatch.setenv("TOOLKIT_INSTANCE_DIR", str(tmp_path))
    path = data_logs.setup_logging("scanner", max_bytes=400, backups=2)
    assert path == tmp_path / "logs" / "scanner.log"

    log = logging.getLogger("toolkit.test")
    log.info("banner from ünïcode host")
    assert "banner from ünïcode host" in _read_logs(path)
    for number in range(20):
        log.info("line %d %s", number, "x" * 40)
    assert sorted(p.name for p in path.parent.iterdir()) == \
        ["scanner.log", "scanner.log.1", "scanner.log.2"]


def test_line_format(tmp_path, monkeypatch, clean_logging):
    monkeypatch.setenv("TOOLKIT_INSTANCE_DIR", str(tmp_path))
    path = data_logs.setup_logging("web")
    logging.getLogger("toolkit.core.services.ids.split").warning("Split %s", "stratified")
    line = _read_logs(path).strip()
    assert line.endswith(" WARNING toolkit.core.services.ids.split: Split stratified")
    assert line[4] == "-" and line[10] == "T"  # ISO date and time first


def test_level_from_environment(tmp_path, monkeypatch, clean_logging):
    assert log_level({}) == logging.INFO
    assert log_level({"TOOLKIT_LOG_LEVEL": " debug "}) == logging.DEBUG
    for bad in ("loud", "10", "trace"):
        with pytest.raises(ValueError, match="TOOLKIT_LOG_LEVEL"):
            log_level({"TOOLKIT_LOG_LEVEL": bad})

    monkeypatch.setenv("TOOLKIT_INSTANCE_DIR", str(tmp_path))
    monkeypatch.setenv("TOOLKIT_LOG_LEVEL", "WARNING")
    path = data_logs.setup_logging("web")
    log = logging.getLogger("toolkit.test")
    log.info("not written")
    log.warning("written")
    text = _read_logs(path)
    assert "written" in text and "not written" not in text


def test_third_party_debug_is_capped(tmp_path, monkeypatch, clean_logging):
    monkeypatch.setenv("TOOLKIT_INSTANCE_DIR", str(tmp_path))
    monkeypatch.setenv("TOOLKIT_LOG_LEVEL", "DEBUG")
    path = data_logs.setup_logging("scanner")
    logging.getLogger("urllib3.connectionpool").debug("GET /rest/json/cves/2.0?keyword=x")
    logging.getLogger("toolkit.recon").debug("our own debug line")
    text = _read_logs(path)
    assert "our own debug line" in text and "connectionpool" not in text


def test_secrets_redacted_also_in_tracebacks(tmp_path, monkeypatch, clean_logging):
    monkeypatch.setenv("TOOLKIT_INSTANCE_DIR", str(tmp_path))
    monkeypatch.setenv("ABUSEIPDB_API_KEY", "abuse-secret-1234567890")
    path = data_logs.setup_logging("scanner")
    data_logs.add_secrets("nvd-secret-0987654321", "short")
    log = logging.getLogger("toolkit.test")

    log.warning("headers %s", {"Key": "header-secret-abcdef12", "Accept": "application/json"})
    log.warning("Authorization: Bearer sk-or-v1-0123456789abcdef")
    log.warning("url https://example.test/check?api_key=query-secret-77777777&ip=1.2.3.4")
    log.warning("env value abuse-secret-1234567890 and nvd-secret-0987654321")
    try:
        raise RuntimeError("request failed with apiKey=trace-secret-55555555")
    except RuntimeError:
        log.exception("lookup failed")
    log.warning("a short value is kept: short")

    text = _read_logs(path)
    for secret in ("header-secret-abcdef12", "sk-or-v1-0123456789abcdef", "query-secret-77777777",
                   "abuse-secret-1234567890", "nvd-secret-0987654321", "trace-secret-55555555"):
        assert secret not in text
    assert text.count(data_logs.REDACTED) >= 6
    assert "'Accept': 'application/json'" in text and "ip=1.2.3.4" in text
    assert "a short value is kept: short" in text  # too short to redact safely


def test_setup_twice_does_not_duplicate_lines(tmp_path, monkeypatch, clean_logging):
    monkeypatch.setenv("TOOLKIT_INSTANCE_DIR", str(tmp_path))
    data_logs.setup_logging("web")
    path = data_logs.setup_logging("web")
    logging.getLogger("toolkit.test").warning("once")
    assert _read_logs(path).count("once") == 1


def test_library_warnings_reach_the_log(tmp_path, monkeypatch, clean_logging):
    monkeypatch.setenv("TOOLKIT_INSTANCE_DIR", str(tmp_path))
    path = data_logs.setup_logging("ids_cli")
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.warn("model saved with scikit-learn 1.6.1", UserWarning, stacklevel=1)
    assert "model saved with scikit-learn 1.6.1" in _read_logs(path)


def test_no_dataset_values_in_logs(tmp_path, monkeypatch, fixtures_dir, clean_logging):
    """A cell value planted in the data must never reach the log, even at DEBUG."""
    from toolkit.core.services.ids.datasets import load_dataset
    from toolkit.core.services.ids.runs import TrainSettings, train

    marker = "zqmarkerzq"
    lines = (fixtures_dir / "nslkdd_train_sample.txt").read_text(encoding="utf-8").splitlines()
    lines[1] = lines[1].replace(",http,", f",{marker},", 1)
    data = tmp_path / "train.txt"
    data.write_text("\n".join(lines) + "\n", encoding="utf-8")
    monkeypatch.setenv("TOOLKIT_INSTANCE_DIR", str(tmp_path / "instance"))
    monkeypatch.setenv("TOOLKIT_LOG_LEVEL", "DEBUG")
    path = data_logs.setup_logging("web")

    loaded = load_dataset(data)
    assert marker in set(loaded.frame["service"])
    train(loaded.frame, profile=loaded.profile,
          settings=TrainSettings(models=("decision_tree",), workers=1))

    text = _read_logs(path)
    assert "Trained decision_tree" in text  # the log really was written at DEBUG/INFO
    assert marker not in text


def test_tools_use_setup_logging_not_basic_config(repo_root):
    for name in TOOL_FILES:
        tree = ast.parse((repo_root / name).read_text(encoding="utf-8"))
        calls = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and n.func.attr == "basicConfig"]
        assert calls == [], f"{name}: logging.basicConfig at lines {calls}; use setup_logging"


LOG_CALLS = {"debug", "info", "warning", "error", "exception", "critical", "log"}


def test_toolkit_logs_through_module_loggers(repo_root):
    """Step 1.3d: no ``logging.info(...)`` on the root logger in ``toolkit/``; a module that
    logs has ``log = logging.getLogger(__name__)``, so each line names its module."""
    for path in sorted((repo_root / "toolkit").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        name = path.relative_to(repo_root).as_posix()
        root_calls = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call)
                      and isinstance(n.func, ast.Attribute) and n.func.attr in LOG_CALLS
                      and isinstance(n.func.value, ast.Name) and n.func.value.id == "logging"]
        assert root_calls == [], f"{name}: root logger at lines {root_calls}"
        log_calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Attribute) and n.func.attr in LOG_CALLS
                     and isinstance(n.func.value, ast.Name) and n.func.value.id == "log"]
        if log_calls:
            assert "log = logging.getLogger(__name__)" in ast.unparse(tree), name
