"""v5.2 step 2.3o: no Python or JavaScript file of toolkit/ is longer than ``MAX_LINES``,
unless ``tests/exceptions_inventory.py`` lists it in ``LONG_FILES`` with its exact length and
a reason. The limit is a signal to look at the file: split it when that helps, or record why
it stays. The stylesheet and the scripts in scripts/ are not counted."""

import exceptions_inventory as inventory

MAX_LINES = 400


def _lengths(root):
    """Lines of every Python and JavaScript file of ``root/toolkit``, by its path."""
    return {path.relative_to(root).as_posix(): len(path.read_text(encoding="utf-8").splitlines())
            for pattern in ("*.py", "*.js") for path in sorted((root / "toolkit").rglob(pattern))
            if "__pycache__" not in path.parts}


def _problems(lengths, long_files):
    problems = [f"{name}: {lines} lines (limit {MAX_LINES}): split it, or list it in LONG_FILES "
                "with its length and a reason"
                for name, lines in lengths.items()
                if lines > MAX_LINES and name not in long_files]
    for name, (ceiling, _) in long_files.items():
        lines = lengths.get(name)
        if lines is None or lines <= MAX_LINES:
            problems.append(f"{name}: listed in LONG_FILES but no longer above the limit")
        elif lines != ceiling:
            problems.append(f"{name}: {lines} lines, LONG_FILES says {ceiling} (it may not "
                            "grow; lower the count when it shrinks)")
    return problems


def test_no_file_is_too_long(repo_root):
    assert _problems(_lengths(repo_root), inventory.LONG_FILES) == []


def test_every_long_file_has_a_reason():
    for name, (lines, reason) in inventory.LONG_FILES.items():
        assert lines > MAX_LINES and len(reason.split()) >= 5, name


def test_the_check_sees_each_case(tmp_path):
    (tmp_path / "toolkit" / "ui").mkdir(parents=True)
    (tmp_path / "toolkit" / "a.py").write_text("x = 1\n" * MAX_LINES, encoding="utf-8")
    (tmp_path / "toolkit" / "ui" / "b.js").write_text("x;\n" * (MAX_LINES + 1), encoding="utf-8")
    (tmp_path / "toolkit" / "ui" / "c.css").write_text("a{}\n" * 999, encoding="utf-8")
    lengths = _lengths(tmp_path)
    assert lengths == {"toolkit/a.py": MAX_LINES, "toolkit/ui/b.js": MAX_LINES + 1}
    assert [p.split(":")[0] for p in _problems(lengths, {})] == ["toolkit/ui/b.js"]
    reason = "a reason of five words"
    assert _problems(lengths, {"toolkit/ui/b.js": (MAX_LINES + 1, reason)}) == []
    grew = _problems(lengths, {"toolkit/ui/b.js": (MAX_LINES, reason)})
    assert len(grew) == 1 and "may not grow" in grew[0]
    stale = _problems(lengths, {"toolkit/ui/b.js": (MAX_LINES + 1, reason),
                                "toolkit/a.py": (MAX_LINES, reason)})
    assert len(stale) == 1 and "no longer above" in stale[0]
