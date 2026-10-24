"""docs/architecture.md and docs/model_card.md stay true to the code and the report
(v5.1 step 1.5c, plan D20)."""

import re

ARCHITECTURE = "docs/architecture.md"
MODEL_CARD = "docs/model_card.md"
REPORT = ("docs/report/README.md", "docs/report/FINDINGS.md")
PERCENT = re.compile(r"(\d+(?:\.\d+)?) ?%")


def _read(repo_root, rel):
    return (repo_root / rel).read_text(encoding="utf-8")


def test_architecture_names_every_module(repo_root):
    """A new module fails this test until the module map says what it is for."""
    text = _read(repo_root, ARCHITECTURE)
    modules = sorted(p.relative_to(repo_root).as_posix()
                     for p in (repo_root / "toolkit").rglob("*.py")
                     if "__pycache__" not in p.parts)
    assert len(modules) > 40
    missing = [m for m in modules if f"`{m}`" not in text]
    assert not missing, f"not in {ARCHITECTURE}: {missing}"


def test_files_the_docs_name_or_link_exist(repo_root):
    for doc in (ARCHITECTURE, MODEL_CARD):
        text = _read(repo_root, doc)
        named = set(re.findall(r"`((?:tests|toolkit|scripts|docs)/[\w./-]+\.\w+)", text))
        folder = (repo_root / doc).parent
        linked = {(folder / target).resolve().relative_to(repo_root.resolve()).as_posix()
                  for target in re.findall(r"\]\(([^)#:]+)(?:#[^)]*)?\)", text)}
        missing = sorted(p for p in named | linked if not (repo_root / p).exists())
        assert not missing, f"{doc} names files that don't exist: {missing}"


def test_model_card_quotes_only_numbers_the_report_has(repo_root):
    """Every percentage in the card is one the ML report measured (no rounding, no
    numbers from memory)."""
    report = " ".join(_read(repo_root, rel) for rel in REPORT)
    measured = set(PERCENT.findall(report)) | set(re.findall(r"\| (\d+\.\d+) ", report))
    quoted = PERCENT.findall(_read(repo_root, MODEL_CARD))
    assert quoted, "the card quotes the report's results"
    unknown = sorted(set(quoted) - measured)
    assert not unknown, f"{MODEL_CARD} quotes numbers that are not in docs/report: {unknown}"
