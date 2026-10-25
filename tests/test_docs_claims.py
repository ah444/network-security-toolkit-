"""Step 0.2: published claims must match the measured baseline (docs/baseline/)."""

import pytest

MODELS_README = "docs/ids_cli_models.md"


def _read(repo_root, rel):
    return (repo_root / rel).read_text(encoding="utf-8")


def test_readme_has_no_withdrawn_claims_or_placeholders(repo_root):
    text = _read(repo_root, "README.md")
    assert "High accuracy (~97%)" not in text
    assert "yourusername" not in text
    assert '"# network-security-toolkit-"' not in text
    # 5.0: no link to the missing LICENSE; since step 1.5g it exists and is linked
    assert "](LICENSE)" in text and (repo_root / "LICENSE").is_file()


def test_readme_states_leakage_and_limitations(repo_root):
    text = _read(repo_root, "README.md")
    assert "withdrawn" in text
    assert "Known limitations" in text
    assert "docs/baseline/FINDINGS.md" in text


def test_models_readme_withdraws_accuracy(repo_root):
    text = _read(repo_root, MODELS_README)
    assert "withdrawn" in text
    assert "| Test Accuracy | ~98% |" not in text
    assert "label-derived" in text


def test_dataset_licences_recorded(repo_root):
    """Step 0.5b: each dataset's terms and citation, checked on its official page."""
    text = _read(repo_root, "docs/datasets.md")
    licences = text[text.index("## Licences and citations"):]
    assert "To confirm" not in licences and "still to complete" not in licences
    for page, citation in (
        ("https://www.unb.ca/cic/datasets/ids-2017.html", "Sharafaldin"),
        ("https://www.unb.ca/cic/datasets/nsl.html", "Tavallaee"),
        ("https://research.unsw.edu.au/projects/unsw-nb15-dataset", "Moustafa"),
    ):
        assert page in licences and citation in licences
    assert "academic research" in licences and "Derived models" in licences
    # committed real-data fixtures are redistributed, so they carry the citations
    fixtures = _read(repo_root, "tests/fixtures/README.md")
    assert "Sharafaldin" in fixtures and "Tavallaee" in fixtures
    assert "UNSW" not in fixtures, "UNSW-NB15 rows may not be redistributed"


@pytest.mark.parametrize("audit_path", ["docs/release/5.0-section8-audit.md",
                                        "docs/release/5.1-section8-audit.md"])
def test_section8_audit_names_real_tests(repo_root, audit_path):
    """5.0 final and 5.1: every test the §8 audit points to exists (no stale references)."""
    import re

    audit = _read(repo_root, audit_path)
    missing = []
    for line in audit.splitlines():
        current = None
        for match in re.finditer(r"`(tests/[\w/]+\.py)?(?:::(test_\w+))?", line):
            path, name = match.groups()
            if path:
                current = path
                assert (repo_root / path).is_file(), path
            if name:
                source = (repo_root / current).read_text(encoding="utf-8")
                if f"def {name}(" not in source:
                    missing.append(f"{current}::{name}")
    assert missing == []
