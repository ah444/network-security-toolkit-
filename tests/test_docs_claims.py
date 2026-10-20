"""Step 0.2: published claims must match the measured baseline (docs/baseline/)."""

MODELS_README = "tools/IDS/IDS_CLI - CIC-2017/models/README.md"


def _read(repo_root, rel):
    return (repo_root / rel).read_text(encoding="utf-8")


def test_readme_has_no_withdrawn_claims_or_placeholders(repo_root):
    text = _read(repo_root, "README.md")
    assert "High accuracy (~97%)" not in text
    assert "yourusername" not in text
    assert '"# network-security-toolkit-"' not in text
    assert "](LICENSE)" not in text  # no link to a file that doesn't exist


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
