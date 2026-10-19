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
