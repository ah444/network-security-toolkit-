"""S15: API keys from the environment or .env, with api_keys.txt as a fallback."""

import logging


def test_env_overrides_file(scanner, tmp_path, caplog):
    key_file = tmp_path / "api_keys.txt"
    key_file.write_text("# comment\nopenrouter=from-file\nabuseipdb=your_abuseipdb_api_key_here\n"
                        "nvd = file-nvd \n", encoding="utf-8")
    env_file = tmp_path / ".env"
    env_file.write_text('export NVD_API_KEY="from-dotenv"\nABUSEIPDB_API_KEY=\n',
                        encoding="utf-8")
    environ = {"OPENROUTER_API_KEY": "from-env"}
    keys = scanner.load_api_keys(environ, key_file, env_file)
    assert keys == {"openrouter": "from-env", "abuseipdb": "", "nvd": "from-dotenv"}
    assert scanner.load_api_keys({}, key_file, tmp_path / "none.env")["nvd"] == "file-nvd"
    assert scanner.load_api_keys({"OPENROUTER_API_KEY": "your_openrouter_api_key_here"},
                                 key_file, env_file)["openrouter"] == "from-file"


def test_missing_key_files_are_not_errors(scanner, tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        keys = scanner.load_api_keys({}, tmp_path / "missing.txt", tmp_path / "missing.env")
    assert keys == {"openrouter": "", "abuseipdb": "", "nvd": ""}
    assert caplog.records == []


def test_key_files_stay_out_of_git(repo_root):
    import subprocess

    for name in (".env", "tools/Port_scan/api_keys.txt"):
        result = subprocess.run(["git", "check-ignore", "-q", name], cwd=repo_root, check=False)
        assert result.returncode == 0, f"{name} must be gitignored"
    template = subprocess.run(["git", "check-ignore", "-q", ".env.example"], cwd=repo_root,
                              check=False)
    assert template.returncode == 1, ".env.example (no secrets) must be committed"
    assert (repo_root / ".env.example").is_file()


def test_ai_model_from_env_file(scanner, tmp_path, monkeypatch):
    """OPENROUTER_MODEL works from .env, as the help text says (found in step 1.1b)."""
    env_file = tmp_path / ".env"
    env_file.write_text("OPENROUTER_MODEL=vendor/model-from-dotenv\n", encoding="utf-8")
    assert scanner.ai_model({}, env_file) == "vendor/model-from-dotenv"
    assert scanner.ai_model({"OPENROUTER_MODEL": "vendor/env"}, env_file) == "vendor/env"
    assert scanner.ai_model({}, tmp_path / "none.env") == scanner.DEFAULT_AI_MODEL

    monkeypatch.setattr(scanner, "ENV_PATH", env_file)
    intel = scanner.Intel({"openrouter": "test-key"}, use_ai=True)
    assert intel.model == "vendor/model-from-dotenv"
