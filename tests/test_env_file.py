"""The shared .env reader in toolkit.config (plan D17, v5.1 step 1.1b)."""

import logging

from toolkit.config import env_value, read_env_file


def test_read_env_file_rules(tmp_path):
    path = tmp_path / ".env"
    path.write_text("# comment\n\nexport NVD_API_KEY = \"quoted-nvd\" \nOPENROUTER_MODEL='m/x'\n"
                    "ABUSEIPDB_API_KEY=your_abuseipdb_api_key_here\nEMPTY=\nno equals sign\n"
                    "URL=https://a.test/?x=1\n", encoding="utf-8")
    assert read_env_file(path) == {"NVD_API_KEY": "quoted-nvd", "OPENROUTER_MODEL": "m/x",
                                   "URL": "https://a.test/?x=1"}


def test_missing_file_is_silent_unreadable_file_warns(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        assert read_env_file(tmp_path / "missing.env") == {}
    assert caplog.records == []

    broken = tmp_path / "broken.env"
    broken.write_bytes(b"NVD_API_KEY=\xff\xfe\n")
    with caplog.at_level(logging.WARNING):
        assert read_env_file(broken) == {}
    assert "Can't read" in caplog.text and "NVD_API_KEY" not in caplog.text


def test_environment_wins_over_env_file(tmp_path):
    path = tmp_path / ".env"
    path.write_text("OPENROUTER_MODEL=from-file\n", encoding="utf-8")
    assert env_value("OPENROUTER_MODEL", {"OPENROUTER_MODEL": " from-env "}, path) == "from-env"
    assert env_value("OPENROUTER_MODEL", {}, path) == "from-file"
    assert env_value("OPENROUTER_MODEL", {"OPENROUTER_MODEL": "your_model_here"}, path) == \
        "from-file"  # a placeholder left from the example counts as unset
    assert env_value("NVD_API_KEY", {}, path) == ""
    assert env_value("NVD_API_KEY", {}, tmp_path / "missing.env") == ""
