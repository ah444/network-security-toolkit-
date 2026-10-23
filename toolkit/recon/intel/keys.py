"""API keys and the AI model setting (S15): the environment, else ``.env`` (project
folder). All keys are optional.

The old fallback file ``tools/Port_scan/api_keys.txt`` is no longer read (v5.1 step 1.4a,
``tools/`` was removed); if it is still on disk, the scanner says once to move its keys.
"""

from collections.abc import Mapping
from pathlib import Path

from ...config import ENV_FILE, REPO_ROOT, env_value

LEGACY_KEY_FILE = REPO_ROOT / "tools" / "Port_scan" / "api_keys.txt"
ENV_PATH = ENV_FILE
DEFAULT_AI_MODEL = "deepseek/deepseek-chat:free"

# short name -> environment variable (the same name is read from .env)
KEY_VARIABLES = {
    "openrouter": "OPENROUTER_API_KEY",
    "abuseipdb": "ABUSEIPDB_API_KEY",
    "nvd": "NVD_API_KEY",
}


def load_api_keys(environ: Mapping[str, str] | None = None,
                  env_file: str | Path | None = ENV_PATH) -> dict[str, str]:
    """Each key from the environment, else .env; '' when missing."""
    return {name: env_value(variable, environ, env_file)
            for name, variable in KEY_VARIABLES.items()}


def legacy_key_file_notice(path: str | Path = LEGACY_KEY_FILE) -> str | None:
    """One line for the user when the old api_keys.txt is still there (it is not read)."""
    if not Path(path).is_file():
        return None
    return ("tools/Port_scan/api_keys.txt is no longer read: move its keys into .env "
            "(see .env.example).")


def ai_model(environ: Mapping[str, str] | None = None,
             env_file: str | Path | None = None) -> str:
    """OpenRouter model: OPENROUTER_MODEL from the environment or .env, else the default."""
    return env_value("OPENROUTER_MODEL", environ,
                     ENV_PATH if env_file is None else env_file) or DEFAULT_AI_MODEL
