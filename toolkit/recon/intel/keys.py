"""API keys and the AI model setting (S15): the environment, else ``.env`` (project
folder), else ``tools/Port_scan/api_keys.txt``. All keys are optional.
"""

from collections.abc import Mapping
from pathlib import Path

from ...config import ENV_FILE, REPO_ROOT, env_value, read_env_file

API_KEYS_PATH = REPO_ROOT / "tools" / "Port_scan" / "api_keys.txt"
ENV_PATH = ENV_FILE
DEFAULT_AI_MODEL = "deepseek/deepseek-chat:free"

# name in api_keys.txt -> environment variable (the same name is read from .env)
KEY_VARIABLES = {
    "openrouter": "OPENROUTER_API_KEY",
    "abuseipdb": "ABUSEIPDB_API_KEY",
    "nvd": "NVD_API_KEY",
}


def load_api_keys(environ: Mapping[str, str] | None = None, key_file: str | Path = API_KEYS_PATH,
                  env_file: str | Path | None = ENV_PATH) -> dict[str, str]:
    """Each key from the environment, else .env, else api_keys.txt; '' when missing."""
    legacy = read_env_file(key_file)  # same "name=value" format, short names
    return {name: env_value(variable, environ, env_file) or legacy.get(name, "")
            for name, variable in KEY_VARIABLES.items()}


def ai_model(environ: Mapping[str, str] | None = None,
             env_file: str | Path | None = None) -> str:
    """OpenRouter model: OPENROUTER_MODEL from the environment or .env, else the default."""
    return env_value("OPENROUTER_MODEL", environ,
                     ENV_PATH if env_file is None else env_file) or DEFAULT_AI_MODEL
