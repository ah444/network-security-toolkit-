"""v5.1 step 1.2a: the web trainer's settings in one place (toolkit.data.settings.WebConfig)."""

import pytest

from toolkit.data.settings import WebConfig, web_config


def test_defaults_without_environment():
    assert web_config({}) == WebConfig()
    assert WebConfig().max_upload_mb == 2048 and WebConfig().port == 5000
    assert not WebConfig().https and not WebConfig().debug


def test_values_from_environment():
    config = web_config({"TOOLKIT_MAX_UPLOAD_MB": "64", "TOOLKIT_WORKSPACE_IDLE_MIN": "5",
                         "TOOLKIT_MAX_WORKSPACES": "2", "TOOLKIT_MAX_DATA_MB": "512",
                         "TOOLKIT_JOB_MAX_MIN": "30",
                         "TOOLKIT_PORT": "5055", "TOOLKIT_HTTPS": "1", "TOOLKIT_DEBUG": "1",
                         "TOOLKIT_SECRET_KEY": "k" * 32})
    assert config == WebConfig(max_upload_mb=64, workspace_idle_minutes=5, max_workspaces=2,
                               max_data_mb=512, job_max_minutes=30, port=5055, https=True,
                               debug=True,
                               secret_key="k" * 32)


def test_only_1_switches_https_and_debug_on():
    config = web_config({"TOOLKIT_HTTPS": "0", "TOOLKIT_DEBUG": ""})
    assert not config.https and not config.debug
    # any other value is an error naming the setting (v5.1 step 1.5e; it used to mean off)
    for variable, value in (("TOOLKIT_HTTPS", "yes"), ("TOOLKIT_DEBUG", "true")):
        with pytest.raises(ValueError, match=f"{variable} must be 1 \\(on\\) or 0 \\(off\\)"):
            web_config({variable: value})


def test_bad_number_is_an_error():
    with pytest.raises(ValueError):
        web_config({"TOOLKIT_PORT": "fivethousand"})


def test_app_uses_the_config(make_app):
    app = make_app(WebConfig(max_upload_mb=1, workspace_idle_minutes=7, max_workspaces=3,
                             max_data_mb=10))
    assert app.config["MAX_CONTENT_LENGTH"] == 1024 * 1024
    store = app.extensions["toolkit"].workspaces
    assert store.idle_seconds == 7 * 60
    assert (store.max_workspaces, store.max_data_bytes) == (3, 10 * 2**20)
