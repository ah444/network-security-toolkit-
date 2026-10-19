import platform
import sys

import pytest

from platforms import PLATFORMS


@pytest.mark.parametrize("name", sorted(PLATFORMS))
def test_patched_platform_is_seen(fake_platform, name):
    fake_platform(name)
    system, sys_platform = PLATFORMS[name]
    assert platform.system() == system
    assert sys.platform == sys_platform
