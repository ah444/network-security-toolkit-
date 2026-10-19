"""OS names used by the ``fake_platform`` fixture.

name -> (platform.system() value, sys.platform value)
"""

PLATFORMS = {
    "windows": ("Windows", "win32"),
    "linux": ("Linux", "linux"),
    "darwin": ("Darwin", "darwin"),
}
