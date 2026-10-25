"""B9: TLS banners without certificate checks, for banners only."""

import ssl
from pathlib import Path

from toolkit.core.data.recon import probes
from toolkit.core.services.recon import intel as recon_intel


def test_banner_context_unverified_only_for_banners(scanner, repo_root):
    context = probes.banner_tls_context()
    assert context.verify_mode == ssl.CERT_NONE and context.check_hostname is False
    assert ssl.create_default_context().verify_mode == ssl.CERT_REQUIRED, "global default kept"
    files = sorted((repo_root / "toolkit").rglob("*.py"))
    sources = [path.read_text(encoding="utf-8") for path in files]
    # certificate checks are switched off in exactly one place of the toolkit: the banner context
    assert sum(source.count("CERT_NONE") for source in sources) == 1
    for source in sources:
        assert "verify=False" not in source and "_create_unverified_context" not in source
    source = Path(probes.__file__).read_text(encoding="utf-8")
    start = source.index("def banner_tls_context")
    assert "CERT_NONE" in source[start:source.index("\ndef ", start + 1)]
    # web lookups go through requests, which verifies certificates
    assert recon_intel.Intel({}).session.verify is True
