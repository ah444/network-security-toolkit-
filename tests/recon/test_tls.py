"""B9: TLS banners without certificate checks, for banners only."""

import ssl
from pathlib import Path


def test_banner_context_unverified_only_for_banners(scanner):
    context = scanner.banner_tls_context()
    assert context.verify_mode == ssl.CERT_NONE and context.check_hostname is False
    assert ssl.create_default_context().verify_mode == ssl.CERT_REQUIRED, "global default kept"
    source = Path(scanner.__file__).read_text(encoding="utf-8")
    # certificate checks are switched off in exactly one place: the banner context
    assert source.count("CERT_NONE") == 1
    assert "verify=False" not in source and "_create_unverified_context" not in source
    start = source.index("def banner_tls_context")
    assert "CERT_NONE" in source[start:source.index("\ndef ", start + 1)]
    # web lookups go through requests, which verifies certificates
    assert scanner.Intel({}).session.verify is True
