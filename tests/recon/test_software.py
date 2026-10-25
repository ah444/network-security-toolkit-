"""S12: the software in a banner, the only part of it that leaves the computer (step A1).

The banners are the forms real services send (SSH, HTTP Server headers, FTP and SMTP
greetings, the MySQL/MariaDB greeting as the reader shows it, Redis, Memcached).
"""

import re

import pytest

from toolkit.core.data.recon import software

SAFE = re.compile(r"^[A-Za-z][A-Za-z0-9\-]*(?: Server)? \d[\d.]*$")  # a cut version may end in "."


@pytest.mark.parametrize(("banner", "found"), [
    # SSH: only the token after the protocol
    ("SSH-2.0-OpenSSH_8.9p1 Ubuntu-3ubuntu0.1", "OpenSSH 8.9"),
    ("SSH-2.0-OpenSSH_4.7p1 Debian-8ubuntu1", "OpenSSH 4.7"),
    ("SSH-2.0-OpenSSH_for_Windows_8.1", "OpenSSH 8.1"),  # was "Windows 8.1"
    ("SSH-2.0-OpenSSH_for_Windows_9.5", "OpenSSH 9.5"),
    ("SSH-2.0-dropbear_2020.81", "dropbear 2020.81"),
    ("SSH-2.0-libssh-0.9.6", "libssh 0.9.6"),
    ("SSH-1.99-Cisco-1.25", "Cisco 1.25"),
    ("SSH-2.0-Go", None),
    ("SSH-2.0-SSHD Ubuntu-3ubuntu0.1", None),  # no version, and the comment is not taken
    # HTTP Server headers
    ("Apache/2.4.41 (Ubuntu)", "Apache 2.4.41"),
    ("Apache/2.4.6 (CentOS) OpenSSL/1.0.2k-fips", "Apache 2.4.6"),
    ("nginx/1.18.0 (Ubuntu)", "nginx 1.18.0"),
    ("Microsoft-IIS/10.0", "Microsoft-IIS 10.0"),
    ("Microsoft-HTTPAPI/2.0", "Microsoft-HTTPAPI 2.0"),
    ("lighttpd/1.4.28", "lighttpd 1.4.28"),
    ("Apache-Coyote/1.1", "Apache-Coyote 1.1"),
    ("SimpleHTTP/0.6 Python/3.14.3", "SimpleHTTP 0.6"),
    ("Server: Jetty(9.4.z-SNAPSHOT)", None),
    ("HTTP/1.1 200 OK", None),
    ("nginx", None),
    # FTP and SMTP greetings
    ("220 (vsFTPd 3.0.3)", "vsFTPd 3.0.3"),
    ("220 (vsFTPd 2.3.4)", "vsFTPd 2.3.4"),
    ("220 ProFTPD 1.3.5 Server (Debian) [::ffff:192.168.56.101]", "ProFTPD 1.3.5"),
    ("220-FileZilla Server 1.7.0", "FileZilla Server 1.7.0"),  # was "Server 1.7.0"
    ("220 mail.example.com ESMTP Exim 4.92 #3 Mon, 25 Oct 2026 16:20:07 +0000", "Exim 4.92"),
    ("220 mail ESMTP Sendmail 8.15.2/8.15.2; Mon, 25 Oct 2026", "Sendmail 8.15.2"),
    ("220 mail.example.com ESMTP Postfix (Ubuntu)", None),
    ("220 Microsoft FTP Service", None),
    ("+OK Dovecot ready.", None),
    # the MySQL reader's "MySQL <version>"
    ("MySQL 8.0.33", "MySQL 8.0.33"),
    ("MySQL 8.0.33-0ubuntu0.22.04.2", "MySQL 8.0.33"),
    ("MySQL 5.0.51a-3ubuntu5", "MySQL 5.0.51"),
    ("MySQL 5.5.5-10.6.12-MariaDB", "MariaDB 10.6.12"),  # was "MySQL 5.5.5"
    ("MySQL 5.5.5-10.6.12-MariaDB-0ubuntu0.22.04.1", "MariaDB 10.6.12"),
    ("MySQL 10.11.6-MariaDB", "MariaDB 10.11.6"),
    # request/answer services
    ("# Server\r\nredis_version:7.0.11\r\n", "Redis 7.0.11"),
    ("VERSION 1.6.21", "Memcached 1.6.21"),  # was "VERSION 1.6.21"
    ("", None),
])
def test_product_version(banner, found):
    assert software.product_version(banner) == found


@pytest.mark.parametrize("banner", [
    "220 Welcome to 192.168.1.20 FTP",
    "220 host-01 10.0.0.5 ready",
    "srv_172.16.0.9 up",
    "Welcome 192.168.56.101",
])
def test_an_address_never_leaves_as_a_version(banner):
    assert software.product_version(banner) is None


@pytest.mark.parametrize("banner", [
    "\x1b[31mOpenSSH\x1b[0m_8.9 \x07",
    "A" * 100 + "/1.0",
    "Name/" + ".".join(["1"] * 80),
    "<script>alert(1)</script>/1.0 <img src=x onerror=alert(1)>",
    "Apache/2.4.41\r\nSet-Cookie: session=secret",
    "x" * 10_000,
])
def test_what_leaves_is_short_and_allow_listed(banner):
    found = software.product_version(banner)
    assert found is None or (len(found) <= software.MAX_LENGTH and SAFE.match(found)), found


def test_rules_come_before_the_generic_pattern():
    assert all(isinstance(rule, software.Rule) for rule in software.RULES)
    # the generic pattern alone would have taken "Windows 8.1" and "MySQL 5.5.5"
    assert software._generic("SSH-2.0-OpenSSH_for_Windows_8.1") == "Windows 8.1"
    assert software._generic("MySQL 5.5.5-10.6.12-MariaDB") == "MySQL 5.5.5"
