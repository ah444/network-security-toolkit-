"""Expect-style walk-through of every launcher menu path (release gate).

Runs main.py with piped stdin/stdout, waits for each prompt before answering
(so the launcher never reads lines meant for the tool it starts) and writes the
whole session to a transcript. Local only: the recon target is 127.0.0.1 with
closed ports, so no AbuseIPDB / CVE / AI lookups are made.

Usage:  python scripts/menu_walkthrough.py [transcript.md]
Exit code 1 (with the transcript tail) if a prompt doesn't appear in time.
"""

import os
import platform
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPT = sys.argv[1] if len(sys.argv) > 1 else "menu-walkthrough.md"
PORT = 5055

# The CLI uses this install's exports (nothing is shipped): pick the smallest one with
# real examples (3+ classes), so the walk-through never waits for a multi-GB model.
sys.path.insert(0, str(ROOT))
from toolkit.bundle_files import list_bundles  # noqa: E402

EXPORTS = list_bundles()
USABLE = [i for i, e in enumerate(EXPORTS) if len(e.manifest.get("examples", [])) >= 3]
if not USABLE:
    sys.exit("No usable export: train a model with 3+ classes in the web trainer and click "
             "Export first.")
EXPORT_CHOICE = str(min(USABLE, key=lambda i: EXPORTS[i].size) + 1)
env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", TOOLKIT_PORT=str(PORT))
proc = subprocess.Popen([sys.executable, "main.py"], cwd=ROOT, env=env, stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
buffer, lock, pos = [], threading.Lock(), 0


def reader():
    for chunk in iter(lambda: proc.stdout.read1(4096), b""):
        with lock:
            buffer.append(chunk.decode("utf-8", "replace"))


threading.Thread(target=reader, daemon=True).start()


def text():
    with lock:
        return "".join(buffer)


def expect(*patterns, timeout=120):
    """Wait for the first of the patterns after the last match; return its index."""
    global pos
    deadline = time.time() + timeout
    while time.time() < deadline:
        current = text()
        hits = [(m.end(), i) for i, p in enumerate(patterns)
                if (m := re.compile(re.escape(p)).search(current, pos))]
        if hits:
            end, index = min(hits)
            pos = end
            return index
        time.sleep(0.1)
    proc.kill()
    raise SystemExit(f"TIMEOUT waiting for {patterns!r}\n--- tail ---\n{text()[-1500:]}")


def send(line):
    proc.stdin.write((line + "\n").encode())
    proc.stdin.flush()
    with lock:
        buffer.append(f"«{line}»\n")  # mark what was typed in the transcript


def answer(prompt, line, timeout=120):
    expect(prompt, timeout=timeout)
    send(line)


checks = []


def web_check(label):
    """Wait for the web app, fetch both pages, then stop the server process."""
    expect(f"http://127.0.0.1:{PORT}/", timeout=120)
    for page in ("/", "/dynamic_ids"):
        for _ in range(60):
            try:
                status = urllib.request.urlopen(f"http://127.0.0.1:{PORT}{page}", timeout=5).status
                break
            except OSError:
                time.sleep(0.5)
        checks.append(f"{label}: GET {page} -> {status}")
    pids = _listening_pids(PORT)
    for pid in pids:
        if platform.system() == "Windows":
            subprocess.run(["taskkill", "/PID", pid, "/F"], capture_output=True)
        else:
            os.kill(int(pid), signal.SIGTERM)
    checks.append(f"{label}: server stopped (pid {', '.join(sorted(pids))})")


def _listening_pids(port):
    """PIDs listening on 127.0.0.1:port (netstat on Windows, lsof on Linux/macOS)."""
    if platform.system() == "Windows":
        out = subprocess.run(["netstat", "-ano", "-p", "TCP"], capture_output=True,
                             text=True).stdout
        return {line.split()[-1] for line in out.splitlines()
                if f"127.0.0.1:{port}" in line and "LISTENING" in line}
    out = subprocess.run(["lsof", "-ti", f"tcp:{port}", "-sTCP:LISTEN"], capture_output=True,
                         text=True).stdout
    return set(out.split())


def cli_session(label, full):
    answer("Choose an export (", EXPORT_CHOICE)
    answer("(or press Enter for ", "", timeout=180)
    if full:
        # two real examples from the model's test split, switch model, one more example
        for option, prompt, value in (("2", "Choose example (", "1"), ("2", "Choose example (", "2"),
                                      ("3", "(or press Enter for ", "2"),
                                      ("2", "Choose example (", "3")):
            answer("Select option: ", option)
            answer(prompt, value, timeout=180)
        answer("Select option: ", "1")  # manual input, prefilled from example 1
        answer("Start from example (", "")
        while expect("]: ", "PREDICTION RESULT") == 0:
            send("")
    answer("Select option: ", "5")
    expect("Goodbye!")
    checks.append(f"{label}: CLI IDS ran and exited normally")


# 1 -> recon on 127.0.0.1 (closed ports) -> "n" must return to the main menu (B15)
answer("Enter your choice [1-4]: ", "1")
answer("Enter target IP or domain", "127.0.0.1")
answer("Enter start port: ", "65500")
answer("Enter end port: ", "65502")
answer("Select protocol for scanning (TCP/UDP/both): ", "tcp")
answer("Enable verbose mode? (Y/N): ", "n")
if expect("scan this host anyway? (Y/N): ", "Scan Summary Report") == 0:
    send("y")
answer("Proceed to Intrusion Detection? (y/n): ", "n")
expect("Returning to Main Menu")
checks.append("Menu 1 (recon) -> 'n': back to main menu (B15 fixed)")

# 1 -> recon -> "y" -> intrusion menu: 1 CLI, repeat, 2 web, repeat, 3 back
answer("Enter your choice [1-4]: ", "1")
answer("Enter target IP or domain", "127.0.0.1")
answer("Enter start port: ", "65500")
answer("Enter end port: ", "65500")
answer("Select protocol for scanning (TCP/UDP/both): ", "both")
answer("Enable verbose mode? (Y/N): ", "y")
if expect("scan this host anyway? (Y/N): ", "Scan Summary Report") == 0:
    send("y")
answer("Proceed to Intrusion Detection? (y/n): ", "y")
answer("Enter your choice [1-3]: ", "1")
cli_session("Intrusion menu 1", full=True)
answer("another detection? (y/n): ", "y")
answer("Enter your choice [1-3]: ", "2")
web_check("Intrusion menu 2 (web)")
answer("another detection? (y/n): ", "y")
answer("Enter your choice [1-3]: ", "9")
expect("Invalid choice")
answer("Enter your choice [1-3]: ", "3")
checks.append("Intrusion menu: invalid choice handled; 3 returns to main menu")

# 2 -> CLI directly; 3 -> web directly; invalid; 4 exit
answer("Enter your choice [1-4]: ", "2")
cli_session("Main menu 2", full=False)
answer("another detection? (y/n): ", "maybe")
expect("Please enter 'y' or 'n'.")
send("n")
answer("Enter your choice [1-4]: ", "3")
web_check("Main menu 3 (web)")
answer("another detection? (y/n): ", "n")
answer("Enter your choice [1-4]: ", "x")
expect("Invalid choice")
answer("Enter your choice [1-4]: ", "4")
expect("Goodbye!")
code = proc.wait(30)
checks.append(f"Main menu 4: exit code {code}")

with open(TRANSCRIPT, "w", encoding="utf-8", newline="\n") as out:
    if TRANSCRIPT.endswith(".md"):
        out.write(f"# Menu walk-through (transcript)\n\nRecorded {time.strftime('%Y-%m-%d')} on "
                  f"{platform.system()} {platform.release()}, Python {platform.python_version()}, "
                  "by `python scripts/menu_walkthrough.py`. Text in «» is what the script typed."
                  "\n\n" + "\n".join(f"- {check}" for check in checks) + "\n\n```text\n"
                  + text().rstrip("\n") + "\n```\n")
    else:
        out.write(text())
print("\n".join(checks))
