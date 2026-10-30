"""Expect-style walk-through of every launcher menu path (release gate).

Runs ``python -m toolkit`` and ``python main.py`` with piped stdin/stdout, waits for each
prompt before answering (so the launcher never reads lines meant for the tool it starts) and
writes the whole session to a transcript. Local only: the recon target is 127.0.0.1 with
closed ports, so no AbuseIPDB / CVE / AI lookups are made.

The tools use ``TOOLKIT_INSTANCE_DIR`` when it is set; otherwise a new temporary folder, so
this install's ``instance/`` is never read. When that folder has no export with real examples,
one is made from the CIC-IDS2017 test fixture (15 classes) through the web trainer's own routes (in this process).

Usage:  python scripts/menu_walkthrough.py [transcript.md]
Exit code 1 (with the transcript tail) if a prompt doesn't appear in time.
"""

import os
import platform
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPT = sys.argv[1] if len(sys.argv) > 1 else "menu-walkthrough.md"
PORT = 5055
MENU = "Enter your choice [1-4]: "
REMOVED = ("Proceed to Intrusion Detection", "another detection", "Enter your choice [1-3]")

if not os.environ.get("TOOLKIT_INSTANCE_DIR", "").strip():
    os.environ["TOOLKIT_INSTANCE_DIR"] = tempfile.mkdtemp(prefix="toolkit-walkthrough-")
INSTANCE = os.environ["TOOLKIT_INSTANCE_DIR"]
sys.path.insert(0, str(ROOT))
from toolkit.core.services.ids.exports import list_bundles  # noqa: E402


def usable_exports():
    return [i for i, e in enumerate(list_bundles())
            if len(e.manifest.get("examples", [])) >= 3]


def make_export():
    """Train 2 models on the 15-class fixture and export them, as the page does (token
    included)."""
    from toolkit.web.controllers import create_app

    app = create_app()
    # the app answers only its own address (v5.2 step 2.4a): the test client names it
    app.config["SERVER_NAME"] = f"127.0.0.1:{app.extensions['toolkit'].config.port}"
    client = app.test_client()
    page = client.get("/dynamic_ids").get_data(as_text=True)
    headers = {"X-CSRF-Token": re.search(r'<meta name="csrf-token" content="([^"]+)">',
                                         page).group(1)}
    with open(ROOT / "tests" / "fixtures" / "cic2017_original_sample.csv", "rb") as data:
        uploaded = client.post("/api/v1/dataset", headers=headers,
                               data={"dataset": (data, "cic2017_original_sample.csv")},
                               content_type="multipart/form-data").get_json()
    target = next(c for c in uploaded["columns"] if c.strip() == "Label")
    job = client.post("/api/v1/jobs/training", headers=headers, json={
        "features": [c for c in uploaded["columns"] if c != target], "target": target,
        "balancing": "none", "test_size": 0.3,
        "models": ["decision_tree", "random_forest"]}).get_json()
    while (state := client.get(job["url"]).get_json())["state"] not in ("done", "failed",
                                                                         "cancelled"):
        time.sleep(0.2)
    if state["state"] != "done":
        sys.exit(f"Could not train the walk-through model: {state}")
    client.post("/api/v1/run/export", headers=headers, json={"name": "walkthrough"})


if not usable_exports():
    make_export()
USABLE = usable_exports()
if not USABLE:
    sys.exit("No usable export (3+ real examples) in " + INSTANCE)
EXPORT_CHOICE = str(min(USABLE, key=lambda i: list_bundles()[i].size) + 1)
ENV = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", TOOLKIT_PORT=str(PORT),
           TOOLKIT_NO_BROWSER="1")

buffer, lock, pos, proc = [], threading.Lock(), 0, None
checks = []


def start(*command):
    """Start the launcher (``python <command>``) and read its output in a thread."""
    global proc, pos
    with lock:
        buffer.append(f"\n$ python {' '.join(command)}\n")
        pos = len("".join(buffer))
    proc = subprocess.Popen([sys.executable, *command], cwd=ROOT, env=ENV,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    output = proc.stdout

    def reader():
        for chunk in iter(lambda: output.read1(4096), b""):
            with lock:
                buffer.append(chunk.decode("utf-8", "replace"))

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    return thread, pos


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


def finish(label, reader, since):
    """Wait for the launcher's end; check its code, one banner, no traceback, no old prompt."""
    code = proc.wait(30)
    reader.join(10)
    session = text()[since:]
    banners = session.count("Network Recon and Detection Tool")
    problems = [p for p in ("Traceback", *REMOVED) if p in session]
    checks.append(f"{label}: exit code {code}, banner shown {banners}x, menu shown "
                  f"{session.count('1. Network Recon')}x"
                  + (f", FOUND {problems}" if problems else ", no traceback, no old prompt"))
    if code != 0 or banners != 1 or problems:
        raise SystemExit(f"{label} failed: {checks[-1]}")


LOCAL = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # 127.0.0.1: no proxy


def fail(message):
    """Stop the launcher and end the walk-through with exit code 1 and the transcript tail."""
    proc.kill()
    raise SystemExit(f"{message}\n--- tail ---\n{text()[-1500:]}")


def web_check(label):
    """Wait for the web trainer, fetch its pages (the port scanner's and its options too, v5.2),
    then stop its process. A port already in
    use or a page that never answers is a failed check, so a page from another server that
    holds the port is never counted."""
    if expect(f"http://127.0.0.1:{PORT}/", "is already in use", timeout=120) == 1:
        fail(f"{label} failed: port {PORT} is already in use (another server holds it)")
    try:
        for page in ("/", "/dynamic_ids", "/scan", "/api/v1/scan/options"):
            status = None
            for _ in range(60):
                try:
                    status = LOCAL.open(f"http://127.0.0.1:{PORT}{page}", timeout=5).status
                    break
                except OSError:
                    time.sleep(0.5)
            checks.append(f"{label}: GET {page} -> {status}")
            if status != 200:
                fail(f"{label} failed: {checks[-1]}")
    finally:
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


def scan(label, end_port, protocol, verbose):
    """1: an interactive scan of closed ports on 127.0.0.1 (no export question)."""
    answer(MENU, "1")
    answer("Enter target IP", "127.0.0.1")
    answer("Enter start port: ", "65500")
    answer("Enter end port: ", end_port)
    answer("Select protocol for scanning (TCP/UDP/both): ", protocol)
    # step N4: the engine is asked only when nmap is installed; Enter keeps the built-in one
    if expect("Scan engine: the toolkit's own scanner or nmap?",
              "if you are authorised to scan") == 0:
        send("")
        expect("if you are authorised to scan")
    send("yes")  # S16
    answer("Enable verbose mode? (Y/N): ", verbose)
    if expect("scan this host anyway? (Y/N): ", "Scan Summary Report") == 0:
        send("y")
    expect("No open port results to export.")
    checks.append(f"{label}: scan of 127.0.0.1 ({protocol}) ended normally")


def cli_session(label, full):
    """2: the CLI IDS with the smallest usable export, then its Exit."""
    answer(MENU, "2")
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


# Each answer at MENU also checks that the menu came back after the tool before it.
# python -m toolkit: a wrong choice, 1 scan, 2 CLI (full), 3 web, q
reader, since = start("-m", "toolkit")
answer(MENU, "9")
expect("is not a choice")
scan("python -m toolkit, 1", "65502", "tcp", "n")
cli_session("python -m toolkit, 2", full=True)
answer(MENU, "3")
web_check("python -m toolkit, 3 (web)")
answer(MENU, "q")
expect("Goodbye!")
finish("python -m toolkit, q", reader, since)

# python main.py: 1 scan (both protocols, verbose), 2 CLI (short), 3 web, x, 4
reader, since = start("main.py")
scan("python main.py, 1", "65500", "both", "y")
cli_session("python main.py, 2", full=False)
answer(MENU, "3")
web_check("python main.py, 3 (web)")
answer(MENU, "x")
expect("is not a choice")
answer(MENU, "4")
expect("Goodbye!")
finish("python main.py, 4", reader, since)

# python -m toolkit: end of input at the menu
reader, since = start("-m", "toolkit")
expect(MENU)
proc.stdin.close()
expect("Goodbye!")
finish("python -m toolkit, end of input", reader, since)

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
