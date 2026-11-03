# Reference

Every command, flag, exit code and setting of the Network Security Toolkit (v5.1). The web
trainer's HTTP API is in [`web_api.md`](web_api.md); how the code is organised is in
[`architecture.md`](architecture.md). `tests/test_docs_reference.py` checks that every flag
and every `TOOLKIT_*` setting in the code is here, and that nothing here is missing from
the code.

## 1. Starting the toolkit

These three are the same program; the help and the examples write `python -m toolkit`:

| Form | Where |
|---|---|
| `python -m toolkit` | from the project folder (with the venv active) |
| `python main.py` | from the project folder |
| `toolkit` | anywhere, once installed with `pip install .` or `pip install -e .` (with that venv active) |

Without a command it shows the menu. With one, everything after the command goes to that
tool: `python -m toolkit scan --help`, `python -m toolkit ids --list`. `python -m toolkit
--help` lists the commands and names the data folder (exports, runs, signing key, logs).

## 2. The menu

| Choice | What it does |
|---|---|
| `1` | Network Recon: `scan`, asking its questions |
| `2` | Intrusion Detection (CLI): `ids`, choosing an export |
| `3` | Intrusion Detection (Web): `web`, until Ctrl+C |
| `4`, `q`, `quit`, `exit` | leaves the menu |

Each tool runs as its own process and the menu comes back when it ends, also after Ctrl+C
or an error in it. Ctrl+C at the menu itself says "Goodbye!" and leaves.

## 3. `scan`: the port scanner

`python -m toolkit scan [TARGET] [flags]`. Only scan systems you own or have written
permission to test.

- Without a `TARGET` it asks its questions: the target, the first and last port, the
  protocol, the engine (only when nmap is installed; for nmap its service detection, its
  scripts and, with raw packets, OS detection), the authorization, the progress line and
  the CVE lookups in NVD, the AI explanation (only with a key); during the scan, the
  AbuseIPDB check of a public IP (only with a key) and whether to scan a host that doesn't
  answer ping; at the end, the export. The questions of the outside services are yes on
  Enter (`(Y/n)`).
- With a `TARGET` it asks nothing but the authorization question. NVD, the AI and AbuseIPDB
  are on when their key is set (NVD needs none) unless `--no-nvd`, `--no-ai` or
  `--no-abuseipdb` turns them off (RD5). A service that fails during the scan is not asked
  again in it: the summary says `Unavailable: NVD (no connection)` and its ports
  `NOT CHECKED`.
- The authorization question needs the answer `yes`; `--i-am-authorized` answers it for
  scripts.

| Flag | Default | What it does |
|---|---|---|
| `TARGET` | (asked) | IPv4 or IPv6 address, network (`192.168.1.0/24`) or domain name |
| `-p`, `--ports SPEC` | `1-1024` | `80`, `1-1024` or a list such as `22,80,8000-8100` |
| `--protocol {tcp,udp,both}` | `tcp` | which protocols to check |
| `-v`, `--verbose` | off | the progress line while scanning |
| `--no-nvd` | NVD on | no CVE lookups (NVD gets the detected software name and version); every port with software says `NOT CHECKED` |
| `--no-ai` | on with the key | no AI explanation (OpenRouter gets the detected software names, never the raw banner; needs `OPENROUTER_API_KEY`) |
| `--no-abuseipdb` | on with the key | no reputation check (AbuseIPDB gets public target IPs only; needs `ABUSEIPDB_API_KEY`) |
| `--ai`, `--abuseipdb` | | the default now; still accepted so older commands work (refused together with its `--no-` form) |
| `--scan-unresponsive` | off | also scans hosts that don't answer ping |
| `--export {txt,json,csv}` | none | saves the report in `instance/scans/` (a new file, never overwritten) |
| `--i-am-authorized` | off | confirms you are authorised to scan the target (for scripts) |
| `--engine {builtin,nmap}` | `builtin` | the toolkit's own scanner, or nmap when it is installed (version 7.80 or newer); with `nmap` and no nmap found, the scan stops before anything is sent and says how to install it (never falls back to the built-in engine) |
| `--service-detection {light,normal,all}` | `normal` | nmap only: how hard `-sV` tries (`--version-intensity` 2, 7 or 9) |
| `--timing {polite,normal,aggressive}` | `normal` | nmap only: `-T2`, `-T3` or `-T4` |
| `--scripts {safe}` | none | nmap only: also runs the toolkit's 8 allowed scripts (`banner`, `http-title`, `http-server-header`, `ssl-cert`, `ssh-hostkey`, `smb-protocols`, `smb2-time`, `rdp-ntlm-info`) |
| `--os-detect` | off | nmap only: OS detection (`-O`); needs raw packets |

The flags other than `--i-am-authorized` work only with a `TARGET`, and the four nmap-only
flags only with `--engine nmap`. Limits: `TOOLKIT_SCAN_MAX_HOSTS` hosts and
`TOOLKIT_SCAN_MAX_PROBES` host×port checks (TCP and UDP count separately); nmap also stops
after `TOOLKIT_SCAN_MAX_MIN` minutes per host.

With `--engine nmap`:

- nmap is run by its full path (`TOOLKIT_NMAP_PATH`, else the folders on `PATH`, else its
  install folder), never through a shell, with only the options above (`-n -Pn`, no DNS and
  no ping of its own).
- nmap's service detection (`-sV`) also runs nmap's own version scripts (48 in nmap 7.992,
  such as `rpcinfo`), each only on a service it recognised; nmap can't turn them off. One,
  `iec61850-mms`, is in nmap's "intrusive" category: it talks to IEC 61850 industrial devices
  on port 102. The scan's opening notice says so every time.
- Raw packets (the SYN scan, UDP, `-O`): on Windows Npcap usable by this user (it comes with
  nmap's installer; an administrator's prompt if Npcap is set to administrators only), on
  Linux and macOS root. Without them nmap does a TCP connect scan, and UDP or `--os-detect`
  stops before anything is sent.
- What is looked up in NVD is nmap's product and version, cleaned with the same rules as a
  banner (for example `OpenSSH for_Windows_8.1` → `OpenSSH 8.1`, `Apache httpd 2.4.49` →
  `Apache 2.4.49`); the report also carries the engine and nmap's CPE names (not looked
  up), and nmap's OS match joins the OS hint.

```
python -m toolkit scan 192.168.1.10 -p 1-1024 --protocol both --export csv
python -m toolkit scan 127.0.0.1 -p 1-1024 --engine nmap --scripts safe
```

### 3.1 The record of every scan

Every scan, from `scan` or from the web page, writes one line to its tool's log when it
starts and one when its scanning ends (HS3): `instance/logs/scanner.log` for `scan`,
`web.log` for the web page. Nothing is written for a scan that was refused before it started
(no authorization, a bad target, a limit). The values are JSON, in ASCII, so a target typed
with a line break or quotes stays inside its own line:

```
2026-11-03T10:00:01 INFO toolkit.core.services.recon.record: Scan started {"scan_id": "k2Pq9xAb", "source": "192.168.56.20", "target": "10.0.0.0/30", "addresses": ["10.0.0.1", "10.0.0.2"], "ports": "22,80-82,443", "protocols": ["tcp"], "engine": "builtin", "online": {"nvd": "on, the software name and version only", "ai": "off: not asked for (--ai)", "abuseipdb": "off: no public address (a private one is never sent)"}}
2026-11-03T10:00:09 INFO toolkit.core.services.recon.record: Scan ended {"scan_id": "k2Pq9xAb", "outcome": "done", "open_ports": 3, "engine_failed": 0, "seconds": 8.1}
```

- `source`: `this computer` (always for `scan`), or the address of the other computer whose
  request it was, when `web --host` opened the toolkit to the network. An address, not a
  person: there are no user accounts (RD1).
- `addresses`: what the target was resolved to; `ports` as ranges; `online`: what the scan
  may send to NVD, the AI and AbuseIPDB, as its opening notice says.
- `outcome`: `done`, `stopped` (Cancel, the time limit or Ctrl+C) or `failed` (an error
  nobody expected); `engine_failed`: targets the engine could not scan; `open_ports` is 0
  when the scan didn't end `done`. The same `scan_id` joins the two lines.
- No API key or access token is written.
- How far back: each log rotates at 1 MB with three older files, the oldest deleted (about
  4 MB per tool; a scan of 256 addresses writes a start line of about 4 KB).
  `TOOLKIT_LOG_LEVEL` above `INFO` leaves the lines out.
- The targets and ports say what was scanned: treat the logs as sensitive, like the rest of
  `instance/` (which other Windows accounts can read is task HS5).

## 4. `ids`: intrusion detection with an export

`python -m toolkit ids [flags]`, with a model bundle exported by the web trainer on this
computer (bundles are signed with this install's key; one from another computer is
refused).

- Without `--csv`, `--json` or `--info`: it asks which export to use (Enter: the newest)
  and which model, then offers: 1 enter features manually, 2 test with a real example
  from the model's test split, 3 switch model, 4 predict a CSV file, 5 exit.
- With `--csv`, `--json` or `--info` it asks nothing.
- `--runs` and `--compare` read the saved runs (`instance/runs`, one record per export: its
  settings and scores) and open no model; they exit with 2 and the message for a bad run id
  or a run that isn't saved. The importance results beside the records
  (`<run id>.importance.<model id>.json`, HR2) are not runs and are not counted.
- `--delete RUN_ID` (HR4a) names the run's files and their sizes, then asks
  `Delete these files? [y/N]`; only `y` or `yes` deletes, anything else (or the end of the
  input) deletes nothing. Its bundle is kept unless `--with-bundle` is given. A file that
  can't be removed exits with 2 and its name; the run is kept and can be deleted again.
- `--storage` (HR4b) shows what the toolkit keeps on this computer, by kind, with the total
  and the free space, and names the leftovers; it deletes nothing. `--register RUN_ID`
  brings back a run deleted with its bundle kept, from the record the bundle carries, once
  its signature is checked; it exits with 2 and why for no bundle, a changed or incomplete
  one, or a run that is already saved.

| Flag | Default | What it does |
|---|---|---|
| `--bundle BUNDLE` | ask (without a prompt: the newest whole export) | the `.bundle` file to use |
| `--list` | | lists the exports (newest first) without loading them, then exits; a bundle that is only part of its file is listed with `[INCOMPLETE: the export was interrupted; ...]` and never removed on its own |
| `--runs` | | lists the saved runs, newest first and 50 at a time, then exits |
| `--before RUN_ID` | | with `--runs`: the runs saved before this one (the list's last line names it) |
| `--compare RUN_ID RUN_ID` | | the two saved runs A and B side by side, then exits: each model of both, with Δ = B − A only when both scored the same test rows (otherwise why not), the models of one run only and the settings that differ |
| `--delete RUN_ID` | | deletes a saved run after naming its files and asking: its importance results, then its bundle with `--with-bundle`, its record last |
| `--with-bundle` | the bundle is kept | with `--delete`: delete the run's bundle too |
| `--storage` | | shows what is kept, by kind, the total and the free space, and the leftovers by name, then exits; it deletes nothing |
| `--register RUN_ID` | | brings back a run deleted with its bundle kept (its signature checked first), then exits; it never replaces a record |
| `--model MODEL` | the bundle's best | which model of the bundle |
| `--csv CSV` | | labels every row of this CSV file |
| `--output OUTPUT` | `<file>.predictions.csv` | where `--csv` writes (a new file, never overwritten) |
| `--json JSON` | | rows as JSON (an object or a list, inline or a `.json` file); prints the results as JSON |
| `--info` | | shows the model card, then exits |

```
python -m toolkit ids --list
python -m toolkit ids --csv traffic.csv --model random_forest
python -m toolkit ids --json rows.json
python -m toolkit ids --runs
python -m toolkit ids --compare 20261030T120509Z-a75da585 20261031T090000Z-1738e0a0
python -m toolkit ids --delete 20261030T120509Z-a75da585
python -m toolkit ids --storage
python -m toolkit ids --register 20261030T120509Z-a75da585
```

A bundle above 500 MB says first that loading takes a while: large models can take minutes
and need several times their size in memory.

## 5. `web`: the web trainer and the port scanner

`python -m toolkit web [flags]`: serves the trainer (`/dynamic_ids`), the port scanner
(`/scan`) and the saved runs (`/runs`: the trainings' records, one shown or two
compared; it reads them only) on this computer only (`127.0.0.1`) unless `--host` opens them to the network,
prints the address and opens it in the default browser once the port answers. Ctrl+C
stops it ("Web trainer stopped.").

| Flag | Default | What it does |
|---|---|---|
| `--port PORT` | `TOOLKIT_PORT`, else `5000` | the port (1-65535) |
| `--no-browser` | off | doesn't open the browser (also `TOOLKIT_NO_BROWSER=1`) |
| `--host ADDRESS` | `127.0.0.1` | also listens on this IP address of this computer (`0.0.0.0`: all of its IPv4 addresses), so other computers on your network can use the toolkit: see below. A host name is refused. No environment variable sets it: it is chosen at every start |

**Opened to the network (`--host`).** Only on a network you trust, such as your virtual
machines' host-only network: anyone who can reach the toolkit and has its access token can
upload data, train models and run port scans **from this computer**, and the connection is
plain HTTP, so someone who can read the network's traffic could take over a session.

- At the start the terminal prints a warning, the addresses other computers open and an
  **access token** such as `K7QX-M2PD-9RTA-W4HE`: 16 letters and digits (no 0, O, 1 or I),
  new at every start, shown only there (never in a log). Every page shows a banner while
  the toolkit is open to the network.
- Another computer's browser is sent to `/unlock`, where the token is typed once (spaces,
  dashes and case don't matter); its session stays unlocked until the toolkit stops or a new
  token is made. The API answers it `LOCKED` (401) before that. Five wrong tokens from one address within a
  minute are refused with `TOO_MANY_TRIES` (429) for the rest of the minute; each wrong
  try is logged with its address. The token is never in an address (URL).
- This computer's own browser, at `127.0.0.1` (which stays bound), never needs the token.
  The address of a request is its connection's: `X-Forwarded-For` is never trusted.
- A holder of the token has the owner's rights over the saved runs, bundles and scans, not
  only a view of the pages: there are no user accounts (RD1). The terminal and the unlock
  page say so.
- **New token** (HS1), in the **Other computers** section of the home page on this computer
  only: a new token, printed in the terminal, and every other computer is locked out at once,
  without stopping the toolkit. A job one of them asked for that still waits is cancelled and
  never starts; a running one goes on, and the section lists it with **Stop**; this
  computer's jobs are not touched. The routes are in [web_api.md](web_api.md).
- `TOOLKIT_DEBUG=1` is refused with `--host` (exit code 2): the debug server's debugger can
  run any code on this computer.
- An address this computer doesn't have is a message and exit code 2. On Windows the
  firewall may ask, the first time, whether to let Python accept connections.

A port already in use is a message naming the next port to try. One web server runs per
data folder (HR3b): a second one on the same `instance/` stops at once with a message naming
the first one's process and address (exit code 2); give it its own folder with
`TOOLKIT_INSTANCE_DIR`. The lock is `instance/web.lock`, released by the system when the
server ends, however it ends; at its start the server removes the temporary files a stopped
export left in `instance/runs/` and `instance/bundles/`. The command line takes no lock: it
writes neither. Each job's ask, start and end are a line in `instance/jobs.journal` (HR3c): at
the next start the server tells every job the last one cut off, in the terminal and once in
each browser session on that kind's page (`INTERRUPTED: a training asked for at ... was cut
off while it ran ...`), then empties the journal. A line cut off by a crash counts for
nothing. Its settings are
`TOOLKIT_*` variables (section 7); the pages and API are in [`web_api.md`](web_api.md).

## 6. Exit codes

| Code | `scan` | `ids` | `web` | the menu, `toolkit` |
|---|---|---|---|---|
| 0 | the scan ended (also with nothing found) | done; also Ctrl+C or "5. Exit" | stopped with Ctrl+C | left the menu; `--help` |
| 1 | nmap could not scan a target (missing or failed; the other targets were still scanned) | | | |
| 2 | a bad flag or setting, a bad target or port list, over a limit, not authorised, the report not saved for lack of disk space (`TOOLKIT_MIN_FREE_MB`) | a bad flag or setting, no export, a bundle refused (signature, format), an unknown model, a missing or unreadable file, an output file that exists, a run not deleted or not registered, a record not saved for lack of disk space | a bad flag or setting, a bad port, the port in use, an address this computer doesn't have, `TOOLKIT_DEBUG=1` with `--host` | an unknown command |
| 130 | cancelled (Ctrl+C, or the end of input at a question) | | | |

`--help` exits with 0 everywhere. A bad flag is reported by the command's own parser with
its usage line. Every bad setting is one line naming it, before the tool starts (section
7.1).

## 7. Settings

### 7.1 `TOOLKIT_*` settings

Read from the environment only (not from `.env`). Each command checks the settings it
reads before it starts. A bad value prints one line naming the setting, for example
`TOOLKIT_MAX_UPLOAD_MB must be a whole number >= 1, not 'abc'`, and the command exits with
code 2. A setting only another tool reads is not checked, so a typo in a web setting
doesn't stop a scan, and `--help` always works.

- **whole number:** 1 or more.
- **on/off:** `1` is on; `0` or unset is off. Anything else (`true`, `yes`) is an error.

| Setting | Read by | Allowed | Default | What it does |
|---|---|---|---|---|
| `TOOLKIT_LOG_LEVEL` | all | `DEBUG`, `INFO`, `WARNING`, `ERROR` (any case) | `INFO` | how much the log files in `instance/logs/` get |
| `TOOLKIT_STORAGE_WARN_MB` | all | whole number | `20480` (20 GB) | past this size of `instance/`, the storage report (Saved Runs, `ids --storage`) and the web server's log at its start warn; nothing is deleted |
| `TOOLKIT_MIN_FREE_MB` | all | whole number | `1024` (1 GB) | a new file (a record, an importance result, a bundle, a scan report) is refused when the disk would keep less free space than this: "Not saved: only … MB free …"; nothing is deleted |
| `TOOLKIT_INSTANCE_DIR` | all | a folder | `instance/` in the project folder (installed: section 7.3) | where exports, runs, the signing key, scans and logs are kept |
| `TOOLKIT_SCAN_MAX_HOSTS` | `scan` | whole number | `256` | hosts in one scan (one /24 network) |
| `TOOLKIT_SCAN_MAX_PROBES` | `scan` | whole number | `65536` | host×port checks in one scan |
| `TOOLKIT_SCAN_MAX_MIN` | `scan` | whole number | `60` | minutes the nmap engine may spend on one host (nmap's `--host-timeout`) |
| `TOOLKIT_NMAP_PATH` | `scan` | a full path (nmap or its folder) | `PATH`'s folders, then where nmap's installer puts it; never the current folder | the nmap the nmap engine runs |
| `TOOLKIT_PORT` | `web` | 1-65535 | `5000` | the web trainer's port (`--port` wins) |
| `TOOLKIT_NO_BROWSER` | `web` | on/off | off | `1`: don't open the browser |
| `TOOLKIT_MAX_UPLOAD_MB` | `web` | whole number | `2048` | the largest dataset upload |
| `TOOLKIT_MAX_PREDICT_MB` | `web` | whole number | `512` | the largest file for "Predict a CSV file" (read into memory) |
| `TOOLKIT_MAX_WORKSPACES` | `web` | whole number | `4` | browser sessions with a dataset or bundle kept at once |
| `TOOLKIT_MAX_DATA_MB` | `web` | whole number | `4096` | loaded datasets of all sessions together (models not counted) |
| `TOOLKIT_WORKSPACE_IDLE_MIN` | `web` | whole number | `240` | minutes before an unused session's data is dropped (never while it trains) |
| `TOOLKIT_JOB_MAX_MIN` | `web` | whole number | `180` | minutes before a training is stopped; nothing is kept |
| `TOOLKIT_MAX_WORKERS` | `web` | whole number | all CPU cores but one | CPU workers for training (never more than the machine has) |
| `TOOLKIT_HTTPS` | `web` | on/off | off | `1`: the session cookie is marked `Secure` (only behind HTTPS) |
| `TOOLKIT_DEBUG` | `web` | on/off | off | `1`: Flask's debug server, development only (its debugger can run code) |
| `TOOLKIT_SECRET_KEY` | `web` | any text | a new random key at each start | signs the session cookie; with a fixed key the cookie stays valid after a restart (the session's data, kept in memory, does not) |

### 7.2 API keys and the AI model

The scanner's optional services. Each is read from the environment, else from `.env` in
the project folder (an installed copy: its data folder, section 7.3); the environment wins.
Copy `.env.example` to `.env` and fill in what you use; `.env` is gitignored. Values left
as `your_..._here` count as unset. Keys never reach the log files.

| Variable | Default | What it does |
|---|---|---|
| `NVD_API_KEY` | none | faster CVE lookups (without one, NVD allows about one every 6 seconds) |
| `OPENROUTER_API_KEY` | none | the AI explanation (on with the key, unless `--no-ai` or no to the question) |
| `OPENROUTER_MODEL` | `deepseek/deepseek-chat:free` | which OpenRouter model explains |
| `ABUSEIPDB_API_KEY` | none | the IP reputation check (on with the key, unless `--no-abuseipdb` or no to the question) |

### 7.3 Where the data folder is

From the project folder (a clone, or `pip install -e .`): `instance/` and `.env` there.
An installed copy (`pip install .`) uses a per-user folder named `network-security-toolkit`:

| System | Folder |
|---|---|
| Windows | `%LOCALAPPDATA%\network-security-toolkit` (without `LOCALAPPDATA`: `~\AppData\Local\network-security-toolkit`) |
| macOS | `~/Library/Application Support/network-security-toolkit` |
| Linux and others | `$XDG_DATA_HOME/network-security-toolkit` when `XDG_DATA_HOME` is an absolute path, else `~/.local/share/network-security-toolkit` |

`TOOLKIT_INSTANCE_DIR` replaces the `instance/` folder in every case; `.env` stays where
it was. `python -m toolkit --help` prints the folder in use. Back up `signing.key` in it:
without it, earlier exports can't be loaded. What each file is: [`architecture.md`,
section 9](architecture.md#9-data-on-disk-and-what-leaves-the-computer).

### 7.4 Setting a variable

For the commands started from that window:

| Shell | Example |
|---|---|
| PowerShell (Windows) | `$env:TOOLKIT_PORT = "5001"`, then `python -m toolkit web`; unset with `Remove-Item Env:TOOLKIT_PORT` |
| cmd (Windows) | `set TOOLKIT_PORT=5001`, then `python -m toolkit web`; unset with `set TOOLKIT_PORT=` |
| macOS, Linux | `TOOLKIT_PORT=5001 python -m toolkit web` (one command), or `export TOOLKIT_PORT=5001`; unset with `unset TOOLKIT_PORT` |

## 8. Web API

The routes, requests, answers, error codes and limits of `/api/v1`: [`web_api.md`](web_api.md).
