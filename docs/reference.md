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
  protocol, the authorization, the progress line and the AI explanation (only with a key);
  during the scan, the AbuseIPDB check of a public IP (only with a key) and whether to scan
  a host that doesn't answer ping; at the end, the export.
- With a `TARGET` it asks nothing but the authorization question, and sends nothing online
  but the CVE lookups unless `--ai` or `--abuseipdb` says so.
- The authorization question needs the answer `yes`; `--i-am-authorized` answers it for
  scripts.

| Flag | Default | What it does |
|---|---|---|
| `TARGET` | (asked) | IPv4 or IPv6 address, network (`192.168.1.0/24`) or domain name |
| `-p`, `--ports SPEC` | `1-1024` | `80`, `1-1024` or a list such as `22,80,8000-8100` |
| `--protocol {tcp,udp,both}` | `tcp` | which protocols to check |
| `-v`, `--verbose` | off | the progress line while scanning |
| `--ai` | off | sends the detected software names (never the raw banner) to OpenRouter for an explanation; needs `OPENROUTER_API_KEY` |
| `--abuseipdb` | off | sends public target IPs to AbuseIPDB for a reputation check; needs `ABUSEIPDB_API_KEY` |
| `--scan-unresponsive` | off | also scans hosts that don't answer ping |
| `--export {txt,json,csv}` | none | saves the report in `instance/scans/` (a new file, never overwritten) |
| `--i-am-authorized` | off | confirms you are authorised to scan the target (for scripts) |

The flags other than `--i-am-authorized` work only with a `TARGET`. Limits:
`TOOLKIT_SCAN_MAX_HOSTS` hosts and `TOOLKIT_SCAN_MAX_PROBES` host×port checks (TCP and UDP
count separately).

```
python -m toolkit scan 192.168.1.10 -p 1-1024 --protocol both --export csv
```

## 4. `ids`: intrusion detection with an export

`python -m toolkit ids [flags]`, with a model bundle exported by the web trainer on this
computer (bundles are signed with this install's key; one from another computer is
refused).

- Without `--csv`, `--json` or `--info`: it asks which export to use (Enter: the newest)
  and which model, then offers: 1 enter features manually, 2 test with a real example
  from the model's test split, 3 switch model, 4 predict a CSV file, 5 exit.
- With `--csv`, `--json` or `--info` it asks nothing.

| Flag | Default | What it does |
|---|---|---|
| `--bundle BUNDLE` | ask (without a prompt: the newest export) | the `.bundle` file to use |
| `--list` | | lists the exports (newest first) without loading them, then exits |
| `--model MODEL` | the bundle's best | which model of the bundle |
| `--csv CSV` | | labels every row of this CSV file |
| `--output OUTPUT` | `<file>.predictions.csv` | where `--csv` writes (a new file, never overwritten) |
| `--json JSON` | | rows as JSON (an object or a list, inline or a `.json` file); prints the results as JSON |
| `--info` | | shows the model card, then exits |

```
python -m toolkit ids --list
python -m toolkit ids --csv traffic.csv --model random_forest
python -m toolkit ids --json rows.json
```

A bundle above 500 MB says first that loading takes a while: large models can take minutes
and need several times their size in memory.

## 5. `web`: the web trainer

`python -m toolkit web [flags]`: serves the trainer on this computer only (`127.0.0.1`),
prints its address and opens it in the default browser once the port answers. Ctrl+C
stops it ("Web trainer stopped.").

| Flag | Default | What it does |
|---|---|---|
| `--port PORT` | `TOOLKIT_PORT`, else `5000` | the port (1-65535) |
| `--no-browser` | off | doesn't open the browser (also `TOOLKIT_NO_BROWSER=1`) |

A port already in use is a message naming the next port to try. Its settings are
`TOOLKIT_*` variables (section 7); the pages and API are in [`web_api.md`](web_api.md).

## 6. Exit codes

| Code | `scan` | `ids` | `web` | the menu, `toolkit` |
|---|---|---|---|---|
| 0 | the scan ended (also with nothing found) | done; also Ctrl+C or "5. Exit" | stopped with Ctrl+C | left the menu; `--help` |
| 2 | a bad flag or setting, a bad target or port list, over a limit, not authorised | a bad flag or setting, no export, a bundle refused (signature, format), an unknown model, a missing or unreadable file, an output file that exists | a bad flag or setting, a bad port, the port in use | an unknown command |
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
| `OPENROUTER_API_KEY` | none | the AI explanation (`--ai`, or yes to the question) |
| `OPENROUTER_MODEL` | `deepseek/deepseek-chat:free` | which OpenRouter model explains |
| `ABUSEIPDB_API_KEY` | none | the IP reputation check (`--abuseipdb`) |

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
