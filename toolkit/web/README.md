# `toolkit/web/`: the web trainer

Upload a dataset, train and compare models, export a signed bundle, predict with it
(`python -m toolkit web`). It uses [`core/`](../core/README.md) and never imports
[`cli/`](../cli/README.md).

```
web/
  ui/
    templates/       the pages (home.html, dynamic_ids.html, scan.html, runs.html; unlock.html
                     for other computers when --host opens the toolkit to the network)
    static/          css/app.css (one stylesheet, colours as tokens) and js/ (ES modules;
                     api.js holds the only fetch)
    console.py       what the web command prints in the terminal
  controllers/
    command.py       the web command: its settings, --port, --no-browser, then the server
    __init__.py      create_app(): the Flask app with its own state
    server.py        waitress on 127.0.0.1, the port bound exclusively, the browser opened
    pages.py         the HTML pages: home, the trainer, the port scanner, the saved runs
    access.py        the unlock page and route: other computers type the access token
    datasets.py      upload, preview, class filter, the analyses before training
    training.py      the model list, a training job started, its results
    jobs.py          any job's state, progress and cancel (a scan's log too)
    scan.py          the port scan: its options, a scan started, the report download
    bundles.py       export, download and load bundles; predictions
    resume.py        GET /api/v1/workspace, so a reload resumes
    runs.py          the saved runs: the list, one run, two side by side (records only)
    workspace.py     the calling session's workspace, locked for the request
    state.py         what one app keeps between requests (no module globals)
    security.py      the security headers and CSP, the CSRF token, the cross-site check
    schemas.py       one small request schema per route
    api.py           request and error helpers; the HTTP status of each ServiceError
  services/
    workspace.py     one in-memory workspace per browser session, with memory caps
    jobs.py          the background jobs: a queue per kind (training, scan), progress, cancel
    datasets.py      uploads read, files combined, the class filter, the training split
    training.py      training settings checked, the job queued, the run kept
    scan.py          the port scan: its choices checked (the authorization first), the job,
                     the last scan kept for the download
    scan_view.py     what the page sees of a running scan: its log, progress and answer
    access.py        the access token of a start (--host): made, checked, wrong tries counted
    prediction.py    one row or a whole file predicted
    errors.py        the error codes and ServiceError
  data/
    uploads.py       uploads on disk, only while they are read
    net.py           the listening socket, and opening the page in the browser
    instance_lock.py one web server per instance (HR3b)
```

Rules that matter here:

- The page loads only its own files: the CSP allows `script-src 'self'` and
  `style-src 'self'`, so no inline script, style or event handler
  (`tests/security/test_headers.py`).
- Every change to the server's state sends the page's CSRF token and passes the
  cross-site check (`tests/security/test_csrf.py`).
- The server listens on 127.0.0.1 only.
- The `/api/v1` routes are exactly the ones in [docs/web_api.md](../../docs/web_api.md)
  (`tests/web/test_api_routes.py`).
- Only `controllers/` uses Flask (and only `command.py` argparse); only `ui/` draws or
  prints; `services/` holds the work and answers with a `ServiceError` code, never an HTTP
  status.

Where a change goes: [docs/architecture.md, section 11](../../docs/architecture.md#11-where-to-make-a-change).
