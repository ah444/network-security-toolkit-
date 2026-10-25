# Web trainer API (`/api/v1`)

The web trainer's page (`/dynamic_ids`) talks to the server only through these routes.
They work on the calling browser session's **workspace** (its dataset, trained run and
loaded bundle); the session cookie holds only the workspace ID. A workspace is created
only by an upload or a bundle load; other routes answer `NO_DATASET`, `NOT_TRAINED`, ...
without creating one. The requests of one session take turns; during a training the
session can still read and predict, but not change its dataset. Everything runs on this
computer (`127.0.0.1`).

- **Requests:** `POST` bodies are JSON objects (`Content-Type: application/json`), except
  the file uploads (`multipart/form-data`). `GET` routes read the query string.
- **Limits:** lists hold at most 10,000 items, texts at most 1,000 characters; `rows`,
  `k` and `test_size` are moved into their range instead of being refused. Uploads are
  limited by `TOOLKIT_MAX_UPLOAD_MB` (default 2,048 MB), a file to predict by
  `TOOLKIT_MAX_PREDICT_MB` (default 512 MB; `TOO_LARGE`, 413).
- **Memory:** at most `TOOLKIT_MAX_WORKSPACES` workspaces (default 4) and
  `TOOLKIT_MAX_DATA_MB` of loaded datasets in all of them together (default 4,096 MB;
  trained models are not counted). When a new workspace or dataset doesn't fit, other
  sessions' workspaces are dropped, least recently used first, but never one that is
  training; if that isn't enough, the request is refused. Workspaces unused for
  `TOOLKIT_WORKSPACE_IDLE_MIN` minutes (default 240) are dropped, checked every 5 minutes.
- **Security token:** every `POST`/`DELETE` must send the session's token in an
  `X-CSRF-Token` header, otherwise it is refused (`CSRF_TOKEN`, 403). See below.
- **Cross-site requests:** a `POST`/`DELETE` whose `Origin` header names another site is
  refused (`CROSS_SITE`, 403).
- **Caching:** every API answer (the bundle download too) and the trainer page are sent
  with `Cache-Control: no-store`, so datasets, results and models never land in a cache.
  Every response also carries `Cross-Origin-Opener-Policy` and
  `Cross-Origin-Resource-Policy: same-origin` and a `Permissions-Policy` that turns off
  camera, microphone, location, payment, USB and motion sensors.

## Routes

| Method | Route | Request | Answer |
|---|---|---|---|
| POST | `/api/v1/dataset` | multipart `dataset`: one or more `.csv`/`.txt` files | columns, detected profile, memory used, message |
| GET | `/api/v1/dataset/preview` | `rows` (1-100, default 10) | first rows, column stats |
| GET | `/api/v1/dataset/classes` | `target` | class values with counts |
| GET | `/api/v1/dataset/distribution` | `target`, `test_size` (0.1-0.5, default 0.2) | class counts, balancing recommendation (training split) |
| POST | `/api/v1/dataset/filter` | `target`, `remove`: classes to drop | rows removed, classes left |
| POST | `/api/v1/dataset/balancing-preview` | `target`, `balancing`, `features`, `test_size` | class counts before/after, on the training split only |
| POST | `/api/v1/dataset/feature-scores` | `target`, `k` (1-10,000, default 10), `test_size` | feature scores on the training split (`"inf"` for a feature with one value per class: JSON has no infinity) |
| GET | `/api/v1/models` | none | the models that can be trained here |
| POST | `/api/v1/jobs/training` | `target`, `features`, `models`, `balancing`, `test_size` | **202** and the job (see below) |
| GET | `/api/v1/jobs/<job_id>` | none | the job's state and progress; when done, the result: leaderboard, split, CV, leakage findings, confusion matrices |
| DELETE | `/api/v1/jobs/<job_id>` | none | **202**: a queued job is cancelled at once, a running one at its next step |
| GET | `/api/v1/run/confusion` | `model` (default: the selected one) | test-split confusion matrix |
| POST | `/api/v1/run/predict` | `values`: {feature: value}, `model` | prediction |
| POST | `/api/v1/run/predict-file` | multipart `file` (`.csv`/`.txt`), `model` | every row predicted: counts per class, matches, download (see below) |
| POST | `/api/v1/run/export` | `name` (letters, digits, `_`, `-`; max 80) | signed bundle saved in `instance/bundles/` |
| GET | `/api/v1/run/bundle` | none | the exported bundle as a download |
| POST | `/api/v1/bundle` | multipart `file`: a `.bundle` exported by this install | the loaded models and their scores |
| POST | `/api/v1/bundle/predict` | `values`, `model` | prediction with the loaded bundle |
| POST | `/api/v1/bundle/predict-file` | multipart `file`, `model` | the same with the loaded bundle |
| GET | `/api/v1/predictions` | none | the last predicted file's labelled CSV as a download (404 `NOT_FOUND` before any) |
| GET | `/api/v1/workspace` | none | what the session holds, to resume after a reload (see below) |
| DELETE | `/api/v1/workspace` | none | the session's dataset and models are forgotten |

## Batch prediction (v5.1 step 1.4f)

`POST /api/v1/run/predict-file` and `/api/v1/bundle/predict-file` label every row of an
uploaded `.csv`/`.txt` file with `model` (default: the run's selected model). The file is
read like a dataset upload and deleted at once; it needs the model's feature columns
(otherwise `UNKNOWN_COLUMN`, 400, naming them), and extra columns are ignored. The answer:

```json
{"message": "2,000 rows predicted with Random Forest.", "source": "run", "model": "random_forest",
 "file": "flows.csv", "rows": 2000,
 "counts": [{"label": "BENIGN", "count": 1600, "share": 0.8}, ...],
 "matches": {"matching": 1995, "rows": 2000, "share": 0.9975}, "label_column": "Label",
 "download_url": "/api/v1/predictions", "download_name": "flows.predictions.csv"}
```

`counts` are largest first; `matches` and `label_column` are null when the file has no
label column. `GET /api/v1/predictions` downloads the labelled CSV with the columns of
`python -m toolkit ids --csv`: `row, prediction, confidence[, true_label]` (both are made
by `toolkit.core.services.ids.batch`). Only the last result is kept, in the session's workspace;
`GET /api/v1/workspace` repeats its answer as `predictions`.

## Security token

Each browser session has one random token, kept in the signed session cookie and shown
in the trainer page as `<meta name="csrf-token" content="...">`. The page sends it with
every `POST` and `DELETE`; another site can't read it, so it can't make the browser change
anything here (CSRF). The token stays the same for the whole session (tabs of one browser
share it, clearing the workspace keeps it). After a server restart the old page's token
no longer matches: the answer is `CSRF_TOKEN` with "Reload the page".

A script on this computer works the same way: keep the cookies, `GET /dynamic_ids`, read
the token from the meta tag, and send it in the `X-CSRF-Token` header. `GET` routes need
no token.

## Resuming after a reload

`GET /api/v1/workspace` answers what the session's workspace holds, each part in the same
shape its own route answered, or `null`:

```json
{"dataset": {"columns": [], "message": "...", "dataset": {}},
 "run": {"run_id": "...", "leaderboard": [], "confusion_matrices": {}},
 "job": {"id": "...", "url": "/api/v1/jobs/...", "state": "running"},
 "export": {"download_url": "/api/v1/run/bundle", "download_name": "..."},
 "bundle": {"run_id": "...", "models": [], "features_info": []},
 "predictions": {"source": "run", "rows": 2000, "counts": [], "download_url": "/api/v1/predictions"}}
```

- `dataset`: as `POST /api/v1/dataset` answered (after a class filter: the rows left).
- `run`: the last finished training, as its job's `result`.
- `job`: a training that is still `queued` or `running` (as `GET /api/v1/jobs/<id>`).
- `export`: as `POST /api/v1/run/export` answered, once the run was exported.
- `bundle`: as `POST /api/v1/bundle` answered.
- `predictions`: as the last `POST /api/v1/run/predict-file` or `/api/v1/bundle/predict-file`
  answered (its `source` says which); the labelled CSV is still at `/api/v1/predictions`.

A session without a workspace gets all `null`s; nothing is created. The page uses this
route on every load; results of the explore routes (distribution, previews, scores) are
not kept and are asked for again.

## Training jobs

A training runs in the background as a **job**. The server runs one job at a time (each
uses `TOOLKIT_MAX_WORKERS` cores); others wait in a queue. While a session's job waits or
runs, that session can read and predict, but not change its dataset or start a second
job (`BUSY`). A job is visible only to its own session (another session's job id answers
`NOT_FOUND`); each session keeps its last 5 jobs.

```json
{"id": "...", "url": "/api/v1/jobs/...", "state": "running", "done": 7, "total": 22,
 "percent": 32, "phase": "Cross-validation, fold 2 of 3", "elapsed_seconds": 72.4}
```

- `state`: `queued` (with `queue_position`: jobs that run first; at 0 the phase is
  `Starting`, as the worker is about to pick it up), `running`, `cancelling`
  (cancel asked, waiting for the current model's fit to end), `done` (with `result`),
  `failed` or `cancelled` (with `error`: `{"code", "message"}`).
- Steps: 2 (split, leakage check) + models x (CV folds + final fit), so 22 for 5 models
  and 3 folds; failed or skipped models count too.
- Cancel is cooperative: it takes effect at the next step, after the current model's fit.
  A job that doesn't end in `done` keeps nothing; the session's previous run stays.
- A job running longer than `TOOLKIT_JOB_MAX_MIN` minutes (default 180) is stopped the same
  way, with the code `TIME_LIMIT`.

## Errors

Every error answer has the same shape; `details` is present only when there is more to say:

```json
{"error": {"code": "TRAINING_FAILED", "message": "No model could be trained. ...", "details": {"failures": []}}}
```

| Code | HTTP | Meaning |
|---|---|---|
| `BAD_REQUEST` | 400 | the body is not a JSON object, or a value is missing, of the wrong type or too long |
| `NO_DATASET` | 400 | no dataset uploaded in this session yet |
| `UNKNOWN_COLUMN` | 400 | the target or a feature is not a column of the dataset |
| `UNREADABLE_FILE` | 400 | an uploaded file can't be read as a CSV/TXT dataset |
| `TRAINING_FAILED` | in a job | training could not run with these settings |
| `CANCELLED` | in a job | the job was cancelled; nothing was kept |
| `TIME_LIMIT` | in a job | the job ran longer than `TOOLKIT_JOB_MAX_MIN`; nothing was kept |
| `NOT_TRAINED` | 400 (404 for the download) | no model trained or exported yet in this session |
| `NO_MODEL_LOADED` | 400 | no bundle loaded yet in this session |
| `NOT_A_BUNDLE`, `BAD_SIGNATURE`, `EMPTY`, ... | 400 | a bundle was refused before anything in it was loaded |
| `CROSS_SITE` | 403 | the request came from another site |
| `CSRF_TOKEN` | 403 | the `X-CSRF-Token` header is missing or isn't this session's token (reload the page) |
| `NOT_FOUND` | 404 | no such API route |
| `METHOD_NOT_ALLOWED` | 405 | the route exists, the method doesn't |
| `BUSY` | 409 | a training run is in progress in this workspace: a second training, an upload, a class filter or clearing the workspace has to wait for it |
| `TOO_LARGE` | 413 | the upload is above `TOOLKIT_MAX_UPLOAD_MB` |
| `DATA_LIMIT` | 413 | the dataset doesn't fit in `TOOLKIT_MAX_DATA_MB` next to the workspaces that are training |
| `BUSY` | 503 | a new workspace would go over `TOOLKIT_MAX_WORKSPACES` and all others are training |
| `INTERNAL` | 500 | an unexpected error; the details are only in `instance/logs/web.log` |

Messages are written for the user and never contain exception text, tracebacks or server
paths.
