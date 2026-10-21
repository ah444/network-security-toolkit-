# Web trainer API (`/api/v1`)

The web trainer's page (`/dynamic_ids`) talks to the server only through these routes.
They work on the calling browser session's **workspace** (its dataset, trained run and
loaded bundle); the session cookie holds only the workspace ID. Everything runs on this
computer (`127.0.0.1`).

- **Requests:** `POST` bodies are JSON objects (`Content-Type: application/json`), except
  the two file uploads (`multipart/form-data`). `GET` routes read the query string.
- **Limits:** lists hold at most 10,000 items, texts at most 1,000 characters; `rows`,
  `k` and `test_size` are moved into their range instead of being refused. Uploads are
  limited by `TOOLKIT_MAX_UPLOAD_MB` (default 2,048 MB).
- **Cross-site requests:** a `POST`/`DELETE` whose `Origin` header names another site is
  refused (`CROSS_SITE`, 403).

## Routes

| Method | Route | Request | Answer |
|---|---|---|---|
| POST | `/api/v1/dataset` | multipart `dataset`: one or more `.csv`/`.txt` files | columns, detected profile, message |
| GET | `/api/v1/dataset/preview` | `rows` (1-100, default 10) | first rows, column stats |
| GET | `/api/v1/dataset/classes` | `target` | class values with counts |
| GET | `/api/v1/dataset/distribution` | `target`, `test_size` (0.1-0.5, default 0.2) | class counts, balancing recommendation (training split) |
| POST | `/api/v1/dataset/filter` | `target`, `remove`: classes to drop | rows removed, classes left |
| POST | `/api/v1/dataset/balancing-preview` | `target`, `balancing`, `features`, `test_size` | class counts before/after, on the training split only |
| POST | `/api/v1/dataset/feature-scores` | `target`, `k` (1-10,000, default 10), `test_size` | feature scores on the training split |
| GET | `/api/v1/models` | none | the models that can be trained here |
| POST | `/api/v1/run` | `target`, `features`, `models`, `balancing`, `test_size` | leaderboard, split, CV, leakage findings, confusion matrices |
| GET | `/api/v1/run/confusion` | `model` (default: the selected one) | test-split confusion matrix |
| POST | `/api/v1/run/predict` | `values`: {feature: value}, `model` | prediction |
| POST | `/api/v1/run/export` | `name` (letters, digits, `_`, `-`; max 80) | signed bundle saved in `instance/bundles/` |
| GET | `/api/v1/run/bundle` | none | the exported bundle as a download |
| POST | `/api/v1/bundle` | multipart `file`: a `.bundle` exported by this install | the loaded models and their scores |
| POST | `/api/v1/bundle/predict` | `values`, `model` | prediction with the loaded bundle |
| DELETE | `/api/v1/workspace` | none | the session's dataset and models are forgotten |

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
| `TRAINING_FAILED` | 400 | training could not run with these settings |
| `NOT_TRAINED` | 400 (404 for the download) | no model trained or exported yet in this session |
| `NO_MODEL_LOADED` | 400 | no bundle loaded yet in this session |
| `NOT_A_BUNDLE`, `BAD_SIGNATURE`, `EMPTY`, ... | 400 | a bundle was refused before anything in it was loaded |
| `CROSS_SITE` | 403 | the request came from another site |
| `NOT_FOUND` | 404 | no such API route |
| `METHOD_NOT_ALLOWED` | 405 | the route exists, the method doesn't |
| `BUSY` | 409 | a training run is already in progress in this workspace |
| `TOO_LARGE` | 413 | the upload is above `TOOLKIT_MAX_UPLOAD_MB` |
| `INTERNAL` | 500 | an unexpected error; the details are only in `instance/logs/web.log` |

Messages are written for the user and never contain exception text, tracebacks or server
paths.
