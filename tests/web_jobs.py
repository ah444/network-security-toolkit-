"""Test helpers for training jobs (v5.1 step 1.2d): submit one and wait for its end."""

import time

TRAIN_URL = "/api/v1/jobs/training"
ENDED = ("done", "failed", "cancelled")


class Finished:
    """A finished job, shaped like the old synchronous answer: 200 + the result when done,
    400 + ``{"error": ...}`` when it failed or was cancelled."""

    def __init__(self, status_code, body, job):
        self.status_code, self._body, self.job = status_code, body, job

    def get_json(self):
        return self._body


def wait_for(client, url, timeout=120):
    """Poll a job until it ends; returns its last state."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(url).get_json()
        if job["state"] in ENDED:
            return job
        time.sleep(0.02)
    raise AssertionError(f"the job at {url} didn't end within {timeout} s")


def train_and_wait(client, body, timeout=120):
    """Train through the job routes. A refused submit (400, 409, ...) is returned as is."""
    submitted = client.post(TRAIN_URL, json=body)
    if submitted.status_code != 202:
        return submitted
    job = wait_for(client, submitted.get_json()["url"], timeout)
    if job["state"] == "done":
        return Finished(200, job["result"], job)
    return Finished(400, {"error": job["error"]}, job)
