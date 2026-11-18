"""
datatrust/web/jobs.py  (Phase 10)

Runs analyses in background threads, so an upload returns at once and the browser shows
"Analysing..." instead of waiting 5-120 seconds for model training or an AI summary.

    job_id = jobs.submit(fn, label="insurance.csv")     # fn() returns the new run id
    jobs.get(job_id)  ->  {"id", "label", "state", "run_id", "error", "error_kind", "elapsed"}

States: queued -> running -> done | failed.

Design (interview talking points):
  - Standard library only (concurrent.futures). No Celery/Redis to install and run.
  - Each job opens its own database connection (repository() per call), so threads never
    share a connection.
  - Job ids are random 128-bit values: nobody can look at another upload's status by guessing.
  - Errors are classified: a bad file ('user'), the database ('database'), anything else
    ('internal', full traceback in the log, a short message on the page).
  - Limits: jobs live in this process's memory. Fine for `runserver` and a single-process server;
    a restart forgets running jobs (their uploads are not saved), and several server processes
    would each have their own list. A shared queue (database table or Redis) is the next step
    if this ever runs on more than one process.
"""
from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from datatrust.logger import get_logger

logger = get_logger(__name__)

MAX_KEPT = 200


class JobManager:
    def __init__(self, workers: int = 2):
        self.workers = max(1, int(workers))
        self._jobs: OrderedDict[str, dict] = OrderedDict()
        self._lock = threading.Lock()
        self._done: dict[str, threading.Event] = {}
        self._executor: ThreadPoolExecutor | None = None

    def _pool(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="datatrust-job")
        return self._executor

    def submit(self, fn: Callable[[], int], label: str, owner: str | None = None) -> str:
        """owner (public demo): the visitor's tag; the view shows the job only to that visitor."""
        job_id = uuid.uuid4().hex
        with self._lock:
            self._jobs[job_id] = {"id": job_id, "label": label, "owner": owner, "state": "queued", "run_id": None,
                                  "error": None, "error_kind": None, "submitted": time.time(),
                                  "started": None, "finished": None}
            self._done[job_id] = threading.Event()
            while len(self._jobs) > MAX_KEPT:
                old, _ = self._jobs.popitem(last=False)
                self._done.pop(old, None)
        self._pool().submit(self._run, job_id, fn)
        logger.info("Job %s queued for %s", job_id[:8], label)
        return job_id

    def _update(self, job_id: str, **fields) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].update(fields)

    def _run(self, job_id: str, fn: Callable[[], int]) -> None:
        from datatrust.storage import StorageError
        from datatrust.web.services import UploadRejected
        self._update(job_id, state="running", started=time.time())
        try:
            run_id = fn()
            self._update(job_id, state="done", run_id=run_id, finished=time.time())
            logger.info("Job %s done: run #%s", job_id[:8], run_id)
        except UploadRejected as exc:
            self._update(job_id, state="failed", error=str(exc), error_kind="user", finished=time.time())
        except StorageError as exc:
            self._update(job_id, state="failed", error=f"{exc} Check the database settings in .env.",
                         error_kind="database", finished=time.time())
            logger.error("Job %s: database unavailable: %s", job_id[:8], exc)
        except Exception:                                    # never let a thread die silently
            logger.exception("Job %s failed unexpectedly", job_id[:8])
            self._update(job_id, state="failed", error="An unexpected error occurred. The details are in "
                         "logs/datatrust.log.", error_kind="internal", finished=time.time())
        finally:
            event = self._done.get(job_id)
            if event:
                event.set()

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            job = dict(job) if job else None
        if job:
            end = job["finished"] or time.time()
            job["elapsed"] = round(end - (job["started"] or job["submitted"]), 1)
        return job

    def pending(self, owner: str | None = None) -> int:
        """Jobs queued or running: all of them, or one visitor's (public demo limits)."""
        with self._lock:
            return sum(1 for j in self._jobs.values()
                       if j["state"] in ("queued", "running") and (owner is None or j["owner"] == owner))

    def wait(self, job_id: str, timeout: float = 60) -> dict | None:
        """Block until the job finishes (tests and scripts)."""
        event = self._done.get(job_id)
        if event:
            event.wait(timeout)
        return self.get(job_id)

    def shutdown(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
