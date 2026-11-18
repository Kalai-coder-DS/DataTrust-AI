"""
datatrust/web/demo.py  (Deployment: the open public demo)

Everything that makes it safe to let strangers upload files without a login. Standard library
only (plus the project's own loader); the Django middleware and views are thin wrappers.

1. Private uploads, without accounts
   Every browser gets a random 128-bit visitor id in a signed cookie (HMAC-SHA256 with the
   SECRET_KEY, HttpOnly, Secure, SameSite=Lax). Runs are saved with an OWNER TAG derived from
   that id (another HMAC), and pages only show public runs plus the visitor's own. Someone
   else's run id gives the same 404 as a run that doesn't exist. The cookie value itself is
   never stored, so even a copy of the database doesn't let anyone pose as a visitor.

2. Limits (DemoLimits in deploy.py)
   Size (while streaming), rows and columns, text-only content, uploads per hour per visitor,
   per address and for the whole server, and a cap on queued analyses. In-memory counters:
   the demo runs ONE server process (gunicorn --workers 1), so they see every request.

3. Sample data, loaded idempotently
   seed_sample_data() adds the sample CSVs as PUBLIC runs only if a public run with exactly
   those bytes (same SHA-256) isn't there yet. Running it at every start-up and at build time
   never creates duplicates.

4. Data persistence (Render free plan)
   The disk is ephemeral: a restart, a redeploy or the 15-minute sleep resets it to the files
   created at BUILD time. The build seeds the sample data, so after every restart the demo is
   back to "samples only"; visitor uploads are gone (they were temporary anyway, and are also
   pruned after DATATRUST_DEMO_RETENTION_HOURS while the server stays up).
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
import threading
import time
from collections import deque
from pathlib import Path

from datatrust.config import PROJECT_ROOT
from datatrust.logger import get_logger
from datatrust.web.services import UploadRejected

logger = get_logger(__name__)

VISITOR_COOKIE = "datatrust_visitor"
VISITOR_MAX_AGE = 7 * 24 * 3600
SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"

# The sample datasets every visitor sees. Same options as the CLI examples in the README.
SAMPLES = (
    {"file": "customers_dirty.csv", "target": "churned", "baseline": True, "summary": "template"},
    {"file": "customers_clean.csv", "target": "churned", "baseline": True, "summary": "template"},
)


# ----------------------------------------------------------------- 1. visitors
def _mac(secret: str, purpose: str, value: str) -> str:
    return hmac.new(secret.encode("utf-8"), f"{purpose}|{value}".encode("utf-8"), hashlib.sha256).hexdigest()


def new_visitor_cookie(secret: str) -> tuple[str, str]:
    """A fresh visitor: returns (visitor_id, cookie_value)."""
    visitor_id = secrets.token_hex(16)                       # 128 random bits
    return visitor_id, f"{visitor_id}.{_mac(secret, 'visitor', visitor_id)}"


def read_visitor_cookie(value: str | None, secret: str) -> str | None:
    """The visitor id if the cookie is genuine, else None (missing, edited or signed with another key)."""
    if not value or value.count(".") != 1:
        return None
    visitor_id, signature = value.split(".")
    if len(visitor_id) != 32 or any(c not in "0123456789abcdef" for c in visitor_id):
        return None
    return visitor_id if hmac.compare_digest(signature, _mac(secret, "visitor", visitor_id)) else None


def owner_tag(visitor_id: str, secret: str) -> str:
    """What is stored in run_owners: derived from the id, so the database never holds the cookie."""
    return _mac(secret, "owner", visitor_id)[:32]


def client_address(meta: dict) -> str:
    """
    The visitor's address. Behind Render's proxy REMOTE_ADDR is the proxy, and the client is the
    first X-Forwarded-For entry. A client can fake that header, so the address limit is only one
    of three (visitor, address, whole server); the server-wide limits hold whatever it says.
    """
    forwarded = (meta.get("HTTP_X_FORWARDED_FOR") or "").split(",")[0].strip()
    return forwarded[:64] or meta.get("REMOTE_ADDR") or "unknown"


# ------------------------------------------------------------------- 2. limits
class UploadRateLimiter:
    """Sliding one-hour window per key ('visitor:..', 'ip:..') and for the whole server."""

    WINDOW = 3600.0

    def __init__(self):
        self._events: dict[str, deque] = {}
        self._lock = threading.Lock()

    def _recent(self, key: str, now: float) -> deque:
        q = self._events.setdefault(key, deque())
        while q and now - q[0] >= self.WINDOW:
            q.popleft()
        return q

    def check_and_record(self, keys: list[str], per_key: int, total: int, now: float | None = None) -> int:
        """
        0 = allowed (and counted). Otherwise the number of seconds until the next upload is
        allowed; nothing is counted then, so a refused attempt doesn't extend the wait.
        """
        now = time.time() if now is None else now
        with self._lock:
            checks = [(self._recent(k, now), per_key) for k in keys] + [(self._recent("*", now), total)]
            waits = [int(self.WINDOW - (now - q[0])) + 1 for q, limit in checks if len(q) >= limit]
            if waits:
                return max(waits)
            for q, _ in checks:
                q.append(now)
            # Forget idle keys so the dictionary can't grow without bound.
            if len(self._events) > 10_000:
                for k in [k for k, q in self._events.items() if not q]:
                    del self._events[k]
            return 0


def check_public_upload(path: Path, name: str, limits) -> None:
    """
    Content checks for a file a stranger uploaded (the size was already enforced while saving it).
    Raises UploadRejected with a message for the form.
      - text only: a NUL byte means a binary file (Excel, zip, image) renamed to .csv;
      - rows and columns within the demo limits (memory and CPU on a small server).
    Parsing errors are left to the normal pipeline, which already explains them.
    """
    with path.open("rb") as fh:
        head = fh.read(64 * 1024)
    if b"\x00" in head:
        raise UploadRejected(f"'{name}' doesn't look like a text CSV file (it contains binary data). "
                            "Export it as CSV (comma-separated values) and try again.")
    from datatrust.exceptions import DataTrustError
    from datatrust.ingestion import CSVLoader
    try:
        df = CSVLoader().load(path).df
    except DataTrustError:
        return                       # the pipeline reports this with its usual, specific message
    rows, columns = df.shape
    if columns > limits.max_columns:
        raise UploadRejected(f"'{name}' has {columns} columns; the public demo accepts up to {limits.max_columns}. "
                            "Run DataTrust AI on your own computer for bigger files.")
    if rows > limits.max_rows:
        raise UploadRejected(f"'{name}' has {rows:,} rows; the public demo accepts up to {limits.max_rows:,}. "
                            "Run DataTrust AI on your own computer for bigger files.")


# ---------------------------------------------------------------- 3. sample data
def seed_sample_data(repo, samples=SAMPLES, sample_dir: Path = SAMPLE_DIR) -> list[tuple[str, str]]:
    """
    Make sure each sample dataset has a public run. Idempotent: a sample whose exact bytes
    (SHA-256) already have a public run is skipped. Returns [(file name, 'added'|'present'|'missing')].
    """
    from datatrust.pipeline import analyze_file, file_sha256
    results = []
    for sample in samples:
        path = Path(sample_dir) / sample["file"]
        if not path.is_file():
            logger.warning("Sample %s not found; skipped", path)
            results.append((sample["file"], "missing"))
            continue
        if repo.find_public_run(sample["file"], file_sha256(path)) is not None:
            results.append((sample["file"], "present"))
            continue
        analysis = analyze_file(path, target=sample.get("target"), baseline=sample.get("baseline", False),
                                summary=sample.get("summary"))
        run_id = repo.save(analysis)                 # no owner: public
        logger.info("Sample %s loaded as public run #%s", sample["file"], run_id)
        results.append((sample["file"], "added"))
    return results


def startup_upkeep(environ: dict | None = None) -> None:
    """
    Run once when the demo server starts (in a background thread, see webapp wsgi.py):
    load any missing sample data and delete expired visitor uploads. Never raises: a problem
    is logged and the site still starts.
    """
    from datatrust.web.deploy import DemoLimits
    from datatrust.web.services import repository
    try:
        limits = DemoLimits.from_env(environ)
        with repository(environ) as repo:
            results = seed_sample_data(repo)
            pruned = repo.prune_owned_runs(limits.retention_hours, limits.max_stored_runs)
        logger.info("Demo start-up: samples %s; %d expired upload(s) removed", results, pruned)
    except Exception:
        logger.exception("Demo start-up upkeep failed (the site still runs)")
