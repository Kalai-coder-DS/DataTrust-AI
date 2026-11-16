"""
datatrust/pipeline.py

ONE function that runs the whole analysis: load -> profile -> quality -> statistics -> scores.

Why: before this, main.py wired the steps together itself. The HTML report, the MySQL
storage (Phase 5b) and the Django app (Phase 6) all need the same sequence; putting it in
one place means they can never drift apart (e.g. one forgetting to pass the target).

The result is a plain dict (the same structure as the JSON export), so it can be saved to
a file, stored in a database, or rendered as HTML without the original DataFrame.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

from datatrust import __version__
from datatrust.ingestion import CSVLoader
from datatrust.logger import get_logger
from datatrust.profiling import DataProfiler
from datatrust.quality import QualityAnalyzer
from datatrust.scoring import score_dataset
from datatrust.stats import StatisticsAnalyzer

logger = get_logger(__name__)


def analyze_file(path: str | Path, target: str | None = None) -> dict:
    """
    Run every phase on a CSV file and return the full analysis as a JSON-serialisable dict.

    Raises DataTrustError for unusable files and ValueError for an invalid target.
    """
    result = CSVLoader().load(path)
    df = result.df
    profile = DataProfiler().profile(df)
    quality = QualityAnalyzer().analyze(df, profile)
    stats = StatisticsAnalyzer().analyze(df, profile, target=target)
    scores = score_dataset(quality, stats, profile, target=target)
    logger.info("Analysis complete for %s: trust %.1f, ML readiness %.1f",
                result.file_name, scores.trust.score, scores.ml_readiness.score)
    return {
        "meta": {
            "tool": "DataTrust AI",
            "version": __version__,
            "analyzed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "target": target,
        },
        "file": {**result.summary(), "sha256": file_sha256(path)},
        "profile": profile.to_dict(),
        "quality": quality.to_dict(),
        "statistics": stats.to_dict(),
        "scores": scores.to_dict(),
    }


def file_sha256(path: str | Path) -> str:
    """Fingerprint of the file's bytes: same fingerprint = identical data (used to track changes)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
