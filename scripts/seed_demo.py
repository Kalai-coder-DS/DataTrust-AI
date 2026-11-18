"""
scripts/seed_demo.py  (Deployment)

Load the sample datasets into the database as PUBLIC runs, once. Render runs this in the
BUILD step (render.yaml), so the SQLite file with the samples is part of the deployed files:
after every restart or sleep (which wipe runtime changes) the demo starts with the samples.

    python scripts/seed_demo.py          # safe to run any number of times

Idempotent: a sample whose exact bytes (SHA-256) already have a public run is skipped.

The samples are analysed with the SAME model settings as on a laptop (3 cross-validation
repeats, 5 importance shuffles), so the demo's sample pages show the numbers in the README and
the study guide. Visitor uploads on the demo use the lighter settings from render.yaml.
"""
import os
import sys
from pathlib import Path

# Must happen before datatrust is imported: its settings read these variables at import time.
for name in ("DATATRUST_ML_CV_REPEATS", "DATATRUST_ML_IMPORTANCE_REPEATS", "DATATRUST_ML_MAX_ROWS"):
    os.environ.pop(name, None)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datatrust.web.demo import seed_sample_data  # noqa: E402
from datatrust.web.services import repository  # noqa: E402


def main() -> int:
    with repository() as repo:
        for file_name, status in seed_sample_data(repo):
            print(f"{file_name}: {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
