#!/usr/bin/env python
"""
Django's command-line utility for the DataTrust AI web app (Phase 6).

Run from the project root:
    python webapp/manage.py runserver       # http://127.0.0.1:8000/
    python webapp/manage.py test dashboard  # the web app's tests
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent   # so `import datatrust` works from anywhere


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(1, str(PROJECT_ROOT))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "datatrust_site.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Django is not installed in this environment. Run:  pip install -r requirements.txt"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
