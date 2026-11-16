"""WSGI entry point (for a production server such as gunicorn or waitress)."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))      # webapp/
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "datatrust_site.settings")

from django.core.wsgi import get_wsgi_application  # noqa: E402

application = get_wsgi_application()
