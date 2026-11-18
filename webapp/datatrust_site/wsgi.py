"""
WSGI entry point (for a production server: gunicorn on Render, see render.yaml).

On the public demo (DATATRUST_PUBLIC_DEMO=1) start-up also loads any missing sample data and
removes expired visitor uploads, in a background thread so the server answers at once.
Normally the build step has already loaded the samples (scripts/seed_demo.py), so this only
checks, and the check is idempotent: it never adds a second copy.
"""
import os
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))      # webapp/
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "datatrust_site.settings")

from django.conf import settings  # noqa: E402
from django.core.wsgi import get_wsgi_application  # noqa: E402

application = get_wsgi_application()

if getattr(settings, "DATATRUST_PUBLIC_DEMO", False):
    from datatrust.web.demo import startup_upkeep  # noqa: E402
    threading.Thread(target=startup_upkeep, name="datatrust-demo-startup", daemon=True).start()
