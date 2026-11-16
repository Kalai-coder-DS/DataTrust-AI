"""
Web layer helpers (Phase 6): everything the Django views need, written WITHOUT Django.

Why a separate, framework-free package?
  - The views stay thin: parse the request, call one function here, render a template.
  - All the logic (upload checks, analysis + save, building page data, the trend chart)
    is tested with plain pytest, no web server or Django test client needed.
  - Same rule as the rest of the project: Django is a delivery mechanism, not the core.

Django talks to the database ONLY through datatrust.storage (repository pattern):
there are no Django ORM models for these tables.
"""
from datatrust.web.services import (
    NotFound,
    UploadRejected,
    analyze_upload,
    dashboard_page,
    dataset_page,
    repository,
    reset_repository_state,
    run_page,
    stored_analysis,
    validate_upload,
)
from datatrust.web.charts import trend_svg

__all__ = [
    "NotFound",
    "UploadRejected",
    "analyze_upload",
    "dashboard_page",
    "dataset_page",
    "repository",
    "reset_repository_state",
    "run_page",
    "stored_analysis",
    "trend_svg",
    "validate_upload",
]
