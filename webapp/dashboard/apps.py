from django.apps import AppConfig


class DashboardConfig(AppConfig):
    """Upload, dashboard, dataset history and run pages. No models: data lives in datatrust.storage."""

    name = "dashboard"
    verbose_name = "DataTrust AI dashboard"
