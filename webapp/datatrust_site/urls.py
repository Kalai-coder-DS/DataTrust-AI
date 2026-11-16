"""Top-level URLs: everything is in the dashboard app."""
from django.urls import include, path

urlpatterns = [
    path("", include("dashboard.urls")),
]

handler404 = "dashboard.views.page_not_found"
handler500 = "dashboard.views.server_error"
