from django.urls import path

from dashboard import views

app_name = "dashboard"

urlpatterns = [
    path("", views.home, name="home"),
    path("upload/", views.upload, name="upload"),
    path("jobs/<str:job_id>/", views.job, name="job"),
    path("datasets/<str:name>/", views.dataset, name="dataset"),
    path("runs/<int:run_id>/", views.run, name="run"),
    path("runs/<int:run_id>/report/", views.report, name="report"),
    path("runs/<int:run_id>/json/", views.analysis_json, name="json"),
]
