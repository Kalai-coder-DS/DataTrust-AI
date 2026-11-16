from django import forms


class UploadForm(forms.Form):
    """The upload page. Deeper checks (type, size while streaming, target column) are in datatrust.web."""

    file = forms.FileField(
        label="CSV file",
        widget=forms.FileInput(attrs={"accept": ".csv,text/csv"}),
        error_messages={"required": "Choose a CSV file to analyse."},
    )
    target = forms.CharField(
        label="Target column (optional)",
        required=False,
        max_length=255,
        strip=True,
        help_text="The column a model would predict, e.g. churned or charges. "
                  "Leave empty for a data-quality check only.",
    )
    baseline = forms.BooleanField(
        label="Also train a baseline model on the target (raw vs cleaned data)",
        required=False,
        initial=True,
        help_text="Needs a target column. Adds about 5-30 seconds, depending on the file size.",
    )
