"""
datatrust/exceptions.py

Custom exception hierarchy.

Why custom exceptions?
- Callers (CLI today, Django views later) can catch `DataTrustError` once and show
  a friendly message, while genuine bugs (KeyError, TypeError...) still surface.
- Each subclass maps to a distinct user-facing problem.
"""


class DataTrustError(Exception):
    """Base class for all expected, user-facing errors in DataTrust AI."""


class FileValidationError(DataTrustError):
    """The file itself is unacceptable (missing, wrong extension, too large, empty)."""


class DataLoadError(DataTrustError):
    """The file exists but could not be parsed into a table."""


class EmptyDatasetError(DataTrustError):
    """The file parsed, but contains no usable rows or columns."""
