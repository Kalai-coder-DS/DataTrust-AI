"""Ingestion layer: safely turns an uploaded file into a pandas DataFrame."""

from datatrust.ingestion.loader import CSVLoader, LoadResult

__all__ = ["CSVLoader", "LoadResult"]
