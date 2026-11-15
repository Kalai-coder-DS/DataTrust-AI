"""
tests/conftest.py

Shared pytest fixtures. pytest discovers this file automatically.

`tmp_path` (built into pytest) gives each test its own empty folder, so edge-case
CSVs are created fresh per test and never pollute the repository.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"


@pytest.fixture(scope="session")
def sample_dir() -> Path:
    """Ensure the generated sample datasets exist (regenerate if missing)."""
    if not (SAMPLE_DIR / "customers_dirty.csv").exists():
        subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "generate_sample_data.py")], check=True)
    return SAMPLE_DIR


@pytest.fixture(scope="session")
def dirty_manifest(sample_dir: Path) -> dict:
    return json.loads((sample_dir / "dirty_manifest.json").read_text())


@pytest.fixture
def write_csv(tmp_path: Path):
    """Factory fixture: write_csv('a,b\\n1,2', name='x.csv', encoding='utf-8') -> Path"""

    def _write(content: str | bytes, name: str = "test.csv", encoding: str = "utf-8") -> Path:
        path = tmp_path / name
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding=encoding, newline="")
        return path

    return _write
