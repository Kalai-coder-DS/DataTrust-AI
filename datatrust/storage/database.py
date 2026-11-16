"""
datatrust/storage/database.py

Database connection for storing analyses. Two backends, one code path:

  mysql   - the real application database (your MySQL 8 server), via PyMySQL
  sqlite  - built into Python; used by the tests and as a zero-setup fallback

All SQL in this project is written with `?` placeholders and converted to `%s` for
MySQL. Values are ALWAYS passed as parameters, never pasted into the SQL string,
which is what prevents SQL injection (a file named  x'; DROP TABLE issues; --.csv
is stored as plain text).

Credentials come from environment variables, optionally loaded from a local `.env`
file that is listed in .gitignore, so the password never reaches GitHub.

    DATATRUST_DB=mysql                (or sqlite; default sqlite)
    DATATRUST_MYSQL_HOST=localhost
    DATATRUST_MYSQL_PORT=3306
    DATATRUST_MYSQL_USER=root
    DATATRUST_MYSQL_PASSWORD=...
    DATATRUST_MYSQL_DATABASE=datatrust
    DATATRUST_SQLITE_PATH=data/datatrust.db
"""
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from datatrust.config import PROJECT_ROOT
from datatrust.exceptions import DataTrustError
from datatrust.logger import get_logger

logger = get_logger(__name__)


class StorageError(DataTrustError):
    """The database could not be reached or used."""


# ---------------------------------------------------------------- configuration
def load_env_file(path: str | Path = PROJECT_ROOT / ".env", environ: dict | None = None) -> dict[str, str]:
    """
    Minimal .env reader (KEY=VALUE per line, # comments, optional quotes).
    Values already set in the real environment win, so a deployment can override the file.
    Returns the variables it set.
    """
    environ = os.environ if environ is None else environ
    path = Path(path)
    loaded: dict[str, str] = {}
    if not path.is_file():
        return loaded
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and key not in environ:
            environ[key] = value
            loaded[key] = value
    return loaded


@dataclass(frozen=True)
class DBConfig:
    backend: str = "sqlite"
    sqlite_path: Path = PROJECT_ROOT / "data" / "datatrust.db"
    host: str = "localhost"
    port: int = 3306
    user: str = "root"
    password: str = field(default="", repr=False)   # never printed in logs or errors
    database: str = "datatrust"

    @classmethod
    def from_env(cls, environ: dict | None = None) -> "DBConfig":
        env = os.environ if environ is None else environ
        backend = env.get("DATATRUST_DB", "sqlite").strip().lower()
        if backend not in ("sqlite", "mysql"):
            raise StorageError(f"DATATRUST_DB must be 'mysql' or 'sqlite', not '{backend}'.")
        sqlite_path = Path(env.get("DATATRUST_SQLITE_PATH", str(cls.sqlite_path)))
        if not sqlite_path.is_absolute():
            sqlite_path = PROJECT_ROOT / sqlite_path
        try:
            port = int(env.get("DATATRUST_MYSQL_PORT", "3306"))
        except ValueError as exc:
            raise StorageError("DATATRUST_MYSQL_PORT must be a number.") from exc
        return cls(
            backend=backend,
            sqlite_path=sqlite_path,
            host=env.get("DATATRUST_MYSQL_HOST", "localhost"),
            port=port,
            user=env.get("DATATRUST_MYSQL_USER", "root"),
            password=env.get("DATATRUST_MYSQL_PASSWORD", ""),
            database=env.get("DATATRUST_MYSQL_DATABASE", "datatrust"),
        )

    def describe(self) -> str:
        """Safe, password-free description for messages."""
        if self.backend == "sqlite":
            return f"SQLite file {self.sqlite_path}"
        return f"MySQL database '{self.database}' on {self.host}:{self.port} as user '{self.user}'"


# ------------------------------------------------------------------- database
class Database:
    """Thin wrapper: one connection, `?` placeholders everywhere, explicit transactions."""

    def __init__(self, config: DBConfig) -> None:
        self.config = config
        self._conn = None

    @property
    def backend(self) -> str:
        return self.config.backend

    def connect(self) -> "Database":
        if self._conn is not None:
            return self
        if self.backend == "sqlite":
            self.config.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.config.sqlite_path)
            self._conn.execute("PRAGMA foreign_keys = ON")  # SQLite enforces FKs only when asked
        else:
            self._conn = self._connect_mysql(with_database=True)
        logger.info("Connected to %s", self.config.describe())
        return self

    def _connect_mysql(self, with_database: bool):
        try:
            import pymysql  # imported here so SQLite users don't need it installed
        except ImportError as exc:
            raise StorageError("MySQL support needs the PyMySQL package: pip install PyMySQL") from exc
        c = self.config
        try:
            return pymysql.connect(
                host=c.host, port=c.port, user=c.user, password=c.password,
                database=c.database if with_database else None,
                charset="utf8mb4", autocommit=False,
            )
        except pymysql.err.OperationalError as exc:
            code = exc.args[0] if exc.args else None
            hints = {
                1045: "Access denied: check DATATRUST_MYSQL_USER and DATATRUST_MYSQL_PASSWORD in .env.",
                1049: "Database does not exist yet: run  python scripts/db.py init",
                2003: "Cannot reach the server: is the MySQL80 service running? Check host and port.",
            }
            raise StorageError(f"{hints.get(code, 'Could not connect to MySQL.')} ({self.config.describe()})") from exc

    def create_mysql_database(self) -> None:
        """CREATE DATABASE IF NOT EXISTS, using a connection without a default database."""
        name = self.config.database
        if not name.replace("_", "").isalnum():
            raise StorageError(f"Unsafe database name '{name}': use letters, digits and underscores only.")
        conn = self._connect_mysql(with_database=False)
        try:
            with conn.cursor() as cur:
                # A database name can't be a query parameter; it is validated above instead.
                cur.execute(f"CREATE DATABASE IF NOT EXISTS `{name}` "
                            "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
            conn.commit()
        finally:
            conn.close()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "Database":
        return self.connect()

    def __exit__(self, *exc) -> None:
        self.close()

    # ---------------------------------------------------------------- queries
    def _sql(self, sql: str) -> str:
        return sql.replace("?", "%s") if self.backend == "mysql" else sql

    def execute(self, sql: str, params: tuple | list = ()) -> int | None:
        """Run one statement; returns the new row id for INSERTs."""
        cur = self._conn.cursor()
        try:
            cur.execute(self._sql(sql), tuple(params))
            return cur.lastrowid
        finally:
            cur.close()

    def executemany(self, sql: str, rows: list[tuple]) -> None:
        if not rows:
            return
        cur = self._conn.cursor()
        try:
            cur.executemany(self._sql(sql), rows)
        finally:
            cur.close()

    def query(self, sql: str, params: tuple | list = ()) -> list[dict[str, Any]]:
        cur = self._conn.cursor()
        try:
            cur.execute(self._sql(sql), tuple(params))
            columns = [d[0] for d in cur.description]
            return [dict(zip(columns, row)) for row in cur.fetchall()]
        finally:
            cur.close()

    @contextmanager
    def transaction(self) -> Iterator["Database"]:
        """All-or-nothing: either every statement inside is saved, or none is."""
        try:
            yield self
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
