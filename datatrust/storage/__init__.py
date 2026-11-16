"""Storage layer: save analyses to MySQL (or SQLite) and read their history back."""

from datatrust.storage.database import Database, DBConfig, StorageError, load_env_file
from datatrust.storage.repository import AnalysisRepository


def open_repository(environ: dict | None = None) -> AnalysisRepository:
    """Read .env + environment, connect, make sure the tables exist, return a repository."""
    load_env_file(environ=environ)
    config = DBConfig.from_env(environ)
    db = Database(config)
    if config.backend == "mysql":
        db.create_mysql_database()
    db.connect()
    repo = AnalysisRepository(db)
    repo.init_schema()
    return repo


__all__ = ["AnalysisRepository", "DBConfig", "Database", "StorageError", "load_env_file", "open_repository"]
