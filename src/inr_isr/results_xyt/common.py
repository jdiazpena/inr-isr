"""Small provenance/file helpers for fresh, independently reproducible attempts."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    """Atomic updates inside a fresh live attempt, never historical output reuse."""
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def fresh_directory(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def source_identity(repository):
    """Bind current scientific source separately from historical sealed guards."""
    repository = Path(repository)
    paths = sorted((repository / "src").rglob("*.py"))
    paths += [repository / "pyproject.toml", repository / "scripts/project_environment.sh"]
    files = {str(path.relative_to(repository)): sha256(path) for path in paths}
    return {"files": files, "identity": canonical_hash(files)}
