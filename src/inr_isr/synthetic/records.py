"""Small source/runtime/command receipts, without bulk execution logs."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def file_identity(path):
    path = Path(path)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest}


def runtime_identity():
    root = Path(__file__).resolve().parents[3]
    paths = sorted(Path(__file__).parent.glob("*.py")) + [root / "src/synthetic_plasma.py"]
    return {"interpreter": sys.executable, "python": sys.version,
            "distributions": {name: importlib.metadata.version(name) for name in
                              ("numpy", "pandas", "matplotlib", "scipy", "torch")},
            "source_files": [file_identity(p) for p in paths],
            "git_head": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
            "git_dirty": bool(subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True).strip())}


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
