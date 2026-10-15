"""Shared provenance primitives without analysis or scheduling dependencies."""

import hashlib
import json
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path


def utc_now() -> str:
    """Return an ISO UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: dict) -> None:
    """Atomically replace JSON using deterministic encoding."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def file_hash(path: Path) -> str:
    """Hash a file in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_identity() -> dict:
    """Record source and environment identity without importing runners."""
    package = Path(__file__).resolve().parent
    repository = package.parent.parent
    sources = {
        f"src/asc_reproduction/{p.name}": file_hash(p)
        for p in sorted(package.glob("*.py"))
    }
    for name in ("analysis_spec.json", "pyproject.toml", "uv.lock"):
        if (repository / name).is_file():
            sources[name] = file_hash(repository / name)
    package_names = (
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "Pillow",
        "seaborn",
        "statannotations",
        "statsmodels",
        "matplotlib-venn",
        "pymaftools",
    )
    packages = {name: version(name) for name in package_names}
    return {
        "source_sha256": sources,
        "packages": packages,
        "python": sys.version,
        "platform": sys.platform,
    }
