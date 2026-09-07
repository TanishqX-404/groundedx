"""Config-driven profiling entry point placeholder for optional GPU runs."""

from __future__ import annotations

from pathlib import Path


def require_model(path: str | Path) -> Path:
    """Validate a model path before a profiling sweep starts."""

    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    return model_path
