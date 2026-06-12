"""Helpers for loading optional ML runtime dependencies on demand."""

from importlib import import_module


ML_RUNTIME_INSTALL_HINT = (
    "ML runtime dependencies are not installed. "
    "Install them with: pip install -r requirements_PY_3_12_ml.txt"
)


def load_ml_builder_module():
    """Import ml_builder only when a training or prediction path actually needs it."""
    try:
        return import_module("ml_builder")
    except ImportError as exc:
        raise ImportError(ML_RUNTIME_INSTALL_HINT) from exc
