"""Helpers for canonicalizing and repairing legacy ticker symbols."""

from __future__ import annotations

EXCHANGE_SUFFIXES = (
    ".CO", ".ST", ".HE", ".OL", ".DE", ".PA", ".L", ".AS", ".MC",
    ".MI", ".BR", ".VI", ".SW", ".LS", ".IR", ".WA", ".PR", ".TO",
    ".AX", ".HK", ".T", ".NS", ".BO",
)


def canonicalize_ticker(ticker: str) -> str:
    """Convert legacy or source-specific ticker formats to Yahoo-compatible form."""
    if ticker is None:
        return ""

    resolved = str(ticker).strip().upper()
    if not resolved:
        return ""

    if resolved.startswith("^"):
        return resolved

    if ":" in resolved:
        resolved = resolved.split(":")[-1].strip()

    resolved = resolved.replace(" ", "-").replace("/", "-")

    for suffix in EXCHANGE_SUFFIXES:
        if resolved.endswith(suffix):
            base = resolved[:-len(suffix)].replace(".", "-")
            return f"{base}{suffix}"

    if "." in resolved:
        left, right = resolved.rsplit(".", 1)
        if right.isalpha() and 1 <= len(right) <= 2:
            return f"{left.replace('.', '-')}-{right}"

    return resolved


def is_legacy_ticker(ticker: str) -> bool:
    """Return True when a ticker needs canonicalization before use."""
    normalized = str(ticker).strip().upper() if ticker is not None else ""
    return bool(normalized) and canonicalize_ticker(normalized) != normalized