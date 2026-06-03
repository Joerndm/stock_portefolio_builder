"""
Data Validation Contracts for Stock Portfolio Builder.

Defines exact DataFrame schemas at each pipeline boundary and provides a
central ``validate_dataframe()`` function to enforce them.

Usage::

    from data_contracts import validate_dataframe, STOCK_PRICE_SCHEMA

    validate_dataframe(df, STOCK_PRICE_SCHEMA, "stock_price_data")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DataFrameSchema:
    """Lightweight schema for a pandas DataFrame."""

    name: str
    required_columns: List[str]
    optional_columns: List[str] = field(default_factory=list)
    min_rows: int = 1
    # column → expected dtype prefix ("float", "int", "object", "datetime")
    dtypes: Dict[str, str] = field(default_factory=dict)
    no_nulls_in: List[str] = field(default_factory=list)


# ── Pipeline boundary schemas ──────────────────────────────────────────────

STOCK_PRICE_SCHEMA = DataFrameSchema(
    name="stock_price_data",
    required_columns=[
        "date", "ticker", "open_Price", "high_Price", "low_Price",
        "close_Price", "trade_Volume",
    ],
    no_nulls_in=["date", "ticker", "close_Price"],
    min_rows=50,
)

STOCK_PRICE_EXPORT_SCHEMA = DataFrameSchema(
    name="stock_price_export_batch",
    required_columns=STOCK_PRICE_SCHEMA.required_columns,
    no_nulls_in=STOCK_PRICE_SCHEMA.no_nulls_in,
    min_rows=1,
)

ML_INPUT_SCHEMA = DataFrameSchema(
    name="ml_input",
    required_columns=[],  # columns are dynamic (selected features)
    min_rows=100,
)

FEATURE_SELECTION_OUTPUT_SCHEMA = DataFrameSchema(
    name="feature_selection_output",
    required_columns=[],  # columns determined at runtime
    min_rows=50,
)

MODEL_PREDICTION_SCHEMA = DataFrameSchema(
    name="model_prediction",
    required_columns=["date", "predicted_1D", "close_Price"],
    no_nulls_in=["date", "predicted_1D"],
    min_rows=1,
)

MONTE_CARLO_SCHEMA = DataFrameSchema(
    name="monte_carlo_output",
    required_columns=["date", "close_Price"],
    min_rows=1,
)


# ── Validation function ────────────────────────────────────────────────────

class DataContractError(Exception):
    """Raised when a DataFrame violates its contract."""


def validate_dataframe(
    df: pd.DataFrame,
    schema: DataFrameSchema,
    context: str = "",
    raise_on_error: bool = True,
) -> List[str]:
    """Validate *df* against a ``DataFrameSchema``.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to validate.
    schema : DataFrameSchema
        Expected schema.
    context : str
        Extra context for error messages (e.g. ticker symbol).
    raise_on_error : bool
        If True, raise ``DataContractError`` on first violation.
        If False, collect and return all violations.

    Returns
    -------
    errors : list[str]
        Empty if valid, otherwise list of violation descriptions.
    """
    errors: List[str] = []
    label = f"[{schema.name}]" + (f" ({context})" if context else "")

    # 1. Required columns
    if schema.required_columns:
        missing = set(schema.required_columns) - set(df.columns)
        if missing:
            errors.append(f"{label} Missing required columns: {sorted(missing)}")

    # 2. Minimum rows
    if len(df) < schema.min_rows:
        errors.append(
            f"{label} Too few rows: {len(df)} < {schema.min_rows}"
        )

    # 3. Null checks
    for col in schema.no_nulls_in:
        if col in df.columns and df[col].isna().any():
            n_null = df[col].isna().sum()
            errors.append(f"{label} Column '{col}' has {n_null} null(s)")

    # 4. Dtype checks
    for col, expected_prefix in schema.dtypes.items():
        if col in df.columns:
            actual = str(df[col].dtype)
            if not actual.startswith(expected_prefix):
                errors.append(
                    f"{label} Column '{col}' dtype is '{actual}', "
                    f"expected '{expected_prefix}*'"
                )

    if errors:
        for e in errors:
            logger.warning(e)
        if raise_on_error:
            raise DataContractError("\n".join(errors))

    return errors
