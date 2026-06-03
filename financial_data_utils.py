"""Focused helpers for normalizing financial DataFrames."""

from __future__ import annotations

import numpy as np
import pandas as pd


def drop_all_null_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Drop only columns that are entirely null."""
    if dataframe.empty:
        return dataframe
    return dataframe.dropna(axis=1, how="all")


def ensure_current_liabilities_column(balancesheet_df: pd.DataFrame) -> pd.DataFrame:
    """Provide stable liquidity columns for downstream balance-sheet calculations."""
    normalized = balancesheet_df.copy()

    if (
        "Derivative Product Liabilities" in normalized.columns
        and "Current Liabilities" not in normalized.columns
    ):
        normalized = normalized.rename(
            columns={"Derivative Product Liabilities": "Current Liabilities"}
        )

    if "Current Liabilities" not in normalized.columns:
        normalized["Current Liabilities"] = 0.0

    cash_columns = (
        "Cash And Cash Equivalents",
        "Cash Cash Equivalents And Short Term Investments",
    )
    if cash_columns[0] in normalized.columns and cash_columns[1] not in normalized.columns:
        normalized[cash_columns[1]] = normalized[cash_columns[0]]
    if cash_columns[1] in normalized.columns and cash_columns[0] not in normalized.columns:
        normalized[cash_columns[0]] = normalized[cash_columns[1]]
    for column in cash_columns:
        if column not in normalized.columns:
            normalized[column] = 0.0

    return normalized


def calculate_available_annual_ratios(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Calculate whichever annual valuation ratios have the required inputs."""
    result = dataframe.copy()
    ratio_specs = {
        "P/S": ("close_Price", "revenue", "average_shares"),
        "P/E": ("close_Price", "eps"),
        "P/B": ("close_Price", "book_Value_Per_Share"),
        "P/FCF": ("close_Price", "free_Cash_Flow_Per_Share"),
    }

    for ratio_name in ratio_specs:
        if ratio_name not in result.columns:
            result[ratio_name] = np.nan

    for ratio_name, required_columns in ratio_specs.items():
        if not set(required_columns).issubset(result.columns):
            continue

        if ratio_name == "P/S":
            result[ratio_name] = result["close_Price"] / (result["revenue"] / result["average_shares"])
        elif ratio_name == "P/E":
            result[ratio_name] = result["close_Price"] / result["eps"]
        elif ratio_name == "P/B":
            result[ratio_name] = result["close_Price"] / result["book_Value_Per_Share"]
        elif ratio_name == "P/FCF":
            result[ratio_name] = result["close_Price"] / result["free_Cash_Flow_Per_Share"]

    ratio_columns = list(ratio_specs.keys())
    result[ratio_columns] = result[ratio_columns].replace([np.inf, -np.inf], np.nan)
    result[ratio_columns] = result[ratio_columns].shift(1)
    return result