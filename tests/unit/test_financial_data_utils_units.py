"""Unit tests for financial_data_utils.py."""

import os
import sys
import unittest

import pandas as pd


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from financial_data_utils import (
    calculate_available_annual_ratios,
    drop_all_null_columns,
    ensure_current_liabilities_column,
)


class TestFinancialDataUtils(unittest.TestCase):
    """Focused regression coverage for financial data normalization helpers."""

    def test_drop_all_null_columns_keeps_partially_populated_financial_fields(self):
        dataframe = pd.DataFrame(
            {
                "average_shares": [1000.0, None],
                "revenue": [500.0, 550.0],
                "all_null_metric": [None, None],
            }
        )

        result = drop_all_null_columns(dataframe)

        self.assertIn("average_shares", result.columns)
        self.assertIn("revenue", result.columns)
        self.assertNotIn("all_null_metric", result.columns)

    def test_ensure_current_liabilities_column_adds_zero_fallback(self):
        balancesheet_df = pd.DataFrame(
            {
                "Date": [pd.Timestamp("2025-12-31")],
                "Current Assets": [125.0],
            }
        )

        result = ensure_current_liabilities_column(balancesheet_df)

        self.assertIn("Current Liabilities", result.columns)
        self.assertEqual(result.loc[0, "Current Liabilities"], 0.0)

    def test_ensure_current_liabilities_column_renames_derivative_liabilities(self):
        balancesheet_df = pd.DataFrame(
            {
                "Date": [pd.Timestamp("2025-12-31")],
                "Derivative Product Liabilities": [42.0],
            }
        )

        result = ensure_current_liabilities_column(balancesheet_df)

        self.assertIn("Current Liabilities", result.columns)
        self.assertNotIn("Derivative Product Liabilities", result.columns)
        self.assertEqual(result.loc[0, "Current Liabilities"], 42.0)

    def test_ensure_current_liabilities_column_populates_cash_aliases(self):
        balancesheet_df = pd.DataFrame(
            {
                "Date": [pd.Timestamp("2025-12-31")],
                "Cash And Cash Equivalents": [17.5],
            }
        )

        result = ensure_current_liabilities_column(balancesheet_df)

        self.assertEqual(result.loc[0, "Cash And Cash Equivalents"], 17.5)
        self.assertEqual(
            result.loc[0, "Cash Cash Equivalents And Short Term Investments"],
            17.5,
        )

    def test_calculate_available_annual_ratios_keeps_partial_ratio_coverage(self):
        dataframe = pd.DataFrame(
            {
                "close_Price": [100.0, 110.0],
                "revenue": [500.0, 550.0],
                "average_shares": [10.0, 10.0],
                "eps": [5.0, 5.5],
                "book_Value_Per_Share": [50.0, 55.0],
            }
        )

        result = calculate_available_annual_ratios(dataframe)

        self.assertTrue(pd.isna(result.loc[0, "P/S"]))
        self.assertAlmostEqual(result.loc[1, "P/S"], 2.0)
        self.assertAlmostEqual(result.loc[1, "P/E"], 20.0)
        self.assertAlmostEqual(result.loc[1, "P/B"], 2.0)
        self.assertTrue(pd.isna(result.loc[1, "P/FCF"]))


if __name__ == "__main__":
    unittest.main()