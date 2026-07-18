"""Unit tests for validate_stock_data.py."""

import json
import os
import sys
import tempfile
import unittest
from datetime import datetime
from unittest.mock import patch

import pandas as pd


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import validate_stock_data


class TestValidateStockData(unittest.TestCase):
    """Focused coverage for ticker-scoped validation."""

    def test_normalizes_selected_tickers(self):
        normalized = validate_stock_data._normalize_selected_tickers([" AAPL ", "", "AAPL", "MSFT"])

        self.assertEqual(normalized, ["AAPL", "MSFT"])

    def test_indicator_warmup_budget_uses_first_non_null_row(self):
        budget = validate_stock_data._indicator_warmup_null_budget(3919, 282, 200)

        self.assertEqual(budget, 281)

    def test_indicator_warmup_budget_falls_back_to_default_when_all_null(self):
        budget = validate_stock_data._indicator_warmup_null_budget(150, None, 34)

        self.assertEqual(budget, 34)

    @patch("validate_stock_data.run_structured_db_checks")
    @patch("validate_stock_data.get_engine")
    @patch("validate_stock_data.run_query")
    def test_validate_all_scopes_queries_to_requested_tickers(self, mock_run_query, mock_get_engine, mock_structured_checks):
        queries = []
        mock_get_engine.return_value = object()
        mock_structured_checks.return_value = {
            "summary": {"duplicate_key_groups": 0, "orphan_ticker_groups": 0}
        }

        def fake_run_query(_engine, sql):
            normalized_sql = " ".join(str(sql).split())
            queries.append(normalized_sql)

            if "FROM stock_info_data" in normalized_sql:
                self.assertIn("WHERE ticker IN ('AAPL')", normalized_sql)
                return pd.DataFrame({
                    "ticker": ["AAPL"],
                    "company_Name": ["Apple Inc."],
                    "industry": ["Technology"],
                })

            if normalized_sql.startswith("SELECT DISTINCT ticker FROM"):
                self.assertIn("WHERE ticker IN ('AAPL')", normalized_sql)
                return pd.DataFrame({"ticker": ["AAPL"]})

            if "FROM stock_price_data WHERE ticker IN ('AAPL') GROUP BY ticker" in normalized_sql and "MIN(date)" in normalized_sql:
                return pd.DataFrame({
                    "ticker": ["AAPL"],
                    "cnt": [150],
                    "earliest": [pd.Timestamp("2024-01-01")],
                    "latest": [pd.Timestamp(datetime.now().date())],
                    "min_close": [100.0],
                    "max_close": [220.0],
                })

            if "SUM(CASE WHEN close_Price IS NULL" in normalized_sql:
                return pd.DataFrame(columns=["ticker", "null_close", "null_open", "null_high", "null_low", "total"])

            if "WHERE ticker IN ('AAPL') AND high_Price < low_Price" in normalized_sql:
                return pd.DataFrame(columns=["ticker", "cnt"])

            if "trade_Volume IS NULL OR trade_Volume = 0" in normalized_sql:
                return pd.DataFrame(columns=["ticker", "zero_vol", "total"])

            if "SUM(CASE WHEN rsi_14 IS NULL" in normalized_sql:
                return pd.DataFrame({
                    "ticker": ["AAPL"],
                    "null_rsi": [0],
                    "null_sma200": [0],
                    "total": [150],
                    "first_rsi_row": [35],
                    "first_sma200_row": [201],
                })

            if "ABS((price - prev_price) / prev_price) > 0.40" in normalized_sql:
                return pd.DataFrame(columns=["ticker", "date", "price", "prev_price", "pct_change"])

            if "WHERE gap_days > 10" in normalized_sql:
                return pd.DataFrame(columns=["ticker", "gap_start", "gap_end", "gap_days"])

            if "FROM stock_income_stmt_data WHERE ticker IN ('AAPL') GROUP BY ticker" in normalized_sql:
                return pd.DataFrame({"ticker": ["AAPL"], "cnt": [4], "null_rev": [0], "null_eps": [0], "null_shares": [0]})

            if "FROM stock_balancesheet_data WHERE ticker IN ('AAPL') GROUP BY ticker" in normalized_sql:
                return pd.DataFrame({"ticker": ["AAPL"], "cnt": [4], "null_assets": [0], "null_bvps": [0]})

            if "FROM stock_cash_flow_data WHERE ticker IN ('AAPL') GROUP BY ticker" in normalized_sql:
                return pd.DataFrame({"ticker": ["AAPL"], "cnt": [4], "null_fcf": [0], "null_fcfps": [0]})

            if "FROM stock_ratio_data WHERE ticker IN ('AAPL') GROUP BY ticker" in normalized_sql:
                return pd.DataFrame({
                    "ticker": ["AAPL"],
                    "cnt": [150],
                    "null_ps": [0],
                    "null_pe": [0],
                    "null_pb": [0],
                    "null_pfcf": [0],
                    "null_fd": [0],
                    "min_ps": [1.0],
                    "max_ps": [4.0],
                    "min_pe": [10.0],
                    "max_pe": [30.0],
                    "min_pb": [2.0],
                    "max_pb": [8.0],
                    "min_pfcf": [5.0],
                    "max_pfcf": [20.0],
                })

            if "FROM stock_income_stmt_quarterly WHERE ticker IN ('AAPL') GROUP BY ticker" in normalized_sql:
                return pd.DataFrame({"ticker": ["AAPL"], "qtrs": [4], "null_rev_ttm": [3]})

            if "FROM stock_cashflow_quarterly WHERE ticker IN ('AAPL') GROUP BY ticker" in normalized_sql:
                return pd.DataFrame({"ticker": ["AAPL"], "qtrs": [4]})

            if "FROM stock_ratio_data r JOIN stock_price_data p" in normalized_sql:
                self.assertIn("WHERE r.ticker IN ('AAPL')", normalized_sql)
                return pd.DataFrame(columns=["ticker", "ratio_end", "price_end"])

            if "FROM stock_income_stmt_data i WHERE i.ticker IN ('AAPL') GROUP BY i.ticker" in normalized_sql:
                return pd.DataFrame({"ticker": ["AAPL"], "inc_p": [4], "bs_p": [4], "cf_p": [4]})

            raise AssertionError(f"Unexpected query: {normalized_sql}")

        mock_run_query.side_effect = fake_run_query

        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = os.path.join(temp_dir, "report.json")
            issues, report = validate_stock_data.validate_all(selected_tickers=["AAPL"], report_file=report_path)

            self.assertEqual(dict(issues), {})
            self.assertEqual(report["summary"]["stock_tickers"], 1)
            self.assertEqual(report["summary"]["total_issues"], 0)
            self.assertEqual(report["validation_scope"]["selected_tickers"], ["AAPL"])

            with open(report_path, "r", encoding="utf-8") as handle:
                written = json.load(handle)

            self.assertEqual(written["validation_scope"]["selected_tickers"], ["AAPL"])
            self.assertTrue(any("WHERE ticker IN ('AAPL')" in query for query in queries))

    @patch("validate_stock_data.run_structured_db_checks")
    @patch("validate_stock_data.get_engine")
    @patch("validate_stock_data.run_query")
    def test_validate_all_separates_stock_and_index_issue_counts(self, mock_run_query, mock_get_engine, mock_structured_checks):
        mock_get_engine.return_value = object()
        mock_structured_checks.return_value = {
            "summary": {"duplicate_key_groups": 0, "orphan_ticker_groups": 0}
        }

        def fake_run_query(_engine, sql):
            normalized_sql = " ".join(str(sql).split())

            if "FROM stock_info_data" in normalized_sql:
                return pd.DataFrame({
                    "ticker": ["AAPL", "^VIX"],
                    "company_Name": ["Apple Inc.", "CBOE Volatility Index"],
                    "industry": ["Technology", "Index"],
                })

            if normalized_sql.startswith("SELECT DISTINCT ticker FROM stock_price_data"):
                return pd.DataFrame({"ticker": ["AAPL", "^VIX"]})

            if normalized_sql.startswith("SELECT DISTINCT ticker FROM"):
                return pd.DataFrame({"ticker": ["AAPL"]})

            if "FROM stock_price_data GROUP BY ticker" in normalized_sql and "MIN(date)" in normalized_sql:
                return pd.DataFrame({
                    "ticker": ["AAPL", "^VIX"],
                    "cnt": [150, 100],
                    "earliest": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-01")],
                    "latest": [pd.Timestamp(datetime.now().date()), pd.Timestamp(datetime.now().date())],
                    "min_close": [100.0, 15.0],
                    "max_close": [220.0, 40.0],
                })

            if "SUM(CASE WHEN close_Price IS NULL" in normalized_sql:
                return pd.DataFrame(columns=["ticker", "null_close", "null_open", "null_high", "null_low", "total"])

            if "high_Price < low_Price" in normalized_sql:
                return pd.DataFrame(columns=["ticker", "cnt"])

            if "trade_Volume IS NULL OR trade_Volume = 0" in normalized_sql:
                return pd.DataFrame({"ticker": ["^VIX"], "zero_vol": [100], "total": [100]})

            if "SUM(CASE WHEN rsi_14 IS NULL" in normalized_sql:
                return pd.DataFrame({
                    "ticker": ["AAPL", "^VIX"],
                    "null_rsi": [0, 0],
                    "null_sma200": [0, 0],
                    "total": [150, 100],
                    "first_rsi_row": [35, 35],
                    "first_sma200_row": [201, None],
                })

            if "ABS((price - prev_price) / prev_price) > 0.40" in normalized_sql:
                return pd.DataFrame(columns=["ticker", "date", "price", "prev_price", "pct_change"])

            if "WHERE gap_days > 10" in normalized_sql:
                return pd.DataFrame(columns=["ticker", "gap_start", "gap_end", "gap_days"])

            if "FROM stock_income_stmt_data" in normalized_sql and "GROUP BY ticker" in normalized_sql and "SUM(CASE WHEN revenue IS NULL" in normalized_sql:
                return pd.DataFrame({"ticker": ["AAPL"], "cnt": [4], "null_rev": [0], "null_eps": [0], "null_shares": [0]})

            if "FROM stock_balancesheet_data" in normalized_sql and "GROUP BY ticker" in normalized_sql:
                return pd.DataFrame({"ticker": ["AAPL"], "cnt": [4], "null_assets": [0], "null_bvps": [0]})

            if "FROM stock_cash_flow_data" in normalized_sql and "GROUP BY ticker" in normalized_sql:
                return pd.DataFrame({"ticker": ["AAPL"], "cnt": [4], "null_fcf": [0], "null_fcfps": [0]})

            if "FROM stock_ratio_data" in normalized_sql and "GROUP BY ticker" in normalized_sql and "MIN(p_s)" in normalized_sql:
                return pd.DataFrame({
                    "ticker": ["AAPL"],
                    "cnt": [150],
                    "null_ps": [0],
                    "null_pe": [0],
                    "null_pb": [0],
                    "null_pfcf": [0],
                    "null_fd": [0],
                    "min_ps": [1.0],
                    "max_ps": [4.0],
                    "min_pe": [10.0],
                    "max_pe": [30.0],
                    "min_pb": [2.0],
                    "max_pb": [8.0],
                    "min_pfcf": [5.0],
                    "max_pfcf": [20.0],
                })

            if "FROM stock_income_stmt_quarterly" in normalized_sql and "GROUP BY ticker" in normalized_sql:
                return pd.DataFrame({"ticker": ["AAPL"], "qtrs": [4], "null_rev_ttm": [3]})

            if "FROM stock_cashflow_quarterly" in normalized_sql and "GROUP BY ticker" in normalized_sql:
                return pd.DataFrame({"ticker": ["AAPL"], "qtrs": [4]})

            if "FROM stock_ratio_data r JOIN stock_price_data p" in normalized_sql:
                return pd.DataFrame(columns=["ticker", "ratio_end", "price_end"])

            if "FROM stock_income_stmt_data i" in normalized_sql and "GROUP BY i.ticker" in normalized_sql:
                return pd.DataFrame({"ticker": ["AAPL"], "inc_p": [4], "bs_p": [4], "cf_p": [4]})

            raise AssertionError(f"Unexpected query: {normalized_sql}")

        mock_run_query.side_effect = fake_run_query

        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = os.path.join(temp_dir, "report.json")
            issues, report = validate_stock_data.validate_all(report_file=report_path)

            self.assertEqual(set(issues.keys()), {"^VIX"})
            self.assertEqual(report["summary"]["stock_tickers"], 1)
            self.assertEqual(report["summary"]["index_tickers"], 1)
            self.assertEqual(report["summary"]["tickers_with_issues"], 0)
            self.assertEqual(report["summary"]["index_tickers_with_issues"], 1)
            self.assertEqual(report["summary"]["all_tickers_with_issues"], 1)
            self.assertEqual(report["summary"]["clean_tickers"], 1)
            self.assertEqual(report["summary"]["clean_index_tickers"], 0)

            with open(report_path, "r", encoding="utf-8") as handle:
                written = json.load(handle)

            self.assertEqual(written["summary"]["tickers_with_issues"], 0)
            self.assertEqual(written["summary"]["index_tickers_with_issues"], 1)
            self.assertEqual(
                written["repair_classification"]["cohorts"]["index_tickers_with_issues"]["tickers"],
                ["^VIX"],
            )
            self.assertEqual(
                written["repair_classification"]["action_groups"]["manual_triage"]["tickers"],
                ["^VIX"],
            )
            self.assertEqual(
                written["repair_classification"]["ticker_actions"]["^VIX"]["primary_action"],
                "manual_triage",
            )


if __name__ == "__main__":
    unittest.main()