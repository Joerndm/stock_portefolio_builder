"""Unit tests for data_quality_audit.py."""

import os
import sys
import unittest
from collections import defaultdict

import pandas as pd


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import data_quality_audit


class TestStructuredDBChecks(unittest.TestCase):
    """Focused coverage for reusable duplicate/orphan diagnostics."""

    def test_reports_duplicate_and_orphan_findings(self):
        logs = []
        issues = defaultdict(list)

        def fake_run_query(_engine, sql):
            sql = " ".join(sql.split())
            if "FROM stock_price_data GROUP BY ticker, date HAVING COUNT(*) > 1" in sql:
                return pd.DataFrame({
                    "ticker": ["AAPL"],
                    "date": [pd.Timestamp("2024-01-02")],
                    "duplicate_count": [2],
                })
            if "FROM stock_ratio_data GROUP BY ticker, date HAVING COUNT(*) > 1" in sql:
                return pd.DataFrame({
                    "ticker": ["MSFT"],
                    "date": [pd.Timestamp("2024-01-03")],
                    "duplicate_count": [3],
                })
            if "FROM stock_price_data data LEFT JOIN stock_info_data info" in sql:
                return pd.DataFrame({"ticker": ["ORPHAN1"], "orphan_rows": [4]})
            if "FROM stock_income_stmt_data data LEFT JOIN stock_info_data info" in sql:
                return pd.DataFrame({"ticker": ["ORPHAN2"], "orphan_rows": [1]})
            return pd.DataFrame()

        diagnostics = data_quality_audit.run_structured_db_checks(
            engine=object(),
            run_query=fake_run_query,
            issues=issues,
            log=logs.append,
            max_sample_rows=5,
        )

        self.assertEqual(diagnostics["summary"]["duplicate_key_groups"], 2)
        self.assertEqual(diagnostics["summary"]["orphan_ticker_groups"], 2)
        self.assertEqual(
            diagnostics["duplicate_key_checks"]["stock_price_duplicate_keys"]["affected_groups"],
            1,
        )
        self.assertEqual(
            diagnostics["orphan_ticker_checks"]["stock_price_orphans"]["affected_tickers"],
            1,
        )
        self.assertTrue(any("Duplicate key group" in item["issue"] for item in issues["AAPL"]))
        self.assertTrue(any("Orphaned ticker rows" in item["issue"] for item in issues["ORPHAN1"]))
        self.assertTrue(any("duplicate primary-key groups" in entry for entry in logs))
        self.assertTrue(any("orphaned ticker rows" in entry.lower() for entry in logs))

    def test_handles_clean_database_results(self):
        issues = defaultdict(list)
        logs = []

        diagnostics = data_quality_audit.run_structured_db_checks(
            engine=object(),
            run_query=lambda _engine, _sql: pd.DataFrame(),
            issues=issues,
            log=logs.append,
        )

        self.assertEqual(diagnostics["summary"]["duplicate_key_groups"], 0)
        self.assertEqual(diagnostics["summary"]["orphan_ticker_groups"], 0)
        self.assertEqual(dict(issues), {})
        self.assertTrue(any("0 duplicate key group(s)" in entry for entry in logs))
        self.assertTrue(any("0 orphan ticker(s)" in entry for entry in logs))


if __name__ == "__main__":
    unittest.main()