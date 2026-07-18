"""Unit tests for repair_ticker_cohorts.py."""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import repair_ticker_cohorts


class FakeOrchestrator:
    """Minimal orchestrator double for repair runner tests."""

    def __init__(self):
        self.calls = []

    def process_ticker(self, ticker, force_full=False, prefer_ttm=True):
        self.calls.append(("process_ticker", ticker, force_full, prefer_ttm))
        return True

    def _fetch_and_export_quarterly_data(self, ticker, force_fetch=False):
        self.calls.append(("fetch_quarterly", ticker, force_fetch))
        return True

    def process_financial_data(self, ticker, prefer_ttm=True):
        self.calls.append(("process_financial_data", ticker, prefer_ttm))
        return True, {"ticker": ticker}

    def process_ratio_data(self, ticker, prefer_ttm=True):
        self.calls.append(("process_ratio_data", ticker, prefer_ttm))
        return True, {"ticker": ticker}


class TestRepairTickerCohorts(unittest.TestCase):
    """Focused coverage for safe repair queueing and execution."""

    def setUp(self):
        self.plan = {
            "baseline_summary": {
                "total_issues": 12,
                "tickers_with_issues": 4,
                "stock_tickers": 5,
            },
            "cohorts": {
                "stale_price_tickers": {"tickers": ["AAPL"]},
                "date_gap_tickers": {"tickers": ["MSFT"]},
                "missing_quarterly_income_tickers": {"tickers": ["MSFT"]},
                "missing_quarterly_cashflow_tickers": {"tickers": []},
                "quarterly_income_null_revenue_tickers": {"tickers": ["NVDA"]},
                "ratio_null_tickers": {"tickers": ["AAPL"]},
                "ratio_lag_tickers": {"tickers": ["GOOG"]},
                "spike_tickers": {"tickers": ["^VIX"]},
                "zero_volume_tickers": {"tickers": ["ILLQ"]},
                "index_tickers_with_issues": {"tickers": ["^VIX"]},
                "financial_sector_ratio_null_tickers": {"tickers": ["JPM"]},
                "annual_only_tickers": {"tickers": ["IPOX"]},
            },
        }

    def test_build_repair_queue_combines_safe_actions(self):
        queue = repair_ticker_cohorts.build_repair_queue(self.plan)

        self.assertEqual([item["ticker"] for item in queue], ["AAPL", "GOOG", "MSFT", "NVDA"])
        self.assertEqual(queue[0]["actions"], ["refresh_incremental_pipeline", "rebuild_ratio_history"])
        self.assertEqual(queue[1]["actions"], ["rebuild_ratio_history"])
        self.assertEqual(queue[2]["actions"], ["refresh_full_pipeline", "refresh_quarterly_and_ratios"])
        self.assertEqual(queue[3]["actions"], ["refresh_quarterly_and_ratios"])

    def test_rejects_manual_only_cohorts(self):
        with self.assertRaises(ValueError) as context:
            repair_ticker_cohorts.build_repair_queue(self.plan, cohorts=["spike_tickers"])

        self.assertIn("Non-executable cohorts", str(context.exception))

    def test_rejects_separate_scope_cohorts(self):
        with self.assertRaises(ValueError) as context:
            repair_ticker_cohorts.build_repair_queue(
                self.plan,
                cohorts=["financial_sector_ratio_null_tickers", "annual_only_tickers"],
            )

        self.assertIn("financial_sector_ratio_null_tickers", str(context.exception))
        self.assertIn("annual_only_tickers", str(context.exception))

    @patch("repair_ticker_cohorts.db_interactions.delete_stock_ratio_data_from_date")
    def test_execute_queue_calls_expected_repair_actions(self, mock_delete_ratio_data):
        fake_orchestrator = FakeOrchestrator()
        mock_delete_ratio_data.return_value = 27

        queue = repair_ticker_cohorts.build_repair_queue(self.plan, tickers=["AAPL", "MSFT"])
        results = repair_ticker_cohorts.execute_repair_queue(
            queue,
            execute=True,
            orchestrator_factory=lambda: fake_orchestrator,
        )

        self.assertEqual([item["status"] for item in results], ["success", "success"])
        self.assertEqual(
            fake_orchestrator.calls,
            [
                ("process_ticker", "AAPL", False, True),
                ("process_ratio_data", "AAPL", True),
                ("process_ticker", "MSFT", True, True),
                ("fetch_quarterly", "MSFT", True),
                ("process_financial_data", "MSFT", True),
                ("process_ratio_data", "MSFT", True),
            ],
        )
        mock_delete_ratio_data.assert_called_once_with("AAPL", "1900-01-01")

    def test_writes_dry_run_report_outputs(self):
        queue = repair_ticker_cohorts.build_repair_queue(self.plan, limit=2)
        results = repair_ticker_cohorts.execute_repair_queue(queue, execute=False)
        report = repair_ticker_cohorts.build_execution_report(
            self.plan,
            "repair_cohorts.json",
            queue,
            results,
            execute=False,
            requested_cohorts=repair_ticker_cohorts.SAFE_EXECUTION_COHORTS,
            prefer_ttm=True,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            json_path = os.path.join(temp_dir, "repair_execution_report.json")
            text_path = os.path.join(temp_dir, "repair_execution_summary.txt")
            repair_ticker_cohorts.write_execution_outputs(report, json_path, text_path)

            with open(json_path, "r", encoding="utf-8") as handle:
                written = json.load(handle)
            with open(text_path, "r", encoding="utf-8") as handle:
                summary = handle.read()

        self.assertEqual(written["summary"]["queued_tickers"], 2)
        self.assertIn("Mode: dry-run", summary)
        self.assertIn("AAPL: planned", summary)


if __name__ == "__main__":
    unittest.main()