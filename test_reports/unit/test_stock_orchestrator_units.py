"""Unit tests for stock_orchestrator.py."""

import datetime
import os
import sys
import threading
import unittest
from unittest.mock import Mock, patch

import pandas as pd


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import stock_orchestrator


class TestStockDataOrchestratorProcessTicker(unittest.TestCase):
    """Focused coverage for ticker success/failure accounting."""

    def _make_orchestrator(self):
        orchestrator = stock_orchestrator.StockDataOrchestrator.__new__(stock_orchestrator.StockDataOrchestrator)
        orchestrator.blacklist = Mock()
        orchestrator.blacklist.is_blacklisted.return_value = False
        orchestrator.processed_tickers = set()
        orchestrator.failed_tickers = []
        orchestrator.error_counts = {}
        orchestrator.processing_stats = {
            "total": 0,
            "success": 0,
            "skipped": 0,
            "errors": 0,
            "blacklisted": 0,
        }
        orchestrator._processed_lock = threading.Lock()
        return orchestrator

    @patch("stock_orchestrator.yf.Ticker")
    def test_process_ticker_fails_when_financial_stage_fails(self, mock_ticker):
        orchestrator = self._make_orchestrator()
        mock_ticker.return_value.info = {"regularMarketPrice": 100.0}

        orchestrator._is_index_ticker = Mock(return_value=False)
        orchestrator.process_stock_info = Mock(return_value=(True, None))
        orchestrator.process_price_data = Mock(return_value=(True, None))
        orchestrator.process_financial_data = Mock(return_value=(False, None))
        orchestrator.process_ratio_data = Mock()
        orchestrator._validate_post_fetch = Mock()

        success = orchestrator.process_ticker("AAPL")

        self.assertFalse(success)
        self.assertEqual(orchestrator.processing_stats["total"], 1)
        self.assertEqual(orchestrator.processing_stats["success"], 0)
        self.assertEqual(orchestrator.processing_stats["errors"], 1)
        self.assertEqual(orchestrator.failed_tickers, [("AAPL", "Financial data processing failed")])
        self.assertNotIn("AAPL", orchestrator.processed_tickers)
        orchestrator.process_ratio_data.assert_not_called()
        orchestrator._validate_post_fetch.assert_not_called()

    @patch("stock_orchestrator.yf.Ticker")
    def test_process_ticker_fails_when_ratio_stage_fails(self, mock_ticker):
        orchestrator = self._make_orchestrator()
        mock_ticker.return_value.info = {"regularMarketPrice": 100.0}

        orchestrator._is_index_ticker = Mock(return_value=False)
        orchestrator.process_stock_info = Mock(return_value=(True, None))
        orchestrator.process_price_data = Mock(return_value=(True, None))
        orchestrator.process_financial_data = Mock(return_value=(True, None))
        orchestrator.process_ratio_data = Mock(return_value=(False, None))
        orchestrator._validate_post_fetch = Mock()

        success = orchestrator.process_ticker("AAPL")

        self.assertFalse(success)
        self.assertEqual(orchestrator.processing_stats["total"], 1)
        self.assertEqual(orchestrator.processing_stats["success"], 0)
        self.assertEqual(orchestrator.processing_stats["errors"], 1)
        self.assertEqual(orchestrator.failed_tickers, [("AAPL", "Ratio data processing failed")])
        self.assertNotIn("AAPL", orchestrator.processed_tickers)
        orchestrator._validate_post_fetch.assert_not_called()


class TestStockDataOrchestratorProcessRatioData(unittest.TestCase):
    """Focused coverage for ratio refresh edge cases."""

    def _make_orchestrator(self):
        orchestrator = stock_orchestrator.StockDataOrchestrator.__new__(stock_orchestrator.StockDataOrchestrator)
        orchestrator.db_con = object()
        orchestrator._reconnect_database = Mock()
        return orchestrator

    @patch("stock_orchestrator.pd.read_sql")
    @patch("stock_orchestrator.db_interactions.import_stock_financial_data")
    @patch("stock_orchestrator.db_interactions.get_last_ratio_financial_date")
    @patch("stock_orchestrator.db_interactions.get_newest_financial_date")
    @patch("stock_orchestrator.db_interactions.does_stock_exists_stock_ratio_data")
    def test_process_ratio_data_treats_no_new_price_rows_as_successful_noop(
        self,
        mock_ratio_exists,
        mock_newest_financial_date,
        mock_last_ratio_financial_date,
        mock_import_financial_data,
        mock_read_sql,
    ):
        orchestrator = self._make_orchestrator()
        mock_ratio_exists.return_value = True
        mock_newest_financial_date.return_value = (datetime.date(2025, 12, 31), "annual")
        mock_last_ratio_financial_date.return_value = (datetime.date(2026, 5, 22), datetime.date(2025, 12, 31))
        mock_import_financial_data.return_value = pd.DataFrame(
            {
                "date": [pd.Timestamp("2025-12-31")],
                "ticker": ["BBOX.L"],
                "average_shares": [100.0],
            }
        )
        mock_read_sql.return_value = pd.DataFrame(columns=["date", "ticker", "close_Price"])

        success, result = orchestrator.process_ratio_data("BBOX.L")

        self.assertTrue(success)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()