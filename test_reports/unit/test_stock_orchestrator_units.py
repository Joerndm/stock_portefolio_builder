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


class TestStockDataOrchestratorUpdatePriceData(unittest.TestCase):
    """Focused coverage for incremental price update sanitation."""

    def _make_orchestrator(self):
        orchestrator = stock_orchestrator.StockDataOrchestrator.__new__(stock_orchestrator.StockDataOrchestrator)
        orchestrator._validate_price_continuity = Mock(side_effect=lambda ticker, price_df: price_df)
        return orchestrator

    @patch("stock_orchestrator.db_interactions.export_stock_price_data")
    @patch("stock_orchestrator.db_interactions.delete_price_dates_across_related_tables")
    @patch("stock_orchestrator.db_interactions.import_stock_price_data")
    @patch("market_hours_utils.should_fetch_new_data")
    @patch("stock_data_fetch.fetch_stock_price_data")
    @patch("stock_data_fetch.calculate_period_returns")
    @patch("stock_data_fetch.add_technical_indicators")
    @patch("stock_data_fetch.add_volume_indicators")
    @patch("stock_data_fetch.add_volatility_indicators")
    @patch("stock_data_fetch.calculate_moving_averages")
    @patch("stock_data_fetch.calculate_standard_diviation_value")
    @patch("stock_data_fetch.calculate_bollinger_bands")
    @patch("stock_data_fetch.calculate_momentum")
    @patch("stock_orchestrator.add_all_technical_patterns")
    def test_update_price_data_drops_invalid_historical_rows_before_recalculation(
        self,
        mock_add_patterns,
        mock_momentum,
        mock_bollinger,
        mock_std,
        mock_ma,
        mock_volatility,
        mock_volume,
        mock_technical,
        mock_returns,
        mock_fetch_price,
        mock_should_fetch,
        mock_import_price,
        mock_delete_bad_dates,
        mock_export_price,
    ):
        orchestrator = self._make_orchestrator()
        orchestrator.db_con = object()

        latest_row = pd.DataFrame(
            {
                "date": [pd.Timestamp("2026-05-31")],
                "ticker": ["AAPL"],
                "close_Price": [100.0],
                "open_Price": [99.0],
                "high_Price": [101.0],
                "low_Price": [98.5],
                "trade_Volume": [1_000_000],
            }
        )
        historical_rows = pd.DataFrame(
            {
                "date": [pd.Timestamp("2026-05-30"), pd.Timestamp("2026-05-31")],
                "ticker": ["AAPL", "AAPL"],
                "close_Price": [None, 100.0],
                "open_Price": [98.0, 99.0],
                "high_Price": [101.0, 101.0],
                "low_Price": [97.5, 98.5],
                "trade_Volume": [900_000, 1_000_000],
                "1D": [0.01, 0.02],
            }
        )
        fresh_rows = pd.DataFrame(
            {
                "date": [pd.Timestamp("2026-06-01")],
                "ticker": ["AAPL"],
                "close_Price": [102.0],
                "open_Price": [101.0],
                "high_Price": [103.0],
                "low_Price": [100.5],
                "trade_Volume": [1_100_000],
            }
        )

        mock_import_price.side_effect = [latest_row, historical_rows]
        mock_should_fetch.return_value = (True, datetime.date(2026, 6, 1), "fetch")
        mock_fetch_price.return_value = fresh_rows
        mock_delete_bad_dates.return_value = {"stock_price_data": 1, "stock_ratio_data": 1}

        def passthrough(df):
            return df

        mock_returns.side_effect = passthrough
        mock_technical.side_effect = passthrough
        mock_volume.side_effect = passthrough
        mock_volatility.side_effect = passthrough
        mock_ma.side_effect = passthrough
        mock_std.side_effect = passthrough
        mock_bollinger.side_effect = passthrough
        mock_momentum.side_effect = passthrough
        mock_add_patterns.side_effect = passthrough

        success, exported_df = orchestrator._update_price_data("AAPL", stock_info={}, is_index=False)

        self.assertTrue(success)
        self.assertIsNotNone(exported_df)
        self.assertEqual(list(exported_df["date"]), [pd.Timestamp("2026-06-01")])
        self.assertEqual(exported_df["close_Price"].tolist(), [102.0])

        recalculation_input = mock_returns.call_args.args[0]
        self.assertEqual(recalculation_input["date"].tolist(), [pd.Timestamp("2026-05-31"), pd.Timestamp("2026-06-01")])
        self.assertFalse(recalculation_input["close_Price"].isna().any())
        mock_delete_bad_dates.assert_called_once()
        mock_export_price.assert_called_once()

    @patch("stock_orchestrator.db_interactions.migrate_legacy_ticker")
    def test_repair_legacy_tickers_canonicalizes_and_deduplicates(self, mock_migrate):
        orchestrator = self._make_orchestrator()
        orchestrator.blacklist = Mock()
        orchestrator.blacklist.is_blacklisted.return_value = False
        orchestrator.db_con = object()

        repaired = orchestrator._repair_legacy_tickers([
            "EURONEXT-BRUSSELS: SOF.BR",
            "BF.B",
            "SOF.BR",
        ])

        self.assertEqual(repaired, ["SOF.BR", "BF-B"])
        self.assertEqual(mock_migrate.call_count, 2)


class TestStockDataOrchestratorDatabaseStartup(unittest.TestCase):
    """Focused coverage for DB startup validation and diagnostics."""

    def _make_orchestrator(self):
        orchestrator = stock_orchestrator.StockDataOrchestrator.__new__(stock_orchestrator.StockDataOrchestrator)
        orchestrator.db_con = None
        return orchestrator

    @patch("builtins.print")
    @patch("stock_orchestrator.db_connectors.pandas_mysql_connector")
    @patch("stock_orchestrator.fetch_secrets.secret_import")
    def test_connect_database_success_probes_before_reporting_success(
        self,
        mock_secret_import,
        mock_db_connector,
        mock_print,
    ):
        orchestrator = self._make_orchestrator()
        orchestrator._probe_database_connection = Mock()

        mock_secret_import.return_value = ("db", "stock_user", "stock_pass", "stock_db")
        fake_engine = Mock()
        mock_db_connector.return_value = fake_engine

        orchestrator._connect_database()

        self.assertIs(orchestrator.db_con, fake_engine)
        orchestrator._probe_database_connection.assert_called_once_with(fake_engine)
        mock_print.assert_any_call("✓ Database connection established")

    @patch("builtins.print")
    @patch("stock_orchestrator.db_connectors.pandas_mysql_connector")
    @patch("stock_orchestrator.fetch_secrets.secret_import")
    def test_connect_database_prints_actionable_hint_for_host_local_db_hostname(
        self,
        mock_secret_import,
        mock_db_connector,
        mock_print,
    ):
        orchestrator = self._make_orchestrator()

        mock_secret_import.return_value = ("db", "stock_user", "stock_pass", "stock_db")
        mock_db_connector.side_effect = Exception("Errno 11001: getaddrinfo failed")

        with patch.object(stock_orchestrator.StockDataOrchestrator, "_is_running_in_container", return_value=False):
            orchestrator._connect_database()

        self.assertIsNone(orchestrator.db_con)
        print_messages = [args[0] for args, _ in mock_print.call_args_list if args]
        self.assertTrue(any("Database connection failed" in msg for msg in print_messages))
        self.assertTrue(any("DB_HOST=db resolves only inside docker compose networking" in msg for msg in print_messages))

    @patch("builtins.print")
    @patch("stock_orchestrator.db_connectors.pandas_mysql_connector")
    @patch("stock_orchestrator.fetch_secrets.secret_import")
    def test_connect_database_skips_host_local_hint_inside_container(
        self,
        mock_secret_import,
        mock_db_connector,
        mock_print,
    ):
        orchestrator = self._make_orchestrator()

        mock_secret_import.return_value = ("db", "stock_user", "stock_pass", "stock_db")
        mock_db_connector.side_effect = Exception("Errno 11001: getaddrinfo failed")

        with patch.object(stock_orchestrator.StockDataOrchestrator, "_is_running_in_container", return_value=True):
            orchestrator._connect_database()

        self.assertIsNone(orchestrator.db_con)
        print_messages = [args[0] for args, _ in mock_print.call_args_list if args]
        self.assertTrue(any("Database connection failed" in msg for msg in print_messages))
        self.assertFalse(any("DB_HOST=db resolves only inside docker compose networking" in msg for msg in print_messages))

    def test_build_db_connection_hint_reports_missing_env_values(self):
        orchestrator = self._make_orchestrator()

        hint = orchestrator._build_db_connection_hint(None, "")

        self.assertIn("Set DB_HOST, DB_USER, DB_PASSWORD", hint)


if __name__ == "__main__":
    unittest.main()