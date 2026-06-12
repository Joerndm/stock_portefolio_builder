"""Unit tests for repair_validation_loop.py."""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import repair_validation_loop


class TestRepairValidationLoop(unittest.TestCase):
    """Focused coverage for batched repair-validation execution."""

    def test_chunk_queue(self):
        queue = [{"ticker": "A"}, {"ticker": "B"}, {"ticker": "C"}]

        chunks = repair_validation_loop._chunk_queue(queue, batch_size=2)

        self.assertEqual(chunks, [[{"ticker": "A"}, {"ticker": "B"}], [{"ticker": "C"}]])

    def test_evaluates_unresolved_targeted_and_critical_issues(self):
        batch_queue = [
            {"ticker": "AHT.L", "cohorts": ["stale_price_tickers"], "actions": ["refresh_incremental_pipeline"]}
        ]
        validation_issues = {
            "AHT.L": [
                {"table": "stock_price_data", "issue": "Stale price data - last update 2026-03-13 (70 days ago)", "severity": "HIGH"},
                {"table": "stock_price_data", "issue": "Price spike 5190.00→399.70 (92%) on 2026-02-26", "severity": "CRITICAL"},
            ]
        }

        unresolved, critical = repair_validation_loop._evaluate_post_validation(batch_queue, validation_issues)

        self.assertEqual(len(unresolved), 1)
        self.assertEqual(unresolved[0]["cohort"], "stale_price_tickers")
        self.assertEqual(len(critical), 1)

    @patch("repair_validation_loop.validate_stock_data.run_validation")
    @patch("repair_validation_loop.repair_ticker_cohorts.execute_repair_queue")
    @patch("repair_validation_loop.repair_ticker_cohorts.build_repair_queue")
    @patch("repair_validation_loop.repair_ticker_cohorts.load_repair_plan")
    def test_runs_batched_repairs_and_writes_state(
        self,
        mock_load_plan,
        mock_build_queue,
        mock_execute_queue,
        mock_run_validation,
    ):
        mock_load_plan.return_value = {"cohorts": {}}
        mock_build_queue.return_value = [
            {"ticker": "AAPL", "cohorts": ["stale_price_tickers"], "actions": ["refresh_incremental_pipeline"]},
            {"ticker": "MSFT", "cohorts": ["date_gap_tickers"], "actions": ["refresh_full_pipeline"]},
            {"ticker": "NVDA", "cohorts": ["ratio_null_tickers"], "actions": ["rebuild_ratio_history"]},
        ]
        mock_execute_queue.side_effect = [
            [
                {"ticker": "AAPL", "status": "success", "actions": ["refresh_incremental_pipeline"]},
                {"ticker": "MSFT", "status": "success", "actions": ["refresh_full_pipeline"]},
            ],
            [
                {"ticker": "NVDA", "status": "success", "actions": ["rebuild_ratio_history"]},
            ],
        ]
        mock_run_validation.side_effect = [
            ({}, {"summary": {"total_issues": 1, "tickers_with_issues": 1, "severity_counts": {"HIGH": 1}}}),
            ({}, {"summary": {"total_issues": 0, "tickers_with_issues": 0, "severity_counts": {}}}),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = os.path.join(temp_dir, "loop_report.json")
            validation_dir = os.path.join(temp_dir, "validations")
            loop_report = repair_validation_loop.run_repair_validation_loop(
                plan_path="repair_cohorts.json",
                batch_size=2,
                execute=True,
                state_file=state_path,
                validation_dir=validation_dir,
            )

            with open(state_path, "r", encoding="utf-8") as handle:
                written = json.load(handle)

        self.assertEqual(loop_report["summary"]["completed_batches"], 2)
        self.assertFalse(loop_report["summary"]["stopped_early"])
        self.assertEqual(len(loop_report["batches"]), 2)
        self.assertEqual(loop_report["batches"][0]["status"], "success")
        self.assertEqual(written["summary"]["completed_batches"], 2)
        self.assertEqual(mock_run_validation.call_count, 2)
        first_validation_call = mock_run_validation.call_args_list[0].kwargs
        self.assertEqual(first_validation_call["selected_tickers"], ["AAPL", "MSFT"])

    @patch("repair_validation_loop.validate_stock_data.run_validation")
    @patch("repair_validation_loop.repair_ticker_cohorts.execute_repair_queue")
    @patch("repair_validation_loop.repair_ticker_cohorts.build_repair_queue")
    @patch("repair_validation_loop.repair_ticker_cohorts.load_repair_plan")
    def test_stops_early_on_failed_batch(
        self,
        mock_load_plan,
        mock_build_queue,
        mock_execute_queue,
        mock_run_validation,
    ):
        mock_load_plan.return_value = {"cohorts": {}}
        mock_build_queue.return_value = [
            {"ticker": "AAPL", "cohorts": ["stale_price_tickers"], "actions": ["refresh_incremental_pipeline"]},
            {"ticker": "MSFT", "cohorts": ["date_gap_tickers"], "actions": ["refresh_full_pipeline"]},
        ]
        mock_execute_queue.return_value = [
            {"ticker": "AAPL", "status": "failed", "actions": ["refresh_incremental_pipeline"]},
            {"ticker": "MSFT", "status": "success", "actions": ["refresh_full_pipeline"]},
        ]
        mock_run_validation.return_value = ({}, {"summary": {"total_issues": 2, "tickers_with_issues": 1, "severity_counts": {"HIGH": 2}}})

        with tempfile.TemporaryDirectory() as temp_dir:
            loop_report = repair_validation_loop.run_repair_validation_loop(
                plan_path="repair_cohorts.json",
                batch_size=5,
                execute=True,
                state_file=os.path.join(temp_dir, "loop_report.json"),
                validation_dir=os.path.join(temp_dir, "validations"),
            )

        self.assertTrue(loop_report["summary"]["stopped_early"])
        self.assertEqual(loop_report["summary"]["failed_batches"], 1)
        self.assertEqual(len(loop_report["batches"]), 1)

    @patch("repair_validation_loop.validate_stock_data.run_validation")
    @patch("repair_validation_loop.repair_ticker_cohorts.execute_repair_queue")
    @patch("repair_validation_loop.repair_ticker_cohorts.build_repair_queue")
    @patch("repair_validation_loop.repair_ticker_cohorts.load_repair_plan")
    def test_marks_batch_failed_when_targeted_issue_remains_after_validation(
        self,
        mock_load_plan,
        mock_build_queue,
        mock_execute_queue,
        mock_run_validation,
    ):
        mock_load_plan.return_value = {"cohorts": {}}
        mock_build_queue.return_value = [
            {"ticker": "AHT.L", "cohorts": ["stale_price_tickers"], "actions": ["refresh_incremental_pipeline"]},
        ]
        mock_execute_queue.return_value = [
            {"ticker": "AHT.L", "status": "success", "actions": ["refresh_incremental_pipeline"]},
        ]
        mock_run_validation.return_value = (
            {
                "AHT.L": [
                    {"table": "stock_price_data", "issue": "Stale price data - last update 2026-03-13 (70 days ago)", "severity": "HIGH"},
                    {"table": "stock_price_data", "issue": "Price spike 5190.00→399.70 (92%) on 2026-02-26", "severity": "CRITICAL"},
                ]
            },
            {"summary": {"total_issues": 2, "tickers_with_issues": 1, "severity_counts": {"CRITICAL": 1, "HIGH": 1}}},
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            loop_report = repair_validation_loop.run_repair_validation_loop(
                plan_path="repair_cohorts.json",
                batch_size=1,
                execute=True,
                state_file=os.path.join(temp_dir, "loop_report.json"),
                validation_dir=os.path.join(temp_dir, "validations"),
            )

        self.assertTrue(loop_report["summary"]["stopped_early"])
        self.assertEqual(loop_report["summary"]["failed_batches"], 1)
        self.assertEqual(loop_report["batches"][0]["status"], "failed")
        self.assertEqual(len(loop_report["batches"][0]["unresolved_validation"]), 1)
        self.assertEqual(len(loop_report["batches"][0]["critical_post_validation"]), 1)


if __name__ == "__main__":
    unittest.main()