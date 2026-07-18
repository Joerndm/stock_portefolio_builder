"""Unit tests for repair_cohort_planner.py."""

import json
import os
import sys
import tempfile
import unittest


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import repair_cohort_planner


class TestRepairCohortPlanner(unittest.TestCase):
    """Focused coverage for repair cohort extraction and dry-run output."""

    def setUp(self):
        self.report = {
            "run_timestamp": "2026-05-19T11:12:12",
            "summary": {
                "stock_tickers": 5,
                "tickers_with_issues": 4,
                "total_issues": 9,
            },
            "diagnostics": {
                "summary": {
                    "duplicate_key_groups": 0,
                    "orphan_ticker_groups": 0,
                }
            },
            "ticker_metadata": {
                "AAPL": {"industry": "Technology", "is_index": False},
                "MSFT": {"industry": "Technology", "is_index": False},
                "^VIX": {"industry": "Index", "is_index": True},
                "NVDA": {"industry": "Technology", "is_index": False},
            },
            "issues_by_ticker": {
                "AAPL": [
                    {
                        "table": "stock_price_data",
                        "issue": "Stale price data - last update 2026-05-01 (18 days ago)",
                        "severity": "HIGH",
                    },
                    {
                        "table": "stock_income_stmt_quarterly",
                        "issue": "No data in stock_income_stmt_quarterly",
                        "severity": "MEDIUM",
                    },
                    {
                        "table": "stock_ratio_data",
                        "issue": "p_e NULL in 20/30 (67%)",
                        "severity": "HIGH",
                    },
                ],
                "MSFT": [
                    {
                        "table": "stock_cashflow_quarterly",
                        "issue": "No data in stock_cashflow_quarterly",
                        "severity": "MEDIUM",
                    },
                    {
                        "table": "cross-table",
                        "issue": "Ratios lag prices by 45d (ratio:2026-03-01, price:2026-04-15)",
                        "severity": "HIGH",
                    },
                ],
                "^VIX": [
                    {
                        "table": "stock_price_data",
                        "issue": "Price spike 20.85→29.55 (42%) on 2026-04-01",
                        "severity": "CRITICAL",
                    }
                ],
                "NVDA": [
                    {
                        "table": "stock_price_data",
                        "issue": "Date gap: 14d between 2026-04-01 and 2026-04-15",
                        "severity": "MEDIUM",
                    },
                    {
                        "table": "stock_income_stmt_quarterly",
                        "issue": "85 unexpected NULL revenue_ttm",
                        "severity": "HIGH",
                    },
                ],
            },
        }

    def test_extracts_expected_repair_cohorts(self):
        plan = repair_cohort_planner.extract_repair_cohorts(self.report)

        self.assertEqual(plan["cohorts"]["stale_price_tickers"]["tickers"], ["AAPL"])
        self.assertEqual(plan["cohorts"]["missing_quarterly_income_tickers"]["tickers"], ["AAPL"])
        self.assertEqual(plan["cohorts"]["missing_quarterly_cashflow_tickers"]["tickers"], ["MSFT"])
        self.assertEqual(plan["cohorts"]["ratio_null_tickers"]["tickers"], ["AAPL"])
        self.assertEqual(plan["cohorts"]["ratio_lag_tickers"]["tickers"], ["MSFT"])
        self.assertEqual(plan["cohorts"]["date_gap_tickers"]["tickers"], ["NVDA"])
        self.assertEqual(plan["cohorts"]["quarterly_income_null_revenue_tickers"]["tickers"], ["NVDA"])
        self.assertEqual(plan["cohorts"]["spike_tickers"]["tickers"], ["^VIX"])
        self.assertEqual(plan["cohorts"]["index_tickers_with_issues"]["tickers"], ["^VIX"])
        self.assertEqual(plan["action_groups"]["safe_auto_repair"]["tickers"], ["AAPL", "MSFT", "NVDA"])
        self.assertEqual(plan["action_groups"]["manual_triage"]["tickers"], ["^VIX"])
        self.assertEqual(plan["action_groups"]["separate_scope"]["tickers"], ["^VIX"])
        self.assertEqual(plan["ticker_actions"]["^VIX"]["primary_action"], "manual_triage")
        self.assertEqual(plan["spike_analysis"]["top_dates"][0], {"date": "2026-04-01", "event_count": 1})
        self.assertTrue(plan["admission_gate"]["requires_manual_spike_review"])
        self.assertTrue(plan["admission_gate"]["requires_quarterly_repair"])
        self.assertTrue(plan["admission_gate"]["requires_ratio_repair"])

    def test_separates_financial_sector_ratio_nulls_from_auto_repair(self):
        report = {
            "run_timestamp": "2026-05-19T11:12:12",
            "summary": {"stock_tickers": 1, "tickers_with_issues": 1, "total_issues": 1},
            "diagnostics": {"summary": {"duplicate_key_groups": 0, "orphan_ticker_groups": 0}},
            "ticker_metadata": {
                "BAC": {"industry": "Banks - Diversified", "is_index": False},
            },
            "issues_by_ticker": {
                "BAC": [
                    {
                        "table": "stock_ratio_data",
                        "issue": "p_e NULL in 12/12 (100%)",
                        "severity": "HIGH",
                    }
                ]
            },
        }

        plan = repair_cohort_planner.extract_repair_cohorts(report)

        self.assertEqual(plan["cohorts"]["ratio_null_tickers"]["tickers"], [])
        self.assertEqual(plan["cohorts"]["financial_sector_ratio_null_tickers"]["tickers"], ["BAC"])
        self.assertEqual(plan["ticker_actions"]["BAC"]["primary_action"], "separate_scope")

    def test_writes_json_and_text_outputs(self):
        plan = repair_cohort_planner.extract_repair_cohorts(self.report)
        with tempfile.TemporaryDirectory() as temp_dir:
            json_path = os.path.join(temp_dir, "repair_cohorts.json")
            text_path = os.path.join(temp_dir, "repair_cohort_summary.txt")

            repair_cohort_planner.write_repair_outputs(plan, json_path, text_path)

            with open(json_path, "r", encoding="utf-8") as handle:
                written = json.load(handle)
            with open(text_path, "r", encoding="utf-8") as handle:
                summary = handle.read()

            self.assertEqual(written["cohorts"]["stale_price_tickers"]["count"], 1)
            self.assertIn("Repair cohorts:", summary)
            self.assertIn("stale_price_tickers: 1 ticker(s)", summary)
            self.assertIn("Spike triage: 1 event(s) across 1 ticker(s)", summary)


if __name__ == "__main__":
    unittest.main()