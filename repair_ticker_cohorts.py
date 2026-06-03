"""Execute safe ticker-scoped repairs from repair cohort plans."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import db_interactions
from stock_orchestrator import StockDataOrchestrator
from validation_issue_classification import NON_EXECUTION_COHORTS, SAFE_AUTO_REPAIR_COHORTS


SAFE_EXECUTION_COHORTS = SAFE_AUTO_REPAIR_COHORTS
MANUAL_ONLY_COHORTS = NON_EXECUTION_COHORTS

QUARTERLY_REPAIR_COHORTS = {
    "missing_quarterly_income_tickers",
    "missing_quarterly_cashflow_tickers",
    "quarterly_income_null_revenue_tickers",
}

RATIO_REPAIR_COHORTS = {
    "ratio_null_tickers",
    "ratio_lag_tickers",
}


def load_repair_plan(plan_path: str | Path) -> dict:
    path = Path(plan_path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalize_requested_cohorts(cohorts: Optional[Sequence[str]]) -> List[str]:
    requested = list(cohorts or SAFE_EXECUTION_COHORTS)
    invalid = [name for name in requested if name not in SAFE_EXECUTION_COHORTS and name not in MANUAL_ONLY_COHORTS]
    if invalid:
        raise ValueError(f"Unknown cohort(s): {', '.join(sorted(invalid))}")

    non_executable = [name for name in requested if name in MANUAL_ONLY_COHORTS]
    if non_executable:
        raise ValueError(
            "Non-executable cohorts require manual or separate-scope review: "
            f"{', '.join(sorted(non_executable))}"
        )

    return requested


def derive_ticker_actions(cohort_names: Sequence[str]) -> List[str]:
    names = set(cohort_names)
    actions: List[str] = []

    if "date_gap_tickers" in names:
        actions.append("refresh_full_pipeline")
    elif "stale_price_tickers" in names:
        actions.append("refresh_incremental_pipeline")

    if names & QUARTERLY_REPAIR_COHORTS:
        actions.append("refresh_quarterly_and_ratios")
    elif names & RATIO_REPAIR_COHORTS:
        actions.append("rebuild_ratio_history")

    return actions


def build_repair_queue(
    plan: dict,
    cohorts: Optional[Sequence[str]] = None,
    tickers: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
    include_indices: bool = False,
) -> List[dict]:
    requested_cohorts = _normalize_requested_cohorts(cohorts)
    ticker_filter = {ticker.strip() for ticker in tickers} if tickers else None

    queue_by_ticker: Dict[str, dict] = {}
    plan_cohorts = plan.get("cohorts", {})

    for cohort_name in requested_cohorts:
        cohort_entry = plan_cohorts.get(cohort_name, {})
        for ticker in cohort_entry.get("tickers", []):
            if ticker_filter and ticker not in ticker_filter:
                continue
            if not include_indices and ticker.startswith("^"):
                continue

            queue_item = queue_by_ticker.setdefault(
                ticker,
                {
                    "ticker": ticker,
                    "cohorts": [],
                },
            )
            if cohort_name not in queue_item["cohorts"]:
                queue_item["cohorts"].append(cohort_name)

    queue = []
    for ticker in sorted(queue_by_ticker):
        queue_item = queue_by_ticker[ticker]
        actions = derive_ticker_actions(queue_item["cohorts"])
        if not actions:
            continue
        queue.append(
            {
                "ticker": ticker,
                "cohorts": queue_item["cohorts"],
                "actions": actions,
            }
        )

    if limit is not None:
        queue = queue[:limit]

    return queue


def _execute_action(orchestrator: StockDataOrchestrator, ticker: str, action: str, prefer_ttm: bool) -> dict:
    if action == "refresh_incremental_pipeline":
        success = bool(orchestrator.process_ticker(ticker, force_full=False, prefer_ttm=prefer_ttm))
        return {"action": action, "success": success}

    if action == "refresh_full_pipeline":
        success = bool(orchestrator.process_ticker(ticker, force_full=True, prefer_ttm=prefer_ttm))
        return {"action": action, "success": success}

    if action == "refresh_quarterly_and_ratios":
        quarterly_success = bool(orchestrator._fetch_and_export_quarterly_data(ticker, force_fetch=True))
        if not quarterly_success:
            return {
                "action": action,
                "success": False,
                "details": {"quarterly_refresh": False},
            }

        financial_success, _ = orchestrator.process_financial_data(ticker, prefer_ttm=prefer_ttm)
        if not financial_success:
            return {
                "action": action,
                "success": False,
                "details": {"quarterly_refresh": True, "financial_refresh": False},
            }

        ratio_success, _ = orchestrator.process_ratio_data(ticker, prefer_ttm=prefer_ttm)
        return {
            "action": action,
            "success": bool(ratio_success),
            "details": {
                "quarterly_refresh": True,
                "financial_refresh": True,
                "ratio_refresh": bool(ratio_success),
            },
        }

    if action == "rebuild_ratio_history":
        deleted_rows = db_interactions.delete_stock_ratio_data_from_date(ticker, "1900-01-01")
        success, _ = orchestrator.process_ratio_data(ticker, prefer_ttm=prefer_ttm)
        return {
            "action": action,
            "success": bool(success),
            "details": {"deleted_ratio_rows": deleted_rows},
        }

    raise ValueError(f"Unsupported repair action: {action}")


def execute_repair_queue(
    queue: Sequence[dict],
    execute: bool = False,
    prefer_ttm: bool = True,
    orchestrator_factory: Callable[[], StockDataOrchestrator] = StockDataOrchestrator,
) -> List[dict]:
    results: List[dict] = []

    if not execute:
        for item in queue:
            results.append(
                {
                    "ticker": item["ticker"],
                    "cohorts": list(item["cohorts"]),
                    "actions": list(item["actions"]),
                    "status": "planned",
                    "action_results": [],
                }
            )
        return results

    orchestrator = orchestrator_factory()
    for item in queue:
        action_results = []
        error = None

        for action in item["actions"]:
            try:
                action_result = _execute_action(orchestrator, item["ticker"], action, prefer_ttm=prefer_ttm)
                action_results.append(action_result)
                if not action_result["success"]:
                    break
            except Exception as exc:
                error = str(exc)
                action_results.append({"action": action, "success": False, "details": {"error": error}})
                break

        status = "success"
        if error or any(not action_result["success"] for action_result in action_results):
            status = "failed"

        results.append(
            {
                "ticker": item["ticker"],
                "cohorts": list(item["cohorts"]),
                "actions": list(item["actions"]),
                "status": status,
                "action_results": action_results,
                "error": error,
            }
        )

    return results


def build_execution_report(
    plan: dict,
    plan_path: str | Path,
    queue: Sequence[dict],
    results: Sequence[dict],
    execute: bool,
    requested_cohorts: Sequence[str],
    prefer_ttm: bool,
) -> dict:
    success_count = sum(1 for item in results if item.get("status") == "success")
    failure_count = sum(1 for item in results if item.get("status") == "failed")

    return {
        "run_timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": "execute" if execute else "dry-run",
        "source_plan": str(plan_path),
        "requested_cohorts": list(requested_cohorts),
        "prefer_ttm": prefer_ttm,
        "baseline_summary": plan.get("baseline_summary", {}),
        "queue": list(queue),
        "results": list(results),
        "summary": {
            "queued_tickers": len(queue),
            "successful_tickers": success_count,
            "failed_tickers": failure_count,
        },
    }


def render_execution_summary(report: dict) -> str:
    lines = []
    summary = report.get("summary", {})
    lines.append("REPAIR COHORT EXECUTION SUMMARY")
    lines.append("=" * 80)
    lines.append(f"Mode: {report.get('mode', 'unknown')}")
    lines.append(f"Source plan: {report.get('source_plan', 'unknown')}")
    lines.append(
        "Queued tickers: "
        f"{summary.get('queued_tickers', 0)} "
        f"(success={summary.get('successful_tickers', 0)}, failed={summary.get('failed_tickers', 0)})"
    )
    lines.append(f"Requested cohorts: {', '.join(report.get('requested_cohorts', []))}")
    lines.append("")

    for item in report.get("results", []):
        lines.append(
            f"- {item['ticker']}: {item['status']} | "
            f"cohorts={', '.join(item.get('cohorts', []))} | "
            f"actions={', '.join(item.get('actions', []))}"
        )
        for action_result in item.get("action_results", []):
            action_status = "ok" if action_result.get("success") else "failed"
            lines.append(f"  {action_result['action']}: {action_status}")

    return "\n".join(lines)


def write_execution_outputs(report: dict, output_json: str | Path, output_text: str | Path) -> None:
    json_path = Path(output_json)
    text_path = Path(output_text)

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    with text_path.open("w", encoding="utf-8") as handle:
        handle.write(render_execution_summary(report))
        handle.write("\n")


def _default_output_paths(plan_path: str | Path) -> tuple[Path, Path]:
    plan_path = Path(plan_path)
    parent = plan_path.parent
    return parent / "repair_execution_report.json", parent / "repair_execution_summary.txt"


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Execute safe ticker repairs from repair cohorts")
    parser.add_argument("--plan", default="repair_cohorts.json", help="Path to repair_cohorts.json")
    parser.add_argument("--cohorts", nargs="+", help="Specific safe cohorts to execute")
    parser.add_argument("--tickers", nargs="+", help="Optional ticker filter")
    parser.add_argument("--limit", type=int, help="Optional limit on queued tickers")
    parser.add_argument("--include-indices", action="store_true", help="Allow index tickers in the queue")
    parser.add_argument("--execute", action="store_true", help="Execute repairs instead of only producing a dry-run plan")
    parser.add_argument("--output-json", help="Optional JSON output path")
    parser.add_argument("--output-text", help="Optional text summary output path")
    parser.add_argument("--prefer-ttm", dest="prefer_ttm", action="store_true", default=True)
    parser.add_argument("--no-prefer-ttm", dest="prefer_ttm", action="store_false")
    args = parser.parse_args(list(argv) if argv is not None else None)

    requested_cohorts = _normalize_requested_cohorts(args.cohorts)
    plan = load_repair_plan(args.plan)
    queue = build_repair_queue(
        plan,
        cohorts=requested_cohorts,
        tickers=args.tickers,
        limit=args.limit,
        include_indices=args.include_indices,
    )
    results = execute_repair_queue(
        queue,
        execute=args.execute,
        prefer_ttm=args.prefer_ttm,
    )
    report = build_execution_report(
        plan,
        args.plan,
        queue,
        results,
        execute=args.execute,
        requested_cohorts=requested_cohorts,
        prefer_ttm=args.prefer_ttm,
    )

    default_json, default_text = _default_output_paths(args.plan)
    output_json = args.output_json or default_json
    output_text = args.output_text or default_text
    write_execution_outputs(report, output_json, output_text)

    print(render_execution_summary(report))
    print(f"\nJSON output: {output_json}")
    print(f"Text summary: {output_text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())