"""Run safe repair cohorts in batches and revalidate touched tickers."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import repair_ticker_cohorts
import validate_stock_data


def _chunk_queue(queue: Sequence[dict], batch_size: int) -> List[List[dict]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return [list(queue[index:index + batch_size]) for index in range(0, len(queue), batch_size)]


def _summarize_validation(report: dict) -> dict:
    summary = report.get("summary", {})
    return {
        "total_issues": summary.get("total_issues", 0),
        "tickers_with_issues": summary.get("tickers_with_issues", 0),
        "severity_counts": summary.get("severity_counts", {}),
    }


def _issue_matches_cohort(issue: dict, cohort_name: str) -> bool:
    table = issue.get("table", "")
    issue_text = issue.get("issue", "")
    issue_lower = issue_text.lower()

    if cohort_name == "stale_price_tickers":
        return table == "stock_price_data" and "stale price data" in issue_lower
    if cohort_name == "date_gap_tickers":
        return table == "stock_price_data" and issue_lower.startswith("date gap")
    if cohort_name == "missing_quarterly_income_tickers":
        return table == "stock_income_stmt_quarterly" and issue_text.startswith("No data in")
    if cohort_name == "missing_quarterly_cashflow_tickers":
        return table == "stock_cashflow_quarterly" and issue_text.startswith("No data in")
    if cohort_name == "quarterly_income_null_revenue_tickers":
        return table == "stock_income_stmt_quarterly" and "null revenue_ttm" in issue_lower
    if cohort_name == "ratio_null_tickers":
        return table == "stock_ratio_data" and "null" in issue_lower
    if cohort_name == "ratio_lag_tickers":
        return table == "cross-table" and "lag" in issue_lower
    return False


def _evaluate_post_validation(batch_queue: Sequence[dict], validation_issues: dict) -> tuple[list[dict], list[dict]]:
    unresolved = []
    critical_issues = []

    for batch_item in batch_queue:
        ticker = batch_item["ticker"]
        issues = list(validation_issues.get(ticker, []))
        for issue in issues:
            if issue.get("severity") == "CRITICAL":
                critical_issues.append({"ticker": ticker, **issue})

            for cohort_name in batch_item.get("cohorts", []):
                if _issue_matches_cohort(issue, cohort_name):
                    unresolved.append({
                        "ticker": ticker,
                        "cohort": cohort_name,
                        **issue,
                    })

    return unresolved, critical_issues


def run_repair_validation_loop(
    plan_path: str | Path = "repair_cohorts.json",
    cohorts: Optional[Sequence[str]] = None,
    tickers: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
    batch_size: int = 5,
    max_batches: Optional[int] = None,
    execute: bool = False,
    stop_on_failure: bool = True,
    prefer_ttm: bool = True,
    include_indices: bool = False,
    state_file: str | Path = "repair_loop_report.json",
    validation_dir: str | Path = "repair_validation_runs",
) -> dict:
    requested_cohorts = repair_ticker_cohorts._normalize_requested_cohorts(cohorts)
    plan = repair_ticker_cohorts.load_repair_plan(plan_path)
    queue = repair_ticker_cohorts.build_repair_queue(
        plan,
        cohorts=requested_cohorts,
        tickers=tickers,
        limit=limit,
        include_indices=include_indices,
    )

    batches = _chunk_queue(queue, batch_size)
    if max_batches is not None:
        batches = batches[:max_batches]

    validation_dir_path = Path(validation_dir)
    validation_dir_path.mkdir(parents=True, exist_ok=True)

    loop_report = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": "execute" if execute else "dry-run",
        "source_plan": str(plan_path),
        "requested_cohorts": list(requested_cohorts),
        "batch_size": batch_size,
        "stop_on_failure": stop_on_failure,
        "prefer_ttm": prefer_ttm,
        "queue_size": len(queue),
        "batches": [],
        "summary": {
            "completed_batches": 0,
            "stopped_early": False,
            "failed_batches": 0,
        },
    }

    for batch_index, batch_queue in enumerate(batches, start=1):
        repair_results = repair_ticker_cohorts.execute_repair_queue(
            batch_queue,
            execute=execute,
            prefer_ttm=prefer_ttm,
        )
        batch_tickers = [item["ticker"] for item in batch_queue]
        repair_failures = [item for item in repair_results if item.get("status") == "failed"]

        validation_output_path = validation_dir_path / f"batch_{batch_index:04d}_validation_output.txt"
        validation_report_path = validation_dir_path / f"batch_{batch_index:04d}_validation_report.json"
        validation_issues, validation_report = validate_stock_data.run_validation(
            selected_tickers=batch_tickers,
            output_file=str(validation_output_path),
            report_file=str(validation_report_path),
        )
        unresolved_validation, critical_post_validation = _evaluate_post_validation(batch_queue, validation_issues)
        batch_status = "success"
        if repair_failures or unresolved_validation or critical_post_validation:
            batch_status = "failed"

        batch_record = {
            "batch_index": batch_index,
            "status": batch_status,
            "tickers": batch_tickers,
            "repair_results": repair_results,
            "repair_failures": [item["ticker"] for item in repair_failures],
            "validation_summary": _summarize_validation(validation_report),
            "validation_report": str(validation_report_path),
            "validation_output": str(validation_output_path),
            "issues_by_ticker": {ticker: list(items) for ticker, items in validation_issues.items()},
            "unresolved_validation": unresolved_validation,
            "critical_post_validation": critical_post_validation,
        }
        loop_report["batches"].append(batch_record)
        loop_report["summary"]["completed_batches"] = batch_index

        if batch_status == "failed":
            loop_report["summary"]["failed_batches"] += 1
            if stop_on_failure:
                loop_report["summary"]["stopped_early"] = True
                break

        with Path(state_file).open("w", encoding="utf-8") as handle:
            json.dump(loop_report, handle, indent=2)

    with Path(state_file).open("w", encoding="utf-8") as handle:
        json.dump(loop_report, handle, indent=2)

    return loop_report


def render_loop_summary(loop_report: dict) -> str:
    lines = []
    summary = loop_report.get("summary", {})
    lines.append("REPAIR VALIDATION LOOP SUMMARY")
    lines.append("=" * 80)
    lines.append(f"Mode: {loop_report.get('mode', 'unknown')}")
    lines.append(f"Source plan: {loop_report.get('source_plan', 'unknown')}")
    lines.append(f"Queue size: {loop_report.get('queue_size', 0)}")
    lines.append(
        f"Completed batches: {summary.get('completed_batches', 0)} | "
        f"failed batches: {summary.get('failed_batches', 0)} | "
        f"stopped early: {summary.get('stopped_early', False)}"
    )
    lines.append("")

    for batch in loop_report.get("batches", []):
        validation_summary = batch.get("validation_summary", {})
        lines.append(
            f"- Batch {batch['batch_index']}: tickers={', '.join(batch.get('tickers', []))} | "
            f"status={batch.get('status', 'unknown')} | "
            f"repair failures={len(batch.get('repair_failures', []))} | "
            f"unresolved targeted={len(batch.get('unresolved_validation', []))} | "
            f"critical post-validation={len(batch.get('critical_post_validation', []))} | "
            f"post-validation issues={validation_summary.get('total_issues', 0)}"
        )

    return "\n".join(lines)


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run safe repair cohorts in batches with revalidation")
    parser.add_argument("--plan", default="repair_cohorts.json", help="Path to repair_cohorts.json")
    parser.add_argument("--cohorts", nargs="+", help="Specific safe cohorts to include")
    parser.add_argument("--tickers", nargs="+", help="Optional ticker subset")
    parser.add_argument("--limit", type=int, help="Optional limit on queued tickers")
    parser.add_argument("--batch-size", type=int, default=5, help="Tickers to repair and revalidate per batch")
    parser.add_argument("--max-batches", type=int, help="Optional cap on number of batches to run")
    parser.add_argument("--include-indices", action="store_true", help="Allow index tickers in the queue")
    parser.add_argument("--execute", action="store_true", help="Execute repairs instead of dry-run planning")
    parser.add_argument("--continue-on-failure", action="store_true", help="Keep running after failed repair batches")
    parser.add_argument("--state-file", default="repair_loop_report.json", help="JSON state/report output path")
    parser.add_argument("--validation-dir", default="repair_validation_runs", help="Directory for per-batch validation outputs")
    parser.add_argument("--prefer-ttm", dest="prefer_ttm", action="store_true", default=True)
    parser.add_argument("--no-prefer-ttm", dest="prefer_ttm", action="store_false")
    args = parser.parse_args(list(argv) if argv is not None else None)

    loop_report = run_repair_validation_loop(
        plan_path=args.plan,
        cohorts=args.cohorts,
        tickers=args.tickers,
        limit=args.limit,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        execute=args.execute,
        stop_on_failure=not args.continue_on_failure,
        prefer_ttm=args.prefer_ttm,
        include_indices=args.include_indices,
        state_file=args.state_file,
        validation_dir=args.validation_dir,
    )
    print(render_loop_summary(loop_report))
    print(f"\nState file: {args.state_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())