"""Build repair-first cohorts from a validation report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from validation_issue_classification import build_repair_classification


def load_validation_report(report_path: str | Path) -> dict:
    path = Path(report_path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def extract_repair_cohorts(report: dict) -> dict:
    precomputed = report.get("repair_classification")
    if precomputed and precomputed.get("cohorts"):
        return precomputed
    return build_repair_classification(report)


def render_repair_summary(plan: dict) -> str:
    lines = []
    baseline = plan.get("baseline_summary", {})
    diagnostics = plan.get("diagnostics_summary", {})
    lines.append("REPAIR-FIRST COHORT SUMMARY")
    lines.append("=" * 80)
    lines.append(f"Validation timestamp: {plan.get('run_timestamp', 'unknown')}")
    lines.append(
        "Baseline: "
        f"{baseline.get('total_issues', 0)} issues across "
        f"{baseline.get('tickers_with_issues', 0)}/{baseline.get('stock_tickers', 0)} stocks"
    )
    lines.append(
        "Structural diagnostics: "
        f"duplicate key groups={diagnostics.get('duplicate_key_groups', 0)}, "
        f"orphan ticker groups={diagnostics.get('orphan_ticker_groups', 0)}"
    )
    lines.append("")
    lines.append("Repair cohorts:")

    for cohort_name, cohort in plan.get("cohorts", {}).items():
        line = f"- {cohort_name}: {cohort['count']} ticker(s)"
        if cohort.get("issue_count") not in {None, cohort['count']}:
            line += f", {cohort['issue_count']} matching issue(s)"
        lines.append(line)

    action_groups = plan.get("action_groups", {})
    if action_groups:
        lines.append("")
        lines.append("Action groups:")
        for action_name, group in action_groups.items():
            lines.append(f"- {action_name}: {group['count']} ticker(s)")

    spike_analysis = plan.get("spike_analysis", {})
    lines.append("")
    lines.append(
        f"Spike triage: {spike_analysis.get('event_count', 0)} event(s) across "
        f"{spike_analysis.get('ticker_count', 0)} ticker(s)"
    )
    for item in spike_analysis.get("top_dates", [])[:5]:
        lines.append(f"  {item['date']}: {item['event_count']} spike event(s)")

    lines.append("")
    lines.append("Recommended execution order:")
    lines.append("1. Refresh stale-price and obvious date-gap cohorts.")
    lines.append("2. Repair quarterly income and cashflow gaps.")
    lines.append("3. Recompute ratio-null and ratio-lag cohorts.")
    lines.append("4. Manually review and then repair suspicious spike cohorts.")

    return "\n".join(lines)


def write_repair_outputs(plan: dict, output_json: str | Path, output_text: str | Path) -> None:
    json_path = Path(output_json)
    text_path = Path(output_text)

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(plan, handle, indent=2)

    with text_path.open("w", encoding="utf-8") as handle:
        handle.write(render_repair_summary(plan))
        handle.write("\n")


def _default_output_paths(report_path: str | Path) -> tuple[Path, Path]:
    report_path = Path(report_path)
    parent = report_path.parent
    return parent / "repair_cohorts.json", parent / "repair_cohort_summary.txt"


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build repair-first cohorts from a validation report")
    parser.add_argument(
        "--report",
        default="data_validation_report.json",
        help="Path to the validation report JSON",
    )
    parser.add_argument("--output-json", help="Optional output path for repair_cohorts.json")
    parser.add_argument("--output-text", help="Optional output path for repair_cohort_summary.txt")
    args = parser.parse_args(list(argv) if argv is not None else None)

    report = load_validation_report(args.report)
    plan = extract_repair_cohorts(report)
    default_json, default_text = _default_output_paths(args.report)
    output_json = args.output_json or default_json
    output_text = args.output_text or default_text
    write_repair_outputs(plan, output_json, output_text)

    print(render_repair_summary(plan))
    print(f"\nJSON output: {output_json}")
    print(f"Text summary: {output_text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())