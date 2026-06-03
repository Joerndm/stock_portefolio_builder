"""Shared classification helpers for validation reports and repair planning."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Mapping


SPIKE_DATE_RE = re.compile(r"on\s+(\d{4}-\d{2}-\d{2})")
SPIKE_PCT_RE = re.compile(r"\(([-+]?\d+(?:\.\d+)?)%\)")

FINANCIAL_INDUSTRY_HINTS = (
    "bank",
    "banks",
    "insurance",
    "financial",
    "capital markets",
    "asset management",
    "consumer finance",
    "credit services",
    "investment",
    "brokerage",
)

COHORT_DEFINITIONS: dict[str, dict[str, str]] = {
    "stale_price_tickers": {
        "description": "Tickers with stale price data that can be refreshed safely.",
        "action_class": "safe_auto_repair",
    },
    "date_gap_tickers": {
        "description": "Tickers with date gaps that likely need price refresh or review.",
        "action_class": "safe_auto_repair",
    },
    "missing_quarterly_income_tickers": {
        "description": "Tickers missing quarterly income data.",
        "action_class": "safe_auto_repair",
    },
    "missing_quarterly_cashflow_tickers": {
        "description": "Tickers missing quarterly cashflow data.",
        "action_class": "safe_auto_repair",
    },
    "quarterly_income_null_revenue_tickers": {
        "description": "Tickers with unexpected NULL quarterly revenue fields.",
        "action_class": "safe_auto_repair",
    },
    "ratio_null_tickers": {
        "description": "Tickers with high-severity ratio NULL findings suitable for ratio recomputation.",
        "action_class": "safe_auto_repair",
    },
    "ratio_lag_tickers": {
        "description": "Tickers where ratios lag price data and should be recomputed from the newest financial date.",
        "action_class": "safe_auto_repair",
    },
    "zero_volume_tickers": {
        "description": "Tickers with high zero or NULL trading volume ratios.",
        "action_class": "manual_triage",
    },
    "spike_tickers": {
        "description": "Tickers with critical price spike findings that require triage before repair.",
        "action_class": "manual_triage",
    },
    "index_tickers_with_issues": {
        "description": "Index-like tickers that should be reviewed separately from stock cohorts.",
        "action_class": "separate_scope",
    },
    "financial_sector_ratio_null_tickers": {
        "description": "Financial-sector tickers with ratio NULL findings that should be tracked separately from common-stock repairs.",
        "action_class": "separate_scope",
    },
    "annual_only_tickers": {
        "description": "Tickers with only short annual history that should be tracked separately from repair automation.",
        "action_class": "separate_scope",
    },
}

ACTION_GROUP_DESCRIPTIONS = {
    "safe_auto_repair": "Tickers with findings suitable for automated ticker-scoped repair.",
    "manual_triage": "Tickers that require manual review before repair.",
    "separate_scope": "Tickers that should be tracked separately from common-stock repair automation.",
    "unclassified": "Tickers with findings that are not yet mapped to a repair cohort.",
}

SAFE_AUTO_REPAIR_COHORTS = tuple(
    name for name, meta in COHORT_DEFINITIONS.items() if meta["action_class"] == "safe_auto_repair"
)
MANUAL_TRIAGE_COHORTS = tuple(
    name for name, meta in COHORT_DEFINITIONS.items() if meta["action_class"] == "manual_triage"
)
SEPARATE_SCOPE_COHORTS = tuple(
    name for name, meta in COHORT_DEFINITIONS.items() if meta["action_class"] == "separate_scope"
)
NON_EXECUTION_COHORTS = MANUAL_TRIAGE_COHORTS + SEPARATE_SCOPE_COHORTS


def _empty_cohort_map() -> dict[str, set[str]]:
    return {key: set() for key in COHORT_DEFINITIONS}


def _empty_issue_count_map() -> dict[str, int]:
    return {key: 0 for key in COHORT_DEFINITIONS}


def _record_cohort_match(
    cohorts: dict[str, set[str]],
    issue_counts: dict[str, int],
    name: str,
    ticker: str,
) -> None:
    cohorts[name].add(ticker)
    issue_counts[name] += 1


def _parse_spike_event(issue_text: str, ticker: str) -> dict[str, Any]:
    date_match = SPIKE_DATE_RE.search(issue_text)
    pct_match = SPIKE_PCT_RE.search(issue_text)
    return {
        "ticker": ticker,
        "issue": issue_text,
        "date": date_match.group(1) if date_match else None,
        "pct_change": float(pct_match.group(1)) if pct_match else None,
    }


def _is_financial_industry(industry: object) -> bool:
    normalized = str(industry or "").strip().lower()
    if not normalized:
        return False
    return any(hint in normalized for hint in FINANCIAL_INDUSTRY_HINTS)


def _cohort_payload(
    cohorts: Mapping[str, set[str]],
    issue_counts: Mapping[str, int],
) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "description": COHORT_DEFINITIONS[name]["description"],
            "action_class": COHORT_DEFINITIONS[name]["action_class"],
            "count": len(tickers),
            "issue_count": issue_counts[name],
            "tickers": sorted(tickers),
        }
        for name, tickers in cohorts.items()
    }


def _build_ticker_actions(
    cohorts: Mapping[str, set[str]],
    ticker_metadata: Mapping[str, Mapping[str, Any]],
    issue_tickers: set[str],
) -> dict[str, dict[str, Any]]:
    cohorts_by_ticker: dict[str, list[str]] = defaultdict(list)
    for cohort_name, tickers in cohorts.items():
        for ticker in tickers:
            cohorts_by_ticker[ticker].append(cohort_name)

    ticker_actions: dict[str, dict[str, Any]] = {}
    for ticker in sorted(issue_tickers):
        cohort_names = sorted(cohorts_by_ticker.get(ticker, []))
        metadata = ticker_metadata.get(ticker, {})

        if any(name in MANUAL_TRIAGE_COHORTS for name in cohort_names):
            primary_action = "manual_triage"
        elif any(name in SEPARATE_SCOPE_COHORTS for name in cohort_names):
            primary_action = "separate_scope"
        elif any(name in SAFE_AUTO_REPAIR_COHORTS for name in cohort_names):
            primary_action = "safe_auto_repair"
        else:
            primary_action = "unclassified"

        ticker_actions[ticker] = {
            "cohorts": cohort_names,
            "primary_action": primary_action,
            "industry": metadata.get("industry"),
            "is_index": bool(metadata.get("is_index")),
        }

    return ticker_actions


def _build_action_groups(
    cohorts: Mapping[str, set[str]],
    ticker_actions: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    grouped_tickers = {
        action_name: set()
        for action_name in ACTION_GROUP_DESCRIPTIONS
        if action_name != "unclassified"
    }
    for cohort_name, tickers in cohorts.items():
        action_class = COHORT_DEFINITIONS[cohort_name]["action_class"]
        grouped_tickers[action_class].update(tickers)

    grouped_tickers["unclassified"] = {
        ticker
        for ticker, action in ticker_actions.items()
        if action["primary_action"] == "unclassified"
    }

    return {
        action_name: {
            "description": ACTION_GROUP_DESCRIPTIONS[action_name],
            "count": len(tickers),
            "tickers": sorted(tickers),
        }
        for action_name, tickers in grouped_tickers.items()
    }


def build_repair_classification(report: Mapping[str, Any]) -> dict[str, Any]:
    cohorts = _empty_cohort_map()
    issue_counts = _empty_issue_count_map()
    spike_events: list[dict[str, Any]] = []
    ticker_metadata = report.get("ticker_metadata", {})
    issues_by_ticker = report.get("issues_by_ticker", {})

    for ticker, issue_list in issues_by_ticker.items():
        metadata = ticker_metadata.get(ticker, {})
        is_index = bool(metadata.get("is_index")) or ticker.startswith("^")
        is_financial = _is_financial_industry(metadata.get("industry"))

        if is_index:
            _record_cohort_match(cohorts, issue_counts, "index_tickers_with_issues", ticker)

        for issue in issue_list:
            table = issue.get("table", "")
            severity = issue.get("severity", "")
            issue_text = issue.get("issue", "")
            issue_lower = issue_text.lower()

            if table == "stock_price_data" and issue_text.startswith("Price spike"):
                _record_cohort_match(cohorts, issue_counts, "spike_tickers", ticker)
                spike_events.append(_parse_spike_event(issue_text, ticker))

            if table == "stock_price_data" and "zero/null volume" in issue_lower:
                _record_cohort_match(cohorts, issue_counts, "zero_volume_tickers", ticker)

            if is_index:
                continue

            if table == "stock_price_data" and "stale price" in issue_lower:
                _record_cohort_match(cohorts, issue_counts, "stale_price_tickers", ticker)

            if table == "stock_price_data" and issue_lower.startswith("date gap"):
                _record_cohort_match(cohorts, issue_counts, "date_gap_tickers", ticker)

            if table == "stock_income_stmt_quarterly" and issue_text.startswith("No data in"):
                _record_cohort_match(cohorts, issue_counts, "missing_quarterly_income_tickers", ticker)

            if table == "stock_cashflow_quarterly" and issue_text.startswith("No data in"):
                _record_cohort_match(cohorts, issue_counts, "missing_quarterly_cashflow_tickers", ticker)

            if table == "stock_income_stmt_quarterly" and "null revenue" in issue_lower:
                _record_cohort_match(cohorts, issue_counts, "quarterly_income_null_revenue_tickers", ticker)

            if table == "stock_ratio_data" and severity in {"HIGH", "CRITICAL"} and "null" in issue_lower:
                if is_financial:
                    _record_cohort_match(cohorts, issue_counts, "financial_sector_ratio_null_tickers", ticker)
                else:
                    _record_cohort_match(cohorts, issue_counts, "ratio_null_tickers", ticker)

            if table == "cross-table" and "lag" in issue_lower:
                _record_cohort_match(cohorts, issue_counts, "ratio_lag_tickers", ticker)

            if table == "stock_income_stmt_data" and issue_text.startswith("Only ") and "annual records" in issue_lower:
                _record_cohort_match(cohorts, issue_counts, "annual_only_tickers", ticker)

    spike_dates = Counter(event["date"] for event in spike_events if event["date"])
    spike_tickers = Counter(event["ticker"] for event in spike_events)
    issue_tickers = set(issues_by_ticker)
    ticker_actions = _build_ticker_actions(cohorts, ticker_metadata, issue_tickers)
    action_groups = _build_action_groups(cohorts, ticker_actions)

    return {
        "run_timestamp": report.get("run_timestamp"),
        "baseline_summary": report.get("summary", {}),
        "diagnostics_summary": report.get("diagnostics", {}).get("summary", {}),
        "cohorts": _cohort_payload(cohorts, issue_counts),
        "action_groups": action_groups,
        "ticker_actions": ticker_actions,
        "spike_analysis": {
            "event_count": len(spike_events),
            "ticker_count": len(cohorts["spike_tickers"]),
            "top_dates": [
                {"date": date, "event_count": count}
                for date, count in spike_dates.most_common(10)
            ],
            "top_tickers": [
                {"ticker": ticker, "event_count": count}
                for ticker, count in spike_tickers.most_common(10)
            ],
            "sample_events": spike_events[:25],
        },
        "admission_gate": {
            "requires_manual_spike_review": bool(spike_events),
            "requires_quarterly_repair": bool(
                cohorts["missing_quarterly_income_tickers"]
                or cohorts["missing_quarterly_cashflow_tickers"]
                or cohorts["quarterly_income_null_revenue_tickers"]
            ),
            "requires_ratio_repair": bool(
                cohorts["ratio_null_tickers"] or cohorts["ratio_lag_tickers"]
            ),
            "requires_manual_triage": bool(action_groups["manual_triage"]["count"]),
            "has_separate_scope_findings": bool(action_groups["separate_scope"]["count"]),
        },
    }