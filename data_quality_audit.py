"""Reusable database audit helpers for stock-data diagnostics."""

from __future__ import annotations

from typing import Callable, Dict, List, MutableMapping

import pandas as pd


IssueMap = MutableMapping[str, List[dict]]
RunQueryFn = Callable[[object, str], pd.DataFrame]
LogFn = Callable[[str], None]


DUPLICATE_KEY_CHECKS = (
    {
        "key": "stock_price_duplicate_keys",
        "table": "stock_price_data",
        "columns": ("ticker", "date"),
        "severity": "CRITICAL",
    },
    {
        "key": "stock_ratio_duplicate_keys",
        "table": "stock_ratio_data",
        "columns": ("ticker", "date"),
        "severity": "CRITICAL",
    },
)

ORPHAN_TICKER_CHECKS = (
    {"key": "stock_price_orphans", "table": "stock_price_data", "severity": "HIGH"},
    {"key": "stock_income_orphans", "table": "stock_income_stmt_data", "severity": "HIGH"},
    {"key": "stock_balance_orphans", "table": "stock_balancesheet_data", "severity": "HIGH"},
    {"key": "stock_cashflow_orphans", "table": "stock_cash_flow_data", "severity": "HIGH"},
    {"key": "stock_ratio_orphans", "table": "stock_ratio_data", "severity": "HIGH"},
    {"key": "stock_income_quarterly_orphans", "table": "stock_income_stmt_quarterly", "severity": "MEDIUM"},
    {"key": "stock_balance_quarterly_orphans", "table": "stock_balancesheet_quarterly", "severity": "MEDIUM"},
    {"key": "stock_cashflow_quarterly_orphans", "table": "stock_cashflow_quarterly", "severity": "MEDIUM"},
)


def add_issue(issues: IssueMap, ticker: str, table: str, issue: str, severity: str) -> None:
    issues[str(ticker)].append({
        "table": table,
        "issue": issue,
        "severity": severity,
    })


def _sample_rows(df: pd.DataFrame, max_sample_rows: int) -> List[dict]:
    if df is None or df.empty:
        return []
    sample = df.head(max_sample_rows).copy()
    sample = sample.astype(object).where(pd.notna(sample), None)
    return sample.to_dict(orient="records")


def _format_key_parts(row: pd.Series, columns: tuple[str, ...]) -> str:
    parts = []
    for column in columns:
        if column in row.index:
            parts.append(f"{column}={row[column]}")
    return ", ".join(parts)


def run_structured_db_checks(
    engine: object,
    run_query: RunQueryFn,
    issues: IssueMap,
    log: LogFn,
    max_sample_rows: int = 10,
) -> Dict[str, dict]:
    """Run reusable duplicate/orphan diagnostics and attach issues."""
    diagnostics: Dict[str, dict] = {
        "duplicate_key_checks": {},
        "orphan_ticker_checks": {},
        "summary": {
            "duplicate_key_groups": 0,
            "orphan_ticker_groups": 0,
        },
    }

    log("\n[2b] Checking duplicate primary-key groups...")
    for check in DUPLICATE_KEY_CHECKS:
        column_sql = ", ".join(check["columns"])
        result = run_query(
            engine,
            f"""
            SELECT {column_sql}, COUNT(*) AS duplicate_count
            FROM {check['table']}
            GROUP BY {column_sql}
            HAVING COUNT(*) > 1
            ORDER BY duplicate_count DESC, {column_sql}
            """,
        )
        group_count = int(len(result))
        diagnostics["duplicate_key_checks"][check["key"]] = {
            "table": check["table"],
            "key_columns": list(check["columns"]),
            "affected_groups": group_count,
            "sample_rows": _sample_rows(result, max_sample_rows),
        }
        diagnostics["summary"]["duplicate_key_groups"] += group_count
        log(f"  {check['table']}: {group_count} duplicate key group(s)")

        for _, row in result.iterrows():
            ticker = row.get("ticker", check["table"])
            key_parts = _format_key_parts(row, check["columns"])
            add_issue(
                issues,
                ticker,
                check["table"],
                f"Duplicate key group for ({column_sql}): {key_parts} appears {int(row['duplicate_count'])} times",
                check["severity"],
            )

    log("\n[2c] Checking orphaned ticker rows...")
    for check in ORPHAN_TICKER_CHECKS:
        result = run_query(
            engine,
            f"""
            SELECT data.ticker, COUNT(*) AS orphan_rows
            FROM {check['table']} data
            LEFT JOIN stock_info_data info
                ON data.ticker = info.ticker
            WHERE info.ticker IS NULL
            GROUP BY data.ticker
            ORDER BY orphan_rows DESC, data.ticker
            """,
        )
        orphan_count = int(len(result))
        diagnostics["orphan_ticker_checks"][check["key"]] = {
            "table": check["table"],
            "affected_tickers": orphan_count,
            "sample_rows": _sample_rows(result, max_sample_rows),
        }
        diagnostics["summary"]["orphan_ticker_groups"] += orphan_count
        log(f"  {check['table']}: {orphan_count} orphan ticker(s)")

        for _, row in result.iterrows():
            add_issue(
                issues,
                row["ticker"],
                check["table"],
                f"Orphaned ticker rows: {int(row['orphan_rows'])} row(s) exist without a stock_info_data record",
                check["severity"],
            )

    return diagnostics