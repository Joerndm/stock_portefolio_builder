"""
Comprehensive Stock Data Validation Script (Optimized)

Scans every stock in the database using optimized queries.
Results saved to validation_output.txt and data_validation_report.json.
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import json

import fetch_secrets
import db_connectors

OUTPUT_FILE = "validation_output.txt"
REPORT_FILE = "data_validation_report.json"

_logfile = None

def log(msg=""):
    print(msg, flush=True)
    if _logfile:
        _logfile.write(msg + "\n")
        _logfile.flush()

def get_engine():
    db_host, db_user, db_pass, db_name = fetch_secrets.secret_import()
    return db_connectors.pandas_mysql_connector(db_host, db_user, db_pass, db_name)

def run_query(engine, sql):
    return pd.read_sql(sql, con=engine)

def validate_all():
    engine = get_engine()
    issues = defaultdict(list)

    log("=" * 80)
    log("STOCK DATABASE COMPREHENSIVE VALIDATION")
    log(f"Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 80)

    # ---------------------------------------------------------------
    # 1. Get all tickers
    # ---------------------------------------------------------------
    log("\n[1/9] Loading all tickers...")
    info_df = run_query(engine, "SELECT ticker, company_Name, industry FROM stock_info_data")
    all_tickers = info_df["ticker"].tolist()
    non_index = info_df[info_df["industry"] != "Index"]["ticker"].tolist()
    index_tickers = info_df[info_df["industry"] == "Index"]["ticker"].tolist()
    log(f"  Total: {len(all_tickers)} ({len(non_index)} stocks, {len(index_tickers)} indices)")

    for _, row in info_df.iterrows():
        if pd.isna(row["company_Name"]) or row["company_Name"] == "":
            issues[row["ticker"]].append({"table": "stock_info_data", "issue": "Missing company_Name", "severity": "LOW"})
        if pd.isna(row["industry"]) or row["industry"] == "":
            issues[row["ticker"]].append({"table": "stock_info_data", "issue": "Missing industry", "severity": "LOW"})

    # ---------------------------------------------------------------
    # 2. Check data presence in all tables
    # ---------------------------------------------------------------
    log("\n[2/9] Checking data presence across tables...")
    tables_check = [
        ("stock_price_data", "ticker", "HIGH"),
        ("stock_income_stmt_data", "ticker", "HIGH"),
        ("stock_balancesheet_data", "ticker", "HIGH"),
        ("stock_cash_flow_data", "ticker", "HIGH"),
        ("stock_ratio_data", "ticker", "HIGH"),
        ("stock_income_stmt_quarterly", "ticker", "MEDIUM"),
        ("stock_balancesheet_quarterly", "ticker", "MEDIUM"),
        ("stock_cashflow_quarterly", "ticker", "MEDIUM"),
    ]

    presence = {}
    for table, col, sev in tables_check:
        tickers_in = set(run_query(engine, f"SELECT DISTINCT {col} FROM {table}")[col].tolist())
        presence[table] = tickers_in
        missing = set(non_index) - tickers_in
        log(f"  {table}: {len(tickers_in)} present, {len(missing)} missing")
        for t in missing:
            issues[t].append({"table": table, "issue": f"No data in {table}", "severity": sev})

    # ---------------------------------------------------------------
    # 3. Price data - basic stats
    # ---------------------------------------------------------------
    log("\n[3/9] Price data: basic stats...")
    price_basic = run_query(engine, """
        SELECT ticker,
               COUNT(*) as cnt,
               MIN(date) as earliest,
               MAX(date) as latest,
               MIN(close_Price) as min_close,
               MAX(close_Price) as max_close
        FROM stock_price_data
        GROUP BY ticker
    """)
    log(f"  Got stats for {len(price_basic)} tickers")

    today = datetime.now().date()
    stale_threshold = today - timedelta(days=7)

    for _, row in price_basic.iterrows():
        t = row["ticker"]
        latest = row["latest"]
        if hasattr(latest, 'date'):
            latest = latest.date()
        if latest and latest < stale_threshold:
            days_stale = (today - latest).days
            issues[t].append({"table": "stock_price_data",
                "issue": f"Stale price data - last update {latest} ({days_stale} days ago)", "severity": "HIGH"})
        if row["cnt"] < 100:
            issues[t].append({"table": "stock_price_data",
                "issue": f"Only {row['cnt']} price rows (earliest: {row['earliest']})", "severity": "MEDIUM"})
        if row["min_close"] is not None and row["min_close"] <= 0:
            issues[t].append({"table": "stock_price_data",
                "issue": f"Has close_Price <= 0 (min={row['min_close']})", "severity": "CRITICAL"})

    # ---------------------------------------------------------------
    # 4. Price data - NULL and quality checks
    # ---------------------------------------------------------------
    log("\n[4/9] Price data: NULL and quality checks...")

    null_ohlc = run_query(engine, """
        SELECT ticker,
               SUM(CASE WHEN close_Price IS NULL THEN 1 ELSE 0 END) as null_close,
               SUM(CASE WHEN open_Price IS NULL THEN 1 ELSE 0 END) as null_open,
               SUM(CASE WHEN high_Price IS NULL THEN 1 ELSE 0 END) as null_high,
               SUM(CASE WHEN low_Price IS NULL THEN 1 ELSE 0 END) as null_low,
               COUNT(*) as total
        FROM stock_price_data
        GROUP BY ticker
        HAVING SUM(CASE WHEN close_Price IS NULL THEN 1 ELSE 0 END) > 0
            OR SUM(CASE WHEN open_Price IS NULL THEN 1 ELSE 0 END) > 0
            OR SUM(CASE WHEN high_Price IS NULL THEN 1 ELSE 0 END) > 0
            OR SUM(CASE WHEN low_Price IS NULL THEN 1 ELSE 0 END) > 0
    """)
    log(f"  Tickers with NULL OHLC: {len(null_ohlc)}")
    for _, row in null_ohlc.iterrows():
        issues[row["ticker"]].append({"table": "stock_price_data",
            "issue": f"NULL OHLC: close={int(row['null_close'])}, open={int(row['null_open'])}, high={int(row['null_high'])}, low={int(row['null_low'])} (of {int(row['total'])})",
            "severity": "HIGH"})

    invalid_ohlc = run_query(engine, """
        SELECT ticker, COUNT(*) as cnt
        FROM stock_price_data
        WHERE high_Price < low_Price
        GROUP BY ticker
    """)
    log(f"  Tickers with high < low: {len(invalid_ohlc)}")
    for _, row in invalid_ohlc.iterrows():
        issues[row["ticker"]].append({"table": "stock_price_data",
            "issue": f"{int(row['cnt'])} rows where high_Price < low_Price", "severity": "CRITICAL"})

    zero_vol = run_query(engine, """
        SELECT ticker,
               SUM(CASE WHEN trade_Volume IS NULL OR trade_Volume = 0 THEN 1 ELSE 0 END) as zero_vol,
               COUNT(*) as total
        FROM stock_price_data
        GROUP BY ticker
        HAVING SUM(CASE WHEN trade_Volume IS NULL OR trade_Volume = 0 THEN 1 ELSE 0 END) / COUNT(*) > 0.2
    """)
    log(f"  Tickers with >20%% zero volume: {len(zero_vol)}")
    for _, row in zero_vol.iterrows():
        pct = row["zero_vol"] / row["total"] * 100
        issues[row["ticker"]].append({"table": "stock_price_data",
            "issue": f"{int(row['zero_vol'])} zero/null volume days ({pct:.0f}%)", "severity": "MEDIUM"})

    # Technical indicator NULLs
    null_tech = run_query(engine, """
        SELECT ticker,
               SUM(CASE WHEN rsi_14 IS NULL THEN 1 ELSE 0 END) as null_rsi,
               SUM(CASE WHEN sma_200 IS NULL THEN 1 ELSE 0 END) as null_sma200,
               COUNT(*) as total
        FROM stock_price_data
        GROUP BY ticker
    """)
    log(f"  Technical indicators: {len(null_tech)} tickers checked")
    for _, row in null_tech.iterrows():
        t = row["ticker"]
        total = row["total"]
        excess_sma = row["null_sma200"] - min(200, total)
        if excess_sma > 10:
            issues[t].append({"table": "stock_price_data",
                "issue": f"~{int(excess_sma)} excess NULL sma_200 beyond warm-up", "severity": "MEDIUM"})
        excess_rsi = row["null_rsi"] - min(34, total)
        if excess_rsi > 10:
            issues[t].append({"table": "stock_price_data",
                "issue": f"~{int(excess_rsi)} excess NULL rsi_14 beyond warm-up", "severity": "MEDIUM"})

    # ---------------------------------------------------------------
    # 4b. Price spike detection (cross-ticker data shuffling)
    # ---------------------------------------------------------------
    log("\n[4b] Price spike detection (>40% day-over-day jumps)...")
    try:
        spike_df = run_query(engine, """
            SELECT ticker, date, price, prev_price,
                   ABS((price - prev_price) / prev_price) AS pct_change
            FROM (
                SELECT ticker, date, close_Price AS price,
                       LAG(close_Price) OVER (PARTITION BY ticker ORDER BY date) AS prev_price
                FROM stock_price_data
            ) sub
            WHERE prev_price IS NOT NULL
              AND prev_price > 0
              AND ABS((price - prev_price) / prev_price) > 0.40
            ORDER BY date DESC
        """)
        log(f"  Found {len(spike_df)} price spikes (>40% day-over-day)")
        if len(spike_df) > 0:
            spike_tickers = spike_df['ticker'].nunique()
            log(f"  Affected tickers: {spike_tickers}")
            for _, row in spike_df.iterrows():
                pct = row['pct_change'] * 100
                issues[row['ticker']].append({
                    "table": "stock_price_data",
                    "issue": f"Price spike {row['prev_price']:.2f}→{row['price']:.2f} ({pct:.0f}%) on {row['date']}",
                    "severity": "CRITICAL"
                })
    except Exception as e:
        log(f"  ⚠ Price spike check failed: {e}")

    # ---------------------------------------------------------------
    # 5. Date gaps
    # ---------------------------------------------------------------
    log("\n[5/9] Checking date gaps...")
    try:
        gap_df = run_query(engine, """
            SELECT ticker, gap_start, gap_end, gap_days FROM (
                SELECT ticker,
                    LAG(date) OVER (PARTITION BY ticker ORDER BY date) as gap_start,
                    date as gap_end,
                    DATEDIFF(date, LAG(date) OVER (PARTITION BY ticker ORDER BY date)) as gap_days
                FROM stock_price_data
            ) sub
            WHERE gap_days > 10
            ORDER BY ticker, gap_end
        """)
        log(f"  Found {len(gap_df)} large gaps (>10 calendar days)")
        for _, row in gap_df.iterrows():
            issues[row["ticker"]].append({"table": "stock_price_data",
                "issue": f"Date gap: {int(row['gap_days'])}d between {row['gap_start']} and {row['gap_end']}",
                "severity": "MEDIUM"})
    except Exception as e:
        log(f"  WARNING: Gap query failed: {e}")

    # ---------------------------------------------------------------
    # 6. Financial statements
    # ---------------------------------------------------------------
    log("\n[6/9] Validating financial statements...")

    inc = run_query(engine, """
        SELECT ticker, COUNT(*) as cnt,
               SUM(CASE WHEN revenue IS NULL THEN 1 ELSE 0 END) as null_rev,
               SUM(CASE WHEN eps IS NULL THEN 1 ELSE 0 END) as null_eps,
               SUM(CASE WHEN average_shares IS NULL THEN 1 ELSE 0 END) as null_shares
        FROM stock_income_stmt_data GROUP BY ticker
    """)
    for _, row in inc.iterrows():
        t = row["ticker"]
        if row["null_rev"] > 0:
            issues[t].append({"table": "stock_income_stmt_data",
                "issue": f"{int(row['null_rev'])}/{int(row['cnt'])} NULL revenue", "severity": "HIGH"})
        if row["null_eps"] > 0:
            issues[t].append({"table": "stock_income_stmt_data",
                "issue": f"{int(row['null_eps'])}/{int(row['cnt'])} NULL eps", "severity": "MEDIUM"})
        if row["null_shares"] > 0:
            issues[t].append({"table": "stock_income_stmt_data",
                "issue": f"{int(row['null_shares'])}/{int(row['cnt'])} NULL average_shares", "severity": "MEDIUM"})
        if row["cnt"] < 3:
            issues[t].append({"table": "stock_income_stmt_data",
                "issue": f"Only {int(row['cnt'])} annual records", "severity": "LOW"})
    log(f"  Income: {len(inc)} tickers")

    bs = run_query(engine, """
        SELECT ticker, COUNT(*) as cnt,
               SUM(CASE WHEN total_Assets IS NULL THEN 1 ELSE 0 END) as null_assets,
               SUM(CASE WHEN book_Value_Per_Share IS NULL THEN 1 ELSE 0 END) as null_bvps
        FROM stock_balancesheet_data GROUP BY ticker
    """)
    for _, row in bs.iterrows():
        t = row["ticker"]
        if row["null_assets"] > 0:
            issues[t].append({"table": "stock_balancesheet_data",
                "issue": f"{int(row['null_assets'])}/{int(row['cnt'])} NULL total_Assets", "severity": "HIGH"})
        if row["null_bvps"] > 0:
            issues[t].append({"table": "stock_balancesheet_data",
                "issue": f"{int(row['null_bvps'])}/{int(row['cnt'])} NULL book_Value_Per_Share", "severity": "MEDIUM"})
    log(f"  Balance sheet: {len(bs)} tickers")

    cf = run_query(engine, """
        SELECT ticker, COUNT(*) as cnt,
               SUM(CASE WHEN free_Cash_Flow IS NULL THEN 1 ELSE 0 END) as null_fcf,
               SUM(CASE WHEN free_Cash_Flow_Per_Share IS NULL THEN 1 ELSE 0 END) as null_fcfps
        FROM stock_cash_flow_data GROUP BY ticker
    """)
    for _, row in cf.iterrows():
        t = row["ticker"]
        if row["null_fcf"] > 0:
            issues[t].append({"table": "stock_cash_flow_data",
                "issue": f"{int(row['null_fcf'])}/{int(row['cnt'])} NULL free_Cash_Flow", "severity": "MEDIUM"})
        if row["null_fcfps"] > 0:
            issues[t].append({"table": "stock_cash_flow_data",
                "issue": f"{int(row['null_fcfps'])}/{int(row['cnt'])} NULL FCF_Per_Share", "severity": "MEDIUM"})
    log(f"  Cash flow: {len(cf)} tickers")

    # ---------------------------------------------------------------
    # 7. Ratio data
    # ---------------------------------------------------------------
    log("\n[7/9] Validating ratios...")

    ratio = run_query(engine, """
        SELECT ticker, COUNT(*) as cnt,
               SUM(CASE WHEN p_s IS NULL THEN 1 ELSE 0 END) as null_ps,
               SUM(CASE WHEN p_e IS NULL THEN 1 ELSE 0 END) as null_pe,
               SUM(CASE WHEN p_b IS NULL THEN 1 ELSE 0 END) as null_pb,
               SUM(CASE WHEN p_fcf IS NULL THEN 1 ELSE 0 END) as null_pfcf,
               SUM(CASE WHEN financial_date_used IS NULL THEN 1 ELSE 0 END) as null_fd,
               MIN(p_s) as min_ps, MAX(p_s) as max_ps,
               MIN(p_e) as min_pe, MAX(p_e) as max_pe,
               MIN(p_b) as min_pb, MAX(p_b) as max_pb,
               MIN(p_fcf) as min_pfcf, MAX(p_fcf) as max_pfcf
        FROM stock_ratio_data GROUP BY ticker
    """)
    log(f"  Checked {len(ratio)} tickers")

    for _, row in ratio.iterrows():
        t = row["ticker"]
        total = row["cnt"]
        if row["null_ps"] == total and row["null_pe"] == total and row["null_pb"] == total and row["null_pfcf"] == total:
            issues[t].append({"table": "stock_ratio_data",
                "issue": f"ALL ratios NULL across {int(total)} rows", "severity": "CRITICAL"})
        else:
            for name, nc in [("p_s", row["null_ps"]), ("p_e", row["null_pe"]),
                             ("p_b", row["null_pb"]), ("p_fcf", row["null_pfcf"])]:
                pct = nc / total * 100 if total > 0 else 0
                if pct > 50:
                    issues[t].append({"table": "stock_ratio_data",
                        "issue": f"{name} NULL in {int(nc)}/{int(total)} ({pct:.0f}%)", "severity": "HIGH"})

        for name, mn, mx in [("p_s", row["min_ps"], row["max_ps"]),
                              ("p_e", row["min_pe"], row["max_pe"]),
                              ("p_b", row["min_pb"], row["max_pb"]),
                              ("p_fcf", row["min_pfcf"], row["max_pfcf"])]:
            if mx is not None and mx > 10000:
                issues[t].append({"table": "stock_ratio_data",
                    "issue": f"Extreme {name} max={mx:.1f}", "severity": "MEDIUM"})
            if mn is not None and mn < -1000:
                issues[t].append({"table": "stock_ratio_data",
                    "issue": f"Extreme negative {name} min={mn:.1f}", "severity": "MEDIUM"})

        if row["null_fd"] > 0:
            pct = row["null_fd"] / total * 100
            issues[t].append({"table": "stock_ratio_data",
                "issue": f"financial_date_used NULL in {int(row['null_fd'])}/{int(total)} ({pct:.0f}%)", "severity": "MEDIUM"})

    # ---------------------------------------------------------------
    # 8. Quarterly data
    # ---------------------------------------------------------------
    log("\n[8/9] Quarterly data...")

    q_inc = run_query(engine, """
        SELECT ticker, COUNT(*) as qtrs,
               SUM(CASE WHEN revenue_ttm IS NULL THEN 1 ELSE 0 END) as null_rev_ttm
        FROM stock_income_stmt_quarterly GROUP BY ticker
    """)
    for _, row in q_inc.iterrows():
        t = row["ticker"]
        if row["qtrs"] < 4:
            issues[t].append({"table": "stock_income_stmt_quarterly",
                "issue": f"Only {int(row['qtrs'])} quarters (need 4 for TTM)", "severity": "MEDIUM"})
        excess = max(0, row["null_rev_ttm"] - min(3, row["qtrs"]))
        if excess > 0:
            issues[t].append({"table": "stock_income_stmt_quarterly",
                "issue": f"{int(excess)} unexpected NULL revenue_ttm", "severity": "HIGH"})
    log(f"  Quarterly income: {len(q_inc)} tickers")

    q_cf = run_query(engine, """
        SELECT ticker, COUNT(*) as qtrs
        FROM stock_cashflow_quarterly GROUP BY ticker
    """)
    for _, row in q_cf.iterrows():
        if row["qtrs"] < 4:
            issues[row["ticker"]].append({"table": "stock_cashflow_quarterly",
                "issue": f"Only {int(row['qtrs'])} quarterly cashflow records", "severity": "MEDIUM"})
    log(f"  Quarterly cashflow: {len(q_cf)} tickers")

    # ---------------------------------------------------------------
    # 9. Cross-table consistency
    # ---------------------------------------------------------------
    log("\n[9/9] Cross-table checks...")

    ratio_lag = run_query(engine, """
        SELECT r.ticker,
               MAX(r.date) as ratio_end,
               MAX(p.date) as price_end
        FROM stock_ratio_data r
        JOIN stock_price_data p ON r.ticker = p.ticker
        GROUP BY r.ticker
        HAVING DATEDIFF(MAX(p.date), MAX(r.date)) > 30
    """)
    log(f"  Ratio lags price by >30d: {len(ratio_lag)} tickers")
    for _, row in ratio_lag.iterrows():
        r_end = row["ratio_end"]
        p_end = row["price_end"]
        if hasattr(r_end, 'date'): r_end = r_end.date()
        if hasattr(p_end, 'date'): p_end = p_end.date()
        gap = (p_end - r_end).days
        issues[row["ticker"]].append({"table": "cross-table",
            "issue": f"Ratios lag prices by {gap}d (ratio:{r_end}, price:{p_end})", "severity": "HIGH"})

    fin_mismatch = run_query(engine, """
        SELECT i.ticker,
               COUNT(DISTINCT i.financial_Statement_Date) as inc_p,
               (SELECT COUNT(DISTINCT b.financial_Statement_Date) FROM stock_balancesheet_data b WHERE b.ticker = i.ticker) as bs_p,
               (SELECT COUNT(DISTINCT c.financial_Statement_Date) FROM stock_cash_flow_data c WHERE c.ticker = i.ticker) as cf_p
        FROM stock_income_stmt_data i GROUP BY i.ticker
    """)
    mismatch_cnt = 0
    for _, row in fin_mismatch.iterrows():
        if not (row["inc_p"] == row["bs_p"] == row["cf_p"]):
            mismatch_cnt += 1
            issues[row["ticker"]].append({"table": "cross-table",
                "issue": f"Period mismatch: income={int(row['inc_p'])}, bs={int(row['bs_p'])}, cf={int(row['cf_p'])}", "severity": "MEDIUM"})
    log(f"  Financial period mismatches: {mismatch_cnt}")

    # ================================================================
    # REPORT
    # ================================================================
    log("\n" + "=" * 80)
    log("VALIDATION RESULTS SUMMARY")
    log("=" * 80)

    sev_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for t, il in issues.items():
        for i in il:
            sev_counts[i["severity"]] += 1

    total_issues = sum(sev_counts.values())
    tickers_w = len(issues)
    clean = len(non_index) - tickers_w

    log(f"\nTotal issues: {total_issues}")
    log(f"  CRITICAL: {sev_counts['CRITICAL']}")
    log(f"  HIGH:     {sev_counts['HIGH']}")
    log(f"  MEDIUM:   {sev_counts['MEDIUM']}")
    log(f"  LOW:      {sev_counts['LOW']}")
    log(f"\nTickers with issues: {tickers_w}/{len(non_index)}")
    log(f"Clean tickers: {clean}/{len(non_index)}")

    cats = defaultdict(int)
    for t, il in issues.items():
        for i in il:
            short = i["issue"]
            for kw in ["NULL OHLC", "Stale price", "No data in", "ALL ratios NULL",
                        "NULL in", "Date gap", "lag", "Period mismatch",
                        "financial_date_used", "Only", "zero/null volume",
                        "excess NULL", "Extreme", "NULL revenue", "NULL eps",
                        "NULL average_shares", "NULL total_Assets", "NULL book_Value",
                        "NULL free_Cash", "NULL FCF", "high_Price < low_Price",
                        "close_Price <= 0", "unexpected NULL revenue_ttm",
                        "quarterly", "Missing company", "Missing industry"]:
                if kw.lower() in short.lower():
                    short = kw
                    break
            cats[f"[{i['severity']}] {i['table']}: {short}"] += 1

    log("\n--- Issue Categories ---")
    for cat, cnt in sorted(cats.items(), key=lambda x: -x[1]):
        log(f"  {cnt:4d}x  {cat}")

    sev_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    sorted_t = sorted(issues.keys(), key=lambda t: min(sev_order[i["severity"]] for i in issues[t]))

    log("\n--- Detailed Issues by Ticker ---")
    log("=" * 80)

    for t in sorted_t:
        il = sorted(issues[t], key=lambda x: sev_order[x["severity"]])
        log(f"\n  {t} [{il[0]['severity']}] - {len(il)} issues:")
        for i in il:
            log(f"    [{i['severity']:8s}] {i['table']:35s} | {i['issue']}")

    report = {
        "run_timestamp": datetime.now().isoformat(),
        "summary": {"total_tickers": len(all_tickers), "stock_tickers": len(non_index),
                     "tickers_with_issues": tickers_w, "clean_tickers": clean,
                     "severity_counts": sev_counts, "total_issues": total_issues},
        "issue_categories": dict(cats),
        "issues_by_ticker": {t: il for t, il in issues.items()},
    }
    with open(REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log(f"\nJSON report: {REPORT_FILE}")

    return issues, report


if __name__ == "__main__":
    _logfile = open(OUTPUT_FILE, "w", encoding="utf-8")
    try:
        issues, report = validate_all()
    except Exception as e:
        log(f"\nFATAL ERROR: {e}")
        import traceback
        log(traceback.format_exc())
    finally:
        _logfile.close()
        _logfile = None
    print(f"\nFull output saved to {OUTPUT_FILE}")
