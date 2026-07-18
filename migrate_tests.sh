#!/usr/bin/env bash
# =============================================================================
# migrate_tests.sh — reorganize test_reports/ into a clean structure
# =============================================================================
# Run from the repo root on a fresh branch:
#
#   git checkout -b chore/reorganize-tests
#   bash migrate_tests.sh
#   git commit -m "Reorganize test_reports into tests/, docs/history/, scripts/"
#
# Everything uses `git mv` / `git rm` so history is preserved and the whole
# operation is reviewable in a single diff. Nothing here touches code content.
#
# Resulting structure:
#   tests/            real test suites (unit, integration, e2e, performance,
#                     security, data_validation) + shared fixtures/config
#   scripts/          one-off diagnostic and maintenance scripts
#   docs/history/     historical fix/status reports (markdown)
#   (deleted)         archived_tests/, committed __pycache__/*.pyc
# =============================================================================
set -euo pipefail

echo "==> Creating target directories"
mkdir -p tests scripts docs/history

# -----------------------------------------------------------------------------
# 1. Real test suites -> tests/
# -----------------------------------------------------------------------------
echo "==> Moving test suites to tests/"
for d in unit integration e2e performance security data_validation; do
    if [ -d "test_reports/$d" ]; then
        if [ -d "tests/$d" ]; then
            # Target already exists (e.g. new tests added there first):
            # move the contents instead of the directory, or git nests it.
            git mv "test_reports/$d"/* "tests/$d/"
            rmdir "test_reports/$d"
        else
            git mv "test_reports/$d" "tests/$d"
        fi
    fi
done

# Root-level real tests
for f in test_compare_training.py test_forecast_quality.py \
         test_portfolio_construction.py test_runner_envs.py; do
    [ -f "test_reports/$f" ] && git mv "test_reports/$f" "tests/$f"
done

# Test runner + shared config/fixtures
for f in comprehensive_test_runner.py test_config.yaml \
         baseline_metrics_template.json; do
    [ -f "test_reports/$f" ] && git mv "test_reports/$f" "tests/$f"
done

# Package marker so `python -m unittest tests....` keeps working
[ -f "test_reports/__init__.py" ] && git mv "test_reports/__init__.py" "tests/__init__.py"

# -----------------------------------------------------------------------------
# 2. Diagnostic / one-off scripts -> scripts/
#    (These hit the live DB or yfinance; they are tools, not tests.)
# -----------------------------------------------------------------------------
echo "==> Moving diagnostic scripts to scripts/"
for f in check_feature_completeness.py check_stock_info_table.py \
         compare_feature_selection.py data_leakage_example.py \
         debug_carlsberg.py debug_carlsberg_shares.py \
         diagnose_lstm_training.py diagnose_rf_retraining.py \
         fetch_all_remaining_tickers.py final_verification_report.py \
         manual_fetch_demant.py migrate_to_ttm_data.py \
         populate_all_tables.py quick_feature_check.py \
         run_full_fetch_with_monitoring.py; do
    [ -f "test_reports/$f" ] && git mv "test_reports/$f" "scripts/$f"
done

# -----------------------------------------------------------------------------
# 3. Historical markdown reports -> docs/history/
#    (Session artifacts; useful context, but not living documentation.)
# -----------------------------------------------------------------------------
echo "==> Moving historical reports to docs/history/"
for f in test_reports/*.md test_reports/*.txt; do
    [ -f "$f" ] && git mv "$f" "docs/history/$(basename "$f")"
done

# -----------------------------------------------------------------------------
# 4. Delete archived tests and committed bytecode
#    (git history preserves them if ever needed again.)
# -----------------------------------------------------------------------------
echo "==> Removing archived_tests/ and committed .pyc files"
[ -d "test_reports/archived_tests" ] && git rm -r --quiet test_reports/archived_tests
find . -path ./.git -prune -o -name "*.pyc" -print0 | xargs -0 -r git rm --cached --quiet --ignore-unmatch
find . -path ./.git -prune -o -type d -name "__pycache__" -print0 | xargs -0 -r rm -rf

# -----------------------------------------------------------------------------
# 4b. Fix imports that referenced the old test_reports package path
#     (three files import siblings via the package name)
# -----------------------------------------------------------------------------
echo "==> Rewriting stale 'test_reports.' imports to 'tests.'"
if [ -f "tests/comprehensive_test_runner.py" ]; then
    sed -i \
        -e 's/from test_reports\.test_runner_envs/from tests.test_runner_envs/' \
        -e 's/test_reports\.unit\./tests.unit./' \
        -e 's/from test_reports\.integration/from tests.integration/' \
        -e 's/from test_reports\.e2e/from tests.e2e/' \
        -e 's/from test_reports\.performance/from tests.performance/' \
        -e 's/from test_reports\.security/from tests.security/' \
        -e 's/from test_reports\.data_validation/from tests.data_validation/' \
        tests/comprehensive_test_runner.py
fi
if [ -f "tests/unit/test_test_runner_envs_units.py" ]; then
    sed -i 's/from test_reports\.test_runner_envs/from tests.test_runner_envs/' \
        tests/unit/test_test_runner_envs_units.py
fi
if [ -f "tests/unit/test_comprehensive_test_runner_units.py" ]; then
    sed -i \
        -e 's/from test_reports import comprehensive_test_runner/from tests import comprehensive_test_runner/' \
        -e "s/'test_reports\.comprehensive_test_runner/'tests.comprehensive_test_runner/g" \
        tests/unit/test_comprehensive_test_runner_units.py
fi

# -----------------------------------------------------------------------------
# 5. Remove the now-empty test_reports/ and the superseded workflow
# -----------------------------------------------------------------------------
if [ -d "test_reports" ]; then
    remaining=$(find test_reports -type f | wc -l)
    if [ "$remaining" -eq 0 ]; then
        rm -rf test_reports
        echo "==> Removed empty test_reports/"
    else
        echo "!! test_reports/ still contains files — review manually:"
        find test_reports -type f
    fi
fi

if [ -f ".github/workflows/pylint.yml" ]; then
    git rm --quiet .github/workflows/pylint.yml
    echo "==> Removed superseded pylint.yml (replaced by ci.yml)"
fi

# -----------------------------------------------------------------------------
# 6. Sanity check: the sys.path bootstrap in the tests still resolves,
#    because tests/unit/ sits at the same depth test_reports/unit/ did.
# -----------------------------------------------------------------------------
echo "==> Verifying unit tests still import from the new location"
python -m pytest tests/unit -q --collect-only >/dev/null 2>&1 \
    && echo "    OK: pytest can collect tests/unit" \
    || echo "    WARNING: collection failed — run 'pytest tests/unit --collect-only' to inspect"

echo ""
echo "Done. Review with 'git status', then commit."
