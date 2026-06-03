---
description: "Use when changing code that touches database access, external services, API clients, yfinance fetching, secrets loading, or cross-system pipeline integration. Prefer mocks or fakes in unit tests, and reserve integration coverage plus the comprehensive suite for real boundary validation."
---

# External Dependency Testing Guidelines

- For unit tests, mock database connectors, db_interactions calls, yfinance/network calls, secrets loading, filesystem side effects, and other external boundaries.
- Keep unit tests deterministic and runnable without live databases, internet access, or external credentials.
- Use integration or end-to-end tests when the real boundary interaction is part of the behavior under change.
- The required final validation command remains `python test_reports/comprehensive_test_runner.py --verbose`; do not substitute a narrow integration run for final validation unless blocked.
- If the full suite is blocked, state the exact command, blocker, and what narrower validation ran instead.
- When fixing a bug, add the narrowest test that would have failed before the fix.