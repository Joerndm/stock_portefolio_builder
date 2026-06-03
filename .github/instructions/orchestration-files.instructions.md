---
description: "Use when modifying large orchestration or pipeline files such as ml_builder.py, stock_orchestrator.py, price_predictor.py, or model_trainer.py. Prefer helper modules, focused services, and explicit interfaces over adding more branches or responsibilities to these files."
applyTo:
  - "ml_builder.py"
  - "stock_orchestrator.py"
  - "price_predictor.py"
  - "model_trainer.py"
---

# Orchestration File Guidelines

- Treat these files as coordination layers. Prefer moving computation, transformation, validation, and export logic into helper modules or service functions.
- Before adding a new block, first check whether an existing helper/module can own the behavior. If not, create a focused helper instead of extending the orchestration file.
- Keep public entry points and existing call contracts stable when practical. Refactor behind the current interface first.
- If a small local edit is safer than extraction, keep it minimal and avoid adding a second responsibility in the same change.
