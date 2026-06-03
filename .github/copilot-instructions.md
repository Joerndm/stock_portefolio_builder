# Project Guidelines

## Change Workflow
- After making code changes, remove temporary or throwaway artifacts created during the task unless the task explicitly requires keeping them.
- Do not leave behind ad hoc debug scripts, scratch files, temporary exports, or transient tuning/log files.
- Treat generated outputs in `generated_forecasts/`, `generated_graphs/`, `prediction_logs/`, and `tuning_dir/` as temporary unless the task is specifically about those artifacts.

## Architecture
- Prefer modular service-style boundaries over expanding monolithic files.
- For new functionality, prefer modular extraction inside this single repo: extract focused modules, helper services, or clearly bounded components instead of adding more responsibilities to large orchestration files.
- When touching large files such as `ml_builder.py`, `stock_orchestrator.py`, or `price_predictor.py`, prefer small extractions and explicit interfaces over more in-file branching.
- Preserve existing public APIs unless the task explicitly calls for a broader refactor.

## Testing And Validation
- When behavior changes or a bug is fixed, add or update tests for the affected path.
- After any code change, run the narrowest relevant validation first, then run the full project test suite before considering the task complete.
- Use the full suite command: `python test_reports/comprehensive_test_runner.py --verbose`
- Before declaring a code task complete, explicitly report what focused validation ran and whether the full suite ran.
- If the full suite cannot run because of environment, dependency, database, or runtime constraints, state that explicitly and report the exact blocker and command.


## Documentation And Dependencies
- Update `README.md` when behavior, workflows, setup steps, or operator-facing usage changes.
- Update `BUILD_INSTRUCTIONS.md` and `test_reports/README.md` when test commands, setup, or validation workflow changes.
- Update `requirements_PY_3_10.txt` and `requirements_PY_3_12.txt` when Python dependencies are added, removed, or renamed.

## Repo-Specific References
- Full test runner: `test_reports/comprehensive_test_runner.py`
- Test suite documentation: `test_reports/README.md`
- Build and environment guidance: `BUILD_INSTRUCTIONS.md`