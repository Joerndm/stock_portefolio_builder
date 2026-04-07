# Stock Portfolio Builder — AI Build Instructions

A complete, structured instruction set for recreating or extending the Stock Portfolio Builder
system using AI-assisted development.  These instructions are designed so that each section
can be handed to an AI agent (or a developer) as a standalone, unambiguous task.

---

## Table of Contents

0. [Prerequisites](#0-prerequisites)
1. [Architecture Overview](#1-architecture-overview)
2. [Data Model / ER Diagram](#2-data-model--er-diagram)
3. [Configuration Management](#3-configuration-management)
4. [Error Handling & Logging Standards](#4-error-handling--logging-standards)
5. [Data Validation Contracts](#5-data-validation-contracts)
6. [Idempotency Requirements](#6-idempotency-requirements)
7. [Testing Standards](#7-testing-standards)
8. [Phase 0 — Environment & Infrastructure Setup](#phase-0--environment--infrastructure-setup)
9. [Phase 1 — Data Pipeline (Data Ingestion)](#phase-1--data-pipeline-data-ingestion)
10. [Phase 2 — Machine Learning Pipeline](#phase-2--machine-learning-pipeline)
11. [Phase 3 — Portfolio Construction](#phase-3--portfolio-construction)
12. [Phase 4 — Streamlit GUI](#phase-4--streamlit-gui)
13. [Phase 5 — Supporting Utilities](#phase-5--supporting-utilities)
14. [ML Module Splitting Guide](#ml-module-splitting-guide)
15. [Ridge & SVR Full Ensemble Integration](#ridge--svr-full-ensemble-integration)

---

## 0. Prerequisites

Before any code is written, the following **non-Python** tools and infrastructure must be
available on the target machine.

| Requirement | Version / Notes |
|---|---|
| **Git** | 2.30+ recommended |
| **conda** or **mamba** | Package / environment manager (Miniconda or Mambaforge) |
| **MySQL Server** | 8.0+ — The application requires a running MySQL instance |
| **MySQL client tools** | `mysql` CLI for executing DDL scripts |
| **NVIDIA GPU Driver** | 470+ *(only if using TensorFlow GPU on the Py 3.10 environment)* |
| **NVIDIA CUDA Toolkit** | **11.2** *(only for TensorFlow 2.10 GPU)* |
| **NVIDIA cuDNN** | **8.1** *(only for TensorFlow 2.10 GPU)* |
| **Python 3.12** | Primary environment (data pipeline, Streamlit, scikit-learn models) |
| **Python 3.10** | Legacy environment (TensorFlow 2.10 GPU — LSTM/TCN training only) |

> **Windows users:** TensorFlow 2.10 is the *last* version with native Windows GPU support.
> For TensorFlow ≥ 2.11 GPU on Windows you must use WSL2.

### Quick validation

```bash
# Verify conda
conda --version

# Verify MySQL
mysql --version

# Verify GPU (optional)
nvidia-smi          # Shows GPU + driver
nvcc --version      # Shows CUDA toolkit
```

---

## 1. Architecture Overview

The system is a **three-phase pipeline** with a Streamlit GUI overlay.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         STREAMLIT GUI                               │
│  ┌───────────┐  ┌──────────────────┐  ┌──────────────┐  ┌────────┐│
│  │ Dashboard  │  │ Portfolio Builder │  │Stock Explorer │  │Data QA ││
│  └───────────┘  └──────────────────┘  └──────────────┘  └────────┘│
└──────────────────────────┬──────────────────────────────────────────┘
                           │ reads from DB
┌──────────────────────────▼──────────────────────────────────────────┐
│                        MySQL DATABASE                               │
│  stock_info ─┬─ stock_price ─ stock_income ─ stock_balancesheet     │
│              │  stock_cashflow ─ stock_ratio ─ quarterly tables      │
│              │  model_hyperparameters ─ stock_predictions            │
│              └─ portfolio_runs ─ portfolio_holdings ─ monte_carlo    │
└──────┬──────────────────┬──────────────────────┬────────────────────┘
       │                  │                      │
       ▼                  ▼                      ▼
┌──────────────┐  ┌────────────────────┐  ┌───────────────────────┐
│ PHASE 1      │  │ PHASE 2            │  │ PHASE 3               │
│ Data Pipeline│  │ ML Training &      │  │ Portfolio Construction │
│              │  │ Prediction         │  │                       │
│ • Fetch      │  │ • Ridge            │  │ • Ranking by predicted│
│   symbols    │  │ • SVR              │  │   return / risk       │
│ • Fetch      │  │ • Random Forest    │  │ • Efficient frontier  │
│   prices     │  │ • XGBoost          │  │   optimization        │
│ • Fetch      │  │ • LSTM / TCN       │  │ • Monte Carlo sim     │
│   financials │  │ • Ensemble (all 5) │  │ • Export holdings     │
│ • Technical  │  │ • Overfitting      │  │   to DB               │
│   indicators │  │   detection        │  │                       │
│ • TTM ratios │  │ • Price prediction  │  │                       │
│ • DB export  │  │ • MC simulation    │  │                       │
└──────────────┘  └────────────────────┘  └───────────────────────┘

Execution order:
  stock_orchestrator.py → model_trainer.py → price_predictor.py → portfolio_builder.py
                          ▲ Runs on Py 3.10     ▲ Runs on Py 3.10
                          │ (GPU environment)    │ (GPU environment)
```

### Key files per phase

| Phase | Entry Point | Modules Used |
|---|---|---|
| **1 — Data** | `stock_orchestrator.py` | `dynamic_index_fetcher`, `enhanced_financial_fetcher`, `ttm_financial_calculator`, `technical_indicators`, `stock_data_fetch`, `db_interactions`, `blacklist_manager` |
| **2 — ML** | `model_trainer.py` | `ml_builder` (or split modules — see §14), `split_dataset`, `dimension_reduction`, `data_scalers`, `monte_carlo_sim` |
| **3 — Portfolio** | `portfolio_builder.py` | `portfolio_config`, `efficient_frontier`, `monte_carlo_sim`, `db_interactions` |
| **GUI** | `streamlit_app.py` | `pages/*`, `gui_data` |

---

## 2. Data Model / ER Diagram

The MySQL Workbench file `database_files/Stock_db diagram.mwb` contains the canonical
ER diagram.  Below is a text representation of the schema.

```
                               ┌─────────────────┐
                               │ stock_info_data  │  ← FK parent for all
                               │─────────────────│
                               │ ticker (PK)     │
                               │ company_Name    │
                               │ industry        │
                               └────────┬────────┘
          ┌──────────┬──────────┬───────┤────────┬──────────┬──────────┐
          ▼          ▼          ▼       ▼        ▼          ▼          ▼
  ┌──────────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────┐ ┌──────────┐
  │stock_price   │ │income    │ │balance   │ │cashflow  │ │ratio  │ │index     │
  │_data         │ │_stmt_data│ │sheet_data│ │_data     │ │_data  │ │membership│
  │(date,ticker) │ │(date,    │ │(date,    │ │(date,    │ │(date, │ │(ticker,  │
  │PK            │ │ ticker)  │ │ ticker)  │ │ ticker)  │ │ticker)│ │index)    │
  │ OHLCV        │ │ revenue  │ │ assets   │ │ OCF      │ │ P/E   │ │ index    │
  │ returns      │ │ EPS      │ │ equity   │ │ FCF      │ │ P/S   │ │ exchange │
  │ SMA/EMA      │ │ margins  │ │ ROA/ROE  │ │ capex    │ │ P/B   │ │ is_cur   │
  │ RSI/MACD     │ │ growth%  │ │ ratios   │ │ growth%  │ │ P/FCF │ │          │
  │ vol/momentum │ │          │ │          │ │          │ │       │ │          │
  └──────────────┘ └──────────┘ └──────────┘ └──────────┘ └───────┘ └──────────┘

  ┌────────────────┐  ┌────────────────┐  ┌────────────────────┐
  │ Quarterly       │  │ stock_ratio    │  │ model_hyper        │
  │ Tables          │  │ _data_ttm      │  │ parameters         │
  │ (3 tables:      │  │ (date, ticker) │  │ (ticker, model_    │
  │  income, BS,    │  │  p_s_ttm       │  │  type) PK          │
  │  cashflow)      │  │  p_e_ttm       │  │  hyperparams JSON  │
  │ + TTM columns   │  │  p_b           │  │  tuning metadata   │
  └────────────────┘  │  p_fcf_ttm     │  │  val_mse/r2/mae    │
                      └────────────────┘  └────────────────────┘

  ┌───────────────────┐  ┌──────────────────┐  ┌────────────────────┐
  │ stock_predictions  │  │ portfolio_runs   │  │ monte_carlo_yearly │
  │ (date, ticker) PK  │  │ (run_id) PK      │  │ (mc_id, ticker) PK │
  │  predicted_price   │  │  risk_level       │  │  year              │
  │  confidence_low    │  │  strategy         │  │  percentiles       │
  │  confidence_high   │  │  sharpe_ratio     │  │  mean/median       │
  │  model_type        │  │  total_return     │  │                    │
  └───────────────────┘  │──────────────────│  └────────────────────┘
                         │ portfolio_holdings│
                         │ (run_id, ticker)  │
                         │  weight, return   │
                         └──────────────────┘
```

### `model_hyperparameters` — model_type ENUM

The current schema defines `model_type` as:

```sql
ENUM('rf', 'xgb', 'lstm', 'tcn')
```

**To support Ridge and SVR**, this ENUM must be expanded:

```sql
ALTER TABLE model_hyperparameters
  MODIFY COLUMN model_type ENUM('rf', 'xgb', 'lstm', 'tcn', 'ridge', 'svr') NOT NULL;
```

### DDL Files

| File | Purpose |
|---|---|
| `database_files/ddl.sql` | Complete schema (drop-and-recreate) |
| `database_files/ddl_legacy.sql` | Pre-TTM legacy schema |
| `database_files/migrate_add_quarterly_tables.sql` | Quarterly table migration |
| `database_files/migrate_add_hyperparameter_storage.sql` | Hyperparameter table migration |
| `database_files/migrate_add_prediction_mc_tables.sql` | Prediction + Monte Carlo tables |
| `database_files/migrate_add_financial_date_used.sql` | Financial date metadata |
| `database_files/migrate_add_quarterly_fetch_metadata.sql` | Quarterly fetch tracking |

---

## 3. Configuration Management

**Problem:** The codebase contains many magic numbers spread across multiple files
(`time_steps=30`, `validation_size=0.20`, `overfitting_threshold=0.15`,
`max_retrains=150`, `MIN_ENSEMBLE_WEIGHT=0.005`, `MAX_DAILY_RETURN=0.20`, etc.).

**Requirement:** Create a centralized configuration module that all other modules import from.

### 3.1 Create `pipeline_config.py`

This file consolidates every tunable parameter.  It loads overrides from environment
variables (via `dev.env`) but always provides sensible defaults.

```
pipeline_config.py
├── DataPipelineConfig     (dataclass)
│   ├── indices: List[str] = ['SP500', 'C25']
│   ├── max_workers: int = 4
│   ├── prefer_ttm: bool = True
│   └── blacklist_path: str = 'blacklisted_tickers.json'
│
├── MLTrainingConfig       (dataclass)
│   ├── time_steps: int = 30
│   ├── validation_size: float = 0.20
│   ├── test_size: float = 0.10
│   ├── min_training_rows: int = 252
│   ├── max_retrains: int = 150
│   ├── overfitting_threshold: float = 0.15
│   ├── min_ensemble_weight: float = 0.005
│   ├── max_daily_return_clip: float = 0.20
│   │
│   ├── # Per-model tuning budgets
│   ├── rf_trials: int = 100
│   ├── rf_retrain_increment: int = 25
│   ├── xgb_trials: int = 60
│   ├── xgb_retrain_increment: int = 10
│   ├── ridge_trials: int = 40
│   ├── ridge_retrain_increment: int = 10
│   ├── svr_trials: int = 40
│   ├── svr_retrain_increment: int = 10
│   ├── lstm_trials: int = 50
│   ├── lstm_executions: int = 10
│   ├── lstm_epochs: int = 500
│   ├── lstm_retrain_trials_increment: int = 10
│   ├── lstm_retrain_executions_increment: int = 2
│   ├── tcn_trials: int = 30
│   ├── tcn_epochs: int = 100
│   └── tcn_retrain_increment: int = 10
│
├── PredictionConfig       (dataclass)
│   ├── prediction_days: int = 252
│   ├── mc_dropout_enabled: bool = True
│   ├── mc_iterations: int = 30
│   ├── monte_carlo_simulations: int = 1000
│   └── max_prediction_age_days: int = 1
│
├── PortfolioConfig        (dataclass — extends InvestorProfile)
│   ├── risk_free_rate: float = 0.04
│   └── investment_years: int = 5
│
├── GPUConfig              (dataclass)
│   ├── memory_limit_mb: int = 7168
│   └── tf_log_level: str = '2'
│
└── DatabaseConfig         (loaded from dev.env)
    ├── DB_HOST, DB_USER, DB_PASS, DB_NAME
```

### 3.2 Rules

1. **No magic numbers** in any module.  Every tunable constant is defined in `pipeline_config.py`.
2. Each module imports the relevant config dataclass at the top.
3. Environment-variable overrides are loaded via `python-dotenv` inside `pipeline_config.py`.
4. The configuration classes use `@dataclass` with type hints for IDE/AI autocompletion.
5. Add a `validate()` method on each config that raises `ValueError` for out-of-range values.

---

## 4. Error Handling & Logging Standards

### 4.1 Replace all `print()` calls with `logging`

The current codebase mixes `print()` statements (hundreds of them) with no log-level
control.  Every module must:

1. Create a module-level logger at the top of the file:
   ```python
   import logging
   logger = logging.getLogger(__name__)
   ```

2. Use appropriate log levels:
   | Level | Use for |
   |---|---|
   | `logger.debug()` | Detailed diagnostic output (feature values, shape checks) |
   | `logger.info()` | Normal operation events (ticker processed, model trained) |
   | `logger.warning()` | Recoverable issues (missing data, fallback used) |
   | `logger.error()` | Failures that affect a single ticker but not the pipeline |
   | `logger.critical()` | Failures that halt the entire pipeline |

3. Configure the root logger in each entry-point script (`stock_orchestrator.py`,
   `model_trainer.py`, `price_predictor.py`, `portfolio_builder.py`):
   ```python
   logging.basicConfig(
       level=logging.INFO,
       format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
       datefmt="%Y-%m-%d %H:%M:%S"
   )
   ```

### 4.2 Exception handling

1. **Never silently swallow exceptions.**  At minimum, log the traceback:
   ```python
   except Exception as e:
       logger.error("Failed to train %s: %s", ticker, e, exc_info=True)
   ```
2. Use specific exception types (`ValueError`, `KeyError`, `ConnectionError`) rather than
   bare `except Exception` wherever the failure mode is known.
3. Wrap all database operations in try/except with rollback:
   ```python
   try:
       with engine.begin() as conn:
           conn.execute(...)
   except sqlalchemy.exc.SQLAlchemyError:
       logger.error("DB write failed for %s", ticker, exc_info=True)
   ```

---

## 5. Data Validation Contracts

Every pipeline boundary must validate data before passing it downstream.  The following
defines the **exact DataFrame schemas** expected at each stage.

### 5.1 Stock Price Data — `stock_price_data` table / DataFrame

After Phase 1, every row must contain:

| Column | Type | Nullable | Description |
|---|---|---|---|
| `date` | `datetime64[ns]` | No | Trading date |
| `ticker` | `str` | No | Stock symbol |
| `open_Price` | `float64` | No | Opening price |
| `high_Price` | `float64` | No | High price |
| `low_Price` | `float64` | No | Low price |
| `close_Price` | `float64` | No | Closing price |
| `trade_Volume` | `int64` | No | Volume |
| `1D` | `float64` | Yes (first row) | 1-day return |
| `sma_5` … `sma_200` | `float64` | Yes (leading rows) | Simple moving averages |
| `ema_5` … `ema_200` | `float64` | Yes (leading rows) | Exponential moving averages |
| `rsi_14` | `float64` | Yes (first 14 rows) | Relative Strength Index |
| `macd` / `macd_signal` / `macd_histogram` | `float64` | Yes (first 26 rows) | MACD family |
| `volatility_5d/20d/60d` | `float64` | Yes (leading rows) | Volatility measures |

### 5.2 ML Training Input — output of `split_dataset.dataset_train_test_split()`

| Output | Shape | Notes |
|---|---|---|
| `scaler_x` | `MinMaxScaler` (fitted) | Feature scaler |
| `scaler_y` | `MinMaxScaler` (fitted) | Target scaler |
| `x_train_scaled` | `(n_train, n_features)` float32 | Scaled features |
| `x_val_scaled` | `(n_val, n_features)` float32 | Scaled features |
| `x_test_scaled` | `(n_test, n_features)` float32 | Scaled features |
| `y_train_scaled` | `(n_train,)` float32 | Scaled close_Price return (1D) |
| `y_val_scaled` | `(n_val,)` float32 | Scaled target |
| `y_test_scaled` | `(n_test,)` float32 | Scaled target |
| `x_predictions` | `(1, n_features)` float32 | Latest row for future prediction |

### 5.3 Feature Selection Output — `dimension_reduction.feature_selection_rf()`

| Output | Shape |
|---|---|
| `x_train_selected` | `(n_train, k)` — `k ≤ n_features` |
| `x_val_selected` | `(n_val, k)` |
| `x_test_selected` | `(n_test, k)` |
| `x_prediction_selected` | `(1, k)` |
| `selector_model` | fitted `RandomForestRegressor` |
| `selected_features_list` | `List[str]` of length `k` |

### 5.4 Model Output — `train_and_validate_models()` return values

```python
models: dict = {
    'sequence_model': <keras.Model>,       # The trained TCN or LSTM
    'sequence_model_type': 'tcn' | 'lstm', # Which sequence model was used
    'lstm': <Model> | None,                # Legacy key
    'tcn': <Model> | None,                 # Legacy key
    'rf': <RandomForestRegressor>,         # Trained RF model
    'xgb': <XGBRegressor>,                 # Trained XGBoost model
    'ridge': <Ridge>,                      # Trained Ridge model  ← NEW
    'svr': <SVR>,                          # Trained SVR model    ← NEW
    'ensemble_weights': {                  # Weight per model for ensemble
        'tcn': float, 'rf': float, 'xgb': float,
        'ridge': float, 'svr': float       # ← NEW
    }
}
```

### 5.5 Validation Functions

Every module that receives a DataFrame must call a shared validation function:

```python
def validate_dataframe(df, required_columns, name=""):
    """Raises ValueError if df is missing columns or has unexpected dtypes."""
```

Place this in a new `data_contracts.py` module.

---

## 6. Idempotency Requirements

**Every pipeline step must be safely re-runnable** without producing duplicate data or
corrupting state.

| Step | Idempotency Mechanism |
|---|---|
| **Symbol fetching** | Monthly cache with timestamp check; re-fetch only if cache expired |
| **Price data fetch** | Upsert (`INSERT … ON DUPLICATE KEY UPDATE`) keyed on `(date, ticker)` |
| **Financial data fetch** | Upsert keyed on `(financial_Statement_Date, ticker)` |
| **Ratio calculation** | Upsert keyed on `(date, ticker)` |
| **TTM calculation** | Recalculate from quarterly data on every run; upsert results |
| **Model training** | Check `model_hyperparameters` freshness first; skip if fresh |
| **Prediction** | Check `stock_predictions` freshness; skip if `< max_prediction_age_days` |
| **Portfolio construction** | Create new `portfolio_runs` row each time (append-only) |
| **Blacklist updates** | Merge with existing JSON; never remove unless explicitly cleared |

### Rules

1. **No `TRUNCATE` or `DELETE` before insert** unless explicitly a full-refresh mode.
2. All SQL writes use `INSERT … ON DUPLICATE KEY UPDATE` or `REPLACE INTO`.
3. All pipeline runners (`stock_orchestrator.py`, `model_trainer.py`, etc.) query DB
   for freshness before processing each ticker.
4. Re-running any script with the same parameters produces the same database state.

---

## 7. Testing Standards

The repository currently **lacks automated tests**.  Every new or modified module must
ship with corresponding tests.

### 7.1 Directory Structure

```
tests/
├── conftest.py              # Shared fixtures (mock DB, sample DataFrames)
├── test_data_pipeline/
│   ├── test_stock_data_fetch.py
│   ├── test_dynamic_index_fetcher.py
│   ├── test_technical_indicators.py
│   ├── test_ttm_financial_calculator.py
│   └── test_blacklist_manager.py
├── test_ml/
│   ├── test_split_dataset.py
│   ├── test_dimension_reduction.py
│   ├── test_model_definitions.py   # Unit tests for build_* functions
│   ├── test_model_tuning.py        # Integration: tune + train small set
│   ├── test_overfitting_detection.py
│   ├── test_ridge_integration.py   # ← NEW: Ridge in ensemble
│   ├── test_svr_integration.py     # ← NEW: SVR in ensemble
│   ├── test_ensemble_weights.py    # ← NEW: 5-model weighting
│   └── test_predict_future.py
├── test_portfolio/
│   ├── test_efficient_frontier.py
│   ├── test_monte_carlo_sim.py
│   └── test_portfolio_builder.py
├── test_database/
│   ├── test_db_interactions.py
│   └── test_db_connectors.py
└── test_gui/
    └── test_gui_data.py
```

### 7.2 Rules

1. Use **pytest** as the test runner.
2. Add `pytest` and `pytest-cov` to both `requirements_PY_3_10.txt` and
   `requirements_PY_3_12.txt`.
3. **Unit tests** must run without a database or GPU.  Mock external dependencies.
4. **Integration tests** (marked with `@pytest.mark.integration`) may use the database.
5. Minimum coverage targets:
   - Data pipeline: 80%
   - ML module: 70% (model training is expensive; test build/validate functions)
   - Portfolio: 80%
   - Database interactions: 60% (mock-heavy)
6. Every bug fix must include a regression test.
7. Run the test suite with:
   ```bash
   pytest tests/ -v --cov=. --cov-report=term-missing
   ```

### 7.3 Test Data

Create `tests/fixtures/` with small, deterministic datasets:
- `sample_stock_data.csv` — 500 rows of synthetic OHLCV + indicators for one ticker
- `sample_financials.json` — Quarterly income/balance/cashflow for one ticker
- `sample_ml_input.npz` — Pre-split x_train/x_val/x_test/y_train/y_val/y_test arrays

---

## Phase 0 — Environment & Infrastructure Setup

### Step 0.1: Create two conda environments

```bash
# Primary environment (Python 3.12)
conda create -n stock_env python=3.12
conda activate stock_env
pip install -r requirements_PY_3_12.txt

# GPU/DL environment (Python 3.10) — only if LSTM/TCN training needed
conda create -n stock_env_gpu python=3.10
conda activate stock_env_gpu
pip install -r requirements_PY_3_10.txt
```

### Step 0.2: Configure database credentials

Create `dev.env` in the project root:

```ini
DB_HOST=localhost
DB_USER=stock_user
DB_PASS=your_password
DB_NAME=stock_portefolio_builder
```

### Step 0.3: Initialize the database

```bash
mysql -u root -p < database_files/ddl.sql
```

Then apply any pending migrations (in order):

```bash
mysql -u root -p stock_portefolio_builder < database_files/migrate_add_quarterly_tables.sql
mysql -u root -p stock_portefolio_builder < database_files/migrate_add_hyperparameter_storage.sql
mysql -u root -p stock_portefolio_builder < database_files/migrate_add_prediction_mc_tables.sql
mysql -u root -p stock_portefolio_builder < database_files/migrate_add_financial_date_used.sql
mysql -u root -p stock_portefolio_builder < database_files/migrate_add_quarterly_fetch_metadata.sql
```

### Step 0.4: Expand model_type ENUM for Ridge and SVR

```sql
ALTER TABLE model_hyperparameters
  MODIFY COLUMN model_type ENUM('rf', 'xgb', 'lstm', 'tcn', 'ridge', 'svr') NOT NULL
  COMMENT 'Model type';
```

---

## Phase 1 — Data Pipeline (Data Ingestion)

### Step 1.1: Secrets Loader (`fetch_secrets.py`)

- Load `DB_HOST`, `DB_USER`, `DB_PASS`, `DB_NAME` from `dev.env` using `python-dotenv`.
- Expose as module-level variables.
- **Test:** Verify `ValueError` raised if any variable is missing.

### Step 1.2: Database Connectors (`db_connectors.py`)

- Create a `get_engine()` function returning a SQLAlchemy `create_engine()` instance.
- Use `mysql+mysqlconnector` dialect.
- Connection pool size: 5, max overflow: 10.
- **Test:** Mock `create_engine`, verify connection string format.

### Step 1.3: Database Interactions (`db_interactions.py`)

- Implement import/export functions for every table in the schema.
- All exports must use `INSERT … ON DUPLICATE KEY UPDATE` (upsert pattern).
- Key functions:
  - `import_ticker_list()` → `List[str]`
  - `import_stock_dataset(ticker)` → `pd.DataFrame`
  - `export_stock_price_data(df)` → upsert
  - `export_stock_ratio_data(df)` → upsert
  - `save_hyperparameters(ticker, model_type, hyperparameters, …)` → upsert
  - `load_hyperparameters(ticker, model_type, max_age_days, num_features)` → `dict | None`
  - `get_tickers_needing_training(max_age_days, required_model_types)` → `dict`
  - `invalidate_hyperparameters(ticker, model_type)` → void
- **Test:** Mock DB, verify SQL patterns, test upsert deduplication.

### Step 1.4: Dynamic Index Fetcher (`dynamic_index_fetcher.py`)

- Scrape Wikipedia for index constituent lists.
- Supported indices: `SP500`, `C25`, `DAX40`, `CAC40`, `FTSE100`, `NASDAQ100`, `DOW30`,
  `AEX25`, `OMX30`, `IBEX35`, `OMXH25`, `SMI`.
- Cache results in a local CSV with monthly expiry.
- Optionally include market indices (`^VIX`, `^GSPC`, `^IXIC`, `^DJI`).
- Return `pd.DataFrame` with columns: `Symbol`, `Company`, `Index`.
- **Test:** Mock HTTP responses, verify parsing for at least SP500 and C25.

### Step 1.5: Blacklist Manager (`blacklist_manager.py`)

- Manage `blacklisted_tickers.json` — a JSON file of tickers to skip.
- Auto-blacklist tickers that fail fetching 3+ times.
- Functions: `get_blacklist()`, `add_to_blacklist(ticker, reason)`, `remove_from_blacklist(ticker)`.
- Thread-safe (file lock or atomic write).
- **Test:** Test add/remove/get cycle; test duplicate add is idempotent.

### Step 1.6: Stock Data Fetch (`stock_data_fetch.py`)

- Use `yfinance` to fetch OHLCV history, stock info, financials.
- Calculate all technical indicators (see `technical_indicators.py`).
- Calculate all return periods (1D through 5Y).
- Calculate valuation ratios (P/E, P/S, P/B, P/FCF) using TTM when available.
- Export to database via `db_interactions`.
- **Test:** Mock `yfinance.Ticker`, verify DataFrame schema matches §5.1.

### Step 1.7: Technical Indicators (`technical_indicators.py`)

- Calculate: RSI-14, ATR-14, MACD (12/26/9), Bollinger Bands (5/20/40/120/200 day),
  SMA/EMA (5/20/40/120/200), VWAP, OBV, volume SMA/EMA, volume ratio,
  volatility (5d/20d/60d), momentum.
- Input: DataFrame with OHLCV columns.
- Output: Same DataFrame with indicator columns appended.
- **Test:** Verify RSI range [0, 100], MACD sign changes.

### Step 1.8: Enhanced Financial Fetcher (`enhanced_financial_fetcher.py`)

- Fetch quarterly income statement, balance sheet, cash flow from yfinance.
- Export quarterly data to quarterly tables in DB.
- **Test:** Mock yfinance, verify quarterly DataFrame structure.

### Step 1.9: TTM Financial Calculator (`ttm_financial_calculator.py`)

- Sum last 4 quarters for income statement and cash flow items.
- Use point-in-time for balance sheet items.
- Calculate TTM ratios: P/E_ttm, P/S_ttm, P/B, P/FCF_ttm.
- Fallback to annual data if < 4 quarters available.
- **Test:** Verify TTM sum = Q1+Q2+Q3+Q4; verify fallback logic.

### Step 1.10: Stock Orchestrator (`stock_orchestrator.py`)

- Main entry point: chains Steps 1.4 → 1.5 → 1.6 → 1.7 → 1.8 → 1.9 → DB export.
- Supports CLI args: `--indices`, `--ticker`, `--update-only`.
- Supports `ThreadPoolExecutor` for parallel ticker processing.
- Tracks stats (processed, skipped, failed, time).
- **Test:** Mock all sub-modules, verify orchestration order.

---

## Phase 2 — Machine Learning Pipeline

### Step 2.1: Data Scalers (`data_scalers.py`)

- Wrap `MinMaxScaler` and `StandardScaler` from scikit-learn.
- Provide `fit_transform()` and `inverse_transform()` wrappers.
- **Test:** Round-trip test: `inverse_transform(fit_transform(x)) ≈ x`.

### Step 2.2: Dataset Splitting (`split_dataset.py`)

- Time-aware split (no shuffling — preserve temporal order).
- Default split: 70% train / 20% validation / 10% test.
- Scale x and y separately (fit on train only, transform val/test).
- Prepare a single-row prediction set from the latest data.
- **Test:** Verify no data leakage (val/test dates > train dates).

### Step 2.3: Dimension Reduction (`dimension_reduction.py`)

- **SelectKBest** using `r_regression` scoring.
- **Random Forest feature importance** — fit RF on train, rank features, select top-k.
- **PCA** — as alternative to feature selection.
- All selectors fit on train only; transform val/test/prediction.
- **Test:** Verify output shapes, verify no data leakage.

### Step 2.4: ML Builder (`ml_builder.py`) — Model Definitions

> **See §14 (ML Module Splitting) for the recommended file split.**

This module defines and tunes all five models plus the ensemble.

#### 2.4.1: Random Forest Regressor

- `build_random_forest_model(hp, constrain_for_overfitting=False)` → `RandomForestRegressor`
- Tunable hyperparameters:
  - `n_estimators`: 100–1500
  - `max_depth`: 3–50 (or None)
  - `min_samples_split`: 2–20
  - `min_samples_leaf`: 1–10
  - `max_features`: ['sqrt', 'log2', 0.3–1.0]
  - `criterion`: ['squared_error', 'absolute_error', 'friedman_mse']
  - `bootstrap`: [True, False]
  - `max_samples`: 0.5–1.0 (when bootstrap=True)
- Tuning: Keras Tuner `Sklearn` wrapper with `Hyperband`, up to `rf_trials` trials.
- Overfitting constraint mode: restrict `max_depth` ≤ 15, force `min_samples_leaf` ≥ 3.
- Operates on **unscaled** y data.
- Save best hyperparameters to DB (`model_type='rf'`).
- **Test:** Build with mock `hp`, verify returned model type.

#### 2.4.2: XGBoost Regressor

- `build_xgboost_model(hp, constrain_for_overfitting=False)` → `XGBRegressor`
- Tunable hyperparameters:
  - `n_estimators`: 100–2000
  - `max_depth`: 3–12
  - `learning_rate`: 0.001–0.3 (log scale)
  - `subsample`: 0.5–1.0
  - `colsample_bytree`: 0.3–1.0
  - `min_child_weight`: 1–10
  - `gamma`: 0–5
  - `reg_alpha`: 0–10 (L1)
  - `reg_lambda`: 0–10 (L2)
- Tuning: Keras Tuner `Sklearn` wrapper with `BayesianOptimization`, up to `xgb_trials` trials.
- Overfitting constraint mode: restrict `max_depth` ≤ 6, force `reg_lambda` ≥ 1.
- Operates on **unscaled** y data.
- Save best hyperparameters to DB (`model_type='xgb'`).
- **Test:** Build with mock `hp`, verify returned model type.

#### 2.4.3: Ridge Regression ← NEW (Full Ensemble Integration)

- `build_ridge_model(hp, constrain_for_overfitting=False)` → `Ridge`
- Tunable hyperparameters:
  - `alpha`: 1e-3–1e4 (log scale) — regularization strength
  - `fit_intercept`: [True, False]
  - `solver`: ['auto', 'svd', 'cholesky', 'lsqr', 'saga']
  - `max_iter`: 1000–10000 (for iterative solvers)
- Tuning: Keras Tuner `Sklearn` wrapper with `BayesianOptimization`, up to `ridge_trials` trials.
- Overfitting constraint mode: force `alpha` ≥ 10 to increase regularization.
- Operates on **unscaled** y data (Ridge is not scale-invariant — features are already
  MinMax scaled, so this is acceptable).
- Save best hyperparameters to DB (`model_type='ridge'`).
- Retrain loop: same overfitting detection as RF/XGBoost.
- **Test:** Build with mock `hp`, verify model type is `Ridge`. Test that
  `alpha` constraint is applied in overfitting mode.

#### 2.4.4: SVR (Support Vector Regression) ← NEW (Full Ensemble Integration)

- `build_svr_model(hp, constrain_for_overfitting=False)` → `SVR`
  - **Note:** The current codebase has `build_svm_model(hp)` and `tune_svm_model()` at
    lines 764–824 of `ml_builder.py`.  These must be **refactored** to match the RF/XGBoost
    pattern (overfitting constraint support, DB hyperparameter caching, retrain loop).
- Tunable hyperparameters:
  - `kernel`: ['linear', 'rbf', 'poly']
  - `C`: 1e-3–1e3 (log scale)
  - `gamma`: 1e-4–1e1 (log scale) — only for rbf/poly
  - `epsilon`: 0.001–1.0 (log scale)
  - `degree`: 2–5 (only for poly kernel)
- Tuning: Keras Tuner `Sklearn` wrapper with `BayesianOptimization`, up to `svr_trials` trials.
- Overfitting constraint mode: force `C` ≤ 10, force `epsilon` ≥ 0.01.
- Operates on **unscaled** y data.
- Save best hyperparameters to DB (`model_type='svr'`).
- Retrain loop: same overfitting detection as RF/XGBoost.
- **Test:** Build with mock `hp`, verify model type is `SVR`. Test kernel-specific params.

#### 2.4.5: LSTM (Long Short-Term Memory)

- `build_lstm_model(hp, input_shape)` → `keras.Model`
- Bidirectional architecture with multiple stacked layers.
- Batch normalization, L2 regularization, dropout, gradient clipping.
- Tunable: layer count (1–4), units (32–256), dropout (0.1–0.5), learning rate,
  optimizer (Adam/RMSprop), loss function (MAE/MSE/Huber/MAPE).
- Tuning: Keras Tuner `BayesianOptimization`, up to `lstm_trials` trials.
- Requires 3D input: `(samples, time_steps, features)`.
- Operates on **scaled** y data.
- Save best hyperparameters to DB (`model_type='lstm'`).
- **Test:** Build with mock `hp`, verify output shape.

#### 2.4.6: TCN (Temporal Convolutional Network)

- `build_tcn_model(hp, input_shape)` → `keras.Model`
- Causal dilated convolutions with residual connections.
- Tunable: filters, kernel size, dilation rates, dropout.
- Same tuning/caching/retraining pattern as LSTM.
- Save best hyperparameters to DB (`model_type='tcn'`).
- **Test:** Build with mock `hp`, verify output shape.

### Step 2.5: Overfitting Detection

- Multi-metric system comparing train/validation/test performance:
  - **MSE degradation** (35% weight)
  - **R² degradation** (25% weight)
  - **MAE degradation** (30% weight)
  - **Consistency score** (10% weight)
- Combined score > `overfitting_threshold` triggers constrained retraining.
- Maximum `max_retrains` attempts.
- Early stop if hyperparameters converge (3 consecutive identical sets).
- **Applied to all 5 models individually**, plus the ensemble.
- **Test:** Feed known overfitting metrics, verify detection triggers.

### Step 2.6: `train_and_validate_models()` — the Master Training Function

This function orchestrates training of **all five models** and computes ensemble weights.

**Current state:** Trains TCN/LSTM + RF + XGBoost (3 models).

**Required changes for Ridge + SVR integration:**

1. **Add Ridge training loop** (same pattern as RF):
   ```
   ridge_model = None
   ridge_overfitted = False
   for attempt in range(max_retrains):
       ridge_model = tune_ridge_model(...)
       ridge_metrics = evaluate_model(ridge_model, ...)
       ridge_overfitted, score = detect_overfitting(...)
       if not ridge_overfitted:
           break
       constrain = True
       ridge_trials += ridge_retrain_increment
   ```

2. **Add SVR training loop** (same pattern as RF):
   ```
   svr_model = None
   svr_overfitted = False
   for attempt in range(max_retrains):
       svr_model = tune_svr_model(...)
       svr_metrics = evaluate_model(svr_model, ...)
       svr_overfitted, score = detect_overfitting(...)
       if not svr_overfitted:
           break
       constrain = True
       svr_trials += svr_retrain_increment
   ```

3. **Update ensemble weight calculation** to include all 5 models:
   ```python
   # Inverse MSE weights (lower MSE = higher weight)
   inv_mse = {
       'seq': 1 / seq_val_mse,
       'rf':  1 / rf_val_mse,
       'xgb': 1 / xgb_val_mse,
       'ridge': 1 / ridge_val_mse,
       'svr': 1 / svr_val_mse,
   }
   total = sum(inv_mse.values())
   weights = {k: v / total for k, v in inv_mse.items()}

   # Zero out negligible weights (< MIN_ENSEMBLE_WEIGHT)
   # Renormalize remaining weights
   ```

4. **Update ensemble prediction** to be a 5-model weighted average:
   ```python
   ensemble_pred = (
       weights['seq'] * seq_pred +
       weights['rf']  * rf_pred +
       weights['xgb'] * xgb_pred +
       weights['ridge'] * ridge_pred +
       weights['svr']  * svr_pred
   )
   ```

5. **Update `models` dict return value** to include `'ridge'` and `'svr'` keys.

6. **Update `training_history` dict** to include `'ridge': []` and `'svr': []`.

7. **Update `training_history['ensemble']['weights']`** to include all 5 model keys.

8. **Update `required_model_types`** in `model_trainer.py`:
   ```python
   required_model_types=['rf', 'xgb', 'ridge', 'svr', 'tcn' if use_tcn else 'lstm']
   ```

9. **Update `predict_future_price_changes()`** to extract and use Ridge + SVR models
   from the `model` dict alongside the existing 3 models.

### Step 2.7: Model Trainer (`model_trainer.py`)

- Phase 1 entry point.
- Query DB for untrained → stale → fresh tickers.
- For each ticker: fetch data → split → feature select → `train_and_validate_models()`.
- Configure GPU (TF 2.10 with memory limit).
- CLI args: `--max-age`, `--max-stocks`, `--use-lstm`, `--time-steps`.
- **Test:** Mock `ml_builder`, verify training loop logic.

### Step 2.8: Price Predictor (`price_predictor.py`)

- Phase 2 entry point.
- For tickers with trained models but missing/stale predictions:
  1. Load hyperparameters from DB.
  2. Rebuild models from cached hyperparameters.
  3. Generate day-by-day forecasts using `predict_future_price_changes()`.
  4. Run Monte Carlo simulation.
  5. Export predictions + MC results to DB.
- **Test:** Mock model loading, verify prediction DataFrame schema.

### Step 2.9: Monte Carlo Simulation (`monte_carlo_sim.py`)

- Geometric Brownian Motion (GBM) simulation.
- Configurable: seed, number of simulations, forecast years.
- Output percentiles: 5th, 16th, 50th (median), 84th, 95th.
- Shapiro-Wilk normality test on returns.
- **Test:** Verify output shape, verify seed reproducibility.

---

## Phase 3 — Portfolio Construction

### Step 3.1: Portfolio Config (`portfolio_config.py`)

- `RiskLevel` enum: LOW, MEDIUM, HIGH.
- `InvestmentStrategy` enum: BALANCED, DIVIDEND, GROWTH, VALUE, GARP, QUALITY, MOMENTUM.
- `InvestorProfile` dataclass with validation.
- Volatility caps per risk level (15% / 25% / 40%).
- **Test:** Verify validation rejects out-of-range values.

### Step 3.2: Efficient Frontier (`efficient_frontier.py`)

- Monte Carlo simulation + scipy `minimize(method='SLSQP')`.
- Find: minimum variance portfolio, maximum Sharpe portfolio, risk-constrained max return.
- Input: covariance matrix + expected returns from historical data.
- Output: optimal weights, return, volatility, Sharpe ratio.
- **Test:** Verify weights sum to 1.0, verify Sharpe formula.

### Step 3.3: Portfolio Builder (`portfolio_builder.py`)

- Phase 3 entry point.
- Steps:
  1. Query DB for tickers with recent predictions.
  2. Rank by predicted return / risk score (strategy-aware).
  3. Select top N stocks.
  4. Fetch historical prices.
  5. Run efficient frontier optimization.
  6. Run portfolio-level Monte Carlo simulation.
  7. Export holdings + metrics to DB.
- **Test:** Mock DB queries, verify ranking logic and output schema.

---

## Phase 4 — Streamlit GUI

### Step 4.1: Main App (`streamlit_app.py`)

- Multi-page Streamlit app with dark theme.
- Define pages: Dashboard, Portfolio Builder, Stock Explorer, Data Quality.

### Step 4.2: Dashboard (`pages/1_Dashboard.py`)

- Market overview metrics.
- Portfolio summary if one exists.
- Recent pipeline run status.

### Step 4.3: Portfolio Builder (`pages/2_Portfolio_Builder.py`)

- Interactive investor profile controls (risk level, strategy, horizon).
- Trigger portfolio construction.
- Display optimized weights, expected return, Sharpe ratio.

### Step 4.4: Stock Explorer (`pages/3_Stock_Explorer.py`)

- Per-stock deep dive: price chart, predictions, model performance.
- Display individual model contributions to ensemble.

### Step 4.5: Data Quality (`pages/4_Data_Quality.py`)

- Pipeline health monitoring.
- Missing data detection.
- TTM vs annual comparison.

---

## Phase 5 — Supporting Utilities

### Step 5.1: ML Data Integrity (`ml_data_integrity.py`)

- Pre-training data validation.
- Verify TTM vs annual consistency.
- Check for data staleness.

### Step 5.2: Market Hours Utils (`market_hours_utils.py`)

- Determine if a market is open.
- Generate trading day calendars.

### Step 5.3: Data Scalers (`data_scalers.py`)

- Wrappers around scikit-learn scalers with serialization support.

### Step 5.4: GUI Data (`gui_data.py`)

- Data preparation functions specific to Streamlit pages.
- Query helpers for GUI-specific views.

### Step 5.5: Validate Stock Data (`validate_stock_data.py`)

- Post-fetch validation of stock data completeness.
- Schema checks against §5.1 contract.

---

## 14. ML Module Splitting Guide

**Problem:** `ml_builder.py` is ~5,000+ lines and contains model definitions, tuning logic,
overfitting detection, ensemble weighting, and prediction — all in one file.

**Requirement:** Split into focused, testable modules while maintaining backward compatibility.

### Recommended Split

```
ml/
├── __init__.py                  # Re-export public API for backward compat
├── model_definitions.py         # build_random_forest_model(), build_xgboost_model(),
│                                # build_ridge_model(), build_svr_model(),
│                                # build_lstm_model(), build_tcn_model()
│
├── model_tuning.py              # tune_rf_model(), tune_xgb_model(),
│                                # tune_ridge_model(), tune_svr_model(),
│                                # tune_lstm_model(), tune_tcn_model()
│                                # HP caching (save/load from DB)
│
├── overfitting_detection.py     # detect_overfitting(), check_data_health()
│                                # Multi-metric scoring logic
│
├── ensemble.py                  # calculate_ensemble_weights()
│                                # create_ensemble_predictions()
│                                # Zero-out negligible weights logic
│
├── training_pipeline.py         # train_and_validate_models()
│                                # Orchestrates model_tuning + overfitting + ensemble
│
├── prediction.py                # predict_future_price_changes()
│                                # predict_with_uncertainty()
│                                # multi_run_prediction()
│                                # apply_mean_reversion()
│
├── sequence_utils.py            # create_sequences()
│                                # LSTM/TCN data preparation
│
└── diagnostics.py               # Data health checks, feature importance
```

### Backward Compatibility

To avoid breaking existing imports (`from ml_builder import train_and_validate_models`),
keep `ml_builder.py` as a thin re-export wrapper:

```python
# ml_builder.py (after split)
from ml.training_pipeline import train_and_validate_models
from ml.prediction import predict_future_price_changes
from ml.model_definitions import *
from ml.model_tuning import *
from ml.overfitting_detection import detect_overfitting, check_data_health
from ml.ensemble import calculate_ensemble_weights
```

---

## 15. Ridge & SVR Full Ensemble Integration

This section provides the **complete specification** for integrating Ridge and SVR as
first-class ensemble members alongside the existing RF, XGBoost, and TCN/LSTM models.

### 15.1 Current State (3-Model Ensemble)

The current ensemble in `train_and_validate_models()` combines:
1. **Sequence model** (TCN or LSTM) — operates on scaled data
2. **Random Forest** — operates on unscaled data
3. **XGBoost** — operates on unscaled data

Weights are computed as inverse-MSE on the validation set (unscaled space).

### 15.2 Target State (5-Model Ensemble)

The ensemble must combine:
1. **Sequence model** (TCN or LSTM) — operates on scaled data
2. **Random Forest** — operates on unscaled data
3. **XGBoost** — operates on unscaled data
4. **Ridge** — operates on unscaled data ← NEW
5. **SVR** — operates on unscaled data ← NEW

### 15.3 Database Changes

```sql
-- Expand ENUM to include new model types
ALTER TABLE model_hyperparameters
  MODIFY COLUMN model_type ENUM('rf', 'xgb', 'lstm', 'tcn', 'ridge', 'svr') NOT NULL;
```

### 15.4 New Functions Required

#### `build_ridge_model(hp, constrain_for_overfitting=False)`

```
Input:
  - hp: Keras Tuner hyperparameters object
  - constrain_for_overfitting: bool

Hyperparameter search space:
  Normal mode:
    alpha:         Float(1e-3, 1e4, sampling='log')
    fit_intercept: Choice([True, False])
    solver:        Choice(['auto', 'svd', 'cholesky', 'lsqr', 'saga'])

  Constrained mode (overfitting detected):
    alpha:         Float(10, 1e4, sampling='log')   ← stronger regularization
    fit_intercept: Fixed(True)
    solver:        Choice(['auto', 'svd', 'lsqr'])  ← drop cholesky for stability

Output: sklearn.linear_model.Ridge instance
```

#### `tune_ridge_model(stock_symbol, x_train, y_train, x_val, y_val, max_trials, constrain_for_overfitting=False)`

```
Pattern: Same as existing tune_rf_model / tune_xgb_model
  1. Check DB for cached hyperparameters (load_hyperparameters(ticker, 'ridge', ...))
  2. If fresh and valid → rebuild model from cached HPs, skip tuning
  3. Otherwise → run Keras Tuner Sklearn wrapper with BayesianOptimization
  4. Save best HPs to DB (save_hyperparameters(ticker, 'ridge', ...))
  5. Evaluate on train/val/test → return metrics + model

Tuner config:
  oracle: kt.oracles.BayesianOptimization(max_trials=ridge_trials)
  scoring: neg_mean_squared_error
  cv: PredefinedSplit (train vs val)

Output: (model, train_metrics, val_metrics, test_metrics)
```

#### `build_svr_model(hp, constrain_for_overfitting=False)`

```
Input:
  - hp: Keras Tuner hyperparameters object
  - constrain_for_overfitting: bool

Hyperparameter search space:
  Normal mode:
    kernel:  Choice(['linear', 'rbf', 'poly'])
    C:       Float(1e-3, 1e3, sampling='log')
    gamma:   Float(1e-4, 1e1, sampling='log')  [rbf/poly only]
    epsilon: Float(0.001, 1.0, sampling='log')
    degree:  Int(2, 5)                          [poly only]

  Constrained mode:
    C:       Float(1e-3, 10, sampling='log')    ← restrict complexity
    epsilon: Float(0.01, 1.0, sampling='log')   ← wider margin
    kernel:  Choice(['linear', 'rbf'])          ← drop poly

Output: sklearn.svm.SVR instance
```

#### `tune_svr_model(stock_symbol, x_train, y_train, x_val, y_val, max_trials, constrain_for_overfitting=False)`

```
Same pattern as tune_ridge_model above, but for SVR.
  - DB caching with model_type='svr'
  - Tuner: BayesianOptimization, up to svr_trials trials
  - PredefinedSplit for validation

Output: (model, train_metrics, val_metrics, test_metrics)
```

### 15.5 Changes to `train_and_validate_models()`

#### New parameters:
```python
def train_and_validate_models(
    ...,
    # Existing params remain unchanged
    # New params:
    ridge_trials: int = 40,
    ridge_retrain_increment: int = 10,
    svr_trials: int = 40,
    svr_retrain_increment: int = 10,
):
```

#### New training blocks (insert after XGBoost block, before ensemble):

```
# ─── RIDGE TRAINING ───
ridge_model = None
ridge_overfitted = False
ridge_constrained = False
for ridge_attempt in range(max_retrains):
    ridge_model, ridge_train_m, ridge_val_m, ridge_test_m = tune_ridge_model(
        stock_symbol, x_train_df, y_train_unscaled,
        x_val_df, y_val_unscaled, x_test_df, y_test_unscaled,
        max_trials=ridge_trials, constrain=ridge_constrained
    )
    ridge_overfitted, ridge_score = detect_overfitting(
        ridge_train_m, ridge_val_m, ridge_test_m,
        "Ridge", overfitting_threshold
    )
    training_history['ridge'].append({...})
    if not ridge_overfitted: break
    ridge_constrained = True
    ridge_trials += ridge_retrain_increment

# ─── SVR TRAINING ───
svr_model = None
svr_overfitted = False
svr_constrained = False
for svr_attempt in range(max_retrains):
    svr_model, svr_train_m, svr_val_m, svr_test_m = tune_svr_model(
        stock_symbol, x_train_df, y_train_unscaled,
        x_val_df, y_val_unscaled, x_test_df, y_test_unscaled,
        max_trials=svr_trials, constrain=svr_constrained
    )
    svr_overfitted, svr_score = detect_overfitting(
        svr_train_m, svr_val_m, svr_test_m,
        "SVR", overfitting_threshold
    )
    training_history['svr'].append({...})
    if not svr_overfitted: break
    svr_constrained = True
    svr_trials += svr_retrain_increment
```

#### Updated ensemble weight calculation:

```python
# Align Ridge/SVR predictions with sequence model (trim first time_steps-1)
ridge_train_pred = ridge_model.predict(x_train_df.values)[time_steps-1:]
ridge_val_pred   = ridge_model.predict(x_val_df.values)[time_steps-1:]
ridge_test_pred  = ridge_model.predict(x_test_df.values)[time_steps-1:]

svr_train_pred = svr_model.predict(x_train_df.values)[time_steps-1:]
svr_val_pred   = svr_model.predict(x_val_df.values)[time_steps-1:]
svr_test_pred  = svr_model.predict(x_test_df.values)[time_steps-1:]

# Compute validation MSE for all 5 models (all in unscaled space)
ridge_val_mse = mean_squared_error(y_val_aligned, ridge_val_pred)
svr_val_mse   = mean_squared_error(y_val_aligned, svr_val_pred)

# Inverse MSE weighting for 5 models
inv_mse = {
    seq_model_key: 1 / seq_val_mse,
    'rf':    1 / rf_val_mse,
    'xgb':   1 / xgb_val_mse,
    'ridge': 1 / ridge_val_mse,
    'svr':   1 / svr_val_mse,
}
total = sum(inv_mse.values())
weights = {k: v / total for k, v in inv_mse.items()}

# Zero out negligible weights (< MIN_ENSEMBLE_WEIGHT = 0.005)
for name in list(weights.keys()):
    if weights[name] < MIN_ENSEMBLE_WEIGHT:
        weights[name] = 0.0
remaining = sum(weights.values())
if remaining > 0:
    weights = {k: v / remaining for k, v in weights.items()}
```

#### Updated ensemble predictions:

```python
ensemble_train_pred = (
    weights[seq_model_key] * seq_train_pred +
    weights['rf']    * rf_train_pred +
    weights['xgb']   * xgb_train_pred +
    weights['ridge'] * ridge_train_pred +
    weights['svr']   * svr_train_pred
)
# Same for val and test
```

#### Updated return dict:

```python
models = {
    'sequence_model': sequence_model,
    'sequence_model_type': seq_model_key,
    'lstm': lstm_model if not use_tcn else None,
    'tcn': tcn_model if use_tcn else None,
    'rf': rf_model,
    'xgb': xgb_model,
    'ridge': ridge_model,    # ← NEW
    'svr': svr_model,        # ← NEW
    'ensemble_weights': weights  # Now includes 'ridge' and 'svr' keys
}
```

### 15.6 Changes to `predict_future_price_changes()`

The prediction function must be updated to use all 5 models:

1. **Extract Ridge and SVR from model dict:**
   ```python
   ridge_model = model.get('ridge', None)
   svr_model = model.get('svr', None)
   ```

2. **Get weights for all 5 models:**
   ```python
   ridge_weight = ensemble_weights.get('ridge', 0.0)
   svr_weight = ensemble_weights.get('svr', 0.0)
   ```

3. **Generate predictions from Ridge and SVR in the day-by-day loop:**
   ```python
   # Inside the prediction loop, after RF and XGBoost predictions:
   if ridge_model is not None:
       forecast_ridge = ridge_model.predict(x_input_rf_df.values)[0]
       forecast_ridge = np.clip(forecast_ridge, -MAX_DAILY_RETURN, MAX_DAILY_RETURN)
   else:
       forecast_ridge = 0.0
       ridge_weight = 0.0

   if svr_model is not None:
       forecast_svr = svr_model.predict(x_input_rf_df.values)[0]
       forecast_svr = np.clip(forecast_svr, -MAX_DAILY_RETURN, MAX_DAILY_RETURN)
   else:
       forecast_svr = 0.0
       svr_weight = 0.0
   ```

4. **Update the ensemble combination:**
   ```python
   raw_ensemble = (
       seq_weight   * forecast_seq +
       rf_weight    * forecast_rf +
       xgb_weight   * forecast_xgb +
       ridge_weight * forecast_ridge +
       svr_weight   * forecast_svr
   )
   ```

5. **Update uncertainty estimation** to include Ridge/SVR variance:
   ```python
   ensemble_std = np.sqrt(
       seq_weight**2   * forecast_seq_std**2 +
       rf_weight**2    * (historical_std * 0.5)**2 +
       xgb_weight**2   * (historical_std * 0.5)**2 +
       ridge_weight**2 * (historical_std * 0.5)**2 +
       svr_weight**2   * (historical_std * 0.5)**2
   )
   ```

6. **Update print/log statements** to show all 5 model predictions.

7. **Handle the 2-model fallback** (when XGBoost is unavailable) — update to handle
   variable numbers of available models. Renormalize weights among available models.

### 15.7 Changes to `model_trainer.py`

```python
# Update required_model_types to include ridge and svr
training_needs = db_interactions.get_tickers_needing_training(
    max_age_days=max_model_age_days,
    required_model_types=['rf', 'xgb', 'ridge', 'svr', 'tcn' if use_tcn else 'lstm']
)
```

### 15.8 Changes to `price_predictor.py`

When rebuilding models from cached hyperparameters, also load and rebuild Ridge and SVR:

```python
ridge_hp = db_interactions.load_hyperparameters(ticker, 'ridge', max_age_days, num_features)
svr_hp = db_interactions.load_hyperparameters(ticker, 'svr', max_age_days, num_features)
```

### 15.9 Changes to `db_interactions.py`

- Update `invalidate_hyperparameters()` — already supports arbitrary model_type strings.
- Update `save_hyperparameters()` — validation of model_type must include 'ridge' and 'svr'.
- Update `load_hyperparameters()` — validation of model_type must include 'ridge' and 'svr'.

### 15.10 Testing Requirements for Ridge & SVR Integration

| Test | File | Validates |
|---|---|---|
| `test_build_ridge_model` | `test_model_definitions.py` | Ridge model creation with mock HP |
| `test_build_ridge_constrained` | `test_model_definitions.py` | Overfitting constraints applied |
| `test_build_svr_model` | `test_model_definitions.py` | SVR model creation with mock HP |
| `test_build_svr_constrained` | `test_model_definitions.py` | Overfitting constraints applied |
| `test_tune_ridge_model` | `test_model_tuning.py` | Full tuning on small dataset |
| `test_tune_svr_model` | `test_model_tuning.py` | Full tuning on small dataset |
| `test_ridge_hp_caching` | `test_model_tuning.py` | Save/load hyperparameters from DB |
| `test_svr_hp_caching` | `test_model_tuning.py` | Save/load hyperparameters from DB |
| `test_5_model_ensemble_weights` | `test_ensemble_weights.py` | 5-model inverse-MSE weighting |
| `test_ensemble_zero_weight` | `test_ensemble_weights.py` | Negligible weights zeroed out |
| `test_prediction_5_models` | `test_predict_future.py` | All 5 models used in prediction loop |
| `test_prediction_missing_ridge` | `test_predict_future.py` | Graceful fallback if Ridge is None |
| `test_prediction_missing_svr` | `test_predict_future.py` | Graceful fallback if SVR is None |
| `test_db_model_type_enum` | `test_db_interactions.py` | 'ridge' and 'svr' accepted |
| `test_required_model_types` | `test_model_trainer.py` | All 5 types in freshness check |

---

## Summary

This document provides complete, unambiguous instructions for building or extending the
Stock Portfolio Builder.  The key additions beyond the existing README are:

1. **Prerequisites** — all non-Python requirements listed explicitly.
2. **Architecture diagram** — visual overview of the 3-phase pipeline.
3. **Data model** — ER diagram with all tables and relationships.
4. **Ridge & SVR integration** — complete specification for expanding the ensemble from
   3 models to 5 models, including hyperparameter tuning, overfitting detection, DB
   schema changes, ensemble weighting, prediction updates, and testing.
5. **Testing standards** — directory structure, coverage targets, fixture requirements.
6. **Error handling & logging** — replace `print()` with `logging`, structured exceptions.
7. **Configuration management** — centralized `pipeline_config.py` with all magic numbers.
8. **ML module splitting** — decompose `ml_builder.py` into 8 focused modules.
9. **Data validation contracts** — exact DataFrame schemas at each boundary.
10. **Idempotency requirements** — every step safely re-runnable.
