# Stock Portfolio Builder

A comprehensive stock data pipeline and machine learning system for fetching financial data, calculating metrics, and predicting stock prices.

## Features

- **Automated Data Fetching**: Dynamically fetches index constituents from Wikipedia (S&P 500, C25, etc.)
- **TTM Financial Calculations**: Trailing Twelve Months metrics from quarterly reports
- **Blacklist Management**: Automatic handling of delisted/invalid tickers
- **ML Price Prediction**: LSTM, Random Forest, and XGBoost models for price forecasting
- **Database Storage**: MySQL database for persistent data storage

## Prerequisites

- Python 3.10 or 3.12 (see [Python Environment](#python-environment) below)
- MySQL database server
- Database credentials configured in `dev.env`

## Python Environment

This project supports two Python environments depending on your use case:

### Python 3.12 (Recommended)

Use this environment for the **data pipeline, Streamlit UI, and general usage**. It includes:

- **Data & Visualization**: pandas, numpy, matplotlib, scipy, scikit-learn, plotly
- **Financial Data**: yfinance, pandas-ta, lxml
- **Web UI**: Streamlit
- **Database**: SQLAlchemy, mysql-connector-python

This environment does **not** include TensorFlow or GPU-accelerated ML libraries.

### Python 3.10 (Legacy — GPU / Deep Learning)

Use this environment **only if you need TensorFlow GPU support** for LSTM model training. It includes:

- **Deep Learning**: TensorFlow 2.10 (GPU), Keras Tuner
- **ML**: scikit-learn, XGBoost
- **Data**: pandas, numpy, matplotlib, scipy, yfinance

> **GPU Requirement**: TensorFlow 2.10 requires **NVIDIA CUDA 11.2** and **cuDNN 8.1** installed separately. You can also install via `conda install tensorflow-gpu==2.10.0`.

> **Note**: This environment is **not compatible with Python 3.11 or newer**. It does not include Streamlit or the web UI dependencies.

## Installation

1. Clone or download the repository to your local machine.

2. Create a virtual environment (recommended):

    **For Python 3.12 (recommended):**
    ```bash
    conda create -n stock_env python=3.12
    conda activate stock_env
    pip install -r requirements_PY_3_12.txt
    ```

    **For Python 3.10 (GPU/TensorFlow):**
    ```bash
    conda create -n stock_env_gpu python=3.10
    conda activate stock_env_gpu
    pip install -r requirements_PY_3_10.txt
    ```

3. Configure database credentials in `dev.env`:
    ```
    DB_HOST=your_host
    DB_USER=your_user
    DB_PASS=your_password
    DB_NAME=your_database
    ```

4. Initialize the database using the DDL scripts in `database_files/`:
    ```bash
    mysql -u your_user -p your_database < database_files/ddl.sql
    ```

## Usage

### Fetching Stock Data (Recommended)

Use the Stock Data Orchestrator to fetch and update all stock data:

```python
from stock_orchestrator import StockDataOrchestrator

# Initialize with default indices (S&P 500 + C25)
orchestrator = StockDataOrchestrator()

# Run the full data pipeline
stats = orchestrator.run()

# Or run for specific indices
orchestrator = StockDataOrchestrator(indices=['SP500'])
stats = orchestrator.run()
```

**Command line:**
```bash
python stock_orchestrator.py
```

### Programmatic Usage

```python
from stock_orchestrator import run_with_indices

# Fetch data for specific indices
stats = run_with_indices(['SP500', 'C25'], prefer_ttm=True)
```

### Building ML Models

```python
from ml_builder import train_and_validate_models

# Train models for a specific stock
# (Data is fetched from the database)
results = train_and_validate_models(
    stock_symbol='AAPL',
    # ... other parameters
)
```

### Generating Predictions

```python
from price_predictor import run_predictions

summary = run_predictions(max_prediction_age_days=1, investment_years=7)
```

`price_predictor.py` is cache-first: it rebuilds models only from valid cached hyperparameters.
If any required cache entry is missing, stale, or invalid, the ticker returns an explicit status such
as `training_required` or `cache_invalidated` and prediction stops for that ticker instead of tuning
models inside the prediction phase. Run `model_trainer.py` first to refresh missing caches.

If cached flat-model rows (`rf`, `xgb`, `ridge`, `svr`) need to be reconciled after a preprocessing
contract change, use the supported admin command instead of an ad hoc script:

```bash
python model_trainer.py --refresh-cache-contract APP ASML.AS
```

To refresh only a subset of flat-model caches and then verify each ticker can still predict from
cache-only state, run:

```bash
python model_trainer.py --refresh-cache-contract APP ASML.AS --refresh-model-types ridge svr --validate-prediction
```

### Auditing And Repair Planning

Run the stock-data validation first:

```bash
python validate_stock_data.py
```

Then build repair cohorts from the generated validation report:

```bash
python repair_cohort_planner.py --report data_validation_report.json
```

This writes `repair_cohorts.json` and `repair_cohort_summary.txt`, which separate safe ticker-scoped
repair cohorts such as stale prices, missing quarterly data, and ratio lag from spike findings that
should be manually reviewed before any destructive cleanup.

Build the executable repair queue in dry-run mode with:

```bash
python repair_ticker_cohorts.py --plan repair_cohorts.json
```

After reviewing the queue, execute only the safe cohorts with:

```bash
python repair_ticker_cohorts.py --plan repair_cohorts.json --execute
```

This writes `repair_execution_report.json` and `repair_execution_summary.txt` so each repair run leaves
an auditable record of which ticker-scoped actions were planned or executed.

## Project Structure

| File | Description |
|------|-------------|
| `stock_orchestrator.py` | Main entry point - orchestrates the complete data pipeline |
| `dynamic_index_fetcher.py` | Fetches index constituents from Wikipedia |
| `enhanced_financial_fetcher.py` | Fetches quarterly financial data with TTM calculations |
| `ttm_financial_calculator.py` | TTM calculations and ratio computations |
| `db_interactions.py` | Database import/export operations |
| `blacklist_manager.py` | Manages blacklisted (delisted/invalid) tickers |
| `ml_builder.py` | Machine learning model training and prediction |
| `stock_data_fetch.py` | Core data fetching utilities |

## Data Pipeline Flow

1. **Get Symbols**: Fetch index constituents from Wikipedia (cached monthly)
2. **Filter Blacklist**: Remove delisted/invalid tickers
3. **Fetch Stock Info**: Basic company information
4. **Fetch Price Data**: Historical OHLCV data
5. **Fetch Financial Data**: Income statement, balance sheet, cash flow
6. **Fetch Quarterly Data**: Quarterly reports for TTM calculations
7. **Calculate Ratios**: P/E, P/B, P/S, P/FCF using TTM when available

## Blacklist Management

Tickers are automatically blacklisted when:
- Symbol is delisted or not found
- No data available from data sources
- Repeated fetch errors

Blacklisted tickers are stored in `blacklisted_tickers.json` and filtered from future fetches.

## Configuration

### Supported Indices

- `SP500` - S&P 500 (US)
- `C25` - OMX Copenhagen 25 (Denmark)
- Market indices: `^VIX`, `^GSPC`, `^IXIC`, `^DJI`, etc.

### Environment Variables

Configure in `dev.env`:
- `DB_HOST`, `DB_USER`, `DB_PASS`, `DB_NAME` - Database credentials

## License

See [LICENSE](LICENSE) file for details.
