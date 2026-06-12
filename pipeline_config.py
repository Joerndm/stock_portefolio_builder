"""
Centralized Configuration for Stock Portfolio Builder Pipeline.

All magic numbers and tunable parameters live here. Every module imports
from this file instead of hard-coding values.

Environment variable overrides are supported via python-dotenv:
    export SPB_TIME_STEPS=60          # overrides DataPipelineConfig.time_steps
    export SPB_MAX_RETRAINS=200       # overrides MLTrainingConfig.max_retrains
    export SPB_GPU_MEMORY_LIMIT=8192  # overrides GPUConfig.memory_limit_mb

Naming convention for env vars: SPB_{UPPER_SNAKE_CASE_FIELD_NAME}
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv("dev.env")
except ImportError:
    pass


def _env(key: str, default, cast=None):
    """Read an environment variable with optional type casting."""
    val = os.environ.get(f"SPB_{key}")
    if val is None:
        return default
    if cast is not None:
        return cast(val)
    return val


def _env_float(key: str, default: float) -> float:
    return _env(key, default, float)


def _env_int(key: str, default: int) -> int:
    return _env(key, default, int)


def _env_bool(key: str, default: bool) -> bool:
    val = os.environ.get(f"SPB_{key}")
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes")


def _get_db_password() -> str:
    """Return the database password using the normalized env var with fallback."""
    return os.environ.get("DB_PASSWORD") or os.environ.get("DB_PASS", "")


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
@dataclass
class DatabaseConfig:
    host: str = field(default_factory=lambda: os.environ.get("DB_HOST", "127.0.0.1"))
    user: str = field(default_factory=lambda: os.environ.get("DB_USER", "root"))
    password: str = field(default_factory=_get_db_password)
    name: str = field(default_factory=lambda: os.environ.get("DB_NAME", "stock_portefolio_builder"))


# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------
@dataclass
class GPUConfig:
    memory_limit_mb: int = field(default_factory=lambda: _env_int("GPU_MEMORY_LIMIT", 7168))
    tf_log_level: str = "2"


# ---------------------------------------------------------------------------
# Data Pipeline
# ---------------------------------------------------------------------------
@dataclass
class DataPipelineConfig:
    # Sequence model lookback window
    time_steps: int = field(default_factory=lambda: _env_int("TIME_STEPS", 30))

    # Train / validation / test split proportions
    validation_size: float = field(default_factory=lambda: _env_float("VALIDATION_SIZE", 0.20))
    test_size: float = field(default_factory=lambda: _env_float("TEST_SIZE", 0.10))

    # Minimum rows required after cleaning (must cover at least 1 trading year)
    min_rows_floor: int = 252

    # Forecast-out proportion for prediction target
    forecast_out_pct: float = 0.05

    # Outlier removal (z-score threshold, 0 = disabled)
    outlier_zscore_threshold: float = field(default_factory=lambda: _env_float("OUTLIER_ZSCORE", 3.0))

    # Correlated feature removal threshold (0 = disabled)
    correlation_threshold: float = field(default_factory=lambda: _env_float("CORRELATION_THRESHOLD", 0.85))


# ---------------------------------------------------------------------------
# ML Training
# ---------------------------------------------------------------------------
@dataclass
class MLTrainingConfig:
    # Overfitting
    max_retrains: int = field(default_factory=lambda: _env_int("MAX_RETRAINS", 150))
    overfitting_threshold: float = field(default_factory=lambda: _env_float("OVERFITTING_THRESHOLD", 0.15))

    # TCN / LSTM (sequence model)
    use_sequence_model: bool = field(default_factory=lambda: _env_bool("USE_SEQUENCE_MODEL", False))
    use_tcn: bool = field(default_factory=lambda: _env_bool("USE_TCN", True))
    lstm_trials: int = field(default_factory=lambda: _env_int("LSTM_TRIALS", 50))
    lstm_executions: int = field(default_factory=lambda: _env_int("LSTM_EXECUTIONS", 10))
    lstm_epochs: int = field(default_factory=lambda: _env_int("LSTM_EPOCHS", 500))
    lstm_retrain_trials_increment: int = 10
    lstm_retrain_executions_increment: int = 2
    tcn_trials: int = field(default_factory=lambda: _env_int("TCN_TRIALS", 30))
    tcn_epochs: int = field(default_factory=lambda: _env_int("TCN_EPOCHS", 100))
    tcn_retrain_increment: int = 10

    # Random Forest
    rf_trials: int = field(default_factory=lambda: _env_int("RF_TRIALS", 100))
    rf_retrain_increment: int = 25

    # XGBoost
    xgb_trials: int = field(default_factory=lambda: _env_int("XGB_TRIALS", 60))
    xgb_retrain_increment: int = 10

    # Ridge Regression
    ridge_trials: int = field(default_factory=lambda: _env_int("RIDGE_TRIALS", 50))
    ridge_retrain_increment: int = 15

    # SVR
    svr_trials: int = field(default_factory=lambda: _env_int("SVR_TRIALS", 40))
    svr_retrain_increment: int = 10

    # Multi-metric overfitting detection
    use_multi_metric_detection: bool = True

    # Ensemble
    min_ensemble_weight: float = 0.005

    # Required model types for freshness checks
    @property
    def required_model_types(self) -> List[str]:
        base = ["rf", "xgb", "ridge", "svr"]
        if self.use_sequence_model:
            seq = "tcn" if self.use_tcn else "lstm"
            base.append(seq)
        return base


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
@dataclass
class PredictionConfig:
    # Prediction horizon
    prediction_days_multiplier: int = 3  # prediction_days = time_steps * this
    investment_years: int = field(default_factory=lambda: _env_int("INVESTMENT_YEARS", 7))

    # Monte Carlo Dropout
    use_mc_dropout: bool = field(default_factory=lambda: _env_bool("USE_MC_DROPOUT", True))
    mc_iterations: int = field(default_factory=lambda: _env_int("MC_ITERATIONS", 30))

    # Monte Carlo Simulation
    sim_amount: int = field(default_factory=lambda: _env_int("SIM_AMOUNT", 1000))

    # Max model age before retraining (days)
    max_model_age_days: int = field(default_factory=lambda: _env_int("MAX_MODEL_AGE_DAYS", 30))
    max_prediction_age_days: int = field(default_factory=lambda: _env_int("MAX_PREDICTION_AGE_DAYS", 1))

    # Prediction clipping
    max_daily_return: float = 0.20  # ±20%

    # Mean reversion
    mean_reversion_strength: float = 0.10
    mean_reversion_threshold_std: float = 2.5
    mean_reversion_hard_cap_std: float = 4.0

    # Prediction noise / uncertainty
    initial_confidence: float = 0.60
    confidence_decay_per_day: float = 0.003
    min_confidence: float = 0.35

    # Directional balance
    max_same_direction_days: int = 5

    # Feature degradation rates for 90-day predictions
    # Features not listed here are recalculated dynamically (no decay)
    feature_decay_rates: Dict[str, float] = field(default_factory=lambda: {
        "volume_sma_20": 0.02,
        "volume_ema_20": 0.02,
        "volume_ratio": 0.05,
        "vwap": 0.03,
        "obv": 0.01,
        "ATR_14": 0.01,
        "VIX_close": 0.10,
    })


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------
@dataclass
class PortfolioConfig:
    default_risk_free_rate: float = 0.04
    min_portfolio_size: int = 10
    max_portfolio_size: int = 30


# ---------------------------------------------------------------------------
# Walk-Forward Validation
# ---------------------------------------------------------------------------
@dataclass
class WalkForwardConfig:
    enabled: bool = field(default_factory=lambda: _env_bool("WALK_FORWARD_ENABLED", False))
    min_train_years: int = 2
    test_window_years: int = 1
    step_years: int = 1


# ---------------------------------------------------------------------------
# Singleton accessors (create once, reuse everywhere)
# ---------------------------------------------------------------------------
_db_config: Optional[DatabaseConfig] = None
_gpu_config: Optional[GPUConfig] = None
_data_config: Optional[DataPipelineConfig] = None
_ml_config: Optional[MLTrainingConfig] = None
_pred_config: Optional[PredictionConfig] = None
_portfolio_config: Optional[PortfolioConfig] = None
_wf_config: Optional[WalkForwardConfig] = None


def get_db_config() -> DatabaseConfig:
    global _db_config
    if _db_config is None:
        _db_config = DatabaseConfig()
    return _db_config


def get_gpu_config() -> GPUConfig:
    global _gpu_config
    if _gpu_config is None:
        _gpu_config = GPUConfig()
    return _gpu_config


def get_data_config() -> DataPipelineConfig:
    global _data_config
    if _data_config is None:
        _data_config = DataPipelineConfig()
    return _data_config


def get_ml_config() -> MLTrainingConfig:
    global _ml_config
    if _ml_config is None:
        _ml_config = MLTrainingConfig()
    return _ml_config


def get_pred_config() -> PredictionConfig:
    global _pred_config
    if _pred_config is None:
        _pred_config = PredictionConfig()
    return _pred_config


def get_portfolio_config() -> PortfolioConfig:
    global _portfolio_config
    if _portfolio_config is None:
        _portfolio_config = PortfolioConfig()
    return _portfolio_config


def get_wf_config() -> WalkForwardConfig:
    global _wf_config
    if _wf_config is None:
        _wf_config = WalkForwardConfig()
    return _wf_config
