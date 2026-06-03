"""Cache-first contract helpers for prediction runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable


@dataclass(frozen=True)
class PredictionCacheState:
    """Summarize cached hyperparameter availability for a prediction run."""

    ticker: str
    required_model_types: list[str]
    available_model_types: list[str]
    missing_model_types: list[str]
    max_age_days: int | None

    @property
    def cache_status(self) -> str:
        return "cache_hit" if not self.missing_model_types else "training_required"

    def to_result(self) -> dict[str, Any]:
        return {
            "cache_status": self.cache_status,
            "required_model_types": list(self.required_model_types),
            "available_model_types": list(self.available_model_types),
            "missing_model_types": list(self.missing_model_types),
            "cache_max_age_days": self.max_age_days,
        }


class PredictionCacheContractError(RuntimeError):
    """Base error for prediction cache contract failures."""

    def __init__(
        self,
        *,
        ticker: str,
        message: str,
        cache_status: str,
        required_model_types: Iterable[str] | None = None,
        missing_model_types: Iterable[str] | None = None,
        model_type: str | None = None,
        invalidated_count: int = 0,
    ) -> None:
        super().__init__(message)
        self.ticker = ticker
        self.cache_status = cache_status
        self.required_model_types = list(required_model_types or [])
        self.missing_model_types = list(missing_model_types or [])
        self.model_type = model_type
        self.invalidated_count = invalidated_count

    def to_result(self) -> dict[str, Any]:
        return {
            "cache_status": self.cache_status,
            "required_model_types": list(self.required_model_types),
            "missing_model_types": list(self.missing_model_types),
            "failing_model_type": self.model_type,
            "invalidated_cache_rows": self.invalidated_count,
        }


class PredictionTrainingRequiredError(PredictionCacheContractError):
    """Raised when prediction cannot continue because trained cache is missing."""

    def __init__(
        self,
        *,
        ticker: str,
        required_model_types: Iterable[str],
        missing_model_types: Iterable[str],
        message: str | None = None,
    ) -> None:
        missing = list(missing_model_types)
        message = message or (
            f"Prediction requires trained cache for {ticker}; missing cached hyperparameters for: "
            f"{', '.join(missing)}"
        )
        super().__init__(
            ticker=ticker,
            message=message,
            cache_status="training_required",
            required_model_types=required_model_types,
            missing_model_types=missing,
        )


class PredictionCacheInvalidatedError(PredictionCacheContractError):
    """Raised when a cached hyperparameter row is invalid during strict restore."""

    def __init__(
        self,
        *,
        ticker: str,
        model_type: str,
        reason: Any,
        invalidated_count: int = 0,
    ) -> None:
        super().__init__(
            ticker=ticker,
            message=(
                f"Invalid cached {model_type.upper()} hyperparameters for {ticker}; "
                f"training is required before prediction can continue: {reason}"
            ),
            cache_status="cache_invalidated",
            required_model_types=[model_type],
            missing_model_types=[model_type],
            model_type=model_type,
            invalidated_count=invalidated_count,
        )


def inspect_prediction_cache(
    ticker: str,
    required_model_types: Iterable[str],
    *,
    max_age_days: int | None = 30,
    load_hyperparameters: Callable[..., Any] | None = None,
) -> PredictionCacheState:
    """Inspect cached hyperparameter availability for one ticker."""
    required = [str(model_type).lower() for model_type in required_model_types]
    if load_hyperparameters is None:
        from db_interactions import load_hyperparameters as load_hyperparameters_func
    else:
        load_hyperparameters_func = load_hyperparameters

    available: list[str] = []
    missing: list[str] = []

    for model_type in required:
        cached_hp = load_hyperparameters_func(
            ticker=ticker,
            model_type=model_type,
            max_age_days=max_age_days,
        )
        if cached_hp is None:
            missing.append(model_type)
        else:
            available.append(model_type)

    return PredictionCacheState(
        ticker=ticker,
        required_model_types=required,
        available_model_types=available,
        missing_model_types=missing,
        max_age_days=max_age_days,
    )


def require_prediction_cache(
    ticker: str,
    required_model_types: Iterable[str],
    *,
    max_age_days: int | None = 30,
    load_hyperparameters: Callable[..., Any] | None = None,
) -> PredictionCacheState:
    """Require that all prediction model caches exist before rebuilding models."""
    cache_state = inspect_prediction_cache(
        ticker,
        required_model_types,
        max_age_days=max_age_days,
        load_hyperparameters=load_hyperparameters,
    )
    if cache_state.missing_model_types:
        raise PredictionTrainingRequiredError(
            ticker=ticker,
            required_model_types=cache_state.required_model_types,
            missing_model_types=cache_state.missing_model_types,
        )
    return cache_state


def raise_cache_invalidated_error(
    ticker: str,
    model_type: str,
    reason: Any,
    *,
    invalidator: Callable[[str, str, Any], int] | None = None,
) -> None:
    """Invalidate one cached hyperparameter row, then raise a strict restore error."""
    invalidated_count = 0
    if invalidator is None:
        from model_cache_utils import invalidate_hyperparameter_cache as invalidator_func
    else:
        invalidator_func = invalidator

    try:
        invalidated_count = invalidator_func(ticker, model_type, reason)
    except Exception:
        invalidated_count = 0

    raise PredictionCacheInvalidatedError(
        ticker=ticker,
        model_type=model_type,
        reason=reason,
        invalidated_count=invalidated_count,
    )