"""
Portfolio Configuration Module

This module provides configuration classes and settings for portfolio optimization.
It defines investor risk profiles, investment parameters, and validation utilities
for building customized stock portfolios.

Features:
---------
- Risk level enumeration (LOW, MEDIUM, HIGH) with corresponding volatility constraints
- Investor profile configuration with validation
- Default settings for portfolio construction
- Future extensibility for industry/country filtering

Usage:
------
>>> from portfolio_config import InvestorProfile, RiskLevel
>>> profile = InvestorProfile(
...     risk_level=RiskLevel.MEDIUM,
...     investment_years=5,
...     portfolio_size=25
... )
>>> print(profile.get_volatility_cap())
0.25

Author: Stock Portfolio Builder
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict


class RiskLevel(Enum):
    """
    Investor risk tolerance levels with associated portfolio optimization strategies.
    
    Attributes:
        LOW: Conservative strategy - minimize volatility, accept lower returns
             Targets volatility cap of 15%, focuses on stable dividend stocks
        MEDIUM: Balanced strategy - maximize Sharpe ratio with moderate risk
                Targets volatility cap of 25%, balanced growth and stability
        HIGH: Aggressive strategy - maximize returns, accept higher volatility
              Targets volatility cap of 40%, focuses on growth stocks
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class InvestmentStrategy(Enum):
    """
    Investment strategy that determines how stocks are scored and filtered
    using fundamental data from the database.

    Each strategy applies different weights to valuation ratios, growth
    metrics, profitability, and dividend data when ranking candidate stocks.
    """
    BALANCED = "balanced"        # Default: pure ML prediction score
    DIVIDEND = "dividend"        # Favour dividend-paying, low-payout companies
    GROWTH = "growth"            # Favour high revenue/earnings growth
    VALUE = "value"              # Favour low P/E, P/B, P/S multiples
    GARP = "garp"                # Growth At a Reasonable Price (PEG-style)
    QUALITY = "quality"          # Favour high ROE, margins, low debt
    MOMENTUM = "momentum"        # Favour recent price momentum


# Human-readable labels for the GUI
STRATEGY_LABELS: Dict[str, str] = {
    "balanced": "Balanced (ML Only)",
    "dividend": "Dividend",
    "growth": "Growth",
    "value": "Value",
    "garp": "GARP",
    "quality": "Quality",
    "momentum": "Momentum",
}

STRATEGY_DESCRIPTIONS: Dict[str, str] = {
    "balanced": "Rank purely by ML prediction score — no fundamental filter.",
    "dividend": "Favour stocks paying dividends with sustainable payout ratios.",
    "growth": "Favour stocks with high revenue and earnings growth (TTM YoY).",
    "value": "Favour stocks trading at low P/E, P/B, and P/S multiples.",
    "garp": "Growth At a Reasonable Price — blend of growth rate and P/E (PEG).",
    "quality": "Favour high ROE, strong margins, and low leverage.",
    "momentum": "Favour stocks with strong recent price momentum (3M / 6M).",
}


# Volatility constraints by risk level (annualized standard deviation)
VOLATILITY_CAPS: Dict[RiskLevel, float] = {
    RiskLevel.LOW: 0.15,     # 15% annual volatility cap
    RiskLevel.MEDIUM: 0.25,  # 25% annual volatility cap
    RiskLevel.HIGH: 0.40     # 40% annual volatility cap
}

# Minimum acceptable Sharpe ratio by risk level
MIN_SHARPE_RATIOS: Dict[RiskLevel, float] = {
    RiskLevel.LOW: 0.3,
    RiskLevel.MEDIUM: 0.5,
    RiskLevel.HIGH: 0.2  # More lenient for high-risk strategies
}

# Default risk-free rate (annual, e.g., treasury bond yield)
DEFAULT_RISK_FREE_RATE: float = 0.04  # 4%


@dataclass
class InvestorProfile:
    """
    Configuration class for investor preferences and portfolio constraints.
    
    This class encapsulates all settings needed to customize portfolio optimization
    for a particular investor's risk tolerance and investment horizon.
    
    Attributes:
        risk_level: Risk tolerance level (LOW, MEDIUM, HIGH)
        investment_years: Investment horizon in years (1-10)
        portfolio_size: Number of stocks in the portfolio (10-30)
        industries: Optional list of industries to include (None = all)
        countries: Optional list of countries to include (None = all)
        excluded_tickers: Optional list of tickers to exclude from consideration
        min_market_cap: Optional minimum market capitalization filter
        require_dividend: If True, only include dividend-paying stocks
    
    Raises:
        ValueError: If investment_years not in range 1-10
        ValueError: If portfolio_size not in range 10-30
    
    Example:
        >>> profile = InvestorProfile(
        ...     risk_level=RiskLevel.MEDIUM,
        ...     investment_years=5,
        ...     portfolio_size=25
        ... )
        >>> profile.validate()
        True
    """
    risk_level: RiskLevel = RiskLevel.MEDIUM
    investment_years: int = 5
    portfolio_size: int = 25
    strategy: InvestmentStrategy = InvestmentStrategy.BALANCED
    industries: Optional[List[str]] = None
    countries: Optional[List[str]] = None
    excluded_tickers: List[str] = field(default_factory=list)
    min_market_cap: Optional[float] = None
    require_dividend: bool = False
    
    def __post_init__(self):
        """Validate configuration on initialization."""
        self.validate()
    
    def validate(self) -> bool:
        """
        Validate all configuration parameters.
        
        Returns:
            bool: True if all parameters are valid
            
        Raises:
            ValueError: If any parameter is out of valid range
        """
        if not 1 <= self.investment_years <= 10:
            raise ValueError(
                f"Investment years must be between 1 and 10, got {self.investment_years}"
            )
        
        if not 10 <= self.portfolio_size <= 30:
            raise ValueError(
                f"Portfolio size must be between 10 and 30, got {self.portfolio_size}"
            )
        
        if not isinstance(self.risk_level, RiskLevel):
            raise ValueError(
                f"Risk level must be a RiskLevel enum, got {type(self.risk_level)}"
            )
        
        return True
    
    def get_volatility_cap(self) -> float:
        """
        Get the maximum allowed portfolio volatility for this risk profile.
        
        Returns:
            float: Maximum annualized volatility (e.g., 0.25 for 25%)
        """
        return VOLATILITY_CAPS[self.risk_level]
    
    def get_min_sharpe_ratio(self) -> float:
        """
        Get the minimum acceptable Sharpe ratio for this risk profile.
        
        Returns:
            float: Minimum Sharpe ratio threshold
        """
        return MIN_SHARPE_RATIOS[self.risk_level]
    
    def get_monte_carlo_years(self) -> int:
        """
        Get the number of years to run Monte Carlo simulation.
        
        Returns:
            int: Number of years (matches investment horizon)
        """
        return self.investment_years
    
    def to_dict(self) -> Dict:
        """
        Convert profile to dictionary for serialization/logging.
        
        Returns:
            Dict: Profile configuration as dictionary
        """
        return {
            "risk_level": self.risk_level.value,
            "investment_years": self.investment_years,
            "portfolio_size": self.portfolio_size,
            "strategy": self.strategy.value,
            "industries": self.industries,
            "countries": self.countries,
            "excluded_tickers": self.excluded_tickers,
            "min_market_cap": self.min_market_cap,
            "require_dividend": self.require_dividend,
            "volatility_cap": self.get_volatility_cap(),
            "min_sharpe_ratio": self.get_min_sharpe_ratio()
        }
    
    @classmethod
    def from_dict(cls, config: Dict) -> "InvestorProfile":
        """
        Create profile from dictionary configuration.
        
        Args:
            config: Dictionary with profile settings
            
        Returns:
            InvestorProfile: Configured profile instance
        """
        risk_level = RiskLevel(config.get("risk_level", "medium"))
        strategy = InvestmentStrategy(config.get("strategy", "balanced"))
        return cls(
            risk_level=risk_level,
            investment_years=config.get("investment_years", 5),
            portfolio_size=config.get("portfolio_size", 25),
            strategy=strategy,
            industries=config.get("industries"),
            countries=config.get("countries"),
            excluded_tickers=config.get("excluded_tickers", []),
            min_market_cap=config.get("min_market_cap"),
            require_dividend=config.get("require_dividend", False)
        )


@dataclass
class StockPredictionResult:
    """
    Container for individual stock prediction results.
    
    Stores the outcomes of ML predictions and Monte Carlo simulations
    for a single stock, including success/failure status.
    
    Attributes:
        symbol: Stock ticker symbol
        success: Whether prediction completed successfully
        forecast_df: DataFrame with price predictions (None if failed)
        monte_carlo_day_df: Daily Monte Carlo simulation results
        monte_carlo_year_df: Yearly Monte Carlo summary statistics
        error_message: Error description if prediction failed
        execution_time: Time taken for processing in seconds
    """
    symbol: str
    success: bool
    forecast_df: Optional[object] = None  # pd.DataFrame
    monte_carlo_day_df: Optional[object] = None  # pd.DataFrame
    monte_carlo_year_df: Optional[object] = None  # pd.DataFrame
    error_message: Optional[str] = None
    execution_time: float = 0.0
    
    def get_predicted_return(self, days: int = 252) -> Optional[float]:
        """
        Get the predicted return for a given number of trading days.
        
        Args:
            days: Number of trading days (252 = 1 year)
            
        Returns:
            float: Predicted return as decimal (e.g., 0.10 for 10%), or None if unavailable
        """
        if self.forecast_df is None or self.forecast_df.empty:
            return None
        
        try:
            # Get price column (may be named differently)
            price_col = None
            for col in ['close_Price', 'predicted_price', 'price']:
                if col in self.forecast_df.columns:
                    price_col = col
                    break
            
            if price_col is None:
                return None
            
            start_price = self.forecast_df[price_col].iloc[0]
            end_idx = min(days, len(self.forecast_df) - 1)
            end_price = self.forecast_df[price_col].iloc[end_idx]
            
            if start_price > 0:
                return (end_price - start_price) / start_price
            return None
        except (IndexError, KeyError):
            return None
    
    def get_monte_carlo_percentiles(self, year: int) -> Optional[Dict[str, float]]:
        """
        Get Monte Carlo percentile predictions for a specific year.
        
        Args:
            year: Year number (1-10)
            
        Returns:
            Dict with percentile values, or None if unavailable
        """
        if self.monte_carlo_year_df is None:
            return None
        
        try:
            row = self.monte_carlo_year_df.loc[year]
            return {
                "p5": row.get("5th Percentile"),
                "p16": row.get("16th Percentile"),
                "mean": row.get("Mean"),
                "p84": row.get("84th Percentile"),
                "p95": row.get("95th Percentile")
            }
        except (KeyError, IndexError):
            return None


# Default profiles for quick setup
DEFAULT_PROFILES = {
    "conservative": InvestorProfile(
        risk_level=RiskLevel.LOW,
        investment_years=5,
        portfolio_size=25
    ),
    "balanced": InvestorProfile(
        risk_level=RiskLevel.MEDIUM,
        investment_years=5,
        portfolio_size=25
    ),
    "aggressive": InvestorProfile(
        risk_level=RiskLevel.HIGH,
        investment_years=5,
        portfolio_size=20
    )
}


def get_default_profile(profile_name: str = "balanced") -> InvestorProfile:
    """
    Get a pre-configured investor profile by name.
    
    Args:
        profile_name: One of 'conservative', 'balanced', 'aggressive'
        
    Returns:
        InvestorProfile: Pre-configured profile
        
    Raises:
        KeyError: If profile_name not found
    """
    if profile_name not in DEFAULT_PROFILES:
        raise KeyError(
            f"Unknown profile '{profile_name}'. "
            f"Available: {list(DEFAULT_PROFILES.keys())}"
        )
    # Return a copy to avoid modifying defaults
    profile = DEFAULT_PROFILES[profile_name]
    return InvestorProfile(
        risk_level=profile.risk_level,
        investment_years=profile.investment_years,
        portfolio_size=profile.portfolio_size
    )
