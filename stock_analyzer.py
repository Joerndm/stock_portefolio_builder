"""
Module to analyze stock data and predict future stock prices.

This module provides functionality for:
- Fetching and preprocessing stock data from database
- Splitting datasets into training, validation, and test sets
- Performing feature selection using Random Forest importance
- Training multiple ML models (LSTM, Random Forest, XGBoost) with automatic retraining
- Detecting and preventing overfitting through multi-metric validation
- Predicting future stock price changes using ensemble methods
- Running Monte Carlo simulations for risk analysis
- Optimizing portfolio allocation using efficient frontier analysis
- Generating forecasts and visualization graphs
- Exporting results to database for analysis

The main execution flow:
1. Configure investor profile (risk level, investment period, portfolio size)
2. Import stock symbols from database
3. For each stock (with error handling):
   a. Fetch and clean historical stock data
   b. Split data and perform feature engineering
   c. Train and validate multiple ML models with overfitting detection
   d. Generate future price predictions using ensemble approach
   e. Perform Monte Carlo simulations for uncertainty quantification
   f. Export results to database
4. Optimize portfolio using efficient frontier methodology
5. Run portfolio-level Monte Carlo simulation
6. Generate visualization graphs

GPU Configuration:
- Automatically detects and configures TensorFlow GPU devices
- Limits GPU memory to 7GB to prevent out-of-memory errors
- Enables memory growth for efficient GPU utilization

Dependencies:
- fetch_secrets: Secret management for API keys and credentials
- db_connectors: Database connection utilities
- db_interactions: Database CRUD operations
- stock_data_fetch: Stock data retrieval functionality
- split_dataset: Data splitting and scaling utilities
- dimension_reduction: Feature selection and dimensionality reduction
- ml_builder: Machine learning model construction and training
- monte_carlo_sim: Monte Carlo simulation for risk analysis
- efficient_frontier: Portfolio optimization using Modern Portfolio Theory
- portfolio_config: Investor profile configuration and result containers
"""
import os
import time
import datetime
import traceback
from typing import Dict, List, Optional

# Suppress ptxas warnings when CUDA toolkit is not fully installed
# TF falls back to driver-based PTX compilation which works fine
os.environ['TF_PTXAS_UNAVAILABLE'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Suppress INFO-level TF messages

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

import db_interactions
import split_dataset
import dimension_reduction
import ml_builder
import monte_carlo_sim
import efficient_frontier
from portfolio_config import (
    InvestorProfile, 
    RiskLevel, 
    StockPredictionResult,
    get_default_profile,
    DEFAULT_RISK_FREE_RATE
)


def configure_gpu():
    """Configure TensorFlow GPU settings for optimal performance."""
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("[GPU] No GPU detected, using CPU.")
        return False
    try:
        for gpu in gpus:
            # Note: set_memory_growth and set_virtual_device_configuration are
            # mutually exclusive. Only use virtual device config to cap memory.
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7168)]  # Limit to 7GB
            )
        print(f"[GPU] Configured {len(gpus)} GPU(s) with 7GB memory limit.")
        return True
    except RuntimeError as e:
        # GPU config must happen before GPUs are initialized
        print(f"[GPU] Configuration failed ({e}), falling back to CPU.")
        try:
            tf.config.set_visible_devices([], 'GPU')
        except Exception:
            pass
        return False


def get_stock_symbols(excluded_tickers: List[str] = None) -> pd.DataFrame:
    """
    Import stock symbols from database and apply exclusions.
    
    Args:
        excluded_tickers: List of ticker symbols to exclude
        
    Returns:
        DataFrame with Symbol column
    """
    stock_symbols_list = db_interactions.import_ticker_list()
    stock_symbols_df = pd.DataFrame(stock_symbols_list, columns=["Symbol"])
    
    # Apply exclusions
    if excluded_tickers:
        stock_symbols_df = stock_symbols_df[
            ~stock_symbols_df["Symbol"].isin(excluded_tickers)
        ]
    
    return stock_symbols_df.reset_index(drop=True)


def process_single_stock(
    stock_symbol: str,
    investor_profile: InvestorProfile,
    time_steps: int = 30
) -> StockPredictionResult:
    """
    Process a single stock through the complete ML pipeline.
    
    This function handles all ML training, prediction, and Monte Carlo simulation
    for a single stock, with proper error handling.
    
    Args:
        stock_symbol: The stock ticker symbol
        investor_profile: Investor configuration for risk/return parameters
        time_steps: Number of time steps for sequence models
        
    Returns:
        StockPredictionResult containing success status, forecasts, and MC results
    """
    start_time = time.time()
    
    try:
        # Import stock data
        stock_data_df = db_interactions.import_stock_dataset(stock_symbol)
        stock_data_df["date"] = pd.to_datetime(stock_data_df["date"])
        stock_data_df = stock_data_df.dropna(axis=0, how="any")
        stock_data_df = stock_data_df.dropna(axis=1, how="any")
        
        if len(stock_data_df) < 100:
            raise ValueError(f"Insufficient data: only {len(stock_data_df)} rows available")

        # Split the dataset
        validation_size = 0.20
        test_size = 0.10
        scaler_x, scaler_y, x_train_scaled, x_val_scaled, x_test_scaled, \
            y_train_scaled, y_val_scaled, y_test_scaled, x_predictions = \
            split_dataset.dataset_train_test_split(
                stock_data_df, test_size, validation_size=validation_size
            )

        # Inverse-transform y values for Random Forest
        y_train_unscaled = scaler_y.inverse_transform(
            y_train_scaled.reshape(-1, 1)
        ).flatten()
        y_val_unscaled = scaler_y.inverse_transform(
            y_val_scaled.reshape(-1, 1)
        ).flatten()
        y_test_unscaled = scaler_y.inverse_transform(
            y_test_scaled.reshape(-1, 1)
        ).flatten()

        # Convert to DataFrames for feature selection
        x_training_data = pd.DataFrame(x_train_scaled)
        x_val_data = pd.DataFrame(x_val_scaled)
        x_test_data = pd.DataFrame(x_test_scaled)
        y_training_data_df = pd.Series(y_train_unscaled)
        y_val_data_df = pd.Series(y_val_unscaled)
        y_test_data_df = pd.Series(y_test_unscaled)
        prediction_data = x_predictions

        max_features = len(x_training_data.columns)
        feature_amount = max_features

        # Feature selection
        x_training_dataset, x_val_dataset, x_test_dataset, x_prediction_dataset, \
            selected_features_model, selected_features_list = \
            dimension_reduction.feature_selection_rf(
                feature_amount,
                x_training_data,
                x_val_data,
                x_test_data,
                y_training_data_df,
                y_val_data_df,
                y_test_data_df,
                prediction_data,
                stock_data_df
            )

        # Prepare DataFrames for ML
        x_training_dataset_df = pd.DataFrame(
            x_training_dataset, columns=selected_features_list
        )
        y_training_data_df = y_training_data_df.reset_index(drop=True)
        x_val_dataset_df = pd.DataFrame(x_val_dataset, columns=selected_features_list)
        y_val_data_df = y_val_data_df.reset_index(drop=True)
        x_test_dataset_df = pd.DataFrame(x_test_dataset, columns=selected_features_list)
        y_test_data_df = y_test_data_df.reset_index(drop=True)
        x_prediction_dataset_df = pd.DataFrame(
            x_prediction_dataset, columns=selected_features_list
        )

        y_train_scaled_for_lstm = pd.Series(y_train_scaled)
        y_test_scaled_for_lstm = pd.Series(y_test_scaled)
        y_val_scaled_for_lstm = pd.Series(y_val_scaled)

        # Train ML models
        models, training_history, lstm_datasets = ml_builder.train_and_validate_models(
            stock_symbol=stock_symbol,
            x_train=x_training_dataset_df.values,
            x_val=x_val_dataset_df.values,
            x_test=x_test_dataset_df.values,
            y_train_scaled=y_train_scaled_for_lstm.values,
            y_val_scaled=y_val_scaled_for_lstm.values,
            y_test_scaled=y_test_scaled_for_lstm.values,
            y_train_unscaled=y_train_unscaled,
            y_val_unscaled=y_val_unscaled,
            y_test_unscaled=y_test_unscaled,
            time_steps=time_steps,
            scaler_y=scaler_y,
            max_retrains=150,
            overfitting_threshold=0.15,
            lstm_trials=50,
            lstm_executions=10,
            lstm_epochs=500,
            lstm_retrain_trials_increment=10,
            lstm_retrain_executions_increment=2,
            rf_trials=100,
            rf_retrain_increment=25,
            xgb_trials=60,
            xgb_retrain_increment=10,
            use_multi_metric_detection=True,
            use_tcn=True,
            tcn_trials=30,
            tcn_epochs=100,
            tcn_retrain_increment=10
        )

        # Generate predictions
        amount_of_days = time_steps * 3
        forecast_df = ml_builder.predict_future_price_changes(
            ticker=stock_symbol,
            scaler_x=scaler_x,
            scaler_y=scaler_y,
            model=models,
            selected_features_list=selected_features_list,
            stock_df=stock_data_df,
            prediction_days=amount_of_days,
            time_steps=time_steps,
            historical_prediction_dataset_df=x_prediction_dataset_df,
            use_mc_dropout=True,
            mc_iterations=30
        )

        # Analyze prediction performance
        historical_pred_count = len(x_prediction_dataset_df) \
            if x_prediction_dataset_df is not None else 0
        ml_builder.analyze_prediction_performance(stock_data_df, forecast_df, historical_pred_count)

        # Save prediction graph
        save_prediction_graph(stock_data_df, forecast_df)

        # Calculate predicted profit
        ml_builder.calculate_predicted_profit(forecast_df, amount_of_days)

        # Plot detailed graph
        ml_builder.plot_graph(stock_data_df, forecast_df)

        # Run Monte Carlo simulation for investor's investment period
        year_amount = investor_profile.investment_years
        sim_amount = 1000
        monte_carlo_day_df, monte_carlo_year_df = monte_carlo_sim.monte_carlo_analysis(
            0, stock_data_df, forecast_df, year_amount, sim_amount
        )

        # Get current price for database export
        current_price = float(stock_data_df['close_Price'].iloc[-1])
        
        # Export results to database
        prediction_date = datetime.date.today()
        
        try:
            db_interactions.export_stock_prediction_extended(
                ticker=stock_symbol,
                prediction_date=prediction_date,
                forecast_df=forecast_df,
                current_price=current_price,
                model_type="ensemble",
                mc_dropout_used=True,
                mc_iterations=30
            )
            print(f"[DB] Exported predictions for {stock_symbol}")
        except Exception as db_error:
            print(f"[WARNING] Could not export predictions to DB: {db_error}")
        
        try:
            db_interactions.export_monte_carlo_results(
                ticker=stock_symbol,
                simulation_date=prediction_date,
                monte_carlo_year_df=monte_carlo_year_df,
                num_simulations=sim_amount,
                starting_price=current_price
            )
            print(f"[DB] Exported Monte Carlo results for {stock_symbol}")
        except Exception as db_error:
            print(f"[WARNING] Could not export Monte Carlo to DB: {db_error}")

        execution_time = time.time() - start_time
        
        return StockPredictionResult(
            symbol=stock_symbol,
            success=True,
            forecast_df=forecast_df,
            monte_carlo_day_df=monte_carlo_day_df,
            monte_carlo_year_df=monte_carlo_year_df,
            execution_time=execution_time
        )
        
    except Exception as e:
        execution_time = time.time() - start_time
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"[ERROR] Failed processing {stock_symbol}: {error_msg}")
        print(traceback.format_exc())
        
        return StockPredictionResult(
            symbol=stock_symbol,
            success=False,
            error_message=error_msg,
            execution_time=execution_time
        )


def save_prediction_graph(stock_data_df: pd.DataFrame, forecast_df: pd.DataFrame):
    """Save the prediction graph to generated_graphs folder."""
    plt.figure(figsize=(12, 6))
    plt.plot(forecast_df["close_Price"], color="green")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend(["Predicted Stock Price"], loc="best")
    
    stock_name = stock_data_df.iloc[0]["ticker"]
    graph_name = f"future_stock_prediction_of_{stock_name}.png"
    my_path = os.path.abspath(__file__)
    path = os.path.dirname(my_path)
    
    try:
        plt.savefig(
            os.path.join(path, "generated_graphs", graph_name),
            bbox_inches="tight", pad_inches=0.5, transparent=False, format="png"
        )
    except FileNotFoundError:
        print(f"[WARNING] Could not save graph for {stock_name}")
    finally:
        plt.clf()
        plt.close("all")


def collect_portfolio_prices(
    results: Dict[str, StockPredictionResult],
    prediction_days: int = 90
) -> pd.DataFrame:
    """
    Collect forecasted prices from all successful predictions for portfolio optimization.
    
    Args:
        results: Dictionary of StockPredictionResult objects keyed by ticker
        prediction_days: Number of days of forecasts to include
        
    Returns:
        DataFrame with tickers as columns and dates as index
    """
    price_data = {}
    
    for ticker, result in results.items():
        if result.success and result.forecast_df is not None:
            try:
                forecast = result.forecast_df.copy()
                
                # Get the price column
                price_col = 'close_Price' if 'close_Price' in forecast.columns else None
                if price_col is None:
                    continue
                
                # Limit to prediction_days
                prices = forecast[price_col].iloc[:prediction_days].values
                
                if len(prices) > 0:
                    price_data[ticker] = prices
                    
            except Exception as e:
                print(f"[WARNING] Could not extract prices for {ticker}: {e}")
    
    if not price_data:
        return pd.DataFrame()
    
    # Create DataFrame with consistent length
    max_len = max(len(p) for p in price_data.values())
    
    # Pad shorter series with their last value
    for ticker in price_data:
        if len(price_data[ticker]) < max_len:
            last_val = price_data[ticker][-1]
            price_data[ticker] = list(price_data[ticker]) + [last_val] * (max_len - len(price_data[ticker]))
    
    return pd.DataFrame(price_data)


def rank_and_select_stocks(
    results: Dict[str, StockPredictionResult],
    portfolio_size: int = 25,
    risk_level: RiskLevel = RiskLevel.MEDIUM,
    prediction_days: int = 252
) -> List[str]:
    """
    Rank successful stock predictions and select the top N for portfolio construction.
    
    Stocks are ranked by a composite score that weighs predicted return and
    Monte Carlo confidence, adjusted for the investor's risk level.
    
    Ranking strategy by risk level:
        - LOW: Favors stocks with lower volatility and higher MC percentile floors
        - MEDIUM: Balances predicted return with Sharpe-like risk adjustment
        - HIGH: Favors highest predicted returns
    
    Args:
        results: Dictionary of StockPredictionResult keyed by ticker
        portfolio_size: Maximum number of stocks to select
        risk_level: Investor risk level for ranking strategy
        prediction_days: Number of forecast days to use for return calculation
        
    Returns:
        List of selected ticker symbols, ranked best to worst
    """
    rankings = []
    
    for ticker, result in results.items():
        if not result.success or result.forecast_df is None:
            continue
        
        predicted_return = result.get_predicted_return(days=prediction_days)
        if predicted_return is None:
            continue
        
        # Get MC downside risk (5th percentile return relative to start)
        mc_downside = None
        mc_upside = None
        if result.monte_carlo_year_df is not None:
            try:
                # Use year 1 MC for ranking
                start_price = result.monte_carlo_year_df.iloc[0]["Mean"]
                if start_price > 0:
                    yr1_row = result.monte_carlo_year_df.iloc[1] if len(result.monte_carlo_year_df) > 1 else None
                    if yr1_row is not None:
                        mc_downside = (yr1_row["5th Percentile"] - start_price) / start_price
                        mc_upside = (yr1_row["95th Percentile"] - start_price) / start_price
            except (KeyError, IndexError):
                pass
        
        # Calculate composite score based on risk level
        if risk_level == RiskLevel.LOW:
            # Penalize high downside risk heavily, reward positive MC floor
            downside_penalty = min(mc_downside or -0.5, 0)
            score = predicted_return + 2.0 * downside_penalty
        elif risk_level == RiskLevel.HIGH:
            # Pure return focus with a small upside bonus
            upside_bonus = max(mc_upside or 0, 0) * 0.2
            score = predicted_return + upside_bonus
        else:  # MEDIUM
            # Balance: predicted return minus penalty for large downside
            downside_penalty = min(mc_downside or -0.3, 0) * 0.5
            score = predicted_return + downside_penalty
        
        rankings.append({
            'ticker': ticker,
            'predicted_return': predicted_return,
            'mc_downside': mc_downside,
            'mc_upside': mc_upside,
            'score': score
        })
    
    if not rankings:
        return []
    
    # Sort by score descending
    rankings.sort(key=lambda x: x['score'], reverse=True)
    
    # Select top N
    selected = rankings[:portfolio_size]
    
    # Print ranking report
    print(f"\n{'=' * 70}")
    print(f"STOCK RANKING & SELECTION ({risk_level.value.upper()} risk)")
    print(f"{'=' * 70}")
    print(f"Eligible stocks: {len(rankings)}, Selecting top {min(portfolio_size, len(rankings))}")
    print(f"\n{'Rank':<6} {'Ticker':<14} {'Pred Return':<14} {'MC Down':<12} {'MC Up':<12} {'Score':<10} {'Selected'}")
    print(f"{'-' * 80}")
    
    for i, r in enumerate(rankings[:min(30, len(rankings))]):
        is_selected = "  *" if i < portfolio_size else ""
        mc_d = f"{r['mc_downside']:.2%}" if r['mc_downside'] is not None else "N/A"
        mc_u = f"{r['mc_upside']:.2%}" if r['mc_upside'] is not None else "N/A"
        print(f"{i+1:<6} {r['ticker']:<14} {r['predicted_return']:<14.2%} "
              f"{mc_d:<12} {mc_u:<12} {r['score']:<10.4f}{is_selected}")
    
    print(f"{'=' * 70}")
    
    return [r['ticker'] for r in selected]


def collect_historical_prices(
    selected_tickers: List[str],
    min_days: int = 504
) -> pd.DataFrame:
    """
    Collect historical closing prices from the database for selected tickers.
    
    Uses actual historical data (not forecasts) for portfolio optimization,
    as the efficient frontier should be built on historical return distributions.
    
    Args:
        selected_tickers: List of ticker symbols to fetch
        min_days: Minimum number of trading days required per stock
        
    Returns:
        DataFrame with tickers as columns and dates as index (closing prices)
    """
    price_data = {}
    dropped = []
    
    for ticker in selected_tickers:
        try:
            stock_df = db_interactions.import_stock_dataset(ticker)
            if stock_df is None or len(stock_df) < min_days:
                dropped.append(ticker)
                continue
            
            stock_df["date"] = pd.to_datetime(stock_df["date"])
            stock_df = stock_df.set_index("date").sort_index()
            
            # Use close price
            if 'close_Price' in stock_df.columns:
                prices = stock_df['close_Price'].dropna()
            elif stock_df.shape[1] >= 5:
                prices = stock_df.iloc[:, 4].dropna()
            else:
                dropped.append(ticker)
                continue
            
            if len(prices) >= min_days:
                price_data[ticker] = prices
            else:
                dropped.append(ticker)
                
        except Exception as e:
            print(f"[WARNING] Could not fetch historical data for {ticker}: {e}")
            dropped.append(ticker)
    
    if dropped:
        print(f"[INFO] Dropped {len(dropped)} tickers with insufficient historical data: {dropped}")
    
    if not price_data:
        return pd.DataFrame()
    
    # Align all price series to common dates
    price_df = pd.DataFrame(price_data)
    price_df = price_df.dropna(how='any')
    
    print(f"[INFO] Historical price matrix: {len(price_df)} trading days x {len(price_df.columns)} stocks")
    
    return price_df


def print_summary(
    results: Dict[str, StockPredictionResult],
    total_time: float
):
    """Print a summary of the analysis run."""
    successful = sum(1 for r in results.values() if r.success)
    failed = sum(1 for r in results.values() if not r.success)
    
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"Total stocks processed: {len(results)}")
    print(f"Successful predictions: {successful}")
    print(f"Failed predictions: {failed}")
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Average time per stock: {total_time / len(results):.2f} seconds")
    
    if failed > 0:
        print("\nFailed stocks:")
        for ticker, result in results.items():
            if not result.success:
                print(f"  - {ticker}: {result.error_message}")
    
    print("=" * 70)


def run_portfolio_analysis(
    investor_profile: InvestorProfile = None,
    custom_excluded_tickers: List[str] = None
):
    """
    Main function to run the complete portfolio analysis pipeline.
    
    Pipeline:
        1. Configure investor profile and GPU
        2. Import stock symbols and create DB run record
        3. Process each stock through ML + individual Monte Carlo
        4. Rank stocks by predicted return / risk score
        5. Select top N stocks for portfolio
        6. Fetch historical prices for selected stocks
        7. Optimize portfolio via efficient frontier (risk-aware)
        8. Run portfolio-level Monte Carlo simulation
        9. Export holdings and update DB with all metrics
        10. Print summary
    
    Args:
        investor_profile: Investor configuration (uses default balanced if None)
        custom_excluded_tickers: Additional tickers to exclude
        
    Returns:
        Tuple of (results dict, portfolio optimization result dict)
    """
    overall_start_time = time.time()
    
    # Use default profile if not provided
    if investor_profile is None:
        investor_profile = get_default_profile("balanced")
    
    print("\n" + "=" * 70)
    print("STOCK PORTFOLIO BUILDER - ANALYSIS RUN")
    print("=" * 70)
    print(f"Risk Level: {investor_profile.risk_level.value.upper()}")
    print(f"Investment Period: {investor_profile.investment_years} years")
    print(f"Target Portfolio Size: {investor_profile.portfolio_size} stocks")
    print(f"Volatility Cap: {investor_profile.get_volatility_cap():.0%}")
    print("=" * 70 + "\n")
    
    # Configure GPU
    has_gpu = configure_gpu()
    print(f"[GPU] {'GPU acceleration enabled' if has_gpu else 'Running on CPU'}")
    
    # Default excluded tickers (known problematic ones)
    default_excluded = []
    # default_excluded = ["DANSKE.CO", "JYSK.CO", "NDA-DK.CO", "TRYG.CO", "ORSTED.CO"]
    
    # Combine all exclusions
    all_excluded = list(set(
        default_excluded + 
        investor_profile.excluded_tickers + 
        (custom_excluded_tickers or [])
    ))
    
    # Get stock symbols
    stock_symbols_df = get_stock_symbols(excluded_tickers=all_excluded)
    total_stocks = len(stock_symbols_df)
    
    print(f"[INFO] Found {total_stocks} stocks to analyze")
    print(f"[INFO] Excluded {len(all_excluded)} tickers")
    
    # Create portfolio run record in database
    run_id = None
    try:
        run_id = db_interactions.create_portfolio_run(
            risk_level=investor_profile.risk_level.value,
            investment_years=investor_profile.investment_years,
            portfolio_size=investor_profile.portfolio_size,
            excluded_tickers=all_excluded
        )
        print(f"[DB] Created portfolio run ID: {run_id}")
    except Exception as e:
        print(f"[WARNING] Could not create portfolio run record: {e}")
    
    # =========================================================================
    # PHASE 1: Process all stocks (ML predictions + individual MC)
    # =========================================================================
    results: Dict[str, StockPredictionResult] = {}
    time_steps = 30
    
    for index, row in stock_symbols_df.iterrows():
        stock_symbol = row["Symbol"]
        print(f"\n[STOCK] Processing {stock_symbol} ({index+1}/{total_stocks})")
        
        result = process_single_stock(
            stock_symbol=stock_symbol,
            investor_profile=investor_profile,
            time_steps=time_steps
        )
        
        results[stock_symbol] = result
        
        if result.success:
            print(f"[SUCCESS] {stock_symbol} completed in {result.execution_time:.2f}s")
        else:
            print(f"[FAILED] {stock_symbol} - continuing with next stock")
    
    successful_count = sum(1 for r in results.values() if r.success)
    failed_count = sum(1 for r in results.values() if not r.success)
    
    # Update portfolio run with prediction counts
    if run_id:
        try:
            db_interactions.update_portfolio_run(
                run_id=run_id,
                total_stocks_analyzed=total_stocks,
                successful_predictions=successful_count,
                failed_predictions=failed_count
            )
        except Exception as e:
            print(f"[WARNING] Could not update portfolio run: {e}")
    
    # =========================================================================
    # PHASE 2: Stock ranking, selection, and portfolio optimization
    # =========================================================================
    ef_result = None
    pf_mc_result = None
    
    if successful_count >= 2:
        # Step 2a: Rank and select top stocks
        print("\n[PHASE 2] Ranking stocks and selecting portfolio candidates...")
        selected_tickers = rank_and_select_stocks(
            results=results,
            portfolio_size=investor_profile.portfolio_size,
            risk_level=investor_profile.risk_level
        )
        
        if len(selected_tickers) >= 2:
            # Step 2b: Fetch historical prices for selected stocks
            print(f"\n[PHASE 2] Fetching historical prices for {len(selected_tickers)} selected stocks...")
            historical_prices = collect_historical_prices(selected_tickers)
            
            if not historical_prices.empty and len(historical_prices.columns) >= 2:
                # Step 2c: Run efficient frontier optimization
                print(f"\n[PHASE 2] Running risk-aware efficient frontier optimization...")
                try:
                    risk_free_rate = DEFAULT_RISK_FREE_RATE
                    volatility_cap = investor_profile.get_volatility_cap()
                    
                    # Max weight scales inversely with portfolio size
                    max_weight = min(0.25, 1.0 / max(len(historical_prices.columns) * 0.5, 1))
                    max_weight = max(max_weight, 0.05)  # At least 5%
                    
                    ef_result = efficient_frontier.optimize_portfolio(
                        price_df=historical_prices,
                        risk_level=investor_profile.risk_level.value,
                        volatility_cap=volatility_cap,
                        risk_free_rate=risk_free_rate,
                        max_weight_per_stock=max_weight,
                        min_weight_per_stock=0.0,
                        mc_simulations=100000,
                        plot=True
                    )
                    
                    print("[PHASE 2] Efficient frontier optimization complete!")
                    
                    # =========================================================
                    # PHASE 3: Portfolio Monte Carlo + DB exports
                    # =========================================================
                    
                    # Step 3a: Run portfolio-level Monte Carlo
                    print(f"\n[PHASE 3] Running portfolio Monte Carlo simulation...")
                    try:
                        pf_mc_result = monte_carlo_sim.portfolio_monte_carlo(
                            weights=list(ef_result['optimal_weights'].values()),
                            mean_returns=ef_result['mean_returns'].values,
                            cov_matrix=ef_result['cov_matrix'].values,
                            initial_investment=100000,
                            years=investor_profile.investment_years,
                            num_simulations=5000,
                            seed=42
                        )
                        print("[PHASE 3] Portfolio Monte Carlo complete!")
                    except Exception as mc_error:
                        print(f"[ERROR] Portfolio Monte Carlo failed: {mc_error}")
                        traceback.print_exc()
                    
                    # Step 3b: Export portfolio holdings to database
                    if run_id and ef_result.get('holdings_df') is not None:
                        try:
                            holdings_exported = db_interactions.export_portfolio_holdings(
                                run_id=run_id,
                                holdings_df=ef_result['holdings_df']
                            )
                            print(f"[DB] Exported {holdings_exported} portfolio holdings")
                        except Exception as db_error:
                            print(f"[WARNING] Could not export holdings to DB: {db_error}")
                    
                    # Step 3c: Update portfolio run with optimization + MC metrics
                    if run_id:
                        try:
                            update_kwargs = {
                                'run_id': run_id,
                                'expected_return': ef_result['expected_return'],
                                'expected_volatility': ef_result['expected_volatility'],
                                'sharpe_ratio': ef_result['sharpe_ratio'],
                                'status': 'completed',
                            }
                            
                            if pf_mc_result is not None:
                                update_kwargs['mc_return_p5'] = pf_mc_result['final_return_p5']
                                update_kwargs['mc_return_mean'] = pf_mc_result['final_return_mean']
                                update_kwargs['mc_return_p95'] = pf_mc_result['final_return_p95']
                            
                            db_interactions.update_portfolio_run(**update_kwargs)
                            print("[DB] Updated portfolio run with optimization and MC metrics")
                        except Exception as db_error:
                            print(f"[WARNING] Could not update portfolio run metrics: {db_error}")
                    
                except Exception as ef_error:
                    print(f"[ERROR] Efficient frontier optimization failed: {ef_error}")
                    traceback.print_exc()
                    if run_id:
                        try:
                            db_interactions.update_portfolio_run(
                                run_id=run_id,
                                status='failed',
                                error_message=f"EF optimization failed: {str(ef_error)}"
                            )
                        except Exception:
                            pass
            else:
                print(f"[WARNING] Not enough historical price data for portfolio optimization")
        else:
            print(f"[WARNING] Not enough selected stocks ({len(selected_tickers)}) for portfolio optimization")
    else:
        print(f"[WARNING] Not enough successful predictions ({successful_count}) for portfolio optimization")
    
    # Finalize: update run execution time
    overall_time = time.time() - overall_start_time
    if run_id:
        try:
            final_status = 'completed' if ef_result is not None else 'failed'
            db_interactions.update_portfolio_run(
                run_id=run_id,
                status=final_status,
                execution_time_seconds=overall_time
            )
        except Exception as e:
            print(f"[WARNING] Could not finalize portfolio run: {e}")
    
    # Print summary
    print_summary(results, overall_time)
    
    # Print final portfolio summary if optimization succeeded
    if ef_result is not None:
        print(f"\n{'=' * 70}")
        print(f"FINAL OPTIMIZED PORTFOLIO ({investor_profile.risk_level.value.upper()} RISK)")
        print(f"{'=' * 70}")
        print(f"Expected Annual Return:  {ef_result['expected_return']:.2%}")
        print(f"Expected Volatility:     {ef_result['expected_volatility']:.2%}")
        print(f"Sharpe Ratio:            {ef_result['sharpe_ratio']:.4f}")
        print(f"Number of Holdings:      {len(ef_result['holdings_df'])}")
        
        if pf_mc_result is not None:
            print(f"\n{investor_profile.investment_years}-Year Monte Carlo Outlook:")
            print(f"  Mean Return:           {pf_mc_result['final_return_mean']:.2%}")
            print(f"  Downside (5th pct):    {pf_mc_result['final_return_p5']:.2%}")
            print(f"  Upside (95th pct):     {pf_mc_result['final_return_p95']:.2%}")
            print(f"  Mean Final Value:      {pf_mc_result['final_value_mean']:,.0f} (from 100,000)")
        
        print(f"{'=' * 70}")
    
    return results, ef_result


if __name__ == "__main__":
    # Configure investor profile
    # Options: get_default_profile("conservative"), get_default_profile("balanced"), get_default_profile("aggressive")
    # Or create custom:
    profile = InvestorProfile(
        risk_level=RiskLevel.MEDIUM,
        investment_years=7,
        portfolio_size=25
    )
    
    # Run the complete analysis pipeline
    # Returns: (results_dict, ef_result_dict)
    # ef_result contains: optimal_weights, expected_return, expected_volatility,
    #                      sharpe_ratio, holdings_df, frontier_df, etc.
    results, ef_result = run_portfolio_analysis(investor_profile=profile)

