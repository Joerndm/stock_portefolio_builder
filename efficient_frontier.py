"""
Efficient Frontier Portfolio Optimization Module

This module provides functionality for calculating and visualizing the efficient frontier
of a portfolio using both Monte Carlo simulation and analytical optimization. The efficient
frontier represents the set of optimal portfolios that offer the highest expected return
for a given level of risk.

Features:
---------
- Monte Carlo simulation (configurable iterations) to visualize the frontier
- Analytical optimization using scipy to find:
  - Minimum variance portfolio (LOW risk)
  - Maximum Sharpe ratio portfolio (MEDIUM risk)
  - Risk-constrained maximum return portfolio (HIGH risk)
- Integration with InvestorProfile for risk-aware optimization
- Portfolio holdings export with weights, returns, and risk metrics
- Visualization with highlighted optimal portfolios

Functions:
    efficient_frontier_sim: Legacy MC simulation function (preserved for backward compat)
    optimize_portfolio: Main optimization function using investor risk profile
    find_optimal_weights: Core scipy optimization for different objectives

Dependencies:
    - pandas: Data manipulation
    - numpy: Numerical computations
    - scipy: Optimization (minimize with SLSQP)
    - matplotlib: Plotting and visualization

Author: Joern
"""
import os
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


# =============================================================================
# CORE OPTIMIZATION FUNCTIONS
# =============================================================================

def _portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate=0.04):
    """
    Calculate annualized portfolio return, volatility, and Sharpe ratio.

    Args:
        weights: Array of portfolio weights
        mean_returns: Annualized mean returns per asset
        cov_matrix: Annualized covariance matrix
        risk_free_rate: Annual risk-free rate

    Returns:
        Tuple of (return, volatility, sharpe_ratio)
    """
    port_return = np.sum(weights * mean_returns)
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = (port_return - risk_free_rate) / port_volatility if port_volatility > 0 else 0
    return port_return, port_volatility, sharpe


def _neg_sharpe(weights, mean_returns, cov_matrix, risk_free_rate=0.04):
    """Negative Sharpe ratio for minimization."""
    _, _, sharpe = _portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)
    return -sharpe


def _portfolio_volatility(weights, mean_returns, cov_matrix):
    """Portfolio volatility for minimization."""
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))


def _neg_return(weights, mean_returns, cov_matrix):
    """Negative return for minimization (maximize return)."""
    return -np.sum(weights * mean_returns)


def find_optimal_weights(
    mean_returns,
    cov_matrix,
    num_assets,
    objective="max_sharpe",
    risk_free_rate=0.04,
    volatility_cap=None,
    min_weight=0.0,
    max_weight=0.25
):
    """
    Find optimal portfolio weights using scipy optimization.

    Args:
        mean_returns: Annualized mean returns for each asset
        cov_matrix: Annualized covariance matrix
        num_assets: Number of assets in portfolio
        objective: Optimization objective:
            - 'max_sharpe': Maximize Sharpe ratio (best for MEDIUM risk)
            - 'min_volatility': Minimize portfolio volatility (best for LOW risk)
            - 'max_return': Maximize return with volatility constraint (best for HIGH risk)
        risk_free_rate: Annual risk-free rate (default 4%)
        volatility_cap: Maximum allowed portfolio volatility (annualized)
        min_weight: Minimum weight per asset (default 0 = allow zero allocation)
        max_weight: Maximum weight per asset (default 0.25 = max 25% in one stock)

    Returns:
        Dict with keys: weights, return, volatility, sharpe_ratio
    """
    # Initial guess: equal weights
    init_weights = np.array([1.0 / num_assets] * num_assets)

    # Bounds for each weight
    bounds = tuple((min_weight, max_weight) for _ in range(num_assets))

    # Constraint: weights must sum to 1
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]

    # Add volatility constraint if specified
    if volatility_cap is not None:
        constraints.append({
            'type': 'ineq',
            'fun': lambda w: volatility_cap - _portfolio_volatility(w, mean_returns, cov_matrix)
        })

    if objective == "max_sharpe":
        result = minimize(
            _neg_sharpe,
            init_weights,
            args=(mean_returns, cov_matrix, risk_free_rate),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-10}
        )
    elif objective == "min_volatility":
        result = minimize(
            _portfolio_volatility,
            init_weights,
            args=(mean_returns, cov_matrix),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-10}
        )
    elif objective == "max_return":
        result = minimize(
            _neg_return,
            init_weights,
            args=(mean_returns, cov_matrix),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-10}
        )
    else:
        raise ValueError(f"Unknown objective: {objective}")

    if not result.success:
        print(f"[WARN] Optimization did not converge: {result.message}")

    opt_weights = result.x
    opt_return, opt_vol, opt_sharpe = _portfolio_performance(
        opt_weights, mean_returns, cov_matrix, risk_free_rate
    )

    return {
        'weights': opt_weights,
        'return': opt_return,
        'volatility': opt_vol,
        'sharpe_ratio': opt_sharpe
    }


# =============================================================================
# MAIN PORTFOLIO OPTIMIZATION FUNCTION
# =============================================================================

def optimize_portfolio(
    price_df,
    risk_level="medium",
    volatility_cap=0.25,
    risk_free_rate=0.04,
    max_weight_per_stock=0.20,
    min_weight_per_stock=0.0,
    mc_simulations=100000,
    plot=True
):
    """
    Optimize a portfolio using the efficient frontier methodology with investor risk profile.

    This function:
    1. Calculates log returns and covariance from historical/forecast prices
    2. Runs Monte Carlo simulation to visualize the frontier
    3. Uses scipy optimization to find the optimal portfolio for the given risk level
    4. Returns optimal weights, metrics, and the full frontier DataFrame

    Args:
        price_df: DataFrame with stock prices (columns = tickers, rows = dates/days)
        risk_level: One of 'low', 'medium', 'high'
        volatility_cap: Maximum allowed annualized portfolio volatility
        risk_free_rate: Annual risk-free rate
        max_weight_per_stock: Maximum allocation to a single stock (0-1)
        min_weight_per_stock: Minimum allocation to a stock (0-1)
        mc_simulations: Number of MC simulation portfolios to generate for visualization
        plot: Whether to generate and save frontier plots

    Returns:
        Dict with keys:
            - 'optimal_weights': Dict[ticker, weight]
            - 'expected_return': float
            - 'expected_volatility': float
            - 'sharpe_ratio': float
            - 'holdings_df': DataFrame with ticker, weight, rank columns
            - 'frontier_df': Full MC frontier DataFrame
            - 'risk_level': str
            - 'mean_returns': Series of annualized mean returns per asset
            - 'cov_matrix': DataFrame of annualized covariance matrix
    """
    tickers = list(price_df.columns)
    num_assets = len(tickers)

    if num_assets < 2:
        raise ValueError(f"Need at least 2 stocks for portfolio optimization, got {num_assets}")

    print(f"\n{'=' * 60}")
    print(f"EFFICIENT FRONTIER OPTIMIZATION")
    print(f"{'=' * 60}")
    print(f"Stocks in universe: {num_assets}")
    print(f"Risk level: {risk_level.upper()}")
    print(f"Volatility cap: {volatility_cap:.1%}")
    print(f"Max weight per stock: {max_weight_per_stock:.1%}")
    print(f"MC simulations: {mc_simulations:,}")
    print(f"{'=' * 60}")

    # Calculate log returns
    log_returns = np.log(1 + price_df.pct_change(1).dropna())

    # Annualize statistics
    mean_returns = log_returns.mean() * 252
    cov_matrix = log_returns.cov() * 252

    # Handle assets with zero/negative returns or zero variance
    valid_mask = (mean_returns.notna()) & (cov_matrix.values.diagonal() > 0)
    if not valid_mask.all():
        dropped = [t for t, v in zip(tickers, valid_mask) if not v]
        print(f"[WARN] Dropping {len(dropped)} stocks with invalid returns: {dropped}")
        tickers = [t for t, v in zip(tickers, valid_mask) if v]
        mean_returns = mean_returns[tickers]
        cov_matrix = cov_matrix.loc[tickers, tickers]
        num_assets = len(tickers)
        if num_assets < 2:
            raise ValueError("Not enough valid stocks after filtering")

    # ---- Step 1: Monte Carlo simulation for frontier visualization ----
    print(f"\n[MC] Running {mc_simulations:,} portfolio simulations...")
    mc_returns = []
    mc_volatilities = []
    mc_weights_list = []

    for i in range(mc_simulations):
        weights = np.random.random(num_assets)
        weights /= weights.sum()
        ret, vol, _ = _portfolio_performance(weights, mean_returns.values, cov_matrix.values, risk_free_rate)
        mc_returns.append(ret)
        mc_volatilities.append(vol)
        mc_weights_list.append(weights)

        if i % 25000 == 0 and i > 0:
            print(f"   Simulated {i:,} portfolios...")

    mc_sharpes = [
        (r - risk_free_rate) / v if v > 0 else 0
        for r, v in zip(mc_returns, mc_volatilities)
    ]

    # Build frontier DataFrame
    frontier_df = pd.DataFrame({
        'Return': mc_returns,
        'Volatility': mc_volatilities,
        'Sharpe': mc_sharpes
    })
    weight_df = pd.DataFrame(mc_weights_list, columns=tickers)
    frontier_df = pd.concat([frontier_df, weight_df], axis=1)

    # ---- Step 2: Analytical optimization ----
    print(f"\n[OPT] Finding optimal portfolios...")

    # Map risk level to optimization strategy
    if risk_level == "low":
        objective = "min_volatility"
    elif risk_level == "high":
        objective = "max_return"
    else:  # medium
        objective = "max_sharpe"

    # Find the optimal portfolio for this risk level
    optimal = find_optimal_weights(
        mean_returns.values,
        cov_matrix.values,
        num_assets,
        objective=objective,
        risk_free_rate=risk_free_rate,
        volatility_cap=volatility_cap,
        min_weight=min_weight_per_stock,
        max_weight=max_weight_per_stock
    )

    # Also find min-vol and max-sharpe for plotting reference
    min_vol = find_optimal_weights(
        mean_returns.values, cov_matrix.values, num_assets,
        objective="min_volatility",
        risk_free_rate=risk_free_rate,
        min_weight=min_weight_per_stock,
        max_weight=max_weight_per_stock
    )
    max_sharpe = find_optimal_weights(
        mean_returns.values, cov_matrix.values, num_assets,
        objective="max_sharpe",
        risk_free_rate=risk_free_rate,
        min_weight=min_weight_per_stock,
        max_weight=max_weight_per_stock
    )

    # ---- Step 3: Build the optimal holdings DataFrame ----
    holdings_data = []
    for i, ticker in enumerate(tickers):
        weight = optimal['weights'][i]
        if weight > 0.001:  # Only include stocks with meaningful weight (>0.1%)
            stock_return = float(mean_returns[ticker])
            stock_vol = float(np.sqrt(cov_matrix.loc[ticker, ticker]))
            stock_sharpe = (stock_return - risk_free_rate) / stock_vol if stock_vol > 0 else 0

            holdings_data.append({
                'ticker': ticker,
                'weight': round(float(weight), 6),
                'expected_return': round(stock_return, 6),
                'volatility': round(stock_vol, 6),
                'sharpe_ratio': round(stock_sharpe, 4)
            })

    holdings_df = pd.DataFrame(holdings_data)
    holdings_df = holdings_df.sort_values('weight', ascending=False).reset_index(drop=True)
    holdings_df['rank'] = range(1, len(holdings_df) + 1)

    # ---- Step 4: Print results ----
    print(f"\n{'=' * 60}")
    print(f"OPTIMAL PORTFOLIO ({risk_level.upper()} RISK)")
    print(f"{'=' * 60}")
    print(f"Expected Return:     {optimal['return']:>8.2%}")
    print(f"Expected Volatility: {optimal['volatility']:>8.2%}")
    print(f"Sharpe Ratio:        {optimal['sharpe_ratio']:>8.4f}")
    print(f"Number of holdings:  {len(holdings_df)}")
    print(f"{'=' * 60}")

    print(f"\nReference Portfolios:")
    print(f"  Min Volatility: Return={min_vol['return']:.2%}, Vol={min_vol['volatility']:.2%}, "
          f"Sharpe={min_vol['sharpe_ratio']:.4f}")
    print(f"  Max Sharpe:     Return={max_sharpe['return']:.2%}, Vol={max_sharpe['volatility']:.2%}, "
          f"Sharpe={max_sharpe['sharpe_ratio']:.4f}")

    print(f"\nTop 10 Holdings:")
    print(f"{'Rank':<6} {'Ticker':<12} {'Weight':<10} {'Return':<10} {'Volatility':<12} {'Sharpe':<8}")
    print(f"{'-' * 58}")
    for _, row in holdings_df.head(10).iterrows():
        print(f"{int(row['rank']):<6} {row['ticker']:<12} {row['weight']:<10.2%} "
              f"{row['expected_return']:<10.2%} {row['volatility']:<12.2%} "
              f"{row['sharpe_ratio']:<8.4f}")

    # ---- Step 5: Plot ----
    if plot:
        _plot_efficient_frontier(
            frontier_df, optimal, min_vol, max_sharpe,
            risk_level, tickers,
            mean_returns=mean_returns.values,
            cov_matrix=cov_matrix.values,
            risk_free_rate=risk_free_rate,
            min_weight=min_weight_per_stock,
            max_weight=max_weight_per_stock
        )

    return {
        'optimal_weights': dict(zip(tickers, optimal['weights'])),
        'expected_return': optimal['return'],
        'expected_volatility': optimal['volatility'],
        'sharpe_ratio': optimal['sharpe_ratio'],
        'holdings_df': holdings_df,
        'frontier_df': frontier_df,
        'risk_level': risk_level,
        'mean_returns': mean_returns,
        'cov_matrix': cov_matrix
    }


def _plot_efficient_frontier(
    frontier_df, optimal, min_vol, max_sharpe,
    risk_level, tickers, mean_returns=None, cov_matrix=None,
    risk_free_rate=0.04, min_weight=0.0, max_weight=0.25
):
    """
    Generate and save the efficient frontier plot showing the Markowitz bullet
    with efficient/inefficient frontier curves, GMVP, and portfolio markers.
    """
    fig, ax = plt.subplots(figsize=(14, 9))

    # --- Compute analytical efficient frontier ---
    num_assets = len(tickers)

    # Find GMVP (minimum variance portfolio)
    gmvp = find_optimal_weights(
        mean_returns, cov_matrix, num_assets,
        objective="min_volatility",
        risk_free_rate=risk_free_rate,
        min_weight=min_weight,
        max_weight=max_weight
    )
    gmvp_ret = gmvp['return']
    gmvp_vol = gmvp['volatility']

    # Find max return portfolio
    max_ret_port = find_optimal_weights(
        mean_returns, cov_matrix, num_assets,
        objective="max_return",
        risk_free_rate=risk_free_rate,
        min_weight=min_weight,
        max_weight=max_weight
    )
    max_ret = max_ret_port['return']

    # Trace the efficient frontier: for each target return, find min-variance
    n_points = 50
    target_returns = np.linspace(gmvp_ret, max_ret, n_points)
    frontier_vols = []
    frontier_rets = []

    for target_ret in target_returns:
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
            {'type': 'eq', 'fun': lambda w, tr=target_ret: np.sum(w * mean_returns) - tr}
        ]
        bounds = tuple((min_weight, max_weight) for _ in range(num_assets))
        init_w = np.array([1.0 / num_assets] * num_assets)

        result = minimize(
            _portfolio_volatility,
            init_w,
            args=(mean_returns, cov_matrix),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-10}
        )
        if result.success:
            vol = _portfolio_volatility(result.x, mean_returns, cov_matrix)
            ret = np.sum(result.x * mean_returns)
            frontier_vols.append(vol)
            frontier_rets.append(ret)

    # Trace the inefficient frontier (below GMVP)
    # For returns below GMVP, the min-variance solution is on the lower branch
    if gmvp_ret > mean_returns.min():
        target_returns_low = np.linspace(mean_returns.min() * 0.8, gmvp_ret, 30)
        inefficient_vols = []
        inefficient_rets = []

        for target_ret in target_returns_low:
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
                {'type': 'eq', 'fun': lambda w, tr=target_ret: np.sum(w * mean_returns) - tr}
            ]
            bounds = tuple((min_weight, max_weight) for _ in range(num_assets))
            init_w = np.array([1.0 / num_assets] * num_assets)

            result = minimize(
                _portfolio_volatility,
                init_w,
                args=(mean_returns, cov_matrix),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-10}
            )
            if result.success:
                vol = _portfolio_volatility(result.x, mean_returns, cov_matrix)
                ret = np.sum(result.x * mean_returns)
                inefficient_vols.append(vol)
                inefficient_rets.append(ret)
    else:
        inefficient_vols = []
        inefficient_rets = []

    # --- Classify MC portfolios as efficient vs inefficient ---
    # A portfolio is "efficient" if it's close to the frontier curve
    mc_vols = frontier_df['Volatility'].values
    mc_rets = frontier_df['Return'].values

    # Build interpolation of the efficient frontier
    if len(frontier_vols) > 1:
        from scipy.interpolate import interp1d
        # Sort by volatility for interpolation
        sorted_idx = np.argsort(frontier_vols)
        ef_vols_sorted = np.array(frontier_vols)[sorted_idx]
        ef_rets_sorted = np.array(frontier_rets)[sorted_idx]
        frontier_interp = interp1d(ef_vols_sorted, ef_rets_sorted,
                                   bounds_error=False, fill_value='extrapolate')

        # Classify: efficient if within 5% of frontier return at same vol
        frontier_at_mc_vol = frontier_interp(mc_vols)
        ret_range = max(mc_rets) - min(mc_rets) if max(mc_rets) > min(mc_rets) else 1
        proximity = np.abs(mc_rets - frontier_at_mc_vol) / ret_range
        is_efficient = (proximity < 0.03) & (mc_rets >= gmvp_ret)
    else:
        is_efficient = np.zeros(len(mc_vols), dtype=bool)

    # --- Plot ---
    # Inefficient portfolios (circles) - subsample for clarity
    ineff_mask = ~is_efficient
    ineff_idx = np.where(ineff_mask)[0]
    if len(ineff_idx) > 200:
        ineff_idx = np.random.choice(ineff_idx, 200, replace=False)
    ax.scatter(
        mc_vols[ineff_idx], mc_rets[ineff_idx],
        marker='o', s=40, c='black', alpha=0.4,
        label='Inefficient Portfolios', zorder=2
    )

    # Efficient portfolios (squares) - subsample for clarity
    eff_idx = np.where(is_efficient)[0]
    if len(eff_idx) > 30:
        eff_idx = np.random.choice(eff_idx, 30, replace=False)
    ax.scatter(
        mc_vols[eff_idx], mc_rets[eff_idx],
        marker='s', s=60, c='black', alpha=0.7,
        label='Efficient Portfolios', zorder=3
    )

    # Efficient frontier curve (solid)
    if len(frontier_vols) > 1:
        ax.plot(frontier_vols, frontier_rets, 'k-', linewidth=2,
                label='Efficient Frontier', zorder=4)

    # Inefficient frontier curve (dashed)
    if len(inefficient_vols) > 1:
        ax.plot(inefficient_vols, inefficient_rets, 'k--', linewidth=2,
                label='Inefficient Frontier', zorder=4)

    # GMVP marker
    ax.scatter(
        gmvp_vol, gmvp_ret,
        marker='s', s=200, c='black', edgecolors='black', linewidths=2,
        zorder=6
    )
    ax.annotate(
        'GMVP', xy=(gmvp_vol, gmvp_ret),
        xytext=(gmvp_vol - 0.02, gmvp_ret - 0.015),
        fontsize=11, fontweight='bold',
        arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
        zorder=7
    )

    # Mark the selected optimal portfolio
    ax.scatter(
        optimal['volatility'], optimal['return'],
        marker='*', s=400, c='blue', edgecolors='black', linewidths=1.5,
        label=f'Selected ({risk_level.upper()}) — '
              f'R={optimal["return"]:.1%}, V={optimal["volatility"]:.1%}, '
              f'S={optimal["sharpe_ratio"]:.3f}',
        zorder=5
    )

    ax.set_xlabel('Risk (Annualized Volatility)', fontsize=12)
    ax.set_ylabel('Expected Return (Annualized)', fontsize=12)
    ax.set_title(f'Efficient Frontier — {risk_level.upper()} Risk Portfolio '
                 f'({len(tickers)} stocks)', fontsize=14)
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Save
    my_path = os.path.abspath(__file__)
    path = os.path.dirname(my_path)
    graph_name = f"Efficient_frontier_{risk_level}.png"

    try:
        plt.savefig(
            os.path.join(path, "generated_graphs", graph_name),
            bbox_inches="tight", pad_inches=0.5, dpi=150,
            transparent=False, format="png"
        )
        print(f"\n[GRAPH] Saved: generated_graphs/{graph_name}")
    except FileNotFoundError:
        print("[WARN] Could not save frontier graph")
    finally:
        plt.clf()
        plt.close("all")


# =============================================================================
# LEGACY FUNCTION — preserved for backward compatibility
# =============================================================================

def efficient_frontier_sim(price_df):
    """
    Legacy function: calculates the efficient frontier using MC random sampling only.

    This function is preserved for backward compatibility. For new code, use
    optimize_portfolio() instead which adds analytical optimization.

    Args:
        price_df: A pandas DataFrame containing stock prices (columns = tickers).

    Returns:
        DataFrame with portfolio number, weights, returns, and volatilities.
    """
    print("price_df")
    print(price_df)

    log_returns_df = np.log(1 + price_df.pct_change(1).dropna())
    log_returns_mean = log_returns_df.mean() * 252
    log_returns_mean = log_returns_mean.to_frame().rename(columns={0: "Mean"}).transpose()
    log_returns_df = log_returns_df[log_returns_mean.columns]
    log_returns_cov = log_returns_df.cov() * 252

    portefolio_number = []
    portefolio_weight = []
    portfolio_returns = []
    portfolio_volatilities = []

    print("Starting simulation...")
    for sim in range(750000):
        portefolio_number.append(sim)
        weights = np.random.random(len(log_returns_mean.columns))
        weights /= np.sum(weights)
        portefolio_weight.append(weights)
        portfolio_returns.append(np.sum(weights * np.array(log_returns_mean)))
        portfolio_volatilities.append(np.sqrt(np.dot(weights.T, np.dot(log_returns_cov, weights))))
        if sim % 25000 == 0:
            print(f"Simulations: {sim}")

    portefolio_number_df = pd.DataFrame(portefolio_number, columns=["Portefolio number"])
    portefolio_weight_df = pd.DataFrame(portefolio_weight, columns=log_returns_mean.columns)
    portfolio_returns_df = pd.DataFrame({'Return': portfolio_returns, 'Volatility': portfolio_volatilities})
    portefolio_df = pd.concat([portefolio_number_df, portefolio_weight_df, portfolio_returns_df], axis=1)
    portefolio_df = portefolio_df.sort_values(by="Volatility", ascending=True)

    portefolio_df.plot(x='Volatility', y='Return', kind='scatter', figsize=(18, 10))
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.title('Efficient Frontier')

    graph_name = "Efficient_frontier.png"
    my_path = os.path.abspath(__file__)
    path = os.path.dirname(my_path)

    try:
        plt.savefig(os.path.join(path, "generated_graphs", graph_name),
                    bbox_inches="tight", pad_inches=0.5, transparent=False, format="png")
        plt.clf()
        plt.close("all")
    except FileNotFoundError:
        raise FileNotFoundError("The graph could not be saved.")

    if len(price_df.columns) > 2:
        portefolio_df["Volatility"] = portefolio_df["Volatility"].round(3)
        portefolio_df = portefolio_df.groupby("Volatility").max("Return")
        columns = list(portefolio_df.columns)
        columns.append("Volatility")
        portefolio_df = portefolio_df.reset_index()
        portefolio_df = portefolio_df[columns]

        loop = True
        while loop:
            x = 0
            drop_list = []
            for index, row in portefolio_df.iterrows():
                if index > 0:
                    if row["Return"] < portefolio_df.iloc[index - 1]["Return"]:
                        drop_list.append(index)
                        x += 1
            portefolio_df = portefolio_df.drop(drop_list).reset_index(drop=True)
            if x == 0:
                loop = False

        portefolio_df.plot(x="Volatility", y="Return", kind="scatter", figsize=(18, 10))
        plt.xlabel('Expected Volatility')
        plt.ylabel('Expected Return')
        plt.title('Efficient Frontier')

        graph_name = "Efficient_frontier_reduced.png"
        try:
            plt.savefig(os.path.join(path, "generated_graphs", graph_name),
                        bbox_inches="tight", pad_inches=0.5, transparent=False, format="png")
            plt.clf()
            plt.close("all")
        except FileNotFoundError:
            raise FileNotFoundError("The graph could not be saved.")

    return portefolio_df


# Example usage
if __name__ == "__main__":
    import yfinance as yf

    prices = yf.download(
        ["AMBU-B.CO", "ROCK-B.CO", "TRYG.CO", "DEMANT.CO", "GN.CO", "JYSK.CO", "RBREW.CO"]
    )["Open"]

    # New optimized approach
    result = optimize_portfolio(
        prices,
        risk_level="medium",
        volatility_cap=0.25,
        mc_simulations=50000
    )
    print(f"\nOptimal portfolio Sharpe: {result['sharpe_ratio']:.4f}")
    print(result['holdings_df'])
