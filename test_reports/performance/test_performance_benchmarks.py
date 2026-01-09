"""
Performance Tests for Stock Portfolio Builder

This module contains performance tests that verify system meets performance requirements
and benchmarks for various operations.

Test Categories:
- Training time benchmarks
- Prediction latency tests
- Memory usage profiling
- Database query performance
- Large dataset handling
"""

import unittest
import numpy as np
import pandas as pd
import time
import sys
import os
from memory_profiler import profile
import psutil
import gc

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import stock_data_fetch
import split_dataset
import dimension_reduction
import monte_carlo_sim
import efficient_frontier
import data_scalers


class TestDataProcessingPerformance(unittest.TestCase):
    """Performance tests for data processing operations"""
    
    def setUp(self):
        """Set up performance test data"""
        np.random.seed(42)
        
        # Create different sized datasets
        self.small_data = self._create_dataset(100)  # 100 days
        self.medium_data = self._create_dataset(500)  # 500 days (~2 years)
        self.large_data = self._create_dataset(2500)  # 2500 days (~10 years)
    
    def _create_dataset(self, n_samples):
        """Helper to create test dataset"""
        return pd.DataFrame({
            'date': pd.date_range('2010-01-01', periods=n_samples),
            'close_Price': np.cumsum(np.random.randn(n_samples) * 2) + 100,
            'high_Price': np.cumsum(np.random.randn(n_samples) * 2) + 105,
            'low_Price': np.cumsum(np.random.randn(n_samples) * 2) + 95,
            'open_Price': np.cumsum(np.random.randn(n_samples) * 2) + 98,
            'trade_Volume': np.random.randint(1000000, 10000000, n_samples),
            'ticker': ['AAPL'] * n_samples
        })
    
    def test_moving_average_calculation_performance(self):
        """Benchmark moving average calculation"""
        
        # Small dataset
        start = time.time()
        result_small = stock_data_fetch.calculate_moving_averages(self.small_data.copy())
        time_small = time.time() - start
        
        # Medium dataset
        start = time.time()
        result_medium = stock_data_fetch.calculate_moving_averages(self.medium_data.copy())
        time_medium = time.time() - start
        
        # Large dataset
        start = time.time()
        result_large = stock_data_fetch.calculate_moving_averages(self.large_data.copy())
        time_large = time.time() - start
        
        print(f"\n[PERF] Moving Average Calculation:")
        print(f"  Small (100): {time_small:.4f}s")
        print(f"  Medium (500): {time_medium:.4f}s")
        print(f"  Large (2500): {time_large:.4f}s")
        
        # Performance assertions
        self.assertLess(time_small, 1.0, "Small dataset should process in <1s")
        self.assertLess(time_medium, 5.0, "Medium dataset should process in <5s")
        self.assertLess(time_large, 15.0, "Large dataset should process in <15s")
    
    def test_technical_indicator_performance(self):
        """Benchmark technical indicator calculation"""
        
        # Prepare data with moving averages
        data = stock_data_fetch.calculate_moving_averages(self.large_data.copy())
        
        start = time.time()
        result = stock_data_fetch.add_technical_indicators(data)
        elapsed = time.time() - start
        
        print(f"\n[PERF] Technical Indicators (2500 samples): {elapsed:.4f}s")
        
        self.assertLess(elapsed, 10.0, "Tech indicators should compute in <10s")
    
    def test_full_feature_pipeline_performance(self):
        """Benchmark complete feature engineering pipeline"""
        
        start = time.time()
        
        # Complete pipeline
        result = stock_data_fetch.calculate_moving_averages(self.large_data.copy())
        result = stock_data_fetch.add_technical_indicators(result)
        result = stock_data_fetch.add_volume_indicators(result)
        result = stock_data_fetch.add_volatility_indicators(result)
        result = stock_data_fetch.calculate_period_returns(result)
        
        elapsed = time.time() - start
        
        print(f"\n[PERF] Full Feature Pipeline (2500 samples): {elapsed:.4f}s")
        
        self.assertLess(elapsed, 30.0, "Full pipeline should complete in <30s")


class TestDatasetSplittingPerformance(unittest.TestCase):
    """Performance tests for dataset splitting and scaling"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        
        # Create dataset with many features
        n_samples = 1000
        n_features = 50
        
        self.dataset_df = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        # Add required columns
        self.dataset_df['date'] = pd.date_range('2020-01-01', periods=n_samples)
        self.dataset_df['ticker'] = 'AAPL'
        self.dataset_df['currency'] = 'USD'
        self.dataset_df['open_Price'] = np.random.uniform(90, 110, n_samples)
        self.dataset_df['high_Price'] = np.random.uniform(95, 115, n_samples)
        self.dataset_df['low_Price'] = np.random.uniform(85, 105, n_samples)
        self.dataset_df['close_Price'] = np.random.uniform(90, 110, n_samples)
        self.dataset_df['trade_Volume'] = np.random.randint(1000000, 10000000, n_samples)
        self.dataset_df['1D'] = np.random.uniform(-0.05, 0.05, n_samples)
    
    def test_train_test_split_performance(self):
        """Benchmark dataset splitting performance"""
        
        start = time.time()
        
        scaler_x, scaler_y, x_train, x_val, x_test, y_train, y_val, y_test, x_pred = \
            split_dataset.dataset_train_test_split(self.dataset_df.copy())
        
        elapsed = time.time() - start
        
        print(f"\n[PERF] Train/Test Split (1000 samples, 50 features): {elapsed:.4f}s")
        
        self.assertLess(elapsed, 5.0, "Splitting should complete in <5s")
    
    def test_scaling_performance(self):
        """Benchmark data scaling performance"""
        
        X = self.dataset_df[[f'feature_{i}' for i in range(50)]].copy()
        
        # Test fit performance
        start = time.time()
        scaler = data_scalers.data_preprocessing_minmax_scaler_fit(X)
        fit_time = time.time() - start
        
        # Test transform performance
        start = time.time()
        X_scaled = data_scalers.data_preprocessing_minmax_scaler_transform(scaler, X)
        transform_time = time.time() - start
        
        print(f"\n[PERF] Scaling Performance (1000 x 50):")
        print(f"  Fit: {fit_time:.4f}s")
        print(f"  Transform: {transform_time:.4f}s")
        
        self.assertLess(fit_time, 2.0, "Scaler fit should complete in <2s")
        self.assertLess(transform_time, 1.0, "Transform should complete in <1s")


class TestFeatureSelectionPerformance(unittest.TestCase):
    """Performance tests for feature selection"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        
        # Create high-dimensional dataset
        self.n_samples = 500
        self.n_features = 100
        
        self.x_train = pd.DataFrame(
            np.random.randn(self.n_samples, self.n_features),
            columns=[f'feature_{i}' for i in range(self.n_features)]
        )
        self.x_val = pd.DataFrame(
            np.random.randn(100, self.n_features),
            columns=[f'feature_{i}' for i in range(self.n_features)]
        )
        self.x_test = pd.DataFrame(
            np.random.randn(50, self.n_features),
            columns=[f'feature_{i}' for i in range(self.n_features)]
        )
        self.x_pred = np.random.randn(10, self.n_features)
        
        self.y_train = pd.Series(np.random.randn(self.n_samples))
        self.y_val = pd.Series(np.random.randn(100))
        self.y_test = pd.Series(np.random.randn(50))
        
        # Create mock dataset
        self.dataset_df = pd.DataFrame(
            np.random.randn(self.n_samples, self.n_features + 9),
            columns=[f'feature_{i}' for i in range(self.n_features)] +
                    ['date', 'ticker', 'currency', 'open_Price', 'high_Price',
                     'low_Price', 'close_Price', 'trade_Volume', '1D']
        )
    
    def test_selectkbest_performance(self):
        """Benchmark SelectKBest feature selection"""
        
        dimensions = 20
        
        start = time.time()
        
        x_train_reduced, x_val_reduced, x_test_reduced, x_pred_reduced, selector, features = \
            dimension_reduction.feature_selection(
                dimensions, self.x_train, self.x_val, self.x_test,
                self.y_train, self.y_val, self.y_test,
                self.x_pred, self.dataset_df
            )
        
        elapsed = time.time() - start
        
        print(f"\n[PERF] SelectKBest (500 samples, 100→20 features): {elapsed:.4f}s")
        
        self.assertLess(elapsed, 5.0, "SelectKBest should complete in <5s")
    
    def test_random_forest_feature_selection_performance(self):
        """Benchmark Random Forest feature selection"""
        
        dimensions = 20
        
        start = time.time()
        
        x_train_reduced, x_val_reduced, x_test_reduced, x_pred_reduced, model, features = \
            dimension_reduction.feature_selection_rf(
                dimensions, self.x_train, self.x_val, self.x_test,
                self.y_train, self.y_val, self.y_test,
                self.x_pred
            )
        
        elapsed = time.time() - start
        
        print(f"\n[PERF] RF Feature Selection (500 samples, 100→20 features): {elapsed:.4f}s")
        
        self.assertLess(elapsed, 15.0, "RF selection should complete in <15s")


class TestMonteCarloPerformance(unittest.TestCase):
    """Performance tests for Monte Carlo simulations"""
    
    def setUp(self):
        """Set up test data"""
        self.stock_data = pd.DataFrame({
            'ticker': ['AAPL'] * 500,
            'date': pd.date_range('2021-01-01', periods=500),
            'close_Price': np.cumsum(np.random.randn(500) * 2) + 100
        })
        
        self.forecast_df = pd.DataFrame({
            'close_Price': np.cumsum(np.random.randn(500) * 2) + 100
        })
    
    def test_monte_carlo_simulation_performance(self):
        """Benchmark Monte Carlo simulation with different iteration counts"""
        
        sim_counts = [100, 500, 1000]
        
        print(f"\n[PERF] Monte Carlo Simulation Performance:")
        
        for sim_count in sim_counts:
            start = time.time()
            
            price_df, mc_df = monte_carlo_sim.monte_carlo_analysis(
                seed_number=42,
                stock_data_df=self.stock_data,
                forecast_df=self.forecast_df,
                years=1,
                sim_amount=sim_count
            )
            
            elapsed = time.time() - start
            
            print(f"  {sim_count} simulations: {elapsed:.4f}s ({elapsed/sim_count*1000:.2f}ms per sim)")
            
            # Performance assertions
            if sim_count == 100:
                self.assertLess(elapsed, 5.0, "100 sims should complete in <5s")
            elif sim_count == 500:
                self.assertLess(elapsed, 20.0, "500 sims should complete in <20s")


class TestEfficientFrontierPerformance(unittest.TestCase):
    """Performance tests for efficient frontier calculation"""
    
    def setUp(self):
        """Set up portfolio data"""
        dates = pd.date_range('2021-01-01', periods=500)
        
        # Test with different portfolio sizes
        self.small_portfolio = pd.DataFrame({
            'AAPL': np.cumsum(np.random.randn(500) * 2) + 100,
            'GOOGL': np.cumsum(np.random.randn(500) * 2) + 100,
            'MSFT': np.cumsum(np.random.randn(500) * 2) + 100
        }, index=dates)
        
        self.large_portfolio = pd.DataFrame({
            f'Stock{i}': np.cumsum(np.random.randn(500) * 2) + 100
            for i in range(10)
        }, index=dates)
    
    def test_efficient_frontier_performance(self):
        """Benchmark efficient frontier calculation"""
        
        from unittest.mock import patch
        
        print(f"\n[PERF] Efficient Frontier Performance:")
        
        # Small portfolio (3 stocks)
        with patch('matplotlib.pyplot.savefig'), patch('matplotlib.pyplot.show'):
            start = time.time()
            result_small = efficient_frontier.efficient_frontier_sim(self.small_portfolio)
            time_small = time.time() - start
            
            print(f"  3 stocks: {time_small:.4f}s")
            self.assertLess(time_small, 120.0, "3-stock EF should complete in <120s")
        
        # Large portfolio (10 stocks)
        with patch('matplotlib.pyplot.savefig'), patch('matplotlib.pyplot.show'):
            start = time.time()
            result_large = efficient_frontier.efficient_frontier_sim(self.large_portfolio)
            time_large = time.time() - start
            
            print(f"  10 stocks: {time_large:.4f}s")
            self.assertLess(time_large, 180.0, "10-stock EF should complete in <180s")


class TestMemoryUsage(unittest.TestCase):
    """Memory usage tests"""
    
    def test_large_dataset_memory_usage(self):
        """Test memory usage with large datasets"""
        
        process = psutil.Process(os.getpid())
        
        # Get initial memory
        gc.collect()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create large dataset
        n_samples = 10000
        large_data = pd.DataFrame({
            'date': pd.date_range('2000-01-01', periods=n_samples),
            'close_Price': np.random.uniform(90, 110, n_samples),
            'high_Price': np.random.uniform(95, 115, n_samples),
            'low_Price': np.random.uniform(85, 105, n_samples),
            'trade_Volume': np.random.randint(1000000, 10000000, n_samples),
            'ticker': ['AAPL'] * n_samples
        })
        
        # Process data
        result = stock_data_fetch.calculate_moving_averages(large_data)
        result = stock_data_fetch.add_technical_indicators(result)
        
        # Get final memory
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_used = mem_after - mem_before
        
        print(f"\n[PERF] Memory Usage for 10,000 samples:")
        print(f"  Before: {mem_before:.2f} MB")
        print(f"  After: {mem_after:.2f} MB")
        print(f"  Used: {mem_used:.2f} MB")
        
        # Should use reasonable amount of memory
        self.assertLess(mem_used, 500, "Should use <500MB for 10k samples")
        
        # Cleanup
        del large_data, result
        gc.collect()


def run_performance_tests():
    """Run all performance tests and return results"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDataProcessingPerformance))
    suite.addTests(loader.loadTestsFromTestCase(TestDatasetSplittingPerformance))
    suite.addTests(loader.loadTestsFromTestCase(TestFeatureSelectionPerformance))
    suite.addTests(loader.loadTestsFromTestCase(TestMonteCarloPerformance))
    suite.addTests(loader.loadTestsFromTestCase(TestEfficientFrontierPerformance))
    suite.addTests(loader.loadTestsFromTestCase(TestMemoryUsage))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    result = run_performance_tests()
    
    # Print summary
    print("\n" + "="*70)
    print("PERFORMANCE TEST SUMMARY")
    print("="*70)
    print(f"Tests Run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print("="*70)
