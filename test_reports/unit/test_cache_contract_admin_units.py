"""Unit tests for cache contract refresh helpers and CLI dispatch."""

import importlib.util
import os
import sys
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import cache_contract_admin

TENSORFLOW_AVAILABLE = importlib.util.find_spec('tensorflow') is not None
if TENSORFLOW_AVAILABLE:
    import model_trainer
else:
    model_trainer = None


class TestCacheContractAdmin(unittest.TestCase):
    def setUp(self):
        self.prepared_data = types.SimpleNamespace(
            x_training_dataset_df=pd.DataFrame({'f1': [0.1, 0.2, 0.3], 'f2': [0.4, 0.5, 0.6]}),
            y_train_unscaled=pd.Series([0.01, 0.02, 0.03]),
            x_val_dataset_df=pd.DataFrame({'f1': [0.15, 0.25], 'f2': [0.45, 0.55]}),
            y_val_unscaled=pd.Series([0.015, 0.025]),
            selected_features_list=['f1', 'f2'],
        )

    @patch('cache_contract_admin.db_interactions.invalidate_hyperparameters')
    @patch('cache_contract_admin.db_interactions.save_hyperparameters', return_value=True)
    @patch('cache_contract_admin.db_interactions.get_hyperparameter_cache_rows')
    @patch('cache_contract_admin.prepare_modeling_data')
    @patch('cache_contract_admin.db_interactions.import_stock_dataset')
    def test_refresh_ticker_cache_contract_rebuilds_only_mismatched_rows(
        self,
        mock_import_dataset,
        mock_prepare_modeling_data,
        mock_get_cache_rows,
        mock_save_hyperparameters,
        mock_invalidate,
    ):
        mock_import_dataset.return_value = pd.DataFrame({'close_Price': [1.0]})
        mock_prepare_modeling_data.return_value = self.prepared_data
        current_hash = cache_contract_admin._feature_hash(self.prepared_data.selected_features_list)
        mock_get_cache_rows.return_value = [
            {
                'model_type': 'rf',
                'hyperparameters': {'n_estimators': 100},
                'feature_hash': current_hash,
                'num_features': 2,
                'num_trials': 10,
                'best_score': 0.1,
                'tuning_time_seconds': 12.0,
                'is_constrained': False,
            },
            {
                'model_type': 'ridge',
                'hyperparameters': {'alpha': 2.5, 'solver': 'auto'},
                'feature_hash': 'stale-hash',
                'num_features': 3,
                'num_trials': 3,
                'best_score': 0.2,
                'tuning_time_seconds': 2.0,
                'is_constrained': True,
            },
        ]

        ridge_model = Mock()
        ridge_model.predict.return_value = np.array([0.02, 0.03])

        with patch.dict(cache_contract_admin.CACHE_MODEL_BUILDERS, {'ridge': Mock(return_value=ridge_model)}), \
             patch.dict(cache_contract_admin.CACHE_MODEL_SERIALIZERS, {'ridge': Mock(return_value={'alpha': 2.5, 'solver': 'auto'})}):
            result = cache_contract_admin.refresh_ticker_cache_contract('AAPL')

        self.assertTrue(result['success'])
        self.assertEqual(result['unchanged_models'], ['rf'])
        self.assertEqual(result['refreshed_models'], ['ridge'])
        ridge_model.fit.assert_called_once_with(
            self.prepared_data.x_training_dataset_df,
            self.prepared_data.y_train_unscaled,
        )
        self.assertEqual(mock_save_hyperparameters.call_count, 1)
        self.assertEqual(mock_save_hyperparameters.call_args.kwargs['model_type'], 'ridge')
        self.assertEqual(mock_save_hyperparameters.call_args.kwargs['feature_list'], ['f1', 'f2'])
        self.assertEqual(mock_save_hyperparameters.call_args.kwargs['num_features'], 2)
        mock_invalidate.assert_not_called()

    @unittest.skipUnless(TENSORFLOW_AVAILABLE, 'TensorFlow is not installed in this environment')
    @patch('price_predictor.predict_single_stock', return_value={'success': False, 'error_message': 'cache validation failed'})
    @patch('cache_contract_admin.db_interactions.get_hyperparameter_cache_rows')
    @patch('cache_contract_admin.prepare_modeling_data')
    @patch('cache_contract_admin.db_interactions.import_stock_dataset')
    def test_refresh_ticker_cache_contract_fails_when_prediction_validation_fails(
        self,
        mock_import_dataset,
        mock_prepare_modeling_data,
        mock_get_cache_rows,
        _mock_predict_single_stock,
    ):
        mock_import_dataset.return_value = pd.DataFrame({'close_Price': [1.0]})
        mock_prepare_modeling_data.return_value = self.prepared_data
        current_hash = cache_contract_admin._feature_hash(self.prepared_data.selected_features_list)
        mock_get_cache_rows.return_value = [
            {
                'model_type': 'rf',
                'hyperparameters': {'n_estimators': 100},
                'feature_hash': current_hash,
                'num_features': 2,
            },
        ]

        result = cache_contract_admin.refresh_ticker_cache_contract(
            'AAPL',
            validate_prediction=True,
        )

        self.assertFalse(result['success'])
        self.assertEqual(result['error_message'], 'cache validation failed')


@unittest.skipUnless(TENSORFLOW_AVAILABLE, 'TensorFlow is not installed in this environment')
class TestModelTrainerCLI(unittest.TestCase):
    @patch('model_trainer.refresh_cache_contracts')
    def test_main_dispatches_refresh_cache_contract(self, mock_refresh_cache_contracts):
        mock_refresh_cache_contracts.return_value = {'successful_tickers': ['AAPL'], 'failed_tickers': []}

        summary = model_trainer.main([
            '--refresh-cache-contract', 'AAPL', 'MSFT',
            '--refresh-model-types', 'ridge', 'svr',
            '--time-steps', '45',
            '--validate-prediction',
        ])

        self.assertEqual(summary, mock_refresh_cache_contracts.return_value)
        mock_refresh_cache_contracts.assert_called_once_with(
            tickers=['AAPL', 'MSFT'],
            model_types=['ridge', 'svr'],
            time_steps=45,
            validate_prediction=True,
        )


if __name__ == '__main__':
    unittest.main(verbosity=2)
