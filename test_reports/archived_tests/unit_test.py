import unittest
import numpy as np
import pandas as pd
from ml_builder import create_sequences, prepare_lstm_datasets

class TestSequenceCreation(unittest.TestCase):
    
    def test_create_sequences_basic(self):
        """Test basic sequence creation"""
        data = np.array([[1,2], [3,4], [5,6], [7,8], [9,10]])
        sequences = create_sequences(data, time_steps=3)
        
        self.assertEqual(sequences.shape, (3, 3, 2))
        np.testing.assert_array_equal(sequences[0], [[1,2], [3,4], [5,6]])
    
    def test_create_sequences_insufficient_data(self):
        """Test error handling for insufficient data"""
        data = np.array([[1,2], [3,4]])
        
        with self.assertRaises(ValueError):
            create_sequences(data, time_steps=5)
    
    def test_prepare_lstm_datasets_shapes(self):
        """Test that prepared datasets have correct shapes"""
        np.random.seed(42)
        x_train = np.random.rand(100, 5)
        y_train = np.random.rand(100)
        x_val = np.random.rand(30, 5)
        y_val = np.random.rand(30)
        x_test = np.random.rand(20, 5)
        y_test = np.random.rand(20)
        
        time_steps = 10
        datasets = prepare_lstm_datasets(
            x_train, y_train, x_val, y_val, x_test, y_test, time_steps
        )
        
        # Check shapes
        self.assertEqual(datasets['train']['x'].shape, (91, 10, 5))
        self.assertEqual(datasets['train']['y'].shape, (91, 1))
        self.assertEqual(datasets['val']['x'].shape, (21, 10, 5))
        self.assertEqual(datasets['test']['x'].shape, (11, 10, 5))
        
        # Check metadata
        self.assertEqual(datasets['metadata']['time_steps'], 10)
        self.assertEqual(datasets['metadata']['num_features'], 5)
    
    def test_prepare_lstm_datasets_small_dataset(self):
        """Test error handling for datasets smaller than time_steps"""
        x_train = np.random.rand(100, 5)
        y_train = np.random.rand(100)
        x_val = np.random.rand(5, 5)  # Too small
        y_val = np.random.rand(5)
        x_test = np.random.rand(20, 5)
        y_test = np.random.rand(20)
        
        with self.assertRaises(ValueError):
            prepare_lstm_datasets(
                x_train, y_train, x_val, y_val, x_test, y_test, time_steps=10
            )

if __name__ == '__main__':
    unittest.main()