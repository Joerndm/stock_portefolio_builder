"""Unit tests for technical_patterns.py."""

import os
import sys
import unittest

import pandas as pd


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import technical_patterns


class TestDetectBollingerSqueeze(unittest.TestCase):
    def test_requires_full_threshold_window_history(self):
        df = pd.DataFrame(
            {
                "close_Price": [100.0] * 7,
                "bollinger_Band_3_2STD": [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0],
            }
        )

        result = technical_patterns.detect_bollinger_squeeze(
            df,
            period=3,
            percentile=0.20,
            threshold_window=5,
        )

        self.assertEqual(result.iloc[:4].tolist(), [0, 0, 0, 0])

    def test_uses_recent_window_for_threshold(self):
        df = pd.DataFrame(
            {
                "close_Price": [100.0] * 10,
                "bollinger_Band_3_2STD": [1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 10.0, 6.0],
            }
        )

        result = technical_patterns.detect_bollinger_squeeze(
            df,
            period=3,
            percentile=0.20,
            threshold_window=5,
        )

        self.assertEqual(result.iloc[8], 0)
        self.assertEqual(result.iloc[9], 1)


if __name__ == "__main__":
    unittest.main()