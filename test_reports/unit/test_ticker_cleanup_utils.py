"""Unit tests for ticker cleanup helpers."""

import os
import sys
import unittest


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from ticker_cleanup_utils import canonicalize_ticker, is_legacy_ticker


class TestTickerCleanupUtils(unittest.TestCase):
    def test_canonicalize_exchange_prefixed_ticker(self):
        self.assertEqual(canonicalize_ticker("EURONEXT-BRUSSELS: SOF.BR"), "SOF.BR")

    def test_canonicalize_us_share_class_ticker(self):
        self.assertEqual(canonicalize_ticker("BF.B"), "BF-B")
        self.assertEqual(canonicalize_ticker("BRK.B"), "BRK-B")

    def test_canonicalize_exchange_suffix_share_class_ticker(self):
        self.assertEqual(canonicalize_ticker("BT.A.L"), "BT-A.L")

    def test_regular_ticker_is_unchanged(self):
        self.assertEqual(canonicalize_ticker("AAPL"), "AAPL")
        self.assertFalse(is_legacy_ticker("AAPL"))
        self.assertTrue(is_legacy_ticker("BF.B"))


if __name__ == "__main__":
    unittest.main()