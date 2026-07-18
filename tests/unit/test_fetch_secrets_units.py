"""Unit tests for docker-style env loading in fetch_secrets."""

import os
import sys
import unittest
from unittest.mock import patch


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import fetch_secrets


class TestFetchSecretsDockerEnv(unittest.TestCase):
    @patch("fetch_secrets.load_dotenv")
    @patch.dict(
        os.environ,
        {
            "DB_HOST": "db",
            "DB_USER": "stock_user",
            "DB_PASSWORD": "stock_pass",
            "DB_NAME": "stock_portefolio_builder",
        },
        clear=True,
    )
    def test_secret_import_loads_dev_env_and_reads_docker_database_settings(self, mock_load_dotenv):
        result = fetch_secrets.secret_import()

        self.assertEqual(
            result,
            ("db", "stock_user", "stock_pass", "stock_portefolio_builder"),
        )
        mock_load_dotenv.assert_called_once_with("dev.env")

    @patch.dict(os.environ, {"DB_PASSWORD": "preferred", "DB_PASS": "fallback"}, clear=True)
    def test_get_db_password_prefers_db_password(self):
        get_db_password = getattr(fetch_secrets, "_get_db_password")
        self.assertEqual(get_db_password(), "preferred")

    @patch.dict(os.environ, {"DB_PASS": "fallback"}, clear=True)
    def test_get_db_password_falls_back_to_db_pass(self):
        get_db_password = getattr(fetch_secrets, "_get_db_password")
        self.assertEqual(get_db_password(), "fallback")


if __name__ == "__main__":
    unittest.main()