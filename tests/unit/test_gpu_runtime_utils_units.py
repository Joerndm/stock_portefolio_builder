import unittest
from unittest.mock import Mock, patch

import gpu_runtime_utils


class TestConfigureTensorflowGpu(unittest.TestCase):
    def _build_fake_tf(self, gpus=None):
        fake_tf = Mock()
        fake_tf.config.list_physical_devices.return_value = gpus or []
        fake_tf.config.experimental.VirtualDeviceConfiguration.side_effect = (
            lambda memory_limit: {'memory_limit': memory_limit}
        )
        return fake_tf

    def test_returns_false_when_no_gpu_detected(self):
        fake_tf = self._build_fake_tf([])
        logger = Mock()

        with patch.object(gpu_runtime_utils, '_load_tensorflow', return_value=fake_tf):
            configured = gpu_runtime_utils.configure_tensorflow_gpu(7168, logger=logger)

        self.assertFalse(configured)
        logger.info.assert_called_once_with("[GPU] No GPU detected, using CPU.")
        fake_tf.config.experimental.set_virtual_device_configuration.assert_not_called()

    def test_configures_all_detected_gpus(self):
        fake_tf = self._build_fake_tf(['GPU:0', 'GPU:1'])
        logger = Mock()

        with patch.object(gpu_runtime_utils, '_load_tensorflow', return_value=fake_tf):
            configured = gpu_runtime_utils.configure_tensorflow_gpu(7168, logger=logger)

        self.assertTrue(configured)
        self.assertEqual(fake_tf.config.experimental.set_virtual_device_configuration.call_count, 2)
        logger.info.assert_called_once_with(
            "[GPU] Configured %d GPU(s) with %dMB memory limit.",
            2,
            7168,
        )

    def test_runtime_error_falls_back_to_cpu(self):
        fake_tf = self._build_fake_tf(['GPU:0'])
        error = RuntimeError('busy')
        fake_tf.config.experimental.set_virtual_device_configuration.side_effect = error
        logger = Mock()

        with patch.object(gpu_runtime_utils, '_load_tensorflow', return_value=fake_tf):
            configured = gpu_runtime_utils.configure_tensorflow_gpu(4096, logger=logger)

        self.assertFalse(configured)
        fake_tf.config.set_visible_devices.assert_called_once_with([], 'GPU')
        logger.warning.assert_called_once_with(
            "[GPU] Configuration failed (%s), falling back to CPU.",
            error,
        )

    def test_cpu_fallback_suppresses_visibility_reset_errors(self):
        fake_tf = self._build_fake_tf(['GPU:0'])
        error = ValueError('bad config')
        fake_tf.config.experimental.set_virtual_device_configuration.side_effect = error
        fake_tf.config.set_visible_devices.side_effect = ValueError('already initialized')
        logger = Mock()

        with patch.object(gpu_runtime_utils, '_load_tensorflow', return_value=fake_tf):
            configured = gpu_runtime_utils.configure_tensorflow_gpu(4096, logger=logger)

        self.assertFalse(configured)
        fake_tf.config.set_visible_devices.assert_called_once_with([], 'GPU')
        logger.warning.assert_called_once_with(
            "[GPU] Configuration failed (%s), falling back to CPU.",
            error,
        )


if __name__ == '__main__':
    unittest.main()