import logging
from typing import Optional


def _log_info(logger: Optional[logging.Logger], message: str, *args) -> None:
    if logger is None:
        print(message % args if args else message)
        return
    logger.info(message, *args)


def _log_warning(logger: Optional[logging.Logger], message: str, *args) -> None:
    if logger is None:
        print(message % args if args else message)
        return
    logger.warning(message, *args)


def _load_tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is not installed. Install ML dependencies with: "
            "pip install -r requirements_PY_3_12_ml.txt"
        ) from exc

    return tf


def configure_tensorflow_gpu(memory_limit_mb: int, logger: Optional[logging.Logger] = None) -> bool:
    """Configure TensorFlow GPU memory limits and fall back to CPU safely."""
    tf = _load_tensorflow()
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        _log_info(logger, "[GPU] No GPU detected, using CPU.")
        return False

    try:
        for gpu in gpus:
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit_mb)]
            )
        _log_info(
            logger,
            "[GPU] Configured %d GPU(s) with %dMB memory limit.",
            len(gpus),
            memory_limit_mb,
        )
        return True
    except (RuntimeError, ValueError) as error:
        _log_warning(logger, "[GPU] Configuration failed (%s), falling back to CPU.", error)
        try:
            tf.config.set_visible_devices([], 'GPU')
        except (RuntimeError, ValueError):
            pass
        return False