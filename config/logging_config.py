import logging
import logging.config
import os
from typing import Optional

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s"
        },
    },
    "handlers": {
        "file_handler": {
            "class": "logging.FileHandler",
            "formatter": "standard",
            "filename": "logs/dev.log",
            "mode": "a",
            "level": "INFO",
        },
        "console_handler": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "level": "INFO",
        },
    },
    "loggers": {
        "": {
            "handlers": ["file_handler", "console_handler"],
            "level": "INFO",
            "propagate": True,
        }
    },
}


def setup_logging(log_file: Optional[str] = None, log_level: Optional[str] = None):
    if log_file:
        log_file_path = os.path.join("logs", log_file)
        LOGGING_CONFIG["handlers"]["file_handler"]["filename"] = log_file_path
    if log_level:
        LOGGING_CONFIG["handlers"]["file_handler"]["level"] = log_level
        LOGGING_CONFIG["handlers"]["console_handler"]["level"] = log_level
    logging.config.dictConfig(LOGGING_CONFIG)
