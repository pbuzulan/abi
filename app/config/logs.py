import os
import logging
import logging.config
import sys


def setup_logging():
    logging_config = {
        "version": 1,
        "formatters": {
            "default": {
                "format": "%(asctime)s | %(levelname)s | %(threadName)s | %(name)s.%(funcName)s:%(lineno)s | %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S ",
            }
        },
        "handlers": {
            "console": {
                "level": os.getenv("LOG_LEVEL", "info").upper(),
                "class": "logging.StreamHandler",
                "formatter": "default",
                "stream": sys.stdout,
            },
            "file": {
                "level": os.getenv("LOG_LEVEL", "info").upper(),
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "default",
                "filename": os.path.join('logs', 'logging.log'),
                "maxBytes": 1024 * 1024 * 5,  # 5mb
                "backupCount": 10,
            },
        },
        "loggers": {
            # "default": {
            #     "level": os.getenv("LOG_LEVEL", "info").upper(),
            #     "handlers": ["console", "file"],
            #     "propagate": False,
            # },
            None: {
                "level": os.getenv("LOG_LEVEL", "info").upper(),
                "handlers": ["console", "file"],
                "propagate": True,
            },
        },
        "disable_existing_loggers": False,
    }
    logging.config.dictConfig(logging_config)
