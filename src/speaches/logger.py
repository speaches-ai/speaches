import logging
import logging.config


def setup_logger(log_level: str) -> None:
    assert log_level.upper() in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], log_level
    # https://www.youtube.com/watch?v=9L77QExPmI0
    # https://docs.python.org/3/library/logging.config.html
    logging_config = {
        "version": 1,  # required
        "disable_existing_loggers": False,
        "formatters": {
            "simple": {"format": "%(asctime)s:%(levelname)s:%(name)s:%(funcName)s:%(lineno)d:%(message)s"},
        },
        "handlers": {
            "stdout": {
                "class": "logging.StreamHandler",
                "formatter": "simple",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "root": {
                "level": log_level.upper(),
                "handlers": ["stdout"],
            },
            # TODO: do I need to specify `handlers` for each logger?
            "PIL": {
                "level": "INFO",
                "handlers": ["stdout"],
            },
            "httpx": {
                "level": "INFO",
                "handlers": ["stdout"],
            },
            "python_multipart": {
                "level": "INFO",
                "handlers": ["stdout"],
            },
            "httpcore": {
                "level": "INFO",
                "handlers": ["stdout"],
            },
            "aiortc": {
                "level": "INFO",
                "handlers": ["stdout"],
            },
            "aioice": {
                "level": "INFO",
                "handlers": ["stdout"],
            },
            "numba.core": {
                "level": "INFO",
                "handlers": ["stdout"],
            },
        },
    }

    logging.config.dictConfig(logging_config)
