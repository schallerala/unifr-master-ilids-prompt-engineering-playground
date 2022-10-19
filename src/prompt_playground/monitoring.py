import functools
import logging
import time
from typing import Optional


SHARED_MONITORING_LOGGER_NAME = "prompt_playground.monitoring"


def timeit(
    *,
    logger: Optional[logging.Logger] = None,
    logger_name: Optional[str] = None,
    log_level: int = logging.DEBUG,
    message: str = None,
):

    message = f" ({message})" if message else ""

    def timeit_decorator(func):
        log = logger or logging.getLogger(
            logger_name or f"{func.__module__}.monitoring"
        )

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()

            try:
                return func(*args, **kwargs)
            finally:
                elapsed_time = time.time() - start
                log.log(
                    log_level,
                    f"function {func.__name__} took {elapsed_time * 1e+3:.3f} ms"
                    + message,
                )

        return wrapper

    return timeit_decorator
