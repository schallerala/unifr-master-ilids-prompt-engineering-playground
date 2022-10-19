import asyncio
import functools
import logging
from functools import wraps
from typing import Iterable, List, Optional, Callable, Dict


def call_sync(func: Callable, sequence_args, *, logger: logging.Logger):
    try:
        logger.info(f"calling {func.__name__}...")
        if sequence_args:
            sequence_args = sequence_args[0]  # get first tuple
            for args in sequence_args:
                if isinstance(args, Iterable) and not isinstance(args, str):
                    func(*args)
                else:
                    func(args)

        else:
            func()
    except:
        logger.error(f"Issue in {func.__name__}")


class PrecacheRegistry:
    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.functions_registry: Dict[Callable, ...] = dict()
        self.logger = logger or logging.getLogger(__name__)

    def cache(
        self,
        *args: Optional[List],
    ):
        def cache_decorator(func):
            func = functools.cache(func)

            self.functions_registry[func] = args

            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapper

        return cache_decorator

    def register(
        self,
        func: Callable,
        *args: Optional[List],
    ):
        """function is expected to be annotated with functools.cache decorator"""
        self.functions_registry[func] = args

    def call_all_sync(self):
        for func, args1 in self.functions_registry.items():
            call_sync(func, args1, logger=self.logger)

        print("done calling cached functions")

    async def call_all_async(self):
        await asyncio.to_thread(self.call_all_sync)
