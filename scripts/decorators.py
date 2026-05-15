import functools
import logging
from time import perf_counter


def timer_decorator(func):
    """Декоратор для async функций"""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        logging.info(f"[START] {func.__name__}")

        start_time = perf_counter()
        result = await func(*args, **kwargs)
        elapsed = perf_counter() - start_time

        logging.info(f"[TIME] {func.__name__}: {elapsed:.4f} сек")
        return result

    return wrapper
