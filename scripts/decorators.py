import logging
from time import perf_counter
from typing import Callable


def timer[T, **P](func: Callable[P, T]) -> Callable[P, T]:
    """ Декоратор, который измеряет и логирует время выполнения функции.

    Аргументы:
        func (Callable): Функция, время выполнения которой необходимо измерить.

    Возвращается:
        Callable: Обернутая функция с логикой замера времени.
    """

    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        logging.info(f"[Начало] '{func.__name__}'")
        start = perf_counter()
        result = func(*args, **kwargs)
        end = perf_counter() - start
        logging.info(f"[Время] {func.__name__} ({end:.4f}с)")
        return result

    return wrapper
