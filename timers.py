import time
from functools import wraps
from contextlib import contextmanager


def timer(func):
    @wraps(func)
    def decorated_func(*args, **kwargs) -> float:
        start_time = time.time()
        func(*args, **kwargs)
        return time.time() - start_time
    return decorated_func


@contextmanager
def timeit(msg: str) -> None:
    start_time = time.time()
    try:
        yield
    finally:
        print(f'{msg}: {time.time() - start_time}')
