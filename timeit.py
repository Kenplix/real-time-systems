import time
from contextlib import contextmanager


@contextmanager
def timeit(msg: str) -> None:
    start_time = time.time()
    try:
        yield
    finally:
        print(f'{msg}: {time.time() - start_time}')
