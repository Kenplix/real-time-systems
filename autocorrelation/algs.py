import random
import math
from functools import reduce
from typing import List, Union, Optional, Callable

from utilities.logged import logged

Generator = Callable[[Union[int, float]], float]
DataType = List[Union[int, float]]


@logged(separator='\n')
def signal(harmonics: int, omega: int, start: int = 0, end: int = 1) -> Generator:
    """Return signal function"""
    A = [random.uniform(start, end) for _ in range(harmonics)]
    phi = [random.uniform(start, end) for _ in range(harmonics)]

    def generate(t: Union[int, float]) -> float:
        return reduce(lambda ac, i: ac + A[i] * math.sin(omega / harmonics * t * i + phi[i]), range(harmonics))
    return generate


@logged()
def expected_value(data: DataType) -> float:
    return sum(data) / len(data)


@logged()
def dispersion(data: DataType, M: Optional[float] = None) -> float:
    if M is None:
        M = expected_value(data)
    return sum([(i - M) ** 2 for i in data]) / (len(data) - 1)


@logged(separator='\n')
def autocorrelation(gen_x: Generator, gen_y: Generator, parts: int, tau: int = 0) -> float:
    sig_x = [gen_x(t) for t in range(parts)]
    sig_y = [gen_y(t + tau) for t in range(parts)]
    M_x, M_y = expected_value(sig_x), expected_value(sig_y)

    return reduce(lambda ac, i: ac + (sig_x[i] - M_x) * (sig_y[i] - M_y) / (parts - 1), range(parts))