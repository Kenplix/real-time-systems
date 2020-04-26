import random
from typing import Union, List, Callable

import numpy as np

from utilities.logged import logged


Rational = Union[int, float]

AMPLITUDE: Rational = 1
FULL_CIRCLE = 2 * np.pi

HARMONICS: int = 10
FREQUENCY: int = 1100
LAGS: List[int] = range(256)

Ticker = Callable[[Rational], Rational]


@logged(separator='\n')
def generator(harmonics: int, frequency: int) -> Ticker:
    A = np.array([random.uniform(-AMPLITUDE, AMPLITUDE) for _ in range(harmonics)])
    phi = np.array([FULL_CIRCLE * random.uniform(-1, 1) for _ in range(harmonics)])
    w = frequency / harmonics

    def tiker(t: Union[int, float]) -> float:
        return np.sum(A * np.sin(w * np.arange(harmonics) * t + phi))
    return tiker


def show() -> None:
    x_gen = generator(HARMONICS, FREQUENCY)
    y_gen = generator(HARMONICS, FREQUENCY)

    sig_x = np.array([x_gen(lag) for lag in LAGS])
    sig_y = np.array([y_gen(lag) for lag in LAGS])

    import matplotlib.pyplot as plt
    plt.plot(LAGS, sig_x, label='x')
    plt.plot(LAGS, sig_y, label='y')
    plt.title('Random signals')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    show()
