from typing import Union

from autocorrelation.algs import signal

HARMONICS: int = 10
OMEGA: int = 1100
PARTS: int = 256

RANGE_MIN: Union[int, float] = 0
RANGE_MAX: Union[int, float] = 1

x_gen = signal(HARMONICS, OMEGA, RANGE_MIN, RANGE_MAX)
y_gen = signal(HARMONICS, OMEGA, RANGE_MIN, RANGE_MAX)
