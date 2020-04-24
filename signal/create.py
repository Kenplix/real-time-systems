from typing import Union

from autocorrelation.algs import signal

HARMONICS: int = 10
FREQUENCY: int = 1100
TICKS: int = 256

RANGE_MIN: Union[int, float] = 0
RANGE_MAX: Union[int, float] = 1

x_gen = signal(HARMONICS, FREQUENCY, RANGE_MIN, RANGE_MAX)
y_gen = signal(HARMONICS, FREQUENCY, RANGE_MIN, RANGE_MAX)
