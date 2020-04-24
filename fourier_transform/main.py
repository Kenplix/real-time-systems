import time
from contextlib import contextmanager

import numpy as np
import random
import math
import matplotlib.pyplot as plt


AMPLITUDE_MAX = 1
FULL_CIRCLE = 2 * math.pi

# variant 10
HARMONICS = 14
TICKS = 2048
FREQUENCY = 1700


def random_signal(harmonics, ticks, freq):
    generated_signal = np.zeros(ticks)
    for i in range(harmonics):
        fi = FULL_CIRCLE * random.random()
        amplitude = AMPLITUDE_MAX * random.random()
        w = freq - i * freq / harmonics

        x = amplitude * np.sin(np.arange(0, ticks, 1) * w + fi)
        generated_signal += x
    return generated_signal


def w_table(n):
    res = np.empty((n, n), dtype=complex)
    for p in range(n):
        for k in range(n):
            angle = 2 * np.pi * p * k / n
            res[p][k] = np.cos(angle) - 1j * np.sin(angle)
    return res


@contextmanager
def timeit(msg):
    s = time.time()
    try:
        yield
    finally:
        print(f'{msg} Took {time.time()- s}')


if __name__ == '__main__':
    random.seed(10)
    x_line = [i for i in range(TICKS)]
    sig = random_signal(HARMONICS, TICKS, FREQUENCY)
    table = w_table(TICKS)

    with timeit('With precomputed table'):
        dft = np.matmul(table, sig)

    with timeit('With manual coef counting'):
        dft2 = np.matmul(w_table(TICKS), sig)

    with timeit('Default numpy fft'):
        np.fft.fft(sig, n=TICKS)


# draw plots
#     plt.subplot(311)
#     p1 = plt.plot(x_line, sig, label='Random signal')
#     plt.legend(handles=p1)
#
#     plt.subplot(312)
#     plt.title('DFT real')
#     p2 = plt.stem(x_line, np.real(dft), use_line_collection=True)
#
#     plt.subplot(313)
#     plt.title('DFT Imag')
#     p3 = plt.stem(x_line, np.imag(dft), use_line_collection=True)
#
#     plt.show()
