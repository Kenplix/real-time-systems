from typing import Union

import matplotlib.pyplot as plt

from autocorrelation_research import algs

harmonics: int = 10
omega: int = 1100
parts: int = 256

range_min: Union[int, float] = 0
range_max: Union[int, float] = 1

x_gen = algs.signal(harmonics, omega, range_min, range_max)
y_gen = algs.signal(harmonics, omega, range_min, range_max)

max_tau: int = parts


def show_signals():
    sig_x = [x_gen(i) for i in range(parts)]
    sig_y = [y_gen(i) for i in range(parts)]

    plt.plot(range(parts), sig_x, label='first')
    plt.plot(range(parts), sig_y, label='second')
    plt.title('Random signals')
    plt.legend()
    plt.show()


def show_autocorrelation():
    Rxx = [algs.autocorrelation(x_gen, x_gen, parts, tau) for tau in range(max_tau)]
    Rxy = [algs.autocorrelation(x_gen, y_gen, parts, tau) for tau in range(max_tau)]

    plt.plot(range(max_tau), Rxx, label='Rxx(t, tau)')
    plt.plot(range(max_tau), Rxy, label='Rxy(t, tau)')
    plt.title('Autocorrelation')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    show_signals()
    show_autocorrelation()
