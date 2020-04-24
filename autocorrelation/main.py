from typing import Union

import matplotlib.pyplot as plt

from signal.create import *
from autocorrelation import algs

MAX_TAU: int = PARTS


def show_signals():
    sig_x = [x_gen(i) for i in range(PARTS)]
    sig_y = [y_gen(i) for i in range(PARTS)]

    plt.plot(range(PARTS), sig_x, label='first')
    plt.plot(range(PARTS), sig_y, label='second')
    plt.title('Random signals')
    plt.legend()
    plt.show()


def show_autocorrelation():
    Rxx = [algs.autocorrelation(x_gen, x_gen, PARTS, tau) for tau in range(MAX_TAU)]
    Rxy = [algs.autocorrelation(x_gen, y_gen, PARTS, tau) for tau in range(MAX_TAU)]

    plt.plot(range(MAX_TAU), Rxx, label='Rxx(t, tau)')
    plt.plot(range(MAX_TAU), Rxy, label='Rxy(t, tau)')
    plt.title('Autocorrelation')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    show_signals()
    show_autocorrelation()
