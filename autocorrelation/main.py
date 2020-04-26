import numpy as np

from signal.create import *
from utilities.logged import logged


@logged(separator='\n')
def autocorr(x: np.ndarray, y: np.ndarray, mode: str = 'regular') -> np.ndarray:
    """The autocorrelation produces a symmetric signal,
     we only care about the "right half"""
    corr = np.correlate(x, y, mode='full')[len(x) - 1:]
    if mode == 'regular':
        return corr
    elif mode == 'normalized':
        if all(x == y):
            return corr / np.var(x) / len(x)
        else:
            raise ValueError('Different signals')
    else:
        raise ValueError(f'Unknown mode: {mode}')


def show() -> None:
    x_gen = generator(HARMONICS, FREQUENCY)
    y_gen = generator(HARMONICS, FREQUENCY)

    sig_x = np.array([x_gen(lag) for lag in LAGS])
    sig_y = np.array([y_gen(lag) for lag in LAGS])

    Rxx = autocorr(sig_x, sig_x)
    Rxy = autocorr(sig_x, sig_y)

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3, 1)
    axs[0].plot(LAGS, Rxx)
    axs[0].set_xlabel('Lags')
    axs[0].set_ylabel('Rxx(t, lag)')

    axs[1].plot(LAGS, Rxy)
    axs[1].set_xlabel('Lags')
    axs[1].set_ylabel('Rxy(t, lag)')

    axs[2].plot(LAGS, Rxx, LAGS, Rxy)
    axs[2].set_xlabel('Lags')
    axs[2].set_ylabel('Rxx and Rxy')

    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    show()
