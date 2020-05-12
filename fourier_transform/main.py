from timers import timeit
from signal.main import *


def w_table(n: int) -> np.ndarray:
    res = np.empty((n, n), dtype=complex)
    for p in range(n):
        for k in range(n):
            angle = 2 * np.pi * p * k / n
            res[p][k] = np.cos(angle) - 1j * np.sin(angle)
    return res


def show(sig: np.ndarray, ft: np.ndarray, ft_type: str) -> None:
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3, 1)
    axs[0].set_title('Random signal')
    axs[0].plot(LAGS, sig)

    axs[1].set_title(f'{ft_type} real')
    axs[1].plot(LAGS, np.real(ft))

    axs[2].set_title(f'{ft_type} imag')
    axs[2].plot(LAGS, np.imag(ft))

    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    gen = generator(HARMONICS, FREQUENCY)
    sig = np.array([gen(lag) for lag in LAGS])

    with timeit('MY'):
        my = np.matmul(w_table(len(LAGS)), sig)

    with timeit('NP'):
        np_ = np.fft.fft(sig, n=len(LAGS))

    show(sig, np.absolute(my - np_), 'Absolute difference')
