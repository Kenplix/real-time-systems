import time
from contextlib import contextmanager

from signal.main import *


@logged(separator='\n')
def w_table(n: int) -> np.ndarray:
    res = np.empty((n, n), dtype=complex)
    for p in range(n):
        for k in range(n):
            angle = 2 * np.pi * p * k / n
            res[p][k] = np.cos(angle) - 1j * np.sin(angle)
    return res


@contextmanager
def timeit(msg: str) -> None:
    start_time = time.time()
    try:
        yield
    finally:
        print(f'{msg}: {time.time()- start_time}')


def show(sig: np.ndarray, ft: np.ndarray, ft_type: str) -> None:
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3, 1)
    axs[0].set_title('Random signal')
    axs[0].plot(LAGS, sig)

    axs[1].set_title(f'{ft_type} real')
    axs[1].stem(LAGS, np.real(ft), use_line_collection=True)

    axs[2].set_title(f'{ft_type} imag')
    axs[2].stem(LAGS, np.imag(ft), use_line_collection=True)
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    gen = generator(HARMONICS, FREQUENCY)
    sig = np.array([gen(lag) for lag in LAGS])

    msg = 'DFT'
    with timeit(msg):
        dft = np.matmul(w_table(len(LAGS)), sig)
    show(sig, dft, msg)

    msg = 'FFT'
    with timeit(msg):
        fft = np.fft.fft(sig, n=len(LAGS))
    show(sig, fft, msg)
