import json

from timers import timer

from signal.main import *
from autocorrelation.main import autocorr
from fourier_transform.main import w_table

from fermat_factor.main import full_factor


@timer
@logged(separator='\n')
def collector(func, *args, reps=1000, **kwargs) -> float:
    for i in range(reps):
        func(*args, **kwargs)


if __name__ == '__main__':
    x_gen = generator(HARMONICS, FREQUENCY)
    y_gen = generator(HARMONICS, FREQUENCY)

    sig_x = np.array([x_gen(lag) for lag in LAGS])
    sig_y = np.array([y_gen(lag) for lag in LAGS])

    REPS = 100000

    autocorr_mean_time = collector(autocorr, sig_x, sig_y, reps=REPS) / REPS
    dft_mean_time = collector(np.matmul, w_table(len(LAGS)), sig_x, reps=REPS) / REPS
    factor_mean_time = collector(full_factor, 12345, reps=REPS) / REPS

    average_values = {'autocorrelation': autocorr_mean_time,
                      'fourier_transform': dft_mean_time,
                      'fermat_factor': factor_mean_time}

    with open("scheduling/average_values.json", "w") as file:
        json.dump(average_values, file)
