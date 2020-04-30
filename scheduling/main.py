import time
import random
from queue import PriorityQueue
from dataclasses import dataclass, field
from typing import Any

from timers import timer

from signal.main import *
from autocorrelation.main import autocorr
from fourier_transform.main import w_table

from fermat_factor.main import full_factor

REPS: int = 1000

REQUESTS: int = 1000
MIN_PRIORITY: int = 1
MAX_PRIORITY: int = 10
MIN_INTENSITY: Rational = 0.05
MAX_INTENSITY: Rational = 10
STEP: Rational = 0.05

config = {'0': {'name': 'autocorrelation', 'mean': 0},
          '1': {'name': 'fourier_transform', 'mean': 0},
          '2': {'name': 'fermat_factor', 'mean': 0}}


@timer
@logged(separator='\n')
def collector(func, *args, reps=1000, **kwargs) -> float:
    for i in range(reps):
        func(*args, **kwargs)


@dataclass(order=True)
class PrioritizedItem:
    priority: int
    item: Any = field(compare=False)


class Request:
    def __init__(self, task_id: int):
        self.login_time = time.time()
        self.lead_time = config[str(task_id)]['mean']
        self.deadline = self.login_time + random.randint(1, 10 * MAX_INTENSITY) * self.lead_time

    def __repr__(self):
        return f'life time: {self.deadline - self.login_time}'


def main():
    x_gen = generator(HARMONICS, FREQUENCY)
    y_gen = generator(HARMONICS, FREQUENCY)

    sig_x = np.array([x_gen(lag) for lag in LAGS])
    sig_y = np.array([y_gen(lag) for lag in LAGS])

    config['0']['mean'] = collector(autocorr, sig_x, sig_y, reps=REPS) / REPS
    config['1']['mean'] = collector(np.matmul, w_table(len(LAGS)), sig_x, reps=REPS) / REPS
    config['2']['mean'] = collector(full_factor, 12345, reps=REPS) / REPS

    queue = PriorityQueue()

    for _ in range(REQUESTS):
        priority = random.randint(MIN_PRIORITY, MAX_PRIORITY)
        item = Request(str(random.randint(0, 2)))
        wrapper = PrioritizedItem(priority, item)
        queue.put(wrapper)

    while queue:
        print(queue.get())


if __name__ == '__main__':
    main()
