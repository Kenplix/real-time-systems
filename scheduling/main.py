import time
import sys
from queue import PriorityQueue
from dataclasses import dataclass, field
from typing import Any

from timers import timer

from signal.main import *
from autocorrelation.main import autocorr
from fourier_transform.main import w_table

from fermat_factor.main import full_factor


REQUESTS: int = 10

MIN_PRIORITY: int = 1
MAX_PRIORITY: int = 10

MIN_INTENSITY: int = 1
MAX_INTENSITY: int = 10
INTENSITY_STEP: int = 1

REPS: int = 100
CONFIG = dict()


@timer
def collector(func, *args, **kwargs):
    for _ in range(REPS):
        func(*args, **kwargs)


def create_config():
    x_gen = generator(HARMONICS, FREQUENCY)
    y_gen = generator(HARMONICS, FREQUENCY)

    sig_x = np.array([x_gen(lag) for lag in LAGS])
    sig_y = np.array([y_gen(lag) for lag in LAGS])

    local_config = {
        '0': {'func': autocorr, 'params': [sig_x, sig_y]},
        '1': {'func': np.matmul, 'params': [w_table(len(LAGS)), sig_x]},
        '2': {'func': full_factor, 'params': [12345]}
    }

    global CONFIG
    for k, v in local_config.items():
        CONFIG[k] = {'func': (func := v['func']), 'mean': collector(func, *v['params']) / REPS}


def generate_intensity():
    try:
        return random.choice(range(int(MIN_INTENSITY), int(MAX_INTENSITY)))
    except IndexError:
        print('MIN_INTENSITY could not be greater, than MAX_INTENSITY')
        sys.exit(0)



class Request:
    def __init__(self, task_id: int):
        self.login = time.time()
        self.mean = CONFIG[str(task_id)]['mean']

        deadline_coef = generate_intensity() + random.random()
        self.deadline = self.login + deadline_coef * self.mean

    def __repr__(self):
        return f'Available time: {self.deadline - self.login}'


@dataclass(order=True)
class PrioritizedRequest:
    priority: int
    item: Any = field(compare=False)

    def __repr__(self):
        return f'Priority: {self.priority} | {self.item}'


def request_generator():
    for _ in range(REPS):
        try:
            t = 1/generate_intensity()
        except ZeroDivisionError:
            print('MIN_INTENSITY cannot be less than 1')
            sys.exit(0)
        time.sleep(t)

        priority = random.randint(MIN_PRIORITY, MAX_PRIORITY)
        request = Request(random.randint(0, 2))
        yield PrioritizedRequest(priority, request)

def main():

    queue = PriorityQueue()

    for request in request_generator():
        queue.put(request)

    while not queue.empty():
        print(queue.get())

if __name__ == '__main__':
    create_config()
    main()

