import time
from queue import PriorityQueue
from dataclasses import dataclass, field

from timers import timer

from signal.main import *
from autocorrelation.main import autocorr
from fourier_transform.main import w_table

from fermat_factor.main import full_factor

REQUESTS: int = 10000

MIN_PRIORITY: int = 1
MAX_PRIORITY: int = 10

INTENSITY: int = 5000

REPS: int = 100
CONFIG = dict()

processed_requests = []



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
        CONFIG[k] = {'func': (func := v['func']),
                     'params': v['params'],
                     'mean': collector(func, *v['params']) / REPS}


class Request:
    def __init__(self, task_id: Union[int, str], login: float):
        self.task_id = str(task_id)
        self.login = login
        self.mean = CONFIG[str(task_id)]['mean']
        self.deadline = self.login + (1 + random.random()) * self.mean


@dataclass(order=True)
class PrioritizedRequest:
    priority: int
    item: Any = field(compare=False)


def fill_queue(intensity: Union[int, float]):
    current_time = 0
    for _ in range(REQUESTS):
        priority = random.randint(MIN_PRIORITY, MAX_PRIORITY)
        request = Request(random.randint(0, 2), current_time)
        current_time += 1 / intensity
        queue.put(PrioritizedRequest(priority, request))


def execute_queue():
    current_time = 0
    while not queue.empty():
        start_time = time.time()
        request = queue.get()
        func = CONFIG[request.item.task_id]['func']
        params = CONFIG[request.item.task_id]['params']
        func(*params)
        current_time += time.time() - start_time
        if current_time < request.item.deadline:
            processed_requests.append(request)

if __name__ == '__main__':
    queue = PriorityQueue()

    create_config()
    fill_queue(INTENSITY)
    execute_queue()

    print(len(processed_requests))