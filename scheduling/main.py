import time
import logging
from threading import Lock
from queue import PriorityQueue
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from collections import namedtuple

from timers import timer
from autocorrelation.main import autocorr
from fermat_factor.main import full_factor
from fourier_transform.main import w_table
from signal.main import *

from termcolor import colored

REPS: int = 100
REQUESTS: int = 1000
MIN_PRIORITY: int = 3
MAX_PRIORITY: int = 0
INTENSITY: float = 50
DELAY: float = 1 / INTENSITY

CFG = namedtuple('CFG', ['func', 'params'])

# Counters
producer_time: float = 0
qsize_counter: int = 0
current_time: float = 0
waiting_time: float = 0
processed_requests: int = 0
start_time = time.time()


@timer
def collector(func, *args, reps: int, **kwargs) -> None:
    for _ in range(reps):
        func(*args, **kwargs)


def calculate_average(*cfgs: CFG, reps: int) -> Dict[int, float]:
    return {i: collector(cfg.func, *cfg.params, reps=reps) / reps for i, cfg in enumerate(cfgs)}


def build_funcs(*cfgs: CFG) -> Dict[int, CFG]:
    return {index: cfg for index, cfg in enumerate(cfgs)}


def dead_coef(min: int, max: int) -> float:
    if min < 0 or min > max:
        raise ValueError
    return 1 + (random.randint(min, max) * random.random())


@dataclass
class Request:
    task_id: int
    login: float
    average: float
    deadline: float = None

    def __post_init__(self):
        self.deadline = (self.login + self.average) * dead_coef(1, 3)

    # Implementation of EDF algorithm
    def __lt__(self, other):
        return self.deadline < other.deadline

    def __repr__(self):
        return f'{self.login=}, {self.deadline=}'


def producer(queue, average: Dict[int, float], lock) -> None:
    global qsize_counter
    request_number = 0
    producer_time = time.time() - start_time
    while request_number < REQUESTS:
        priority = random.randint(MAX_PRIORITY, MIN_PRIORITY)
        item = Request(id := random.randint(0, 2), producer_time, average[id])
        queue.put((priority, item))
        time.sleep(DELAY)
        prefix = colored('Putting', 'cyan', attrs=['bold'])
        logging.debug(f'{prefix} {current_time=} {item}')

        with lock:
            qsize_counter += queue.qsize()
        producer_time += DELAY
        request_number += 1


def consumer(queue, funcs: Dict[int, CFG], lock) -> None:
    global current_time, waiting_time
    global processed_requests
    request_number = 0
    while request_number < REQUESTS:
        if not queue.empty():
            priority, item = queue.get()
            with lock:
                waiting_time += time.time() - start_time - item.login

            is_executed = False
            if time.time() - start_time < item.deadline:
                cfg = funcs[item.task_id]
                cfg.func(*cfg.params)
                is_executed = True
            queue.task_done()

            prefix = colored('Failed', 'red', attrs=['bold'])
            with lock:
                current_time = time.time() - start_time
                if current_time < item.deadline and is_executed:
                    processed_requests += 1
                    prefix = colored('Passed', 'green', attrs=['bold'])
                request_number += 1
            logging.debug(f'{prefix} {current_time=} {item}')


def main(*, delay: float = 0, buf_size: Optional[int] = None, max_workers: int = 1):
    x_gen = generator(HARMONICS, FREQUENCY)
    y_gen = generator(HARMONICS, FREQUENCY)

    sig_x = np.array([x_gen(lag) for lag in LAGS])
    sig_y = np.array([y_gen(lag) for lag in LAGS])

    c1 = CFG(autocorr, [sig_x, sig_y])
    c2 = CFG(np.matmul, [w_table(len(LAGS)), sig_x])
    c3 = CFG(full_factor, [12345])
    configs = (c1, c2, c3)

    queue = PriorityQueue(buf_size) if buf_size else PriorityQueue()
    average = calculate_average(*configs, reps=REPS)
    lock = Lock()
    funcs = build_funcs(*configs)

    with ThreadPoolExecutor(max_workers) as executor:
        executor.submit(producer, queue, average, lock)
        time.sleep(delay)
        executor.submit(consumer, queue, funcs, lock)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(message)s')
    main(delay=0.2, max_workers=3)
    print(f'Requests {REQUESTS}')
    print(f'Average qsize {qsize_counter / REQUESTS}\n'
          f'Average waiting {waiting_time / REQUESTS}')

    overdue_requests = REQUESTS - processed_requests
    print(f'Number of overdue requests {overdue_requests}\n'
          f'Overdue requests / requests {overdue_requests/REQUESTS}')
