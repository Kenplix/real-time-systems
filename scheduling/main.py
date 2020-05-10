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
import matplotlib.pyplot as plt

REPS: int = 100

REQUESTS: int = 1500
MIN_PRIORITY: int = 5
MAX_PRIORITY: int = 0
INTENSITY: float = 9000
DELAY: float = 1 / INTENSITY

CFG = namedtuple('CFG', ['func', 'params'])

# Counters
start_time: float = 0
current_time: float = 0

qsize: int
waiting_time: float
downtime: float
processed_requests: int

# Data
intensities: List[float] = []
qsize_ds: List[int] = []
waiting_time_ds: List[float] = []
downtime_ds: List[float] = []
overdue_requests_ds: List[int] = []


@timer
def collector(func, *args, reps: int, **kwargs) -> None:
    for _ in range(reps):
        func(*args, **kwargs)


def calculate_average(*cfgs: CFG, reps: int) -> Dict[int, float]:
    return {i: collector(cfg.func, *cfg.params, reps=reps) / reps for i, cfg in enumerate(cfgs)}


def build_funcs(*cfgs: CFG) -> Dict[int, CFG]:
    return {index: cfg for index, cfg in enumerate(cfgs)}


@dataclass
class Request:
    task_id: int
    login: float
    average: float
    deadline: float = None

    def __post_init__(self):
        self.deadline = self.login + self.average + DELAY

    # Implementation of EDF algorithm
    def __lt__(self, other):
        return self.deadline < other.deadline

    def __repr__(self):
        return f'{self.login=}, {self.deadline=}'


def producer(queue, average: Dict[int, float], lock) -> None:
    global qsize
    producer_time = time.time() - start_time
    for _ in range(REQUESTS):
        priority = random.randint(MAX_PRIORITY, MIN_PRIORITY)
        item = Request(id := random.randint(0, 2), producer_time, average[id])
        with lock:
            queue.put((priority, item))
            qsize += queue.qsize()
        prefix = colored('Putting', 'cyan', attrs=['bold'])
        logging.debug(f'{prefix} current_time={time.time() - start_time} {item}')

        time.sleep(DELAY)
        producer_time = time.time() - start_time


def consumer(queue, funcs: Dict[int, CFG], lock) -> None:
    global current_time, waiting_time, downtime
    global processed_requests
    request_number = 0
    downtime_start = time.time()
    while request_number < REQUESTS:
        if not queue.empty():
            with lock:
                priority, item = queue.get()
                downtime += (time.time() - downtime_start) / DELAY
                waiting_time += (time.time() - start_time - item.login) / DELAY

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
            downtime_start = time.time()
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
    start_time = time.time()

    while INTENSITY >= 0:
        INTENSITY -= 100

        qsize: int = 0
        waiting_time: float = 0
        downtime: float = 0
        processed_requests: int = 0
        start_time = time.time()

        main(max_workers=3)

        print(f'{INTENSITY=}')
        avg_qsize = qsize / REQUESTS
        avg_waiting_percent = 100 * waiting_time / REQUESTS
        avg_downtime_percent = 100 * downtime / REQUESTS
        print(f'{avg_qsize=}\n{avg_waiting_percent=}\n{avg_downtime_percent=}')
        overdue_requests = REQUESTS - processed_requests
        print(f'{overdue_requests=} percent: {100 * overdue_requests / REQUESTS}\n')

        qsize_ds.append(avg_qsize)
        waiting_time_ds.append(avg_waiting_percent)
        downtime_ds.append(avg_downtime_percent)
        overdue_requests_ds.append(overdue_requests)
        intensities.append(INTENSITY)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
    #  Make a little extra space between the subplots
    fig.subplots_adjust(hspace=10)

    # The ratio of Queue size to Intensity
    ax1.plot(intensities, qsize_ds)
    ax1.set_xlabel('Intensity')
    ax1.set_ylabel('Queue size')

    # The ratio of Waiting time percent to Intensity relative to DELAY
    ax2.plot(intensities, waiting_time_ds)
    ax2.set_xlabel('Intensity')
    ax2.set_ylabel('Waiting percent')

    # The ratio of Downtime percent to Intensity relative to DELAY
    ax3.plot(intensities, downtime_ds)
    ax3.set_xlabel('Intensity')
    ax3.set_ylabel('Downtime percent')

    # The ratio of Overdue requests to Intensity
    ax4.plot(intensities, overdue_requests_ds)
    ax4.set_xlabel('Intensity')
    ax4.set_ylabel('Overdue requests')

    plt.show()
