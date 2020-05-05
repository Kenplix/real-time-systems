import time
import logging
from threading import Lock
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from collections import namedtuple

from autocorrelation.main import autocorr
from fermat_factor.main import full_factor
from fourier_transform.main import w_table
from signal.main import *

from termcolor import colored
import matplotlib.pyplot as plt

REQUESTS: int = 250
INTENSITY: float = 10
DELAY: float = 1 / INTENSITY

CFG = namedtuple('CFG', ['func', 'params'])

# Counters
start_time: float = 0
current_time: float = 0

waiting_time: float
downtime: float
processed_requests: int

# Data
intensities: List[float] = []
waiting_time_ds: List[float] = []
downtime_ds: List[float] = []


def build_funcs(*cfgs: CFG) -> Dict[int, CFG]:
    return {index: cfg for index, cfg in enumerate(cfgs)}


class Request:
    def __init__(self, task_id, login):
        self.task_id = task_id
        self.login = login

    @property
    def logout(self):
        return self.__dict__['logout']

    @logout.setter
    def logout(self, value):
        self.__dict__['logout'] = value

    def __repr__(self):
        if 'logout' not in self.__dict__:
            return f'Request({self.login=})'
        return f'Request({self.login=}, {self.logout=})'


def producer(queue) -> None:
    global qsize
    producer_time = time.time() - start_time
    for _ in range(REQUESTS):
        item = Request(random.randint(0, 2), producer_time)
        queue.put(item)
        prefix = colored('Putting', 'cyan', attrs=['bold'])
        logging.debug(f'{prefix} current_time={time.time() - start_time} {item}')

        time.sleep(DELAY)
        producer_time = time.time() - start_time


def consumer(queue, funcs: Dict[int, CFG], lock) -> None:
    global current_time, waiting_time, downtime
    global processed_requests
    downtime_start = time.time()
    while processed_requests < REQUESTS:
        if not queue.empty():
            with lock:
                item = queue.get()
                waiting_time += time.time() - start_time - item.login
                downtime += time.time() - downtime_start

            cfg = funcs[item.task_id]
            cfg.func(*cfg.params)

            with lock:
                item.logout = time.time() - start_time
                downtime_start = time.time()
                processed_requests += 1
            queue.task_done()

            prefix = colored('Passed', 'green', attrs=['bold'])
            logging.debug(f'{prefix} {item}')


def main(*, buf_size: Optional[int] = None, max_workers: int = 1):
    x_gen = generator(HARMONICS, FREQUENCY)
    y_gen = generator(HARMONICS, FREQUENCY)

    sig_x = np.array([x_gen(lag) for lag in LAGS])
    sig_y = np.array([y_gen(lag) for lag in LAGS])

    c1 = CFG(autocorr, [sig_x, sig_y])
    c2 = CFG(np.matmul, [w_table(len(LAGS)), sig_x])
    c3 = CFG(full_factor, [12345])
    configs = (c1, c2, c3)

    queue = Queue(buf_size) if buf_size else Queue()
    lock = Lock()
    funcs = build_funcs(*configs)

    with ThreadPoolExecutor(max_workers) as executor:
        executor.submit(producer, queue)
        executor.submit(consumer, queue, funcs, lock)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    while INTENSITY <= REQUESTS * 3:
        waiting_time: float = 0
        downtime: float = 0
        processed_requests: int = 0
        start_time = time.time()

        main(max_workers=2)

        print(f'{INTENSITY=}')
        avg_waiting = 100 * 100 * waiting_time / REQUESTS
        avg_downtime_percent = 100 * downtime / REQUESTS / DELAY
        print(f'{avg_waiting=}\n{avg_downtime_percent=}')

        intensities.append(INTENSITY)
        waiting_time_ds.append(avg_waiting)
        downtime_ds.append(avg_downtime_percent)
        INTENSITY += 20

    fig, (ax1, ax2) = plt.subplots(2, 1)
    #  Make a little extra space between the subplots
    fig.subplots_adjust(hspace=0.5)

    # The ratio of Average waiting time to Intensity
    ax1.plot(intensities, waiting_time_ds)
    ax1.set_xlabel('intensity')
    ax1.set_ylabel('avg wt (ms)')

    # The ratio of Average downtime percent to Intensity relative to DELAY
    ax2.plot(intensities, downtime_ds)
    ax2.set_xlabel('intensity')
    ax2.set_ylabel('avg dt pc')

    plt.show()
