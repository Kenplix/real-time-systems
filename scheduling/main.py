import logging
import time
import threading as th
from queue import PriorityQueue
from dataclasses import dataclass
from collections import namedtuple

from timers import timer
from autocorrelation.main import autocorr
from fermat_factor.main import full_factor
from fourier_transform.main import w_table
from signal.main import *

from termcolor import colored

REPS: int = 100
REQUESTS: int = 2000
MIN_PRIORITY: int = 3
MAX_PRIORITY: int = 0
INTENSITY: float = 10000

CFG = namedtuple('CFG', ['func', 'params'])

# Counters
request_number: int = 0
qsize_counter: int = 0
processed_requests: int = 0


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
        self.deadline = self.login + (1 + random.random()) * self.average

    # Implementation of EDF algorithm
    def __lt__(self, other):
        return self.deadline < other.deadline

    def __repr__(self):
        return f'{self.task_id=}, {self.deadline=}'


class ProducerThread(th.Thread):
    def __init__(self, name, queue, average: Dict[int, float], lock):
        super().__init__()
        self.name = name
        self.queue = queue
        self.average = average
        self.lock = lock

    def run(self):
        request_number = 0
        current_time = 0
        global qsize_counter
        while request_number < REQUESTS:
            if not self.queue.full():
                priority = random.randint(MAX_PRIORITY, MIN_PRIORITY)
                item = Request(id := random.randint(0, 2), current_time, self.average[id])
                self.queue.put((priority, item))

                with self.lock:
                    qsize_counter += self.queue.qsize()

                delay = 1 / INTENSITY
                time.sleep(delay)
                current_time += delay
                request_number += 1

                prefix = colored('Putting', 'cyan', attrs=['bold'])
                logging.debug(f'{prefix} ({priority=}, {item}): '
                              f'{self.queue.qsize()} items in queue, {request_number=}')


class ConsumerThread(th.Thread):
    def __init__(self, name, queue, funcs: Dict[int, CFG], lock):
        super().__init__()
        self.name = name
        self.queue = queue
        self.funcs = funcs
        self.lock = lock

    def run(self):
        current_time = 0
        global request_number, processed_requests
        while request_number < REQUESTS:
            if not self.queue.empty():
                priority, item = self.queue.get()

                is_executed = False
                start_time = time.time()
                if current_time < item.deadline:
                    cfg = self.funcs[item.task_id]
                    cfg.func(*cfg.params)
                    is_executed = True
                current_time += time.time() - start_time

                prefix = colored('Failed', 'red', attrs=['bold'])
                with self.lock:
                    if current_time < item.deadline and is_executed:
                        processed_requests += 1
                        prefix = colored('Passed', 'green', attrs=['bold'])
                    request_number += 1

                logging.debug(f'{prefix} ({priority=}, {item}): '
                              f'{self.queue.qsize()} items in queue, {request_number=}')
                self.queue.task_done()


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
    lock = th.Lock()
    producer = ProducerThread(name='producer', queue=queue, average=average, lock=lock)
    producer.start()
    time.sleep(delay)

    funcs = build_funcs(*configs)
    for _ in range(max_workers):
        consumer = ConsumerThread(name='consumer', queue=queue, funcs=funcs, lock=lock)
        consumer.start()
        consumer.join()

    producer.join()

    print(f'{max_workers=} {processed_requests=}')
    print(f'average qsize {qsize_counter / REQUESTS}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format='(%(threadName)s) %(message)s')

    main(delay=0.1, buf_size=100, max_workers=5)
