import logging
import threading
import time
from queue import PriorityQueue
from dataclasses import dataclass, field
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor

from timers import timer
from autocorrelation.main import autocorr
from fermat_factor.main import full_factor
from fourier_transform.main import w_table
from signal.main import *

from termcolor import colored

REQUESTS: int = 500
request_number = 0

MIN_PRIORITY: int = 10
MAX_PRIORITY: int = 0

INTENSITY: int = 10000

REPS: int = 100

processed_requests = 0


@timer
def collector(func, *args, reps, **kwargs):
    for _ in range(reps):
        func(*args, **kwargs)


def calculate_average(*cfgs, reps):
    return {i: collector(cfg.func, *cfg.params, reps=reps) / reps for i, cfg in enumerate(cfgs)}


def build_funcs(*cfgs):
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


class ProducerThread(threading.Thread):
    def __init__(self, name, queue, average):
        super().__init__()
        self.name = name
        self.queue = queue
        self.average = average

    def run(self):
        request_number = 0
        current_time = 0
        while request_number < REQUESTS:
            if not self.queue.full():
                priority = random.randint(MAX_PRIORITY, MIN_PRIORITY)
                item = Request(id := random.randint(0, 2), current_time, self.average[id])
                self.queue.put((priority, item))
                delay = 1 / INTENSITY
                current_time += delay
                request_number += 1

                prefix = colored('Putting', 'cyan', attrs=['bold'])
                logging.debug(f'{prefix} ({priority=}, {item}): '
                              f'{self.queue.qsize()} items in queue, {request_number=}')
                time.sleep(delay)


def consumer(queue, funcs, lock):
    current_time = 0
    global request_number, processed_requests
    while request_number < REQUESTS:
        if not queue.empty():
            priority, item = queue.get()

            is_executed = False
            start_time = time.time()
            if current_time < item.deadline:
                cfg = funcs[item.task_id]
                cfg.func(*cfg.params)
                is_executed = True
            current_time += time.time() - start_time

            prefix = colored('Failed', 'red', attrs=['bold'])
            with lock:
                if current_time < item.deadline and is_executed:
                    processed_requests += 1
                    prefix = colored('Passed', 'green', attrs=['bold'])
                request_number += 1

            logging.debug(f'{prefix} ({priority=}, {item}): '
                          f'{self.queue.qsize()} items in queue, {request_number=}')
            queue.task_done()


class ConsumerThread(threading.Thread):
    def __init__(self, name, queue, funcs, lock):
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


def main(*, delay: float = 0, buf_size=None, max_workers=1):
    x_gen = generator(HARMONICS, FREQUENCY)
    y_gen = generator(HARMONICS, FREQUENCY)

    sig_x = np.array([x_gen(lag) for lag in LAGS])
    sig_y = np.array([y_gen(lag) for lag in LAGS])

    CFG = namedtuple('CFG', ['func', 'params'])

    c1 = CFG(autocorr, [sig_x, sig_y])
    c2 = CFG(np.matmul, [w_table(len(LAGS)), sig_x])
    c3 = CFG(full_factor, [12345])
    configs = (c1, c2, c3)

    queue = PriorityQueue(buf_size) if buf_size else PriorityQueue()
    average = calculate_average(*configs, reps=REPS)
    producer = ProducerThread(name='producer', queue=queue, average=average)
    producer.start()
    time.sleep(delay)

    funcs = build_funcs(*configs)
    lock = threading.Lock()
    for _ in range(max_workers):
        consumer = ConsumerThread(name='consumer', queue=queue, funcs=funcs, lock=lock)
        consumer.start()
        consumer.join()

    producer.join()

    print(f'{max_workers=} {processed_requests=}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format='(%(threadName)s) %(message)s')

    main(delay=0.01, buf_size=50, max_workers=5)
