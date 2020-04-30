from timers import timer
from logged import logged


def test(a, b):
    return a + b


@timer
@logged(separator='\n')
def collector(func, *args, reps=100, **kwargs) -> float:
    for i in range(reps):
        func(*args, **kwargs)


if __name__ == '__main__':
    print(collector(test, 5, 54, reps=100000))
