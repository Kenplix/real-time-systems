import math
from typing import Tuple

decomposition = []


def factorise(n: int) -> Tuple[int, int]:
    s = math.ceil(math.sqrt(n))
    y = s**2 - n
    while not math.sqrt(y).is_integer():
        s += 1
        y = s**2 - n
    return s + math.sqrt(y), s - math.sqrt(y)


def full_factor(n: int) -> None:
    a, b = factorise(n)
    if b != 1:
        a, b = factorise(n)
        full_factor(a)
        full_factor(b)
    else:
        decomposition.append(a)


if __name__ == '__main__':
    num = 89755
    full_factor(num)
    print(f'{num} ->', '*'.join(map(lambda x: str(int(x)), decomposition)))
