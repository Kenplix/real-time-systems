import time
from enum import Enum, auto
from functools import wraps
from typing import Any, Tuple, Dict

from termcolor import colored


class Modes(Enum):
    WITH_PARAMETERS: int = auto()
    WITHOUT_PARAMETERS: int = auto()


def args_builder(mode: Modes, /, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> str:
    if mode is Modes.WITH_PARAMETERS:
        arg_lst = []
        if args:
            arg_lst.append(colored(', '.join(repr(arg) for arg in args), 'green'))
        if kwargs:
            pairs = [f'{k}={w}' for k, w in sorted(kwargs.items())]
            arg_lst.append(colored(', '.join(pairs), 'cyan'))

        arg_str = ', '.join(arg_lst)
        return colored(f'({arg_str})')
    elif mode is Modes.WITHOUT_PARAMETERS:
        return ''


def logged(time_format='%b %d %Y - %H:%M:%S', separator='', mode: Modes = Modes.WITHOUT_PARAMETERS):
    def decorator(func):
        @wraps(func)
        def decorated_func(*args, **kwargs):
            arg_str = args_builder(mode, args, kwargs)
            func_disp = colored(func.__qualname__, "magenta", attrs=["bold"])
            print(f'{separator}- Running {func_disp}'
                  f'{arg_str} on {time.strftime(time_format)}')

            start_time = time.time()
            result = func(*args, **kwargs)
            print(f'- Finished {func_disp}'
                  f'{arg_str}, execution time = {time.time() - start_time}s{separator}')
            return result
        return decorated_func
    return decorator