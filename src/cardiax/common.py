import jax
import os
import time
from functools import wraps

from cardiax import logger

# A simpler decorator for printing the timing results of a function
def timeit(func):
    """A decorator to wrap a function for timing purposes.

    Args:
        func (callable): The function to be wrapped to log times

    Returns:
        callable: wrapped_function
    """

    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        logger.debug(f'Function {func.__name__} took {total_time:.4f} seconds')
        return result

    return timeit_wrapper


# Wrapper for writing timing results to a file
def walltime(txt_dir=None, filename=None):
    """Check if this is ever used. Mainly using the timeit wrapper.

    Args:


    Returns:
        _type_: _description_
    """

    def decorate(func):

        def wrapper(*list_args, **keyword_args):
            start_time = time.time()
            return_values = func(*list_args, **keyword_args)
            end_time = time.time()
            time_elapsed = end_time - start_time
            platform = jax.lib.xla_bridge.get_backend().platform
            logger.info(
                f"Time elapsed {time_elapsed} of function {func.__name__} "
                f"on platform {platform}"
            )
            if txt_dir is not None:
                os.makedirs(txt_dir, exist_ok=True)
                fname = 'walltime'
                if filename is not None:
                    fname = filename
                with open(os.path.join(txt_dir, f"{fname}_{platform}.txt"),
                          'w') as f:
                    f.write(f'{start_time}, {end_time}, {time_elapsed}\n')
            return return_values

        return wrapper

    return decorate
