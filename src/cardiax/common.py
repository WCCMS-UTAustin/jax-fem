import jax
import os
import time
from functools import wraps

from cardiax import logger
from logging import DEBUG

# A simpler decorator for printing the timing results of a function
def timeit(func):
    """A decorator that times a function call when debugging

    If cardiax.logger has DEBUG-level logging enabled, utilizes JAX's
    recommend block-until-ready to ensure JIT'd compilation time is
    measured rather than initial tracer dispatch time.

    Otherwise, calls the function without timing.

    Args:
        func (callable): The function to be wrapped to log times

    Returns:
        callable: wrapped_function
    """

    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        if logger.isEnabledFor(DEBUG):
            # Only go through the trouble of timing if debugging
            start_time = time.perf_counter()

            result = func(*args, **kwargs)
            jax.block_until_ready(result)

            end_time = time.perf_counter()

            total_time = end_time - start_time
            logger.debug(f'Function {func.__name__} took {total_time:.4f} seconds')
        else:
            # Pass through silently when not debugging
            result = func(*args, **kwargs)

        return result

    return timeit_wrapper
