import unittest
from cardiax import logger
from logging import DEBUG, INFO
from cardiax.common import timeit
from jax.random import normal
from jax import jit
import jax.numpy as jnp


class testsTimeitDebugging(unittest.TestCase):

    def setUp(self) -> None:
        def my_func(x):
            return x ** 2 + x + 1

        self.func_nojit = my_func
        self.func_jit = jit(my_func)

        return super().setUp()


    def _test_variant(self, the_func):
        timed = timeit(the_func)

        # Intentionally big and slow
        x = normal(jnp.array([1337, 1337], dtype=jnp.uint32), (2048, 2048))

        with self.assertLogs(logger, DEBUG) as cm:
            # Make sure logger can be level-adjusted any time
            logger.setLevel(DEBUG)

            y = timed(x)

        # Slightly fragile: second-to-last word in single logger
        # output should be float representation of seconds
        self.assertEqual(len(cm.output), 1)
        time_output = cm.output[0]
        time_float = float(time_output.split(' ')[-2])
        self.assertGreater(time_float, 0.0)

        self.assertTrue((y == x**2 + x + 1).all())


    def test_jitted(self):
        """Tests timeit logs given JIT'd function"""
        self._test_variant(self.func_jit)
        

    def test_not_jitted(self):
        """Tests timeit logs given not-JIT'd function"""
        self._test_variant(self.func_nojit)


class testsTimeitNotDebugging(unittest.TestCase):

    def setUp(self) -> None:
        def my_func(x):
            return x ** 2 + x + 1

        self.func_nojit = my_func
        self.func_jit = jit(my_func)

        return super().setUp()


    def _test_variant(self, the_func):
        timed = timeit(the_func)

        # Intentionally big and slow
        x = normal(jnp.array([1337, 1337], dtype=jnp.uint32), (2048, 2048))

        # It should pass through in this case
        with self.assertNoLogs(logger, DEBUG):
            # Make sure logger can be level-adjusted any time
            logger.setLevel(INFO)

            y = timed(x)

        self.assertTrue((y == x**2 + x + 1).all())


    def test_jitted(self):
        """Tests timeit logs given JIT'd function"""
        self._test_variant(self.func_jit)
        

    def test_not_jitted(self):
        """Tests timeit logs given not-JIT'd function"""
        self._test_variant(self.func_nojit)
