"""

    Kenneth Meyer
    10/1/25
    Testing new kernel architecture in CARDIAX!

    For now, this will be limited to updates required to get
    simulations of structural dynamics working in CARDIAX.
    Additionally, we would like to:
        - 'chop' inputs; i.e. stop passing data around that
           we do not need
        -  iterate through user-defined methods only

    Note that we might not really care about only iterating through
    user-defined methods *only*; this only happens during pre-computations
    and isn't really expensive...

"""


import numpy as onp
import numpy.testing as onptest
import jax
import jax.numpy as np
import meshio
import os
import unittest

from cardiax import rectangle_mesh
from cardiax import FiniteElement, Problem, Newton_Solver
from cardiax import get_F

class Subfns:
    """ testing autodiff with class methods

        this emulates the behavior of computing
        alpha-level velocity and acceleration terms
        in CARDIAX, ensuring that the derivatives
        are computed appropriately.

        NOTE: need to make sure all of these ^ are
              functional!
    """
    def y(self,x):
        return x**2

class Test(unittest.TestCase):
    """ Test multiple FE fields
    
        Attempts to define a CARDIAX problem with multiple
        FE fields. This class tests various functionality like:
            - creating a Problem class instance without failing
            - creating a Problem class instance that should fail

    """

    def test_fn_attr(self):
        """ tests if there is a notion of attributes of functions
        """
        
        def my_fn(x):
            return x + 1
        
        a = my_fn

        y = 2
        a.y = y

        # check that 'y' is an attribute in the class, and that it is populated
        assert a.__dict__['y'] is not None

        # check its value
        onptest.assert_allclose(a.__dict__['y'], y)

    def test_autodiff_fn(self):
        """ checks if autodiff can handle nested functions
        """

        def y(x):
            return x ** 2

        def f(x):
            return y(x) ** 2 + x
        
        # normall expect vector-valued input, testing
        # with scalar for now.
        f_prime = jax.grad(f, allow_int=True)
        f_prime_2 = f_prime(2.0)

        onptest.assert_allclose(f_prime_2, 33)

    def test_autodiff_fn_class(self):
        """ checks if autodiff can nested functions from classes

            this better replicates the environment of incorporating
            time-stepping into CARDIAX.
        """

        y_class = Subfns()

        # y(x) is a method/attribute of y_class.
        def f(x):
            return y_class.y(x) ** 2 + x
        
        # normall expect vector-valued input, testing
        # with scalar for now.
        f_prime = jax.grad(f, allow_int=True)
        f_prime_2 = f_prime(2.0)

        onptest.assert_allclose(f_prime_2, 33)


if __name__ == '__main__':
    unittest.main()