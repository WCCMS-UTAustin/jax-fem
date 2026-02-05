import unittest
import os
parent  = os.getcwd()

suite = unittest.TestLoader().discover(parent)
print(f"Found test suite = {suite}")
unittest.TextTestRunner(verbosity=2).run(suite)
