
import functools

def method_wrapper(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        self._allowed_to_set = True  # Allow setting attributes within the method
        try:
            result = func(self, *args, **kwargs)
        finally:
            self._allowed_to_set = False  # Disallow setting attributes after the method
        return result
    return wrapper

class MethodWrappingMeta(type):
    def __new__(cls, name, bases, dct):
        for key, value in dct.items():
            if callable(value) and not key.startswith('__'):
                dct[key] = method_wrapper(value)
        return super().__new__(cls, name, bases, dct)
