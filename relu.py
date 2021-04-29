from matplotlib import pyplot
from numpy import ndarray
from typing import Callable
import numpy as np


def square(x: ndarray) -> ndarray:
    """Square each element in the array"""
    return np.power(x, 2)


def relu(x: ndarray) -> ndarray:
    """Apply 'Leaky relu' to each element in the array """
    return np.maximum(0.2 * x, x)


x = np.array([n for n in range(-10, 11)])
y = relu(x)


def f(a: Callable[[str], str]) -> str:
    """Run a function  """
    return a("kofi")


def c(b: str) -> str:
    return b + "World"


print(f(c))
# pyplot.plot(x, y)

# pyplot.show()
