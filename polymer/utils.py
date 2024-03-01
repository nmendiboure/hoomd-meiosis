import random
import numpy as np
from math import sin, cos, sqrt


def generateV():
    theta = random.uniform(0, 2 * 3.14)
    phi = random.uniform(0, 3.14)
    return np.array([cos(theta) * sin(phi), sin(theta) * sin(phi), cos(phi)])


def norm(v):
    n = 0
    for el in v:
        n += el * el
    return np.sqrt(n)
