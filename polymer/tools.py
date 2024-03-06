import numpy as np
import bisect
import random
from math import sin, cos


def init_proba_log(P, dx, lrange=[0, 3.14]):
    """
    Generate a uni dimensional probability
    by creating a cumulative sum following the P law
    P must depend only on one variable x
    """

    x = np.arange(lrange[0], lrange[1], dx)

    Plist = np.array([P(xx) for xx in x])
    # print Plist

    Plist -= max(Plist)
    Plist = np.exp(Plist)
    # print Plist

    norm = sum(Plist)
    Plist /= norm
    # print "Norm" , norm
    try:
        index = [Plist[0]]
    except:
        print((lrange[0], lrange[1], dx))
    for w, xx in enumerate(x[1:]):
        index.append(index[-1] + Plist[w + 1])
    return x, index


def generate_point_proba(x, index):
    try:
        u = np.random.rand()
        i = bisect.bisect(index, u)
        # print index[i-1],u,index[i]
        return x[i]
    except:
        print("Constrain to high")
        raise


def generateV():
    theta = random.uniform(0, 2 * 3.14)
    phi = random.uniform(0, 3.14)
    return np.array([cos(theta) * sin(phi), sin(theta) * sin(phi), cos(phi)])


def norm(v):
    n = 0
    for el in v:
        n += el * el
    return np.sqrt(n)