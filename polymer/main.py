import os
import random
import numpy as np
from os.path import join

from polymer import Chromosome
from lsimu import LSimu
from constraint import Sphere, Point
from halley.constraint import Spherical
from halley.vectors import V

np.random.seed(42)
random.seed(42)

if __name__ == "__main__":
    radius = 12
    n_poly = 4
    poly_sizes = [24, 24, 36, 36, 48, 48]

    nucleus = Sphere(position=[0, 0, 0], radius=radius)
    simu = LSimu()

    for x in range(n_poly):
        uid = x+1
        s_nucleus = Spherical(V(0, 0, 0), radius=radius*0.99)
        telo1 = Point(index=0, position=s_nucleus.get_random()._v)
        telo2 = Point(index=poly_sizes[x] - 1, position=s_nucleus.get_random()._v)

        new_chr = Chromosome(
            uid=uid,
            size=poly_sizes[x],
            l_constraints=[telo1, telo2],
            g_constraints=[nucleus],
            shape="linear"
        )

        simu.add(new_chr)

    simu.to_csv("../hoomd4/data/")


