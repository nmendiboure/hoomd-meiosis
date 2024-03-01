import numpy as np
from utils import norm, generateV


class Constrain:
    def __init__(self):
        pass

    def is_inside(pt):
        return True


class Sphere(Constrain):
    def __init__(self, position=[0, 0, 0], radius=1):
        Constrain.__init__(self)
        self.position = position
        self.radius = np.array(radius)

    def is_inside(self, pt, epsilon=1e-7):
        if norm(self.position-np.array(pt)) <= self.radius + epsilon:
            return True
        else:
            return False

    def generate(self):
        r = self.radius * np.random.random()
        return self.position + r*generateV()

    def rescale(self, v=1):
        for i in range(3):
            self.position[i] *= v
        self.radius *= v


class Point:
    def __init__(self, index=0, position=[0, 0, 0]):
        self.index = index
        self.position = position

    def __repr__(self):
        return "%i %s" % (self.index, str(self.position))
