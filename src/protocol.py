import yaml
import os
import sys
import random
import numpy as np


class Protocol(object):

    def __init__(self, path):
        self.seed = None
        self.n_poly = None
        self.l_poly = None
        self.n_breaks = None
        self.persistence_length = None
        self.density = None
        self.sigma = None
        self.rtm_prob = None
        self.rtm_magnitude = None
        self.rtm_period = None
        self.n_fire_blocks = None
        self.n_fire_steps = None
        self.n_run_blocks = None
        self.n_run_steps = None
        self.run_dump_period = None
        self.dt_langevin = None
        self.dt_fire = None
        self.blender = None
        self.force_fire = None

        self._load(path)

        if self.seed is None:
            ValueError(f"Please provide a seed for the random number generator.")
            sys.exit(1)

        random.seed(self.seed)
        np.random.seed(self.seed)

    def _load(self, path):
        if not os.path.exists(path):
            ValueError(f"Impossible to find the configuration file '{path}'.")
            sys.exit(1)

        if not path.endswith('.yaml'):
            ValueError(f"Please provide a valid configuration file in yaml format.")
            sys.exit(1)

        with open(path, 'r') as file:
            data = yaml.safe_load(file)

        for attr, value in data.items():
            setattr(self, attr, value)
