import yaml
import os


class Protocol(object):

    def __init__(self, path):
        self.__SEED = 42
        self.__N_POLY = 4
        self.__L_POLY = [120, 120, 160, 160]
        self.__N_BREAKS = 8

