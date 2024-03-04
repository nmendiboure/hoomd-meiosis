import random
import math
import hoomd
import gsd.hoomd
import sys
import os
import shutil
import numpy as np
import pandas as pd
import random as rdn
from datetime import datetime

from utils import is_debug

seed = 1999
np.random.seed(seed)
random.seed(seed)


if __name__ == "__main__":
    """
    ############################################
    #####       Initial Parameters         #####
    ############################################
    """
    nb_breaks = 10  # nb of DNA breaks
    timepoints = 500    # total number of timestep
    dt = 0.005    # timestep
    radius = 12.0
    calibration_time = 10
    min_distance_binding = 10.2  # required dist for homologous binding
    persistence_length = 10
    period = 1000   # periodic trigger
    mode = 'cpu'    # run on cpu or gpu
    rd_pos = False  # place the polymer randomly or not
    debug = False   # debug mode (useful on IDE)
    notice_level = 2

    if is_debug():
        #   if the debug mode is detected,
        #   increase the notice level
        debug = True
        notice_level = 10

    #   Create folders
    cwd = os.getcwd()
    data = os.path.join(cwd, 'data')
    # simu_id = datetime.now().strftime("%Y:%m:%d-%H%M%S")
    simu_id = "test"
    simu_dir = os.path.join(data, simu_id)
    if os.path.exists(simu_dir):
        shutil.rmtree(simu_dir)
    os.makedirs(simu_dir)

    """
    ############################################
    #####    Atoms Attributes Setup    #####
    ############################################
    """

    # import the particle's attributes
    df_atoms = pd.read_csv('./data/atoms.tsv', sep="\t")
    df_bonds = pd.read_csv('./data/bonds.tsv', sep="\t")
    df_angles = pd.read_csv('./data/angles.tsv', sep="\t")

    atoms_types = {0: 'dna', 1: 'tel', 2: 'dsb'}
    n_atoms = len(df_atoms)
    n_polymers = len(df_atoms['molecule'].unique())
    polymers_sizes = [len(df_atoms[df_atoms['molecule'] == x]) for x in range(1, n_polymers + 1)]
    n_tel = n_polymers * 2
    n_dna = n_atoms - n_tel
    n_bonds = len(df_bonds)
    n_angles = len(df_angles)
    polymers_global_ids, polymers_local_ids, polymers_coords = [], [], []
    polymers_bonds, polymers_bonds_types = [], []
    polymers_angles, polymers_angles_types = [], []
    polymers_atoms_types = [[1] + [0] * (s-2) + [1] for s in polymers_sizes]

    for x in range(1, n_polymers+1):
        global_ids = df_atoms[df_atoms['molecule'] == x]['id'].astype(int).tolist()
        local_ids = df_atoms[df_atoms['molecule'] == x]['type'].astype(int).tolist()
        coords = list(df_atoms[df_atoms['molecule'] == x][['x', 'y', 'z']].values)
        bonds = df_bonds[df_bonds['molecule'] == x][['atom1', 'atom2']].values.astype(int).tolist()
        bonds_types = df_bonds[df_bonds['molecule'] == x]['type'].values.astype(int).tolist()
        angles = df_angles[df_angles['molecule'] == x][['atom1', 'atom2', 'atom3']].values.astype(int).tolist()
        angles_types = df_angles[df_angles['molecule'] == x]['type'].values.astype(int).tolist()
        polymers_global_ids.append(global_ids)
        polymers_local_ids.append(local_ids)
        polymers_coords.append(coords)
        polymers_bonds.append(bonds)
        polymers_bonds_types.append(bonds_types)
        polymers_angles.append(angles)

    pass

