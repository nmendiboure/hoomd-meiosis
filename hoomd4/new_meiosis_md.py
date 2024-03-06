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
from polymer.poly import make_polymer

seed = 42
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
    n_poly = 6
    poly_sizes = [24, 24, 36, 36, 48, 48]
    df_atoms = make_polymer(radius=radius, n_poly=n_poly, poly_sizes=poly_sizes)
    n_atoms = len(df_atoms)
    n_bonds = n_atoms - n_poly
    n_angles = n_atoms - n_poly * 2

    n_mol = len(df_atoms['molecule'].unique())
    n_tel = n_mol * 2
    n_dna = n_atoms - n_tel
    mol_sizes = [len(df_atoms[df_atoms['molecule'] == x]) for x in range(1, n_mol + 1)]

    atoms_types = {0: 'dna', 1: 'tel', 2: 'dsb'}

    mol_global_ids, mol_local_ids, mol_coords = [], [], []
    mol_bonds, mol_bonds_types = [], []
    # mol_angles, mol_angles_types = [], []
    mol_atoms_types = [[1] + [0] * (s-2) + [1] for s in mol_sizes]

    for x in range(1, n_mol+1):
        global_ids = df_atoms[df_atoms['molecule'] == x]['id'].astype(int).tolist()
        local_ids = df_atoms[df_atoms['molecule'] == x]['type'].astype(int).tolist()
        coords = list(df_atoms[df_atoms['molecule'] == x][['x', 'y', 'z']].values)
        # bonds = df_bonds[df_bonds['molecule'] == x][['atom1', 'atom2']].values.astype(int).tolist()
        # bonds_types = df_bonds[df_bonds['molecule'] == x]['type'].values.astype(int).tolist()
        # angles = df_angles[df_angles['molecule'] == x][['atom1', 'atom2', 'atom3']].values.astype(int).tolist()
        # angles_types = df_angles[df_angles['molecule'] == x]['type'].values.astype(int).tolist()
        mol_global_ids.append(global_ids)
        mol_local_ids.append(local_ids)
        mol_coords.append(coords)
        # mol_bonds.append(bonds)
        # mol_bonds_types.append(bonds_types)
        # mol_angles.append(angles)

    pass

