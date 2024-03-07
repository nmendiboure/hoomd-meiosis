import random
import hoomd
import gsd.hoomd
import os
import shutil
import numpy as np

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
    L = 3*radius + 3
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
    n_poly = 6
    poly_sizes = [24, 24, 36, 36, 48, 48]

    # Direct approach for particle IDs and types
    df_particles = make_polymer(radius=radius, n_poly=n_poly, poly_sizes=poly_sizes)
    n_particles = len(df_particles)
    particles_ids = list(range(n_particles))
    particles_types = ['dna', 'tel', 'dsb']
    particles_typeid = [1 if i == 0 or i == s - 1 else 0 for s in poly_sizes for i in range(s)]
    particles_positions = df_particles[['x', 'y', 'z']].values.tolist()
    homologous_pairs = [[df_particles.iloc[i]['id'], df_particles.iloc[i + s]['id']] for s, size in
                        enumerate(poly_sizes) for i in range(size)]

    n_bonds = n_particles - n_poly
    bonds_group = np.zeros((n_bonds, 2), dtype=int)
    bonds_types = ["dna-telo", "dna-dna", "dna-dsb"]
    bonds_typeid = np.array(
        [bonds_types.index("dna-telo") if i == 0 or i == size - 2 else bonds_types.index("dna-dna")
         for size in poly_sizes for i in range(size - 1)])

    n_angles = n_particles - n_poly * 2
    angles_group = np.zeros((n_angles, 3), dtype=int)
    angles_types = ["dna-dna-dna"]
    angles_typeid = np.array([angles_types.index("dna-dna-dna") for size in poly_sizes for i in range(size - 2)])

    # Optimizing loops for bonds and angles
    start_indices = list(np.cumsum([0] + poly_sizes[:-1]))
    b_counter = 0
    a_counter = 0
    for x, start in enumerate(start_indices):
        x_len = poly_sizes[x]
        for b in range(x_len - 1):
            bonds_group[b_counter] = [start + b, start + b + 1]
            b_counter += 1
        for a in range(x_len - 2):
            angles_group[a_counter] = [start + a, start + a + 1, start + a + 2]
            a_counter += 1

    """
    ############################################
    #####         Hoomd Init Frame         #####
    ############################################
    """

    frame = gsd.hoomd.Frame()
    frame.particles.N = n_particles
    frame.particles.types = particles_types
    frame.particles.typeid = particles_typeid
    frame.particles.position = particles_positions

    frame.bonds.N = n_bonds
    frame.bonds.group = bonds_group
    frame.bonds.typeid = bonds_typeid
    frame.bonds.types = bonds_types

    frame.angles.N = n_angles
    frame.angles.group = angles_group
    frame.angles.typeid = angles_typeid
    frame.angles.types = angles_types

    frame.configuration.box = [L, L, L, 0, 0, 0]
    frame.configuration.dimensions = 3

    snapshot_save_path = "./data/lattice.gsd"
    if os.path.exists(snapshot_save_path):
        os.remove(snapshot_save_path)
    with gsd.hoomd.open(name=snapshot_save_path, mode='x') as f:
        f.append(frame)

