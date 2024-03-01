import random
import math
import hoomd
import gsd.hoomd
import sys
import os
import numpy as np
import random as rdn


"""
############################################
#####             Functions            #####
############################################
"""


def is_debug() -> bool:
    """
    Function to see if the script is running in debug mode.
    """
    gettrace = getattr(sys, 'gettrace', None)

    if gettrace is None:
        return False
    else:
        v = gettrace()
        if v is None:
            return False
        else:
            return True


def init_grid(
        r: int | float,
        n: int,
        len_max: int | float
):
    #   Creation of the grid for initialization
    cote = r / math.sqrt(2)
    x_min = -cote / 2
    x_pas = cote / (n - 1)
    x_max = cote / 2
    y_min = -cote / 2
    y_pas = cote / (len_max - 1)
    y_max = cote / 2
    z_min = -cote / 2
    z_pas = cote / (n - 1)
    z_max = cote / 2

    x_list = np.arange(x_min, x_max + x_pas, x_pas)
    y_list = np.arange(y_min, y_max + y_pas, y_pas)
    z_list = np.arange(z_min, z_max + z_pas, z_pas)

    return x_list, y_list, z_list


def get_random_breaks(
        particles_id: list | np.ndarray,
        polymers_lengths: list | np.ndarray,
        dna_id: int,
        num_break: int,
        min_separation: int,
):

    particles_index = np.arange(0, len(particles_id))
    particles_by_polymer = np.concatenate([np.repeat(i, lp) for i, lp in enumerate(polymers_lengths)])
    selected = []
    dna_particles = sum(particles_id[particles_id == 1])
    breaks_per_polymer = [round((pl-2)/dna_particles * num_break) for pl in polymers_lengths]

    for ii in range(len(polymers_lengths)):
        if (polymers_lengths[ii]-2) // breaks_per_polymer[ii] < min_separation*2:
            raise Exception("Cannot interspace {0} breaks every {1} monomers on {2} total polymers long".format(
                breaks_per_polymer[ii], min_separation, sum(particles_by_polymer[particles_by_polymer == ii])))
        bk = 0
        while bk < breaks_per_polymer[ii]:
            p = np.random.choice(particles_index[np.where(particles_by_polymer == ii)[0]])
            if particles_id[p] == dna_id:
                if all(abs(p - s) > min_separation for s in selected):
                    selected.append(p)
                    bk += 1

    return selected


if __name__ == "__main__":
    """
    ############################################
    #####       Initial Parameters         #####
    ############################################
    """
    nb_breaks = 10  # nb of DNA breaks
    timepoints = 500    # total number of timestep
    dt = 0.005    # timestep
    r_nuc = 12.0
    calibration_time = 10
    min_distance_binding = 10.2  # required dist for homologous binding
    persistence_length = 10
    period = 1000   # periodic trigger
    seed = 1
    mode = 'cpu'    # run on cpu or gpu
    rd_pos = False  # place the polymer randomly or not
    debug = False   # debug mode (useful on IDE)
    notice_level = 2    # degrees of notification during the simulation

    if is_debug():
        #   if the debug mode is detected,
        #   increase the notice level
        debug = True
        notice_level = 10

    if seed is not None:
        #   if a seed is given, fix the random generators
        #   for both numpy and random lib
        np.random.seed(seed)
        random.seed(seed)

    #   Create folders
    cwd = os.getcwd()
    output_dir = os.path.join(cwd, 'outputs')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    """
    ############################################
    #####    Particles Attributes Setup    #####
    ############################################
    """
    #   Particles
    polymers_sizes = [i for i in [30, 30] for _ in range(2)]
    N = sum(polymers_sizes) + 1
    particles_types = ['BOX', 'DNA', 'TELO', 'FH']
    nb_telo = len(polymers_sizes) * 2
    nb_dna = N - nb_telo - 1
    particles_proportions = {'BOX': 1, 'DNA': nb_dna, 'TELO': nb_telo, 'FH': 0}
    particles_typeid = np.array([], dtype=int)
    particles_positions = np.array([[0., 0., 0.]] * N, dtype=float)

    #   Nucleus / Sphere
    R = r_nuc + 1
    L = 3*r_nuc + 3
    x, y, z = init_grid(r=R, n=N, len_max=np.max(polymers_sizes))

    #   Bonds
    nb_bonds = N - len(polymers_sizes) - 1
    bonds_types = ['DNA-DNA', 'DNA-TELO', 'FH-FH']
    bonds_group = np.zeros((nb_bonds, 2), dtype=int)
    bonds_typeid = np.zeros(nb_bonds, dtype=int)

    #   Angles
    nb_angles = N - 2*len(polymers_sizes) - 1
    angles_types = ['DNA']
    angles_group = np.zeros((nb_angles, 3), dtype=int)
    angles_typeid = np.zeros(nb_angles, dtype=int)

    counter_bonds = 0
    counter_angles = 0
    for ii_x, x_len in enumerate(polymers_sizes):
        #   Define particles typeid (0, 1, 2, 3)
        for jt in range(x_len):
            if jt not in [0, x_len - 1]:
                particles_typeid = np.append(particles_typeid, particles_types.index('DNA'))
            else:
                particles_typeid = np.append(particles_typeid, particles_types.index('TELO'))

        #   Define particles positions (randomly or not)
        start = sum(polymers_sizes[:ii_x])
        if rd_pos:
            maxi = 2 * R
            while maxi > R:
                p0 = np.array([0., 0., 0.])
                for i in range(x_len):
                    particles_positions[start + i] = np.copy(p0)
                    p0 += 0.2 * (1 - 2 * np.random.rand(3))
                cm = np.mean(particles_positions, axis=0)
                for i in range(x_len):
                    particles_positions[start + i] -= cm[:]
                maxi = np.max(np.abs(particles_positions))
        else:
            x_idx = rdn.randint(0, len(x) - 1)
            z_idx = rdn.randint(0, len(z) - 1)
            for i in range(x_len):
                particles_positions[start + i] = [x[x_idx], y[i], z[z_idx]]
            x = np.delete(x, x_idx)
            z = np.delete(z, z_idx)

        #   Define bonds
        for b in range(x_len-1):
            bonds_group[counter_bonds][0] = start + b
            bonds_group[counter_bonds][1] = start + b + 1
            bonds_typeid[counter_bonds] = bonds_types.index('DNA-DNA')
            counter_bonds += 1

        #   Define angles
        for a in range(x_len-2):
            angles_group[counter_angles][0] = start + a
            angles_group[counter_angles][1] = start + a + 1
            angles_group[counter_angles][2] = start + a + 2
            angles_typeid[counter_angles] = angles_types.index('DNA')
            counter_angles += 1

    particles_typeid = np.append(particles_typeid, particles_types.index('BOX'))

    #   Define Homologous pairs
    homologous = []
    counter = 0
    for jj_x, x_size in enumerate(np.unique(polymers_sizes)):
        for ll in range(1, x_size - 1):
            homologous.append([counter + ll, counter + ll + x_size])
        counter += 2 * x_size
    homologous = np.array(homologous)

    #   Inflict n breaks at random positions for DNA particles
    broken_particles_idx = get_random_breaks(
        particles_id=particles_typeid,
        polymers_lengths=polymers_sizes,
        num_break=nb_breaks,
        dna_id=particles_types.index('DNA'),
        min_separation=4)
    #   Change id from 'DNA' to 'FH' for selected broken particles
    particles_typeid[broken_particles_idx] = particles_types.index('FH')
    homologous_with_breaks = homologous[np.where(np.isin(homologous, broken_particles_idx))[0]]

    """
    ############################################
    #####         Make the snapshot        #####
    ############################################
    """

    frame = gsd.hoomd.Frame()
    frame.particles.N = N
    frame.particles.types = particles_types
    frame.particles.typeid = particles_typeid
    frame.particles.position = particles_positions

    frame.bonds.types = bonds_types
    frame.bonds.N = nb_bonds
    frame.bonds.group = bonds_group
    frame.bonds.typeid = bonds_typeid
    frame.angles.types = angles_types
    frame.angles.N = nb_angles
    frame.angles.group = angles_group
    frame.angles.typeid = angles_typeid

    frame.configuration.box = [L, L, L, 0, 0, 0]

    snapshot_save_path = os.path.join(output_dir, 'lattice.gsd')
    if os.path.exists(snapshot_save_path):
        os.remove(snapshot_save_path)
    with gsd.hoomd.open(name=snapshot_save_path, mode='x') as f:
        f.append(frame)

    particles = frame.particles
    bonds = frame.bonds
    angles = frame.angles

    del x, y, z, a, b, i, jt, jj_x, ll, x_size, x_len, x_idx, z_idx, L, R, start, counter

    """
    ############################################
    #####   Initialize the simulation      #####
    ############################################
    """

    if mode == 'cpu':
        device = hoomd.device.CPU(notice_level=notice_level)
    elif mode == 'gpu':
        device = hoomd.device.GPU(notice_level=notice_level)
    else:
        raise Exception("mode must be 'cpu' or 'gpu'")

    sim = hoomd.Simulation(device=device, seed=seed)
    sim.create_state_from_gsd(snapshot_save_path)
    integrator = hoomd.md.Integrator(dt=dt)

    """
    ############################################
    #####            Define Forces         #####
    ############################################
    """

    # Set up the molecular dynamics simulation
    # Define bond strength and type
    harmonic_bonds = hoomd.md.bond.Harmonic()
    harmonic_bonds.params['DNA-DNA'] = dict(k=800, r0=1)
    harmonic_bonds.params['DNA-TELO'] = dict(k=800, r0=1)
    harmonic_bonds.params['FH-FH'] = dict(k=80, r0=1)
    integrator.forces.append(harmonic_bonds)

    # Define angle energy and type
    # k-parameter acting on the persistence length
    harmonic_angles = hoomd.md.angle.Harmonic()
    harmonic_angles.params['DNA'] = dict(k=persistence_length, t0=math.pi)
    integrator.forces.append(harmonic_angles)

    #   Define pairwise interactions
    telo_group = hoomd.filter.Type(["TELO"])
    box_group = hoomd.filter.Type(["BOX"])
    group_not_telo = hoomd.filter.SetDifference(f=hoomd.filter.All(), g=telo_group)
    first_homologous_group = hoomd.filter.Type(["FH"])
    group_not_box = hoomd.filter.SetDifference(f=hoomd.filter.All(), g=box_group)

    cell = hoomd.md.nlist.Cell(buffer=0.4)
    #   Computes the radially-shifted Lennard-Jones pair force on every particle in the simulation state
    shifted_lj = hoomd.md.pair.ForceShiftedLJ(nlist=cell, default_r_cut=1.5)
    shifted_lj.params[('DNA', 'DNA')] = dict(epsilon=1.0, sigma=1)
    shifted_lj.params[('DNA', 'BOX')] = dict(epsilon=1.0, sigma=0)
    shifted_lj.params[('DNA', 'FH')] = dict(epsilon=1.0, sigma=1.0)
    shifted_lj.params[('DNA', 'TELO')] = dict(epsilon=1.0, sigma=1.0)
    shifted_lj.params[('TELO', 'BOX')] = dict(epsilon=1.0, sigma=0)
    shifted_lj.params[('TELO', 'TELO')] = dict(epsilon=1.0, sigma=1.0)
    shifted_lj.params[('TELO', 'FH')] = dict(epsilon=1.0, sigma=1.0)
    shifted_lj.params[('FH', 'FH')] = dict(epsilon=1.0, sigma=1.0)
    shifted_lj.params[('FH', 'BOX')] = dict(epsilon=1.0, sigma=0)
    shifted_lj.params[('BOX', 'BOX')] = dict(epsilon=1.0, sigma=0)

    shifted_lj.r_cut[('DNA', 'DNA')] = 2**(1.0 / 6.0)
    shifted_lj.r_cut[('DNA', 'DNA')] = 2**(1.0 / 6.0)
    shifted_lj.r_cut[('DNA', 'BOX')] = 2**(1.0 / 6.0)
    shifted_lj.r_cut[('DNA', 'FH')] = 2**(1.0 / 6.0)
    shifted_lj.r_cut[('DNA', 'TELO')] = 2**(1.0 / 6.0)
    shifted_lj.r_cut[('TELO', 'BOX')] = 2**(1.0 / 6.0)
    shifted_lj.r_cut[('TELO', 'TELO')] = 2**(1.0 / 6.0)
    shifted_lj.r_cut[('TELO', 'FH')] = 2**(1.0 / 6.0)
    shifted_lj.r_cut[('FH', 'FH')] = 2**(1.0 / 6.0)
    shifted_lj.r_cut[('FH', 'BOX')] = 2**(1.0 / 6.0)
    shifted_lj.r_cut[('BOX', 'BOX')] = 2**(1.0 / 6.0)

    sphere = hoomd.wall.Sphere(radius=r_nuc, origin=(0., 0., 0.), inside=True)
    walls = [sphere]
    shifted_lj_wall = hoomd.md.external.wall.ForceShiftedLJ(walls)
    # repulsive interaction because cut at the minimum value
    shifted_lj_wall.params[particles_types] = dict(epsilon=1, sigma=2, r_cut=2**(1 / 6))
    shifted_lj_wall.params["TELO"] = dict(epsilon=2, sigma=1, r_extrap=0, r_cut=3)

    #   Record trajectories
    save_path = os.path.join(output_dir, 'poly_d.gsd')
    if os.path.exists(save_path):
        os.remove(save_path)
    hoomd.write.GSD.write(state=sim.state, mode='xb', filter=hoomd.filter.All(), filename=save_path)

    integrator.forces.append(shifted_lj)
    integrator.forces.append(shifted_lj_wall)

    langevin = hoomd.md.methods.Langevin(filter=group_not_box, kT=1)
    langevin.gamma['TELO'] = 0.2
    integrator.methods.append(langevin)

    # Assign the integrator to the simulation
    sim.operations.integrator = integrator

    #   Calibration
    sim.run(1000)
    calibration_save_path = os.path.join(output_dir, 'calibration.gsd')
    if os.path.exists(calibration_save_path):
        os.remove(calibration_save_path)
    hoomd.write.GSD.write(state=sim.state, filename=calibration_save_path, mode='x')

    """
    ############################################
    #####             Simulation           #####
    ############################################
    """

    sim.run(1000)

    # forces = np.zeros((len(homologous), timepoints))
    # pairing_state = {(h1, h2): False for (h1, h2) in homologous}
    # pos_xyz = np.zeros((timepoints, N, 3))

    """
    ############################################
    #####            Saving data           #####
    ############################################
    """

    simulation_save_path = os.path.join(output_dir, 'simulation.gsd')
    if os.path.exists(simulation_save_path):
        os.remove(simulation_save_path)
    hoomd.write.GSD.write(state=sim.state, filename=simulation_save_path, mode='xb')

    forces_dir = os.path.join(output_dir, 'forces')
    distances_dir = os.path.join(output_dir, 'distances')
    positions_dir = os.path.join(output_dir, 'positions')

    if not os.path.exists(forces_dir):
        os.makedirs(forces_dir)
    if not os.path.exists(distances_dir):
        os.makedirs(distances_dir)
    if not os.path.exists(positions_dir):
        os.makedirs(positions_dir)

    # np.save(forces_dir+"/forces_simulation.npy", forces)
    # np.save(forces_dir+"/broken_homologous_pairs.npy", homologous_with_breaks)
    # np.save(positions_dir+"/positions.npy", pos_xyz)
    # np.save(positions_dir+"/polymers_sizes.npy", polymers_sizes)

    print("\n ")
    print("################\n")
    print(" ---- DONE ---- \n")
    print("################\n")
    print("\n")
