import random
import hoomd
import gsd.hoomd
import os
import numpy as np

import init
import utils

# CONSTANTS (Global)
PI = np.pi
SEED = 42
NOTICE_LEVEL = 2
np.random.seed(SEED)
random.seed(SEED)
if utils.is_debug():
    #   if the debug mode is detected,
    #   increase the notice level
    debug = True
    NOTICE_LEVEL = 10


if __name__ == "__main__":

    """
    --------------------------------------------
    Initial Parameters
    --------------------------------------------
    """

    # Inputs parameters
    N_POLY = 4
    L_POLY = [50, 50, 60, 60]
    N_BREAKS = 8
    RADIUS = 16.0
    CALIBRATION_TIME = 10
    MIN_DISTANCE_BINDING = 10.2
    PERSISTENCE_LENGTH = 10
    TIMEPOINTS = 10000
    PERIOD = 100
    SIGMA = 1.0

    # Derived parameters
    L_POLY_SINGLE = [s for s, i in enumerate(L_POLY) if i % 2 == 0]
    N_PARTICLES = sum(L_POLY)
    N_BONDS = N_PARTICLES - N_POLY
    N_ANGLES = N_PARTICLES - N_POLY * 2
    L = 3 * RADIUS
    DT = 0.001

    """
    --------------------------------------------
    Folder Setup
    --------------------------------------------
    """

    #   Create folders
    cwd = os.getcwd()
    data = os.path.join(os.path.dirname(cwd), 'data')
    lattice_init_path = os.path.join(data, "lattice_init.gsd")
    os.makedirs(data, exist_ok=True)
    if os.path.exists(lattice_init_path):
        os.remove(lattice_init_path)

    """
    --------------------------------------------
    Atoms Attributes Setup
    --------------------------------------------
    """

    particles_ids = list(range(N_PARTICLES))
    particles_types = ['dna', 'tel', 'dsb']
    particles_typeid = []
    for s in L_POLY:
        for i in range(s):
            particles_typeid.append(1 if i == 0 or i == s - 1 else 0)
    particles_typeid = np.array(particles_typeid)
    telomeres_ids = np.where(particles_typeid == 1)[0]
    not_telomeres_ids = np.where(particles_typeid == 0)[0]

    particles_positions = []
    for size in L_POLY:
        positions = init.place_polymer_in_sphere(RADIUS - 0.5, size)
        particles_positions.extend(positions)
    particles_positions = np.array(particles_positions)

    homologous_pairs = []
    counter = 0
    l_poly_single = [s for s, i in enumerate(L_POLY) if i % 2 == 0]
    for s, size in enumerate(l_poly_single):
        for ll in range(size):
            homologous_pairs.append([counter + ll, counter + ll + size])
        counter += 2 * size
    homologous_pairs = np.array(homologous_pairs)

    N_BONDS = N_PARTICLES - N_POLY
    bonds_group = np.zeros((N_BONDS, 2), dtype=int)
    bonds_types = ["dna-telo", "dna-dna", "dna-dsb", "dsb-dsb"]
    bonds_typeid = np.array(
        [bonds_types.index("dna-telo") if i == 0 or i == size - 2 else bonds_types.index("dna-dna")
         for size in L_POLY for i in range(size - 1)])

    N_ANGLES = N_PARTICLES - N_POLY * 2
    angles_group = np.zeros((N_ANGLES, 3), dtype=int)
    angles_types = ["dna-dna-dna", "dna-dsb-dna"]
    angles_typeid = np.array([angles_types.index("dna-dna-dna") for size in L_POLY for i in range(size - 2)])

    # Optimizing loops for bonds and angles
    start_indices = list(np.cumsum([0] + L_POLY[:-1]))
    b_counter = 0
    a_counter = 0
    for x, start in enumerate(start_indices):
        x_len = L_POLY[x]
        for b in range(x_len - 1):
            bonds_group[b_counter] = [start + b, start + b + 1]
            b_counter += 1
        for a in range(x_len - 2):
            angles_group[a_counter] = [start + a, start + a + 1, start + a + 2]
            a_counter += 1

    breaks_ids = np.random.choice(not_telomeres_ids, N_BREAKS, replace=False)
    particles_typeid[breaks_ids] = particles_types.index('dsb')
    homologous_breaks_pairs = homologous_pairs[np.where(np.isin(homologous_pairs, breaks_ids))[0]]

    """
    --------------------------------------------
    Hoomd Init Frame
    --------------------------------------------
    """

    frame = gsd.hoomd.Frame()
    frame.particles.N = N_PARTICLES
    frame.particles.types = particles_types
    frame.particles.typeid = particles_typeid
    frame.particles.position = particles_positions

    frame.bonds.N = N_BONDS
    frame.bonds.group = bonds_group
    frame.bonds.typeid = bonds_typeid
    frame.bonds.types = bonds_types

    frame.angles.N = N_ANGLES
    frame.angles.group = angles_group
    frame.angles.typeid = angles_typeid
    frame.angles.types = angles_types

    frame.configuration.box = [L, L, L, 0, 0, 0]
    frame.configuration.dimensions = 3

    with gsd.hoomd.open(name=lattice_init_path, mode='x') as f:
        f.append(frame)

    """
    --------------------------------------------
    Initialize the simulation
    --------------------------------------------    
    """

    device = utils.get_device()
    simulation = hoomd.Simulation(device=device, seed=SEED)
    simulation.create_state_from_gsd(lattice_init_path)
    integrator = hoomd.md.Integrator(dt=DT)

    """
    --------------------------------------------
    Define Forces
    --------------------------------------------
    """

    # Set up the molecular dynamics simulation
    # Define bond strength and type
    harmonic_bonds = hoomd.md.bond.Harmonic()
    harmonic_bonds.params['dna-dna'] = dict(k=800, r0=1)
    harmonic_bonds.params['dna-dsb'] = dict(k=800, r0=1)
    harmonic_bonds.params['dna-telo'] = dict(k=800, r0=1)
    harmonic_bonds.params['dsb-dsb'] = dict(k=80, r0=1)
    integrator.forces.append(harmonic_bonds)

    # Define angle energy and type
    # k-parameter acting on the persistence length
    harmonic_angles = hoomd.md.angle.Harmonic()
    harmonic_angles.params['dna-dna-dna'] = dict(k=PERSISTENCE_LENGTH, t0=PI)
    harmonic_angles.params['dna-dsb-dna'] = dict(k=PERSISTENCE_LENGTH, t0=PI)
    integrator.forces.append(harmonic_angles)

    #   Define pairwise interactions
    group_all = hoomd.filter.All()
    group_tel = hoomd.filter.Type(["tel"])
    group_not_tel = hoomd.filter.SetDifference(f=group_all, g=group_tel)

    nlist = hoomd.md.nlist.Cell(buffer=0.4)
    #   Computes the radially shifted Lennard-Jones pair force on every particle in the simulation state
    shifted_lj = hoomd.md.pair.ForceShiftedLJ(nlist=nlist, default_r_cut=1.5)
    shifted_lj.params[('dna', 'dna')] = dict(epsilon=1.0, sigma=SIGMA)
    shifted_lj.params[('dna', 'dsb')] = dict(epsilon=1.0, sigma=SIGMA)
    shifted_lj.params[('dna', 'tel')] = dict(epsilon=1.0, sigma=SIGMA)
    shifted_lj.params[('tel', 'tel')] = dict(epsilon=1.0, sigma=SIGMA)
    shifted_lj.params[('tel', 'dsb')] = dict(epsilon=1.0, sigma=SIGMA)
    shifted_lj.params[('dsb', 'dsb')] = dict(epsilon=1.0, sigma=SIGMA)

    shifted_lj.r_cut[('dna', 'dna')] = 2 ** (1.0 / 6.0)
    shifted_lj.r_cut[('dna', 'dsb')] = 2 ** (1.0 / 6.0)
    shifted_lj.r_cut[('dna', 'tel')] = 2 ** (1.0 / 6.0)
    shifted_lj.r_cut[('tel', 'tel')] = 2 ** (1.0 / 6.0)
    shifted_lj.r_cut[('tel', 'dsb')] = 2 ** (1.0 / 6.0)
    shifted_lj.r_cut[('dsb', 'dsb')] = 2 ** (1.0 / 6.0)
    integrator.forces.append(shifted_lj)

    sphere = hoomd.md.manifold.Sphere(r=RADIUS, P=(0, 0, 0))
    nve_rattle_telo = hoomd.md.methods.rattle.NVE(filter=group_tel, manifold_constraint=sphere, tolerance=0.01)
    integrator.methods.append(nve_rattle_telo)

    langevin = hoomd.md.methods.Langevin(filter=group_not_tel, kT=1)
    langevin.gamma['dna'] = 0.2
    langevin.gamma['dsb'] = 0.2
    integrator.methods.append(langevin)

    # Define the repulsive wall potential
    walls = [hoomd.wall.Sphere(radius=RADIUS, inside=True)]
    shifted_lj_wall = hoomd.md.external.wall.ForceShiftedLJ(walls=walls)
    shifted_lj_wall.params["dna"] = {"epsilon": 1.0, "sigma": SIGMA, "r_cut": 2**(1/6)}
    shifted_lj_wall.params["dsb"] = {"epsilon": 1.0, "sigma": SIGMA, "r_cut": 2**(1/6)}
    shifted_lj_wall.params["tel"] = {"epsilon": 1.0, "sigma": SIGMA, "r_cut": 2**(1/6)}
    integrator.forces.append(shifted_lj_wall)

    simulation.operations.integrator = integrator

    """
    --------------------------------------------
    Writer Setup
    --------------------------------------------
    """

    # Define the GSD writer to take snapshots every PERIOD steps
    gsd_writer = hoomd.write.GSD(
        filename=os.path.join(data, "trajectory.gsd"),
        trigger=hoomd.trigger.Periodic(PERIOD),
        mode='wb',
        filter=hoomd.filter.All()
    )
    simulation.operations.writers.append(gsd_writer)

    #   Calibration
    print("Calibrating...")
    simulation.run(100000)

