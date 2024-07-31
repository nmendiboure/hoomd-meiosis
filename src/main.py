import os
import time
import hoomd
import gsd.hoomd
import random
import numpy as np

import src.utils as utils
import src.build as build

# CONSTANTS (Global)
PI = np.pi
SEED = 42
NOTICE_LEVEL = 10 if utils.is_debug() else 3
np.random.seed(SEED)
random.seed(SEED)


if __name__ == "__main__":
    """--------------------------------------------
    0 - Initial Parameters
    --------------------------------------------"""

    #TODO : use yaml file to import config and store into a Protocol class

    # Inputs parameters
    N_POLY = 4
    L_POLY = [120, 120, 160, 160]
    N_BREAKS = 8
    RADIUS = 16
    CALIBRATION_TIME = 10
    MIN_DISTANCE_BINDING = 10.2
    PERSISTENCE_LENGTH = 10
    TIMEPOINTS = 10000
    PERIOD = 100
    SIGMA = 1.0

    N_FIRE_BLOCKS = 10
    N_FIRE_STEPS = 1000
    NUM_BLOCKS = 100
    BLOCK_SIZE = 10000


    # Derived parameters
    L_POLY_SINGLE = [s for s, i in enumerate(L_POLY) if i % 2 == 0]
    N_PARTICLES = sum(L_POLY)
    N_BONDS = N_PARTICLES - N_POLY
    N_ANGLES = N_PARTICLES - N_POLY * 2
    L = 3 * RADIUS
    DT_LANGEVIN = 0.001
    DT_FIRE = 0.0005

    #   Create folders
    cwd = os.getcwd()
    data = os.path.join(os.path.dirname(cwd), 'data')
    lattice_init_path = os.path.join(data, "lattice_init.gsd")
    calib_traj_path = os.path.join(data, "calibration.gsd")
    trajectory_path = os.path.join(data, "trajectory.gsd")
    os.makedirs(data, exist_ok=True)
    for path in [lattice_init_path, calib_traj_path, trajectory_path]:
        if os.path.exists(path):
            os.remove(path)

    """--------------------------------------------
    I - Build the chromosomes
    --------------------------------------------"""

    #TODO : determine the best sphere size for the number of particles and
    # then the best box size for the sphere size

    chromosomes_setup = build.set_chromosomes(N_POLY, L_POLY, N_BREAKS, RADIUS, L)
    frame = chromosomes_setup.get('frame', None)
    with gsd.hoomd.open(name=lattice_init_path, mode='x') as f:
        f.append(frame)

    """--------------------------------------------
    II - Initialize the simulation
    --------------------------------------------"""

    device = utils.get_device()
    simulation = hoomd.Simulation(device=device, seed=SEED)
    simulation.create_state_from_gsd(lattice_init_path)

    """--------------------------------------------
    III - Groups
    --------------------------------------------"""
    group_all = hoomd.filter.All()
    group_tel = hoomd.filter.Type(["tel"])
    group_not_tel = hoomd.filter.SetDifference(f=group_all, g=group_tel)

    """--------------------------------------------
    IV - Groups
    --------------------------------------------"""

    """-------------------
    IV_1 - Bonded Forces
    -------------------"""
    # Set up the molecular dynamics simulation
    # Define bond strength and type
    harmonic_bonds = hoomd.md.bond.Harmonic()
    harmonic_bonds.params.default = dict(k=30, r0=1.5, epsilon=1.0, sigma=SIGMA)

    # Define angle energy and type
    # k-parameter acting on the persistence length
    harmonic_angles = hoomd.md.angle.Harmonic()
    harmonic_angles.params['dna-dna-dna'] = dict(k=PERSISTENCE_LENGTH, t0=PI)
    harmonic_angles.params['dna-dsb-dna'] = dict(k=PERSISTENCE_LENGTH, t0=PI)

    """-------------------
    IV_2 - Pairwise Forces
    -------------------"""

    nlist = hoomd.md.nlist.Cell(buffer=0.4)
    shifted_lj = hoomd.md.pair.ForceShiftedLJ(nlist=nlist, default_r_cut=2 ** (1 / 6))
    shifted_lj.params.default = dict(epsilon=1.0, sigma=SIGMA, r_cut=2 ** (1 / 6))

    """-------------------
    IV_3 - External Forces
    -------------------"""
    # Defined a manifold rattle to keep telomere tethered onto the nucleus surface
    sphere = hoomd.md.manifold.Sphere(r=RADIUS, P=(0, 0, 0))
    nve_rattle_telo = hoomd.md.methods.rattle.NVE(filter=group_tel, manifold_constraint=sphere, tolerance=0.01)

    # Define the repulsive wall potential of the nucleus (sphere)
    walls = [hoomd.wall.Sphere(radius=RADIUS, inside=True)]
    shifted_lj_wall = hoomd.md.external.wall.ForceShiftedLJ(walls=walls)
    shifted_lj_wall.params.default = dict(epsilon=1.0, sigma=SIGMA, r_cut=2 ** (1 / 6))

    """-------------------
    IV_4 - Thermostat
    -------------------"""

    langevin = hoomd.md.methods.Langevin(filter=group_not_tel, kT=1)

    """--------------------------------------------
    V - Integrators
    --------------------------------------------"""

    run_integrator = hoomd.md.Integrator(
        dt=DT_LANGEVIN,
        methods=[nve_rattle_telo, langevin],
        forces=[harmonic_bonds, harmonic_angles, shifted_lj, shifted_lj_wall]
    )

    fire_integrator = hoomd.md.minimize.FIRE(
        dt=DT_FIRE,
        force_tol=5e-2,
        angmom_tol=5e-2,
        energy_tol=5e-2,
        forces=[harmonic_bonds, harmonic_angles, shifted_lj, shifted_lj_wall],
        methods=[hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())]
    )

    """--------------------------------------------
    VI - Calibration - Fire
    --------------------------------------------"""
    # perform inital energy minimization with FIRE (fast internal relaxation engine).
    # This helps to remove any overlaps between particles and to relax the initial configuration
    # to avoid large forces that can cause the simulation to crash.

    simulation.operations.integrator = fire_integrator

    # writing is perfomed by the gsd writer object, which saves system states (i.e. snapshots)
    # to a file at regular intervals.
    # These regular intervals are defined by the trigger argument of the writer.
    # Triggers can occur within simulation blocks.
    gsd_optimized_writer = hoomd.write.GSD(
        filename=calib_traj_path,
        trigger=hoomd.trigger.Periodic(N_FIRE_STEPS),
        filter=group_all,
        mode='wb'
    )
    simulation.operations.writers.append(gsd_optimized_writer)

    # hoomd can compute and log thermodynamic properties of the system,
    # such as temperature, pressure, and energy:
    thermodynamic_properties = hoomd.md.compute.ThermodynamicQuantities(filter=group_all)
    simulation.operations.computes.append(thermodynamic_properties)

    # we need to run the simulation (even if for 0 steps) to apply the forces and compute the thermodynamic properties
    simulation.run(0)
    print(f'kin temp = {thermodynamic_properties.kinetic_temperature:.3g}, '
          f'E_P/N = {thermodynamic_properties.potential_energy / N_PARTICLES:.3g}')

    for i in range(N_FIRE_BLOCKS):
        simulation.run(N_FIRE_STEPS)
        print(f'FIRE block #{i + 1} / {N_FIRE_BLOCKS}, '
              f'kin temp = {thermodynamic_properties.kinetic_temperature:.3g}, '
              f'E_P/N = {thermodynamic_properties.potential_energy / N_PARTICLES:.3g}')
        gsd_optimized_writer.write(simulation.state, gsd_optimized_writer.filename)

    # remove the forces and writers from the integrator
    # so that they can be attached to the new integrator
    for _ in range(len(fire_integrator.forces)):
        fire_integrator.forces.pop()
    simulation.operations.writers.pop(0)

    # FIRE reduces the kinetic energy of particles, so we need to re-thermalize the system
    simulation.state.thermalize_particle_momenta(filter=group_all, kT=1.0)

    """--------------------------------------------
    VII - Run the simulation
    --------------------------------------------"""

    simulation.operations.integrator = run_integrator

    # Define the GSD writer to take snapshots every PERIOD steps
    gsd_writer = hoomd.write.GSD(
        filename=trajectory_path,
        trigger=hoomd.trigger.Periodic(BLOCK_SIZE),
        dynamic=['property', 'momentum'],
        filter=group_all,
        mode='wb'
    )

    simulation.operations.writers.append(gsd_writer)
    for i in range(0, NUM_BLOCKS):
        start = time.time()
        simulation.run(BLOCK_SIZE)
        print(f'block #{i+1} / {N_FIRE_BLOCKS}, '
              f'kin temp = {thermodynamic_properties.kinetic_temperature:.3g}, '
              f'E_P/N = {thermodynamic_properties.potential_energy / N_PARTICLES:.3g}')


    # TODO : add RTM (Telomere Rapid Movements)
    # TODO : add synaptonemal complex formation (and disassembly)
    # TODO : extract forces and torques on each particle