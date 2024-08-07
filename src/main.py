import os
import time
import math
import hoomd
import random
import gsd.hoomd
import numpy as np

import src.utils as utils
import src.build as build
import src.blender as blender
import src.protocol as protocol

PI = math.pi


class ComputeRTM(hoomd.custom.Action):
    def __init__(self, telo_ids, prob, mag, manifold):
        self.telo_ids = telo_ids
        self.prob = prob
        self.mag = mag
        self.manifold = manifold

    def act(self, timestep):
        snap = self._state.get_snapshot()
        if snap.communicator.rank == 0:
            for force in simulation.operations.integrator.forces:
                if isinstance(force, hoomd.md.force.ActiveOnManifold):
                    simulation.operations.integrator.forces.remove(force)

            for telo in self.telo_ids:
                if random.random() < self.prob:
                    one_telo = hoomd.filter.Tags([telo])
                    active_force = hoomd.md.force.ActiveOnManifold(
                        filter=one_telo,
                        manifold_constraint=self.manifold,
                    )

                    force = self._get_force(self.mag)
                    active_force.active_force["tel"] = force
                    simulation.operations.integrator.forces.append(active_force)

    @staticmethod
    def _get_force(mag):
        dir_vec = np.random.uniform(-1, 1, 3)
        dir_vec /= np.linalg.norm(dir_vec)
        return mag * dir_vec


if __name__ == "__main__":
    """--------------------------------------------
    I - Initialization
    --------------------------------------------"""

    ptc = protocol.Protocol("config.yaml")
    ptc.blender = True
    l_poly_single = [s for s, i in enumerate(ptc.l_poly) if i % 2 == 0]
    n_particles = sum(ptc.l_poly)

    #   Create folders
    cwd = os.getcwd()
    data = os.path.join(os.path.dirname(cwd), 'data')
    lattice_init_path = os.path.join(data, "lattice_init.gsd")
    calib_traj_path = os.path.join(data, "calibration.gsd")
    trajectory_path = os.path.join(data, "trajectory.gsd")
    os.makedirs(data, exist_ok=True)
    for path in [lattice_init_path, trajectory_path]:
        if os.path.exists(path):
            os.remove(path)

    device = utils.get_device()
    simulation = hoomd.Simulation(device=device, seed=ptc.seed)

    """--------------------------------------------
    II - Build the environment
    --------------------------------------------"""

    """-------------------
    II_1 - Box and Sphere
    -------------------"""
    # 1 - create a simulation box
    # 2 - create a sphere inside of the box
    # 3 - make an inscribed box inside of the sphere to place the chromosomes in lattice
    # 4 - increase by little the size of the simulation box to avoid boundary effects

    simBox = math.ceil((n_particles / ptc.density) ** (1 / 3.0))
    simBox = simBox + 1 if simBox % 2 != 0 else simBox
    radius = simBox / 2
    inscribedBox = math.floor(2 * radius / np.sqrt(3))
    simBox += 4

    """-------------------
    II_2 - Chr placement
    -------------------"""

    # Create a sphere in Blender
    if ptc.blender and not os.path.exists(os.path.join(data, f"sphere{radius}.obj")):
        blender.make_sphere(radius, os.path.join(data, f"sphere{radius}.obj"))

    chromosomes_setup = build.set_chromosomes(ptc.n_poly, ptc.l_poly, ptc.n_breaks, simBox, inscribedBox)
    frame = chromosomes_setup.get('frame', None)
    telo_ids = chromosomes_setup.get('telomeres', None)
    with gsd.hoomd.open(name=lattice_init_path, mode='x') as f:
        f.append(frame)

    """-------------------
    II_3 - Groups
    -------------------"""

    group_all = hoomd.filter.All()
    group_telo = hoomd.filter.Type(["tel"])
    group_not_telo = hoomd.filter.SetDifference(f=group_all, g=group_telo)

    """--------------------------------------------
    III - Forces 
    --------------------------------------------"""

    """-------------------
    III_1 - Bonded Forces
    -------------------"""
    # Set up the molecular dynamics simulation
    # Define bond strength and type
    harmonic_bonds = hoomd.md.bond.Harmonic()
    harmonic_bonds.params.default = dict(k=30, r0=1.0, epsilon=1.0, sigma=1.0)

    # Define angle energy and type
    # k-parameter acting on the persistence length
    harmonic_angles = hoomd.md.angle.Harmonic()
    harmonic_angles.params['dna-dna-dna'] = dict(k=ptc.persistence_length, t0=PI)
    harmonic_angles.params['dna-dsb-dna'] = dict(k=ptc.persistence_length, t0=PI)

    """-------------------
    III_2 - Pairwise Forces
    -------------------"""

    nlist = hoomd.md.nlist.Cell(buffer=0.4)
    shifted_lj = hoomd.md.pair.ForceShiftedLJ(nlist=nlist, default_r_cut=2 ** (1 / 6))
    shifted_lj.params.default = dict(epsilon=1.0, sigma=1.0, r_cut=2 ** (1 / 6))

    """-------------------
    III_3 - External Forces
    -------------------"""
    # Defined a manifold rattle to keep telomere tethered onto the nucleus surface
    sphere = hoomd.md.manifold.Sphere(r=radius, P=(0, 0, 0))
    nve_rattle_telo = hoomd.md.methods.rattle.NVE(filter=group_telo, manifold_constraint=sphere, tolerance=1)

    # Define the repulsive wall potential of the nucleus (sphere)
    walls = [hoomd.wall.Sphere(radius=radius, inside=True)]
    shifted_lj_wall = hoomd.md.external.wall.ForceShiftedLJ(walls=walls)
    shifted_lj_wall.params.default = dict(epsilon=1, sigma=1, r_cut=2 ** (1 / 6))

    """-------------------
    III_4 - Thermostat
    -------------------"""

    langevin = hoomd.md.methods.Langevin(filter=group_not_telo, kT=1)

    """-------------------
    III_5 - Thermodinamics
    -------------------"""
    # Define thermodynamic properties computation early on
    # hoomd can compute and log thermodynamic properties of the system,
    # such as temperature, pressure, and energy:
    thermodynamic_properties = hoomd.md.compute.ThermodynamicQuantities(filter=group_all)
    simulation.operations.computes.append(thermodynamic_properties)

    """--------------------------------------------
    IV - Calibration - Fire
    --------------------------------------------"""

    if os.path.exists(calib_traj_path) and not ptc.force_fire:
        print("Calibration already done. Loading the calibration trajectory...")
        simulation.create_state_from_gsd(calib_traj_path)
        print(f"Calibration {calib_traj_path} loaded. \n")
    else:
        if os.path.exists(calib_traj_path):
            os.remove(calib_traj_path)
        simulation.create_state_from_gsd(lattice_init_path)

        fire_integrator = hoomd.md.minimize.FIRE(
            dt=ptc.dt_fire,
            force_tol=5e-2,
            angmom_tol=5e-2,
            energy_tol=5e-2,
            forces=[harmonic_bonds, harmonic_angles, shifted_lj, shifted_lj_wall],
            methods=[hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())]
        )

        # perform inital energy minimization with FIRE (fast internal relaxation engine).
        # This helps to remove any overlaps between particles and to relax the initial configuration
        # to avoid large forces that can cause the simulation to crash.

        simulation.operations.integrator = fire_integrator

        # writing is perfomed by the gsd writer object,
        # which saves system states (i.e. snapshots)
        # to a file at regular intervals.
        # These regular intervals are defined by the trigger argument of the writer.
        # Triggers can occur within simulation blocks.
        gsd_optimized_writer = hoomd.write.GSD(
            filename=calib_traj_path,
            trigger=hoomd.trigger.Periodic(ptc.n_fire_steps),
            filter=group_all,
            mode='xb'
        )
        simulation.operations.writers.append(gsd_optimized_writer)

        print("Calibrating the system...")
        # we need to run the simulation (even if for 0 steps)
        # to apply the forces and compute the thermodynamic properties
        simulation.run(0)
        print(f'kin temp = {thermodynamic_properties.kinetic_temperature:.3g}, '
              f'E_P/N = {thermodynamic_properties.potential_energy / n_particles:.3g}')

        for i in range(ptc.n_fire_blocks):
            simulation.run(ptc.n_fire_steps)
            print(f'FIRE block #{i + 1} / {ptc.n_fire_blocks}, '
                  f'kin temp = {thermodynamic_properties.kinetic_temperature:.3g}, '
                  f'E_P/N = {thermodynamic_properties.potential_energy / n_particles:.3g}')
            gsd_optimized_writer.write(simulation.state, gsd_optimized_writer.filename)

        # remove the forces and writers from the integrator
        # so that they can be attached to the new integrator
        for _ in range(len(fire_integrator.forces)):
            fire_integrator.forces.pop()
        simulation.operations.writers.pop(0)
        print("Calibration done. \n")

    # FIRE reduces the kinetic energy of particles, so we need to re-thermalize the system
    simulation.state.thermalize_particle_momenta(filter=group_all, kT=1.0)

    """--------------------------------------------
    V - Run the simulation
    --------------------------------------------"""

    rtm_action = ComputeRTM(telo_ids, ptc.rtm_prob, ptc.rtm_magnitude, sphere)
    custom_updater = hoomd.update.CustomUpdater(action=rtm_action, trigger=hoomd.trigger.Periodic(ptc.rtm_period))
    simulation.operations.updaters.append(custom_updater)

    run_integrator = hoomd.md.Integrator(
        dt=ptc.dt_langevin,
        methods=[nve_rattle_telo, langevin],
        forces=[harmonic_bonds, harmonic_angles, shifted_lj, shifted_lj_wall]
    )

    simulation.operations.integrator = run_integrator

    # Define the GSD writer to take snapshots every PERIOD steps
    gsd_writer = hoomd.write.GSD(
        filename=trajectory_path,
        trigger=hoomd.trigger.Periodic(ptc.run_dump_period),
        dynamic=['property', 'momentum'],
        filter=group_all,
        mode='xb'
    )
    simulation.operations.writers.append(gsd_writer)

    print("Running the simulation...")
    for i in range(0, ptc.n_run_blocks):
        start = time.time()
        simulation.run(ptc.n_run_steps)
        print(f'block #{i+1} / {ptc.n_run_blocks}, '
              f'kin temp = {thermodynamic_properties.kinetic_temperature:.3g}, '
              f'E_P/N = {thermodynamic_properties.potential_energy / n_particles:.3g}')

    print("Simulation done. \n")

    # TODO : add synaptonemal complex formation (and disassembly)
    # TODO : extract forces and torques on each particle
