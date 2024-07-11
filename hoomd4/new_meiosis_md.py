import random
import hoomd
import gsd.hoomd
from hoomd.wall import Sphere as WallSphere
from hoomd.md.manifold import Sphere as ManifoldSphere
from hoomd.md.methods.rattle import NVE
import os
import shutil
import numpy as np
import plotly.graph_objects as go


from utils import is_debug

pi = np.pi
seed = 42
np.random.seed(seed)
random.seed(seed)


def random_point_on_sphere(radius):
    theta = np.random.uniform(0, 2 * np.pi)
    phi = np.arccos(1 - 2 * np.random.uniform())
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    return np.array([x, y, z])


def interpolate_points(start, end, num_points):
    vect = end - start
    unit_vect = vect / np.linalg.norm(vect)
    distances = np.linspace(0, np.linalg.norm(vect), num_points)
    return np.array([start + distance * unit_vect for distance in distances])


def place_polymer_in_sphere(radius, num_particles):
    telo1_pos = random_point_on_sphere(radius)
    telo2_pos = random_point_on_sphere(radius)
    while np.linalg.norm(telo1_pos - telo2_pos) < radius:
        telo2_pos = random_point_on_sphere(radius)

    polymer_positions = interpolate_points(telo1_pos, telo2_pos, num_particles - 2)
    polymer_positions = np.vstack([telo1_pos, polymer_positions, telo2_pos])
    return polymer_positions


def plot_polymer(polymer_positions, sizes):

    fig = go.Figure()
    u, v = np.mgrid[0:2 * np.pi:100j, 0:np.pi:50j]
    x = radius * np.cos(u) * np.sin(v)
    y = radius * np.sin(u) * np.sin(v)
    z = radius * np.cos(v)

    fig.add_trace(go.Mesh3d(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        color='lightpink',
        opacity=0.1,
        alphahull=0
    ))

    start = 0
    colors = ['red', 'green', 'blue']
    c = -1
    for s, size in enumerate(sizes):
        positions = polymer_positions[start:start+size]
        start += size
        if s % 2 == 0:
            c += 1
        fig.add_trace(go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode='lines+markers',
            marker=dict(size=4, color=colors[c]),
            line=dict(color=colors[c], width=2)
        ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(nticks=4, range=[-radius - 1, radius + 1]),
            yaxis=dict(nticks=4, range=[-radius - 1, radius + 1]),
            zaxis=dict(nticks=4, range=[-radius - 1, radius + 1]),
            aspectmode='cube'
        ),
        width=1920,
        height=1080,
        margin=dict(r=20, l=10, b=10, t=10)
    )

    fig.show()


if __name__ == "__main__":
    """
    ############################################
    #####       Initial Parameters         #####
    ############################################
    """
    nb_breaks = 8  # nb of DNA breaks
    timepoints = 500    # total number of timestep
    dt = 0.0001    # timestep
    radius = 12.0
    L = 3*radius
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
    poly_sizes_single = [24, 36, 48]

    particles_positions = []
    for size in poly_sizes:
        positions = place_polymer_in_sphere(radius - 0.5, size)
        particles_positions.extend(positions)
    particles_positions = np.array(particles_positions)

    plot_polymer(particles_positions, poly_sizes)

    n_particles = len(particles_positions)
    particles_ids = list(range(n_particles))
    particles_types = ['dna', 'tel', 'dsb']
    particles_typeid = np.array([1 if i == 0 or i == s - 1 else 0 for s in poly_sizes for i in range(s)])

    homologous_pairs = []
    counter = 0
    for s, size in enumerate(poly_sizes_single):
        for ll in range(size):
            homologous_pairs.append([counter + ll, counter + ll + size])
        counter += 2 * size
    homologous_pairs = np.array(homologous_pairs)

    n_bonds = n_particles - n_poly
    bonds_group = np.zeros((n_bonds, 2), dtype=int)
    bonds_types = ["dna-telo", "dna-dna", "dna-dsb", "dsb-dsb"]
    bonds_typeid = np.array(
        [bonds_types.index("dna-telo") if i == 0 or i == size - 2 else bonds_types.index("dna-dna")
         for size in poly_sizes for i in range(size - 1)])

    n_angles = n_particles - n_poly * 2
    angles_group = np.zeros((n_angles, 3), dtype=int)
    angles_types = ["dna-dna-dna", "dna-dsb-dna"]
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

    breaks_ids = np.random.choice(particles_ids, nb_breaks, replace=False)
    particles_typeid[breaks_ids] = particles_types.index('dsb')
    homologous_breaks_pairs = homologous_pairs[np.where(np.isin(homologous_pairs, breaks_ids))[0]]

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

    snapshots_dir = os.path.join(cwd, 'snapshots')
    if not os.path.exists(snapshots_dir):
        os.makedirs(snapshots_dir)
    lattice_init_path = os.path.join(snapshots_dir, "lattice_init.gsd")
    if os.path.exists(lattice_init_path):
        os.remove(lattice_init_path)
    with gsd.hoomd.open(name=lattice_init_path, mode='x') as f:
        f.append(frame)

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

    simulation = hoomd.Simulation(device=device, seed=seed)
    simulation.create_state_from_gsd(lattice_init_path)
    integrator = hoomd.md.Integrator(dt=dt)

    """
    ############################################
    #####            Define Forces         #####
    ############################################
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
    harmonic_angles.params['dna-dna-dna'] = dict(k=persistence_length, t0=pi)
    harmonic_angles.params['dna-dsb-dna'] = dict(k=persistence_length, t0=pi)
    integrator.forces.append(harmonic_angles)

    #   Define pairwise interactions
    group_all = hoomd.filter.All()
    group_tel = hoomd.filter.Type(["tel"])
    group_not_tel = hoomd.filter.SetDifference(f=group_all, g=group_tel)

    cell = hoomd.md.nlist.Cell(buffer=0.4)
    #   Computes the radially shifted Lennard-Jones pair force on every particle in the simulation state
    shifted_lj = hoomd.md.pair.ForceShiftedLJ(nlist=cell, default_r_cut=1.5)
    shifted_lj.params[('dna', 'dna')] = dict(epsilon=1.0, sigma=1.0)
    shifted_lj.params[('dna', 'dsb')] = dict(epsilon=1.0, sigma=1.0)
    shifted_lj.params[('dna', 'tel')] = dict(epsilon=1.0, sigma=1.0)
    shifted_lj.params[('tel', 'tel')] = dict(epsilon=1.0, sigma=1.0)
    shifted_lj.params[('tel', 'dsb')] = dict(epsilon=1.0, sigma=1.0)
    shifted_lj.params[('dsb', 'dsb')] = dict(epsilon=1.0, sigma=1.0)

    shifted_lj.r_cut[('dna', 'dna')] = 2**(1.0 / 6.0)
    shifted_lj.r_cut[('dna', 'dsb')] = 2**(1.0 / 6.0)
    shifted_lj.r_cut[('dna', 'tel')] = 2**(1.0 / 6.0)
    shifted_lj.r_cut[('tel', 'tel')] = 2**(1.0 / 6.0)
    shifted_lj.r_cut[('tel', 'dsb')] = 2**(1.0 / 6.0)
    shifted_lj.r_cut[('dsb', 'dsb')] = 2**(1.0 / 6.0)
    integrator.forces.append(shifted_lj)

    sphere_wall = WallSphere(radius=radius, origin=(0, 0, 0))
    walls = [sphere_wall]
    shifted_lj_wall = hoomd.md.external.wall.ForceShiftedLJ(walls)
    shifted_lj_wall.params[particles_types] = {"epsilon": 1.0, "sigma": 2.0, "r_cut": 2**(1 / 6)}
    integrator.forces.append(shifted_lj_wall)

    manifold = ManifoldSphere(r=radius, P=(0, 0, 0))
    nve_rattle_telo = NVE(filter=hoomd.filter.Type(["tel"]), manifold_constraint=manifold, tolerance=0.01)
    integrator.methods.append(nve_rattle_telo)

    langevin = hoomd.md.methods.Langevin(filter=group_not_tel, kT=1)
    langevin.gamma['dna'] = 0.2
    langevin.gamma['dsb'] = 0.2
    integrator.methods.append(langevin)

    simulation.operations.integrator = integrator

    #   Calibration
    simulation.run(1000)

    pass

    # calibration_save_path = os.path.join(snapshots_dir, 'calibration.gsd')
    # if os.path.exists(calibration_save_path):
    #     os.remove(calibration_save_path)
    # hoomd.write.GSD.write(state=simulation.state, filename=calibration_save_path, mode='x')
    #
