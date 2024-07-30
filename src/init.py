import numpy as np


def random_point_on_sphere(radius):
    theta = np.random.uniform(0, 2 * np.pi)
    phi = np.arccos(1 - 2 * np.random.uniform())
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    return np.array([x, y, z])


def interpolate_points(start, end, num_points, max_deviation=0.5):
    vect = end - start
    unit_vect = vect / np.linalg.norm(vect)
    distances = np.linspace(0, np.linalg.norm(vect), num_points)

    # Introduce random deviation vectors perpendicular to the main vector
    deviate_positions = []
    for dist in distances:
        deviation = np.random.uniform(-max_deviation, max_deviation, 3)
        deviation -= deviation.dot(unit_vect) * unit_vect  # Make sure deviation is perpendicular
        deviate_positions.append(start + dist * unit_vect + deviation)

    return np.array(deviate_positions)


def place_polymer_in_sphere(radius, num_particles):
    telo1_pos = random_point_on_sphere(radius)
    telo2_pos = random_point_on_sphere(radius)
    while np.linalg.norm(telo1_pos - telo2_pos) < radius * 1.8:
        telo2_pos = random_point_on_sphere(radius)

    polymer_positions = interpolate_points(telo1_pos, telo2_pos, num_particles)
    return polymer_positions
