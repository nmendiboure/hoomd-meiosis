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


def rods_in_sphere(radius, num_particles, deviation: float = 0.5):
    telo1_pos = random_point_on_sphere(radius)
    telo2_pos = random_point_on_sphere(radius)
    while np.linalg.norm(telo1_pos - telo2_pos) < radius * 1.8:
        telo2_pos = random_point_on_sphere(radius)

    polymer_positions = interpolate_points(telo1_pos, telo2_pos, num_particles, deviation)
    return polymer_positions


def lattice_cubic(n_particles, radius):
    """
    Inspired from polychrom-hoomd (Mirnylab
    """
    if isinstance(radius, float):
        radius = int(radius)

    t = radius // 2
    a = [(t, t, t), (t, t, t + 1), (t, t + 1, t + 1), (t, t + 1, t)]

    b = np.zeros((radius + 2, radius + 2, radius + 2), int)
    for i in a:
        b[i] = 1

    for i in range((n_particles - len(a)) // 2):
        while True:

            t = np.random.randint(0, len(a))

            if t != len(a) - 1:
                c = np.abs(np.array(a[t]) - np.array(a[t + 1]))
                t0 = np.array(a[t])
                t1 = np.array(a[t + 1])
            else:
                c = np.abs(np.array(a[t]) - np.array(a[0]))
                t0 = np.array(a[t])
                t1 = np.array(a[0])
            cur_direction = np.argmax(c)
            while True:
                direction = np.random.randint(0, 3)
                if direction != cur_direction:
                    break
            if np.random.random() > 0.5:
                shift = 1
            else:
                shift = -1
            shiftar = np.array([0, 0, 0])
            shiftar[direction] = shift
            t3 = t0 + shiftar
            t4 = t1 + shiftar
            if (
                (b[tuple(t3)] == 0)
                and (b[tuple(t4)] == 0)
                and (np.min(t3) >= 1)
                and (np.min(t4) >= 1)
                and (np.max(t3) < radius + 1)
                and (np.max(t4) < radius + 1)
            ):
                a.insert(t + 1, tuple(t3))
                a.insert(t + 2, tuple(t4))
                b[tuple(t3)] = 1
                b[tuple(t4)] = 1
                break
                # print a

    a = np.array(a) - 1
    a = np.asarray(a, dtype=np.float32)
    a -= a.mean(axis=0, keepdims=True)

    return a
