import numpy as np
import gsd.hoomd

import init


def set_chromosomes(n_poly: int, l_poly: list[int], n_breaks: int, simBoxSize: int, inscribedBoxSize) -> dict:

    n_particles = sum(l_poly)
    particles_types = ['dna', 'tel', 'dsb']
    particles_typeid = []
    for s in l_poly:
        for i in range(s):
            particles_typeid.append(1 if i == 0 or i == s - 1 else 0)
    particles_typeid = np.array(particles_typeid)
    telomeres_ids = np.where(particles_typeid == 1)[0]
    not_telomeres_ids = np.where(particles_typeid == 0)[0]
    
    monomer_positions = init.lattice_cubic(n_particles=n_particles, boxSize=inscribedBoxSize)
    
    homologous_pairs = []
    counter = 0
    l_poly_single = [s for s, i in enumerate(l_poly) if i % 2 == 0]
    for s, size in enumerate(l_poly_single):
        for ll in range(size):
            homologous_pairs.append([counter + ll, counter + ll + size])
        counter += 2 * size
    homologous_pairs = np.array(homologous_pairs)
    
    n_bonds = n_particles - n_poly
    bonds_group = np.zeros((n_bonds, 2), dtype=int)
    bonds_types = ["dna-telo", "dna-dna", "dna-dsb", "dsb-dsb"]
    bonds_typeid = np.array(
        [bonds_types.index("dna-telo") if i == 0 or i == size - 2 else bonds_types.index("dna-dna")
         for size in l_poly for i in range(size - 1)])
    
    n_angles = n_particles - n_poly * 2
    angles_group = np.zeros((n_angles, 3), dtype=int)
    angles_types = ["dna-dna-dna", "dna-dsb-dna"]
    angles_typeid = np.array([angles_types.index("dna-dna-dna") for size in l_poly for i in range(size - 2)])
    
    # Optimizing loops for bonds and angles
    start_indices = list(np.cumsum([0] + l_poly[:-1]))
    b_counter = 0
    a_counter = 0
    for x, start in enumerate(start_indices):
        x_len = l_poly[x]
        for b in range(x_len - 1):
            bonds_group[b_counter] = [start + b, start + b + 1]
            b_counter += 1
        for a in range(x_len - 2):
            angles_group[a_counter] = [start + a, start + a + 1, start + a + 2]
            a_counter += 1
    
    breaks_ids = np.random.choice(not_telomeres_ids, n_breaks, replace=False)
    particles_typeid[breaks_ids] = particles_types.index('dsb')
    homologous_breaks_pairs = homologous_pairs[np.where(np.isin(homologous_pairs, breaks_ids))[0]]
    
    frame = gsd.hoomd.Frame()
    frame.particles.N = n_particles
    frame.particles.types = particles_types
    frame.particles.typeid = particles_typeid
    frame.particles.position = monomer_positions
    
    frame.bonds.N = n_bonds
    frame.bonds.group = bonds_group
    frame.bonds.typeid = bonds_typeid
    frame.bonds.types = bonds_types
    
    frame.angles.N = n_angles
    frame.angles.group = angles_group
    frame.angles.typeid = angles_typeid
    frame.angles.types = angles_types
    
    frame.configuration.box = [simBoxSize] * 3 + [0] * 3
    frame.configuration.dimensions = 3
    
    res = {
        "frame": frame,
        "breaks_ids": breaks_ids,
        "homologous_pairs": homologous_pairs,
        "homologous_breaks_pairs": homologous_breaks_pairs,
        "telomeres": telomeres_ids,
        "not_telomeres": not_telomeres_ids
    }

    print("\n")
    print(f"{n_poly} polymers with lengths {l_poly}")
    print(f"{n_particles} particles")
    print(f"{n_breaks} random breaks")
    print(f"{n_bonds} bonds and {n_angles}")
    print(f"HOOMD frame with box size {simBoxSize} generated \n")

    return res
