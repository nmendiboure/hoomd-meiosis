import numpy as np
import gsd.hoomd

import init


def set_chromosomes(n_poly: int, l_poly: list[int], n_breaks: int, simBoxSize: int, inscribedBoxSize):

    n_particles = sum(l_poly)
    particles_types = ['dna', 'tel', 'dsb']
    particles_typeid = []
    for s in l_poly:
        for i in range(s):
            particles_typeid.append(1 if i == 0 or i == s - 1 else 0)
    particles_typeid = np.array(particles_typeid)
    telomere_ids = np.where(particles_typeid == 1)[0]
    not_telomere_ids = np.where(particles_typeid == 0)[0]
    
    monomer_positions = init.lattice_cubic(n_particles=n_particles, boxSize=inscribedBoxSize)
    
    homologous_pairs = []
    counter = 0
    l_poly_single = [s for i, s in enumerate(l_poly) if i % 2 == 0]
    for s, size in enumerate(l_poly_single):
        for ll in range(size):
            homologous_pairs.append([counter + ll, counter + ll + size])
        counter += 2 * size

    # Homologous and broken homologous pairs
    homologous_pairs = np.array(homologous_pairs)
    break_ids = np.random.choice(not_telomere_ids, n_breaks, replace=False)
    particles_typeid[break_ids] = particles_types.index('dsb')
    homologous_break_pairs = homologous_pairs[np.where(np.isin(homologous_pairs, break_ids))[0]]
    
    n_bonds = n_particles - n_poly
    bonds_group = np.zeros((n_bonds, 2), dtype=int)
    bonds_types = ["dna-tel", "dna-dna", "dna-dsb", "dsb-dsb"]
    bonds_typeid = []
    for i, s in enumerate(l_poly):
        this_poly_bonds = np.zeros(s - 1, dtype=np.uint8)
        this_poly_bonds[[0, s-2]] = bonds_types.index("dna-tel")
        this_poly_bonds[1:s-2] = bonds_types.index("dna-dna")
        this_poly_bonds_breaks = np.where(np.isin(np.arange(s), break_ids))[0]
        if this_poly_bonds_breaks:
            for b in this_poly_bonds_breaks:
                this_poly_bonds[[b-1, b]] = bonds_types.index("dna-dsb")

        bonds_typeid.extend(this_poly_bonds)
    bonds_typeid = np.array(bonds_typeid)
    
    n_angles = n_particles - n_poly * 2
    angles_group = np.zeros((n_angles, 3), dtype=int)
    angles_types = ["tel-dna-dna", "dna-dna-dna", "dna-dsb-dna"]
    angles_typeid = []
    for i, s in enumerate(l_poly):
        this_poly_angles = np.zeros(s - 2, dtype=np.uint8)
        this_poly_angles[[0, s-3]] = angles_types.index("tel-dna-dna")
        this_poly_angles[1:s-3] = angles_types.index("dna-dna-dna")
        this_poly_angles_breaks = np.where(np.isin(np.arange(s), break_ids))[0]
        if this_poly_angles_breaks:
            for b in this_poly_angles_breaks:
                this_poly_angles[b-1] = angles_types.index("dna-dsb-dna")

        angles_typeid.extend(this_poly_angles)
    angles_typeid = np.array(angles_typeid)
    
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

    print("\n")
    print(f"{n_poly} polymers with lengths {l_poly}")
    print(f"{n_particles} particles")
    print(f"{n_breaks} random breaks")
    print(f"{n_bonds} bonds and {n_angles} angles")
    print(f"HOOMD frame with box size {simBoxSize} generated \n")

    return frame, break_ids, homologous_pairs, homologous_break_pairs, telomere_ids, not_telomere_ids
