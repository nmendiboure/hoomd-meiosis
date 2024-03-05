import os

import numpy as np

import pandas as pd


class Simu:
    def __init__(self, cmd="lmp -in"):
        self.molecules = []

    def add(self, molecule: Chromosome):
        """
        add a molecule (polymer) and take care of shifting the id bonds
        """
        start_id = 1
        start_bond = 1
        start_angle = 1

        if self.molecules:
            start_id = self.molecules[-1].ids[-1] + 1
            # look for the firsts previous molecule with a bond
            for mol in self.molecules[::-1]:
                if mol.bonds:
                    start_bond = mol.bonds[-1][0] + 1
            if self.molecules[-1].angles:
                start_angle = self.molecules[-1].angles[-1][0]

        add_id = start_id - molecule.ids[0]
        add_bond = 0
        if molecule.bonds:
            add_bond = start_bond - molecule.bonds[0][0]
        add_angle = 0
        if molecule.angles:
            add_angle = start_angle - molecule.angles[0][0]
        molecule.shift(add_id, add_bond, add_angle)
        self.molecules.append(molecule)

    def to_csv(self, dirname: str):

        start_id = 1
        start_bond = 1
        start_angle = 1

        df_atoms, df_bonds, df_angles = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        for molecule in self.molecules:
            # still assigning a start_id, but it should already be correct
            atoms, bonds, angles = [[] for _ in range(3)]
            for ids, l_ids, pos in zip(molecule.ids, molecule.types_beads, molecule.coords):
                X, Y, Z = pos
                atoms.append([ids, molecule.uid, l_ids, X, Y, Z])
            for bid, type_bond, n1, n2 in molecule.bonds:
                bonds.append([bid, molecule.uid, type_bond, n1, n2])
            for aid, type_bond, n1, n2, n3 in molecule.angles:
                angles.append([aid, molecule.uid, type_bond, n1, n2, n3])

            start_id += len(atoms)
            start_bond += len(bonds)
            start_angle += len(angles)

            df_atoms = pd.concat([df_atoms, pd.DataFrame(
                np.asarray(atoms), columns=["id", "molecule", "type", "x", "y", "z"])])
            df_bonds = pd.concat([df_bonds, pd.DataFrame(
                np.asarray(bonds), columns=["id", "molecule", "type", "atom1", "atom2"])])
            df_angles = pd.concat([df_angles, pd.DataFrame(
                np.asarray(angles), columns=["id", "molecule", "type", "atom1", "atom2", "atom3"])])

        os.makedirs(dirname, exist_ok=True)
        df_atoms.to_csv(os.path.join(dirname, "atoms.tsv"), index=False, sep="\t")
        df_bonds.to_csv(os.path.join(dirname, "bonds.tsv"), index=False, sep="\t")
        df_angles.to_csv(os.path.join(dirname, "angles.tsv"), index=False, sep="\t")


