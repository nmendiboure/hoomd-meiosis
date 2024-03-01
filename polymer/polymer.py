import numpy as np
from typing import List

from utils import norm, generateV
from proba import init_proba_log, generate_point_proba


class Chromosome(object):
    def __init__(self, **kwargs):
        self.uid = kwargs.get("uid", 0)
        self.size = kwargs.get("size", 0)
        self.shape = kwargs.get("shape", "linear")
        self.centromere = kwargs.get("centromere", self.size // 2)
        self.l_constraints = kwargs.get("l_constraints", [])
        self.g_constraints = kwargs.get("g_constraints", [])
        self.coords = []
        self.bonds = []
        self.angles = []
        self.types_beads = []
        self.ids = []

        self.make()

    def make(self):
        if self.size <= 0:
            raise ValueError("Chromosome size must be higher than 0")
        ids = [i for i in range(1, self.size + 1)]
        types_beads = [i for i in range(1, self.size + 1)]

        bonds = []
        angles = []
        if self.shape == "linear":
            bonds = [[i, 1, i, i + 1] for i in range(1, self.size)]
            if self.size >= 3:
                angles = [[i, 1, i, i + 1, i + 2] for i in range(1, self.size - 1)]

        elif self.shape == "circular":
            bonds = [[i, 1, i, i + 1] for i in range(1, self.size + 1)]
            bonds[-1][-1] = 0
            if self.size >= 3:
                angles = [[i, 1, i, i + 1, i + 2] for i in range(1, self.size + 1)]
                angles[-1][-2] = 1
                angles[-1][-1] = 0
                angles[-2][-1] = 0

        coords = [
            self.generate_next(
                coords=[],
                g_constraints=self.g_constraints,
                l_constraints=self.l_constraints
            )]

        for bond in bonds:
            coords.append(
                self.generate_next(
                    coords=coords,
                    g_constraints=self.g_constraints,
                    l_constraints=self.l_constraints,
                ))

        self.bonds = bonds
        self.angles = angles
        self.coords = coords
        self.ids = ids
        self.types_beads = types_beads

    def generate_from_local_constraints(
            self,
            coords: List,
            l_constraints: List = None,
            bond_sizes: int | List[int] = 1,
            rc: float = 0.1,
            virtual_lp: float = None,
            rigid_constrain: bool = True
    ):
        pos = []
        index_point = len(coords)

        # To disregard the constraint that where before the actual index, we have to
        # find the first constraint that is useful.
        # The constraint should be organized
        # print [0 if c.index < index_point else 1 for c in l_constraints ]
        is_there_constrain = [0 if c.index < index_point else 1 for c in l_constraints]
        # print virtual_lp
        if isinstance(bond_sizes, int):
            bond_sizes = [bond_sizes for _ in range(self.size - 1)]
        if (1 not in is_there_constrain) and (virtual_lp is None):
            # No more constraint
            start = np.array(coords[-1])
            pos = start + bond_sizes[index_point - 1] * generateV()
            return np.array(pos), []

        else:
            if 1 in is_there_constrain:
                start_constrain = is_there_constrain.index(1)
            else:
                start_constrain = None

        if start_constrain is not None and l_constraints[start_constrain].index == index_point and rigid_constrain:
            # if a constraint matches the index, we return the position
            return np.array(l_constraints[start_constrain].position), []

        redo = []
        for xi in range(3):
            extent = []
            width = []
            if start_constrain is not None:
                for c in l_constraints[start_constrain:]:
                    extent.append(c.position[xi])
                    width.append(np.sum(bond_sizes[index_point:c.index]))
                    if not rigid_constrain and width[-1] == 0:
                        width[-1] = rc
                    elif width[-1] == 0:
                        width[-1] = 1
            if coords:
                extent.append(coords[-1][xi])
                width.append(bond_sizes[index_point - 1])

            if len(coords) >= 2 and virtual_lp:
                # width[-1] is the bond length
                extent.append(
                    coords[-1][xi] + width[-1] * (coords[-1][xi] - coords[-2][xi]) / norm(coords[-1] - coords[-2]))
                width.append(1. / virtual_lp)

            width_quad = np.array(width)  # ** 2  * rc
            cm = 1 / np.sum(1 / width_quad) * np.sum(np.array(extent) / np.array(width_quad))
            gauss = lambda x, c, w: np.exp(-(x - c) ** 2 / (2 * w))  # Not square as we use width_quad

            def Proba(x):
                prod = []
                for pos, w in zip(extent, width_quad):
                    prod.append(gauss(x, pos, w=w))
                return np.prod(prod)

            lngauss = lambda x, c, w: -(x - c) ** 2 / (2 * w)

            def lnProba(x):
                prod = []
                for pos, w in zip(extent, width_quad):
                    prod.append(lngauss(x, pos, w=w))
                return np.sum(prod)

            w = np.min(width)
            w = max(1., w)
            x, index = init_proba_log(lnProba, dx=w / 5., lrange=[cm - 8 * w, cm + 8 * w])
            # x, index = init_proba(Proba,dx=w/5.,lrange=[cm-4*w,cm+4*w])
            p = generate_point_proba(x, index)
            pos.append(p)
            redo.append([x, index])

        return np.array(pos), redo

    def generate_next(
            self,
            coords: List,
            g_constraints: List = None,
            l_constraints: List = None,
            bond_sizes: int | List[int] = 1,
            max_trial: int = 300000,
            rc: float = 0.1,
            virtual_lp: float = None,
            rigid_constrain: bool = True,
            flexible_lp: bool = True
    ):
        N = 0
        pos = []
        redo = []  # The probability density is generated once, and then several trials are extracted from it
        virtual_lp0 = virtual_lp  # virtual_lp0 will be decrease if flexible_lp in True
        while N < max_trial:
            if coords == [] and l_constraints == []:
                if not g_constraints:
                    pos = np.array([0, 0, 0])
                else:
                    pos = np.array(g_constraints[0].generate())

            else:
                if flexible_lp and virtual_lp is not None and N != 0 and N % int(max_trial / 5) == 0:
                    virtual_lp0 /= 2.
                    redo = []
                    print(("Decreasing virtual_lp to ", virtual_lp0))

                if not redo:
                    pos, redo = self.generate_from_local_constraints(
                        coords,
                        l_constraints=l_constraints,
                        bond_sizes=bond_sizes,
                        rc=rc,
                        virtual_lp=virtual_lp0,
                        rigid_constrain=rigid_constrain
                    )
                else:
                    pos = np.array([generate_point_proba(x, index) for x, index in redo])

            # check global constrain
            out = False
            for gc in g_constraints:
                if not gc.is_inside(pos):
                    out = True
            if out:
                N += 1
                if N == max_trial - 1:
                    print(coords)
                    print("constrain not satisfied")
                    raise
                continue
            break
        return pos

    def shift(self, add_id=0, add_bond=0, add_angle=0):
        self.ids = [iid + add_id for iid in self.ids]
        self.bonds = [[b[0] + add_bond, b[1], b[2] + add_id, b[3] + add_id] for b in self.bonds]
        self.angles = [[b[0] + add_angle, b[1], b[2] + add_id, b[3] + add_id, b[4] + add_id] for b in self.angles]
