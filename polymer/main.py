import os
import random
import numpy as np
from os.path import join
from typing import List

import pandas as pd

from constraint import Sphere, Point
from halley.constraint import Spherical
from halley.vectors import V
from utils import norm, generateV
from proba import init_proba_log, generate_point_proba


np.random.seed(42)
random.seed(42)


def generate_from_local_constraints(
        coords: List,
        l_constraints: List = None,
        rc: float = 0.1,
        virtual_lp: float = None,
        rigid_constrain: bool = True
):
    pos = []
    index_point = len(coords)
    bond_size = 1

    # To disregard the constraint that where before the actual index, we have to
    # find the first constraint that is useful.
    # The constraint should be organized
    # print [0 if c.index < index_point else 1 for c in l_constraints ]
    is_there_constrain = [0 if c.index < index_point else 1 for c in l_constraints]
    if (1 not in is_there_constrain) and (virtual_lp is None):
        # No more constraint
        start = np.array(coords[-1])
        pos = start + bond_size * generateV()
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
                width.append(np.sum(np.ones([index_point, c.index])))
                if not rigid_constrain and width[-1] == 0:
                    width[-1] = rc
                elif width[-1] == 0:
                    width[-1] = 1
        if coords:
            extent.append(coords[-1][xi])
            width.append(bond_size)

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
        coords: List,
        g_constraints: List = None,
        l_constraints: List = None,
        rc: float = 0.1,
        virtual_lp: float = None,
        rigid_constrain: bool = True,
        flexible_lp: bool = True
):
    N = 0
    max_trial = 300000
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
                pos, redo = generate_from_local_constraints(
                    coords,
                    l_constraints=l_constraints,
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


if __name__ == "__main__":
    radius = 12
    n_poly = 6
    poly_sizes = [24, 24, 36, 36, 48, 48]

    nucleus = Sphere(position=[0, 0, 0], radius=radius)

    df_atoms = pd.DataFrame()
    poly_coords = []
    for x in range(n_poly):
        s_nucleus = Spherical(V(0, 0, 0), radius=radius*0.99)
        telo1 = Point(index=0, position=s_nucleus.get_random()._v)
        telo2 = Point(index=poly_sizes[x] - 1, position=s_nucleus.get_random()._v)

        coords = [
                generate_next(
                    coords=[],
                    g_constraints=[nucleus],
                    l_constraints=[telo1, telo2],
                )]

        for _ in range(1, poly_sizes[x]):
            coords.append(
                generate_next(
                    coords=coords,
                    g_constraints=[nucleus],
                    l_constraints=[telo1, telo2],
                )
            )

        poly_coords.append(coords)

    for i, coords in enumerate(poly_coords):
        for j, coord in enumerate(coords):
            df_atoms = pd.concat([df_atoms, pd.DataFrame(
                [[i * poly_sizes[i] + j, i, 0, coord[0], coord[1], coord[2]]],
                columns=["id", "molecule", "type", "x", "y", "z"]
            )])
