import argparse
import hoomd
import math
from hoomd import data, init, md, group, dump, deprecated, analyze, comm
from scipy.spatial.distance import cdist
from scipy.spatial import distance
import matplotlib.pyplot as plt
import os
import numpy as np
from math import sqrt


#Récupération des données de forces et de liste de particules cassées pour chaque simulation

list = os.listdir("./forces")
force_simu = []
homo_kept = []
nb_simu = 0
i = 0

while "force_simul"+str(i)+".npy" in list:
    force_simu.append(np.load("./forces/force_simul"+str(i)+".npy"))
    homo_kept.append(np.load("./forces/homo_kept_"+str(i)+".npy"))
    nb_simu = i
    i += 1

plt.figure()
print(homo_kept[-1][0])
for l in homo_kept[-1]:
    x=np.arange(0,1500)
    y=[np.mean(force_simu[-1][l,:]) for i in x]
    plt.figure()
    plt.plot(force_simu[-1][l,:], marker = '+')
    plt.plot(y, color= 'r', label= "moyenne")
    plt.title('force')
    plt.xlabel('pas de temps')
    plt.ylabel('force_bond')
    plt.savefig("./figures/force_"+str(l+1))









