
"""
Created on Mon Sep 13 16:03:13 2021

@author: schay

"""

import argparse
import hoomd
import math
from hoomd import data, init, md, group, dump, deprecated, analyze, comm
from scipy.spatial.distance import cdist
from math import dist
import matplotlib.pyplot as plt
import os
import numpy as np
from math import sqrt
from math import log
from numpy import linalg as la 
from pylab import *
import mdtraj as md
import scipy as sp
from scipy import optimize
import csv
import seaborn as sns





#Récupération données de positions et de longueur des chromosomes par simulation

list = os.listdir("./positions")

pos_xyz=[]
syst_len=[]
i = 0
while "pos_xyz"+str(i)+".npy" in list:
    pos_xyz.append(np.load("./positions/pos_xyz"+str(i)+".npy"))
    syst_len.append(np.load("./positions/syst_len"+str(i)+".npy"))
    i += 1

#pos_xyz[simu, pas de temps, particules, positions]
#syst_len[simu]

"""
Longueur de persistance

"""

moyenne_chomosomes = [] #Works only if all chomosome have same size

#pos_xyz[-1] et syst_len[-1] sigifie qu'on considère la dernièe simulation réalisée 

for iX_len, X_len in enumerate(syst_len[-1]):
    produit_scal_distance=np.ndarray((1000,max(syst_len[-1]))) *np.nan
    moyenne_produit_scal=[]
    start = sum(syst_len[-1][:iX_len])
   
    Vecteur_tangeant = np.array([[(pos_xyz[-1][j,start+i,:] - pos_xyz[-1][j,start+i-1,:])*(1/np.linalg.norm(pos_xyz[-1][j,start+i-1,:] - pos_xyz[-1][j,start+i,:])) for i in range(1,X_len)] for j in range(shape(pos_xyz)[1])]) # vecteur_tangeant[nstep,nb_part,position part]

    for l in range(200,1000): #temps
        for k in range(0,X_len): #ecart
            end=-k
            if k==0:
                end=None
            produit_scal_distance[l,k]=np.mean(np.sum(Vecteur_tangeant[l,k:]*Vecteur_tangeant[l,:end],axis=-1))
           
    moyenne_produit_scal_distance=np.nanmean(produit_scal_distance, axis=0) #temps,distance


k=60 #paramètre théorique de la longueur de persistance 
x=np.arange(0,X_len)
y=moyenne_produit_scal_distance
plot(x,y,label="simu")
plot(x,np.exp(-x/k),label="exp(-x/"+str(k)+")")
plt.ylabel('produit scalaire')
plt.xlabel('distance entre deux particules')
xlim(0,20) 
legend()
plt.savefig('persistance')


"""
Angles 

Mettre en commentaire cette partie lorsque la simulation n'est pas faite avec une paire de chromosomes de 80 particules

"""
 


ecart=[0,10,20,30,40,50,60,78] #mesure de l'angle toutes les 10 particules le long du chromosome
ecart=[0,10,18]
temps=[1,100,200,300,400,500,600,700,800,900]
angle=np.ndarray((len(temps),9))*np.nan

for t in range(len(temps)-1):
    for i in range(len(ecart)-1): 
        angle[t,i]=np.degrees(math.acos(np.dot(Vecteur_tangeant[temps[t],ecart[i]],Vecteur_tangeant[temps[t],ecart[i+1]])))



plt.figure()
sns.distplot(angle,kde=True,hist_kws={"align" : "mid"})
plt.xlabel('angle')
plt.ylabel('fréquence')
plt.title("Répartition angle")
plt.savefig('angle')

plt.figure()




"""
MSD

"""

tau=500 #écart maximal entre les positions d'une particule pour le calcul de MSD 


distance=np.ndarray((shape(pos_xyz[-1])[0],shape(pos_xyz[-1])[1],tau))*np.nan # distance entre particules [pas de temps, particule, tau]

MSD_tau=np.zeros(tau)

for i in range(shape(pos_xyz[-1])[0]):#pas de temps
    for k in range(shape(pos_xyz[-1])[1]):#particule
        for j in range(1,tau):#tau
                if i-j < shape(pos_xyz[-1])[0]:
                    distance[i,k,j] = sum((pos_xyz[-1][i-j,k,:]-pos_xyz[-1][i,k,:])**2)  

for i in range(tau): 
    MSD_tau[i]= distance[:,:,i].mean()
    

np.savetxt('data.csv',MSD_tau) 


plt.figure()
x=np.arange(0,tau)
plt.plot( MSD_tau, label="MSD")
plt.title('msd_tau')
plt.xlabel('tau')
plt.ylabel(' msd')
plt.savefig('msd') 

plt.figure() 




 

