import argparse
import hoomd
import math

from hoomd import data, init, md, group, dump, deprecated, analyze, comm
from scipy.spatial.distance import cdist
from scipy.spatial import distance

import matplotlib.pyplot as plt
import os
import numpy as np
import time as Time
import random
from math import sqrt
#from interval import interval, inf, imath

#♪Paramètres important

nb_bond=100 #nombre de cassures dans ADN
longueur_de_persistance = 10
distance_requise_pour_bond_homologues = 1.2



def initialize_snap(syst, params={}):
    """
    Create the inititial system
    with the correct number of particle, number of bonds and
    box size
    """
    print(syst["len_polymers"])
    syst["np"] = sum(syst["len_polymers"])+1
    
    syst["nbond"] = sum(syst["len_polymers"])-len(syst["len_polymers"])
    syst["nangle"] = sum(syst["len_polymers"])-2*len(syst["len_polymers"])   #(nombre de chaîne)
    
    syst["Rf"] = (sum(syst["len_polymers"])*0.5**3/syst['density'])**0.33 
    print("Radius of the cell", syst["Rf"])
    R = syst["Rf"]+1
    
    snapshot = data.make_snapshot(N=syst["np"], box=data.boxdim(L=3*R+3), bond_types=['polymer'])
 
    syst["bond_list"] = ['DNA-DNA','fh-fh','DNA-Telo']
    syst["plist"] = ["DNA", "Telo","fh","box"]
    syst["angle_list"] = ["DNA"]
    print(snapshot.box.Lx)
    
    return snapshot, syst



def mesure_force(distance, k, r0):
    """
    Compute the force between with paramter k and r0
    """
    force = -k*(distance-r0)
    return force


def init_grid(R, N, len_max):
    """
    Création de la grille pour l'initialisation
    """

    cote = R/sqrt(2)
    
    x_min = -cote/2
    x_pas = cote/(N-1)
    x_max = cote/2

    y_min = -cote/2
    y_pas = cote/(len_max-1)
    y_max = cote/2
    
    z_min = -cote/2
    z_pas = cote/(N-1)
    z_max = cote/2
    
    x = np.arange(x_min, x_max+x_pas, x_pas).tolist()
    y = np.arange(y_min, y_max+y_pas, y_pas).tolist()
    z = np.arange(z_min, z_max+z_pas, z_pas).tolist()
    
    return x, y, z


def create_snapshot(snapshot, syst, seed=False):
    """
    Set the position of the particle
    as well as bonds
    """
    
    snapshot.bonds.types = syst["bond_list"]
    snapshot.bonds.resize(syst["nbond"])
    
    snapshot.particles.types = syst["plist"]
    
    snapshot.angles.resize(syst["nangle"])
    snapshot.angles.types = syst["angle_list"]
    
    R = syst["Rf"]+1
    N = len(syst["len_polymers"])
    len_max = np.max(syst["len_polymers"])
    
    x, y, z = init_grid(R, N, len_max)

    ##################################################
    # Define particle type and positions
    ref_particle = 0
    ref_bond = 0
    ref_angle = 0 
        
    for iX_len, X_len in enumerate(syst["len_polymers"]): #chaine par chaine

        # Particle type
        for i in range(X_len):
            if i not in [0,X_len-1]:
                snapshot.particles.typeid[ref_particle] = syst["plist"].index("DNA")
            else:
                snapshot.particles.typeid[ref_particle] = syst["plist"].index("Telo")
            ref_particle += 1
        
        start = sum(syst["len_polymers"][:iX_len])  #nombre de particules déjà initialisées

        # Particle position
        random_pos = False
        if random_pos:
            maxi = 2*syst["Rf"]
            if seed:
                np.random.seed(0)
            while maxi > syst["Rf"]:
                p0 = np.array([0.0, 0, 0])
                for i in range(X_len):
                    snapshot.particles.position[start+i] = p0
                    p0 += 0.2*(1-2*np.random.rand(3))

                Cm = np.mean(snapshot.particles.position, axis=0)
                for i in range(X_len):
                    snapshot.particles.position[start+i] -= Cm[:]
                maxi = np.max(np.abs(snapshot.particles.position))
        else:
            x_idx = random.randint(0, len(x)-1)
            z_idx = random.randint(0, len(z)-1)
            for i in range(X_len):
               snapshot.particles.position[start+i] = [x[x_idx], y[i], z[z_idx]]
            del(x[x_idx])
            del(z[z_idx])
        #print(interval[-snapshot.box.Ly,snapshot.box.Ly])
            
        # Define bonds
        for i in range(X_len-1):
            snapshot.bonds.group[ref_bond] = [start+i, start+i+1]  
            snapshot.bonds.typeid[ref_bond] = syst["bond_list"].index('DNA-DNA')  # polymer_A
            ref_bond += 1
        
        for i in range(X_len-2):
            snapshot.angles.group[ref_angle] = [start+i, start+i+1,start+i+2]
            snapshot.bonds.typeid[ref_angle] = syst["angle_list"].index("DNA")  
            ref_angle += 1

        #ajout sphère
        snapshot.particles.typeid[-1] = syst["plist"].index("box")
        snapshot.particles.position[-1] = [0,0,0]

        #liste des homologues 
         
        homologues = []
        taille = [syst["len_polymers"][2*i] for i in range((len(syst["len_polymers"])//2))] #taille de la liste syst["len_polymer"] divisé par 2 et on garde les indices pairs

        for (iX_len, X_len) in enumerate(taille):
            start = 2*sum(taille[:iX_len])
            for i in range(1, X_len-1):
                homologues.append((start+i, start+i+X_len))
                
    ##################################################
 

    print(homologues)
                   
    return snapshot,homologues





def simulate(syst, n_steps, data_folder="./repli", params={}, seed=False):
    
    global t0, nb_bond

    t0 = Time.time()
    def time(where):
        global t0
        print(where, "elapsed %.1f" % (Time.time()-t0))
        t0 = Time.time()

    verbose = syst["verbose"]

    data_folder = os.path.join(data_folder)

    os.makedirs(data_folder, exist_ok=True)

    print(data_folder)

    time("Start")
    snapshot, syst = initialize_snap(syst)
    time("Initialize")
    length_steps = syst["length_steps"]  # 50000

    if comm.get_rank() == 0:
        snapshot,homologues = create_snapshot(snapshot, syst, seed=seed)
    
    snapshot.broadcast()
    system = init.read_snapshot(snapshot)
    
    ################################
    #define an other list of homologues from homologues 

    idx = random.randint(0,len(homologues)-1)
    Homologues_kept = [(idx + i*6)%len(homologues) for i in range(nb_bond)]
    for i in Homologues_kept:
        h1,h2=homologues[i]
        snapshot.particles.typeid[h1] = syst["plist"].index("fh")
        snapshot.particles.typeid[h2] = syst["plist"].index("fh")

    
        
    ################################
    # Define bond strength and type
    bond = md.bond.harmonic(name="mybond")
    bond.bond_coeff.set(['DNA-DNA'], k=800, r0=1)
    bond.bond_coeff.set(['DNA-Telo'], k=800, r0=1)
    bond.bond_coeff.set(['fh-fh'], k=80, r0=1) 
    
    ###############################
    #Define angle energy and type
    
    angle = md.angle.harmonic()
    angle.angle_coeff.set(syst["angle_list"], k=longueur_de_persistance, t0=math.pi) #paramètre k agissant sur la longueur de persistance

    
    #############################
    # Define pairwise interactions
    all = group.all()
    Telo_group = hoomd.group.type(name="Telo particles", type="Telo")
    box_group = hoomd.group.type(name="box", type="box")
    group_not_Telo = group.difference(name="not Telo", a=all, b=Telo_group)
    first_homologue_group=hoomd.group.type(name="First Homologue group",type="fh")
    group_not_box = group.difference(name="not box", a=all, b=box_group)

                         
    nl = md.nlist.cell()


    
    slj = md.pair.slj(r_cut=1.5, nlist=nl, d_max = 2.0) 
    slj.pair_coeff.set('DNA', 'DNA', epsilon=1.0, sigma=1,r_cut=2**(1.0/6.0))
    slj.pair_coeff.set('Telo', 'DNA', epsilon=1.0, sigma=1.0,r_cut=2**(1.0/6.0))
    slj.pair_coeff.set('fh', 'fh', epsilon=1.0,sigma=1.0, r_cut=2**(1.0/6.0))
    slj.pair_coeff.set('fh', 'DNA', epsilon=1.0,sigma=1.0, r_cut=2**(1.0/6.0))
    slj.pair_coeff.set('fh', 'Telo', epsilon=1.0,sigma=1.0, r_cut=2**(1.0/6.0))
    slj.pair_coeff.set('Telo', 'Telo', epsilon=1.0,sigma=1.0, r_cut=2**(1.0/6.0))

    slj.pair_coeff.set('DNA', 'box', epsilon=1.0,sigma=0, r_cut=2**(1.0/6.0))
    slj.pair_coeff.set('Telo', 'box', epsilon=1.0,sigma=0, r_cut=2**(1.0/6.0))
    slj.pair_coeff.set('fh', 'box', epsilon=1.0,sigma=0, r_cut=2**(1.0/6.0))
    slj.pair_coeff.set('box', 'box', epsilon=1.0,sigma=0, r_cut=2**(1.0/6.0))
    
    ##################################################
    # wall
    sphere = md.wall.group()
  
    sphere.add_sphere(r=syst["Rf"], origin=(0.0, 0.0, 0.0), inside=True)

    wall_force_slj = md.wall.lj(sphere, r_cut=3)

    #repulsive interaction becasue cut at the minimum value
    wall_force_slj.force_coeff.set(syst["plist"], epsilon=1, sigma=2,
                                  r_cut=2**(1/6) , mode="shift")

    #Attractive for telomeres
    wall_force_slj.force_coeff.set(["Telo"], epsilon=2, sigma=1,r_extrap=0,
                                  r_cut=3)
        
    ###################################################
    #To record the trajectory
    all = group.all()
    period = length_steps

    gsd = dump.gsd(group=all, filename=os.path.join(data_folder, 'poly_d.gsd'),
                   period=period, overwrite=True, dynamic=["attribute", "topology"], phase=0)

    ##################################################

    sim_dt = 0.001

    snp = system
    md.integrate.mode_standard(dt=sim_dt)
    if seed:
        seed = 0
    else:
        seed = np.random.randint(10000)
        
    method = md.integrate.langevin(group=group_not_box, kT=1, seed=seed, dscale=False)
    
 
    method.set_gamma("Telo", gamma=0.2) #par défaut autres particules gamma=1
    group_hic = all  # group.tags(name="hic", tag_min=0, tag_max=Nparticule)

    time("End define all") 
    
    # Do some equilibration
    hoomd.run(syst["equi"], profile=False, quiet=True)
    time("Start loop")
    
    # Run the simulation

    force=np.zeros((len(homologues),n_steps))
    b={(h1,h2) : False for (h1,h2) in homologues}   
    bond=[]
    pos_xyz=np.zeros((n_steps,syst["np"],3))
    for i in range(n_steps):
        #time("Start run")
        
        hoomd.run(length_steps, profile=False, quiet=True) 
        particles_pos = np.array([particle.position for particle in group_hic])
        D1 = cdist(particles_pos, particles_pos) # Compute all distance between particles
        pos_xyz[i,:,:]=np.array([particles_pos[k,:] for k in range(syst["np"])])

         
        bille_connection=True #Booléen pour permettre le lien entre deux particules cassées
        if bille_connection:
            for j in Homologues_kept:
                h1,h2=homologues[j]
                if (D1[h1,h2]<=distance_requise_pour_bond_homologues) and not b[(h1,h2)] : 
                    bond_number=system.bonds.add('fh-fh', h1, h2)
                    force[j,i]=mesure_force(D1[h1,h2],80,1)
                    print("force bond", h1 , "-", h2 , "time_step_force", i)
                    b[(h1,h2)]=True
            for l in range (len(homologues)): 
                h1,h2=homologues[l]
                if b[(h1,h2)]==True :   
                    force[l,i]=mesure_force(D1[h1,h2], 80, 1)
                    Zip = False #Booléen pour activer processus de zipping
                    if Zip : 
                        if force[l,i]>3 and (h1+1,h2+1) in homologues and not b[(h1+1,h2+1)]: 
                            bond_number=system.bonds.add('DNA-DNA', h1+1, h2+1)
                            b[(h1+1,h2+1)]=True
                            print("force bond zip", h1+1, "-", h2+1, "time_step_force", i)
                            #system.bonds.remove(bond_number)  #supprimer le bond         
 
    #Enregistrer forces, distance et positions particules homologues
    if not os.path.exists("./forces"):
        os.makedirs("./forces")
    list=os.listdir("./forces")
    i = 0
    while "force_simul"+str(i)+".npy" in list : 
        i += 1            
    np.save("./forces/force_simul"+str(i)+".npy", force)
    np.save("./forces/homo_kept_"+str(i)+".npy", Homologues_kept)

    if not os.path.exists("./distance"):
        os.makedirs("./distance")
    list_1=os.listdir("./distance")
    j=0
    while "pos_xyz"+str(j)+".npy" in list_1 : 
        j += 1

    if not os.path.exists("./positions"):
        os.makedirs("./positions")
    np.save("./positions/pos_xyz"+str(j)+".npy", pos_xyz)
    np.save("./positions/syst_len"+str(j)+".npy",syst["len_polymers"])
    
 


def test_one(attached=True, args={}):
    syst = {}
    syst["len_polymers"] = [int(i) for i in args["len"] for _ in range(2)]
    syst["density"] = args["density"] 
    syst["verbose"] = False 
    syst["attached"] = attached
    syst["length_steps"] = 1000
    syst["equi"] = 10

    simulate(syst, args["nsteps"], data_folder=args["root"])


if __name__ == "__main__":

    """
    --cpu 
    --nsteps 5000
    --len 24  
    --len 30 
    --len 45 
    --density 0.001 
    --root meiose
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action="store_true")
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--attached', action="store_true")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--nsteps', type=int, default=1000)
    parser.add_argument("--len", action='append', required=True)
    parser.add_argument('--density', type=float, default=0.05)

    parser.add_argument('--root', type=str, default="./repli")

    args = parser.parse_args()
   
    if args.cpu:
        init_cmd = "--mode cpu"
    else:
        init_cmd = "--mode=gpu --gpu=%i " % args.gpu

    if args.debug:
        init_cmd += " --notice-level=10" 
    hoomd.context.initialize(init_cmd)
    my_dict = args.__dict__

    test_one(attached=args.attached, args=args.__dict__)