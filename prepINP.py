#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Abhishek Acharya
        Postdoctoral Fellow
        School of Science
        Constructor University Bremen
        Campus Ring 1, 
        Bremen

Email: abhi117acharya@gmail.con

Note on python version: Python greater than 3.8 is not compatible with vmd-python and rdkit.

Credits: Some functions in the python script runMCPS.py have been sourced from the original implementation of 
the MCPS algorithm described in Haloi N. et al. Chem Sci. 2021 Nov 24; 12(45): 15028–15044,
available on Github (https://github.com/nandanhaloi123/MonteCarloPathwaySearch).

In you use the present implementation of gmxMCPS, please cite the following:

1. Haloi N et al. Rationalizing the generation of broad spectrum antibiotics with the addition 
   of a positive charge. Chem Sci. 2021 Nov 24; 12(45): 15028–15044.
2. Acharya A et al. 

"""

import argparse
import os
from vmd import *
from rdkit import Chem
import math
import numpy as np
import time

def is_tool(name):
    """Check whether `name` is on PATH and marked as executable."""
    from shutil import which
    return which(name) is not None

def write_ie(pie, file):
    """write Interaction Energy to file."""
    fidx=np.arange(len(pie))
    pie=np.array(pie)
    np.savetxt(file, np.vstack([fidx, pie]).T)
    return None

def write_transrot(z, theta, phi, file):
    """Write z, incination and azimuthal to file."""
    z=np.array(z)
    theta=np.array(theta)
    phi=np.array(phi)
    fidx=np.arange(z.shape[0])
    np.savetxt(file, np.vstack([fidx, z, theta, phi]).T)
    return None

def get_ie(file):
    """Extract the interaction energy value from the given file."""
    with open(file, 'r') as f:
        inp=f.readlines()
    out=inp[-1].split()
    return float(out[1])+float(out[2])

def write_posre(posre, serial, kappa):
    """Updates the posre file with the new serial and kappa values."""
    if len(serial)==len(kappa):
        out="""[ position_restraints ]
; atom  type      fx      fy      fz\n"""
        for i, s in enumerate(serial):
            out+=f"{s}     1   {kappa[i]}  {kappa[i]}   {kappa[i]}\n"
        with open(posre, 'w') as f:
            f.write(out)
    return None

def paramfile_parser(file):
    """Reads the parameter file data into a dictionary."""
    param_dict={}
    with open(file, 'r') as f:
        dat=f.readlines()
    for l in dat:
        i=l[:-1].split('=')
        if i[0] in ["SOLUTE", "HEAD_ATM", "TAIL_ATM"]:
            param_dict[i[0]]=i[1]
        elif i[0] in ["NFIB", "N_CHANNELS", "ELF_HEAD_RESID", "ELF_TAIL_RESID"]:
            param_dict[i[0]]=int(i[1])
        elif i[0] in ["X_DIM", "Y_DIM", "Z_DIM"]:
            param_dict[i[0]]=[float(x) for x in i[1][1:-1].split(',')]
        else:
            raise Exception(f"{i[0]} is an invalid parameter string. Check {file}.")
    param_dict["N_RINGS"]=None
    param_dict["RING_ATMS"]={}
    return param_dict

def fib_theta(n):
    return [math.acos(1-(2*(i+1)/float(n))) for i in range(n)]

def fib_phi(n):
    pi=3.141592653589793
    gratio=1.618033988749895
    return [(2*(i+1)*pi)/gratio for i in range(n)]

def format_transout(inp):
    inplist=np.array(inp.replace("{", "").replace("}", "").split(), dtype='float64')
    outlist=inplist.reshape([4, 4]).T.flatten()
    return outlist

def orient(selection, vec1, vec2):
    vecnorm1=vec1/np.linalg.norm(vec1)
    vecnorm2=vec2/np.linalg.norm(vec2)
    rotvec=np.cross(vecnorm1, vecnorm2)
    sine=np.linalg.norm(rotvec)
    cosine=np.dot(vec1, vec2)
    angle=np.arctan2(sine, cosine)
    center=selection.center()
    rotmat=evaltcl(f"trans center  {{ {center[0]} {center[1]} {center[2]} }} axis {{ {rotvec[0]} {rotvec[1]} {rotvec[2]} }} {angle} rad")
    return format_transout(rotmat)

def transaxis(axis, amount, unit='deg'):
    if unit=="rad":
        rotmat=evaltcl(f"transaxis {axis} {amount} rad")
    else:
        rotmat=evaltcl(f"transaxis {axis} {amount}")
    return format_transout(rotmat)

def check_ring_pierce(frame, ring_dict, ring_sels, unqatoms):
    ringatm_residue = ring_sels[0][0].residue[0]
    #set the frame number for all ring atom selections
    for i in ring_sels.keys():
        for j in ring_sels[i]:
            j.frame=frame

    #get the centers for all the updated ring atoms
    ringatom_centers={}
    for i in ring_sels.keys():
        ringatom_centers[i]=[j.center() for j in ring_sels[i]]
    
    #generate the triangle vertices for all ring atoms
    v0_ct=-1
    v0_dict={}
    for i in ringatom_centers.keys():
        for j in range(len(ringatom_centers[i])-2):
            v0_ct+=1
            v0_dict[v0_ct]=ringatom_centers[i][0]
    
    
    v1_ct=-1
    v1_dict={}
    for i in ringatom_centers.keys():
        for j in range(1, len(ringatom_centers[i])-1):
            v1_ct+=1
            v1_dict[v1_ct]=ringatom_centers[i][j]
            
    v2_ct=-1
    v2_dict={}
    for i in ringatom_centers.keys():
        for j in range(2, len(ringatom_centers[i])):
            v2_ct+=1
            v2_dict[v2_ct]=ringatom_centers[i][j]
        
    #get the atom indices of all the protein atoms within a 2 Angstrom radius of the ring atoms
    protsel=atomsel(selection=f"protein and within 2 of (residue {ringatm_residue} and name {' '.join(unqatoms)})")
    protsel.frame=frame
    protsel.update()
    protein_indices=protsel.index
    #loop over protein indices and obtain the associated bonds
    for index in protein_indices:
        indsel=atomsel(selection=f"index {index}")
        indsel.frame=frame
        indsel.update()
        indbonds=indsel.bonds[0]
        indstr=""
        for i in indbonds:
            indstr+=str(i)+" "
        bonds_close=atomsel(selection=f"index {indstr} and within 10 of (residue {ringatm_residue} and name {' '.join(unqatoms)})")
        bonds_close.frame=frame
        bonds_close.update()
        bonds_closelist=bonds_close.index
        
        cent0=indsel.center()
        
        for bond in bonds_closelist:
            b_atom=atomsel(selection=f"index {bond}")
            b_atom.frame=frame
            b_atom.update()
            cent1=b_atom.center()
            #now we iterate over the triangles
            for t in v0_dict.keys():
                out=check_intersect(cent0, cent1, v0_dict[t], v1_dict[t], v2_dict[t])
                if out==1:
                    return 1
    return 0

def vec2d_angle(v1, v2):
    #angle between two 2D vectors
    v1_norm=v1/np.linalg.norm(v1)
    v2_norm=v2/np.linalg.norm(v2)
    dot=np.dot(v1_norm, v2_norm)
    det=np.linalg.det([v1_norm, v2_norm])
    return np.arctan2(det, dot)
        
def check_intersect(c0, c1, v1, v2, v3):
    u=np.subtract(v2, v1)
    v=np.subtract(v3, v1)
    n=np.cross(u, v)
    
    dr=np.subtract(c1, c0)
    w0=np.subtract(c0, v1)
    a=-np.dot(n, w0)
    b=np.dot(n, dr)
    
    if np.absolute(b)<0.01:
        return 0
    if a/b < 0:
        return 0
    if a/b>1.0:
        return 0
    
    ist= np.add(c0, (a/b)*dr)
    uu = np.dot(u, u)
    uv = np.dot(u, v)
    vv = np.dot(v, v)
    w = np.subtract(ist, v1)
    wu = np.dot(w, u)
    wv = np.dot(w, v)
    D = uv*uv - uu*vv
    s = (uv*wv-wu*vv)/D
    
    if s<0 or s>1:
        return 0
    
    t=(uv*wu-uu*wv)/D
    
    if t<0 or (s+t)>1:
        return 0
    
    return 1
    
    
    

def main(args): 
    #read input params
    param_dict=paramfile_parser(args.param_file)
    
    #initial processing of input pdb (extract protein and ligand, remove VS atoms if any.)
    print(">> Reading inputs...")
    system=Molecule.Molecule()
    system.load(filename=args.input_pdb)
    temp=atomsel(selection=f"protein or resname {param_dict['SOLUTE']}")
    if args.remove_vs==True:
        vstypes=" ".join([i for i in set(temp.type) if "M" in i])
        protlig=atomsel(selection=f"(protein or resname {param_dict['SOLUTE']}) and not type {vstypes}")
    else:
        protlig=atomsel(selection=f"(protein or resname {param_dict['SOLUTE']})")
    protlig.write(filename="protlig.pdb", filetype='pdb')
    protein=atomsel("protein")
    n_protresidue=max(protein.residue)+1
    print(">> Analysing solute...")
    print(f"Solute ID: {param_dict['SOLUTE']}")
    #if ring info not provided, extract from ligand structure
    if param_dict['N_RINGS']==None:
        ligtemp=atomsel(selection=f"resname {param_dict['SOLUTE']}")
        ligvstypes=" ".join([i for i in set(ligtemp.type) if "M" in i])
        ligresidues=list(set(ligtemp.residue))
        ligtemp=atomsel(selection=f"(resname {param_dict['SOLUTE']} and residue {ligresidues[0]}) and not type {ligvstypes}")
        ligtypes=np.array(ligtemp.type)
        ligtemp.write(filename="lig.pdb", filetype='pdb')
        lig=Chem.MolFromPDBFile("lig.pdb", removeHs=False)
        ssrlist=Chem.GetSymmSSSR(lig)
        param_dict['N_RINGS']=len(ssrlist)
        for i in range(param_dict['N_RINGS']):
            param_dict['RING_ATMS'][i]=list(ligtypes[list(ssrlist[i])])
        del ligtemp,lig
    if param_dict['N_RINGS']>0:
        print("Rings detected. Will be used for checking ring piercings.")
        print(f"Number of rings: {param_dict['N_RINGS']}")
    
    print(">> Generating Fibonacci points...")
    #generate a list theta and phi values using fibonacci sphere of parameter 'NFIB' points
    theta=fib_theta(param_dict['NFIB'])
    phi=fib_phi(param_dict['NFIB'])
    
    #load the protein and ligand file and perform the conformational search 
    molid=molecule.new("protlig")
    protlig=molecule.read(molid, filename="protlig.pdb", filetype='pdb')
    
    drugs=atomsel(selection="not protein")
    drugresidues=list(set(drugs.residue))
    
    drug_objs=[atomsel(selection=f"not protein and residue {index}") for index in drugresidues]
        
    headatm_objs=[atomsel(selection=f"not protein and name {param_dict['HEAD_ATM']} and residue {index}") for index in drugresidues]
    tailatm_objs=[atomsel(selection=f"not protein and name {param_dict['TAIL_ATM']} and residue {index}") for index in drugresidues]
    
    #the axis vector of the drug is now centered at origin
    headpositions=[np.array(headatm_objs[i].center()) for i in range(len(drug_objs))]
    tailpositions=[np.array(tailatm_objs[i].center()) for i in range(len(drug_objs))]
    
    middlepostions=[(tailpositions[i] + (headpositions[i] - tailpositions[i])*0.5) for i in range(len(headpositions))]
    for i in range(len(drug_objs)):
        drug_objs[i].moveby(-middlepostions[i])
        #align vector to the membrane normal
        headpos=np.array(headatm_objs[i].center())
        transformat=orient(drug_objs[i], headpos, np.array([0, 0, 1]))
        drug_objs[i].move(transformat)
    
    #generate antibiotic conformational sphere
    print(f">> Generating {param_dict['SOLUTE']} fibonacci sphere ...")
    t1=time.time()
    self_rot=60
    j=0
    while j <= 360:
        for a in range(len(drug_objs)):
            drug_objs[a].frame=0
            drug_objs[a].update()
            drug_objs[a].move(transaxis('z', self_rot))
        i=0
        while i < param_dict['NFIB']:
            molecule.dupframe(molid, frame=0)
            for a in range(len(drug_objs)):
                drug_objs[a].frame=int((param_dict['NFIB']*j)/(self_rot) + i + 1)
                drug_objs[a].update()
                drug_objs[a].move(transaxis('y', theta[i], unit='rad'))
                drug_objs[a].move(transaxis('z', phi[i], unit='rad'))
            i+=1
        j+=self_rot
    
    molecule.delframe(molid, first=0, last=0, stride=0)
    molecule.write(molid, filetype="pdb", filename=f"drug_orientations_sphere.pdb")
    ####Translate the drug orientation sphere throughout the channel###
    #define the box
    print("Translating the sphere throughout the specified grid ...")
    protresid = list(range(0, n_protresidue, int(n_protresidue/param_dict["N_CHANNELS"])))
    protresid.append(n_protresidue)
    #get a list of the channel atomsel objects
    channel_objs=[atomsel(selection=f"protein and residue {protresid[i]} to {protresid[i+1]-1}")  for i in range(len(protresid)-1)]
    channel_drugs=[atomsel(selection=f"protein and residue {protresid[i]} to {protresid[i+1]-1} or residue {int(n_protresidue)+i}")  for i in range(len(protresid)-1)]
    channel_gridpos=[]
    #for each channel generate the 3D gridpositions 
    for channel in channel_objs:
        ccent= channel.center()
        min_x=int(ccent[0])-param_dict['X_DIM'][0]
        max_x=int(ccent[0])+param_dict['X_DIM'][1]
        min_y=int(ccent[1])-param_dict['Y_DIM'][0]
        max_y=int(ccent[1])+param_dict['Y_DIM'][1]
        min_z=int(ccent[2])-param_dict['Z_DIM'][0]
        max_z=int(ccent[2])+param_dict['Z_DIM'][1]
        #generate all the positions in the xyz gridspace
        xpos=np.linspace(min_x, max_x, num=int(max_x-min_x+1), endpoint=True)
        ypos=np.linspace(min_y, max_y, num=int(max_y-min_y+1), endpoint=True)
        zpos=np.linspace(min_z, max_z, num=int(max_z-min_z+1), endpoint=True)
        channel_gridpos.append(np.array(np.meshgrid(xpos, ypos, zpos)).T.reshape(-1, 3))
    
    dt=time.time()-t1
    print(f"Done! This step took{dt:.2f} seconds. Time elapsed: {dt:.2f} seconds.")
    #for each channel generate a trajectory with antibiotic at all grid positions and remove clashes
    print("""################# Generating configurations #################\n\n""")
    t2=time.time()
    #generating the atomselections for all ring atoms for all residues
    ringseldict_list=[]
    for i in range(param_dict['N_CHANNELS']):
        ringatom_sels={}
        for a in param_dict['RING_ATMS'].keys():
            ringatom_sels[a]=[atomsel(selection=f"residue {int(n_protresidue)+i} and name {j}") for j in param_dict['RING_ATMS'][a]]
        ringseldict_list.append(ringatom_sels)
    #list of unique ring atom names
    
    ring_name_unq=[]
    for i in param_dict['RING_ATMS'].keys():
        ring_name_unq+=param_dict['RING_ATMS'][i]
    ring_name_unq=set(ring_name_unq)
        
    
    for i, channel in enumerate(channel_objs):
        print(f"For channel {i}")
        molecule.set_frame(molid, frame=0)
        nframes=molecule.numframes(molid)
        fcount=nframes-1
        for pos in channel_gridpos[i]:
            for f in range(nframes):
                fcount+=1
                molecule.dupframe(molid, frame=f)
                drug_objs[i].frame=fcount
                drug_objs[i].update()
                drug_objs[i].moveby(pos)
                #check for a clash
                clash=atomsel(selection=f"protein within 1 of residue {drugresidues[i]}")
                clashcount=len(clash.index)
                status=check_ring_pierce(fcount, param_dict['RING_ATMS'], ringseldict_list[i], ring_name_unq)
                if clashcount > 4:
                    molecule.delframe(molid, first=fcount, last=fcount, stride=0)
                    fcount-=1
                elif status==1:
                    molecule.delframe(molid, first=fcount, last=fcount, stride=0)
                    fcount-=1
                else:
                    pass
        
        molecule.write(molid, filetype="pdb", filename=f"channel_drug_{i+1}.pdb", first=nframes, selection=channel_drugs[i])
        with open(f"NFRAMES_{i+1}.dat", 'w') as f:
            f.write(f"{int(molecule.numframes(molid))}\n")
        
        print(f">> Written trajectory for channel {i}.")
        molecule.delframe(molid, first=nframes)
    
    dt=time.time()-t2
    print(f"Done! Total run took {dt:.2f} seconds. Time elapsed: {dt:.2f} seconds.")
    
    print("Finished!!")
    return None

if __name__=="__main__":
    #setup input commandline arguments
    parser=argparse.ArgumentParser(prog='prepINP.py', 
                                   description="""The script generates all possible configurations of
                                   a solute within a channel(s). The configurations are generated
                                   by first generating rotations of the solute that are points on a
                                   fibonacci sphere, followed by translation of the sphere over 
                                   grid points specified by user. Finally, the configurations
                                   that lead to clashes and ring-piercings are removed. The 
                                   code takes as input a param file that specifies the number of 
                                   channels, solute ID, the number of fibonacci points to use, head 
                                   and tail atoms of the solute, XYZ grid dimensions 
                                   (see manual.pdf for details on params.). An input pdb file 
                                   containing the fully equilibrated system is also required as
                                   input. A fully equilibration is important prior to initiating
                                   any enhanced sampling calculations. For gromacs, you can generate
                                   the pdb by running editconf on the equilibrated configuration.
                                   An optionally, the  --remove_vs flags can be set to True, if the input
                                   has virtual site atoms and one needs to remove those prior to the
                                   enhanced sampling run (although a bit unusual). By default this
                                   is set to False.
                                   """)
    parser.add_argument("-p", "--param_file", action='store', 
                        default='params.dat', type=str, 
                        help="Input parameter files.")
    parser.add_argument("-c", "--input_pdb", action='store', 
                        default='system.pdb', type=str, 
                        help="Input pdb file containing the channel and the drug molecule.")
    parser.add_argument("-rvs", "--remove_vs", action='store', 
                        default=False, type=bool, 
                        help="option to remove virtual site atoms.")
    args=parser.parse_args()
    main(args)
