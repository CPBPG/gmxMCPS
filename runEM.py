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

1. Haloi N. Rationalizing the generation of broad spectrum antibiotics with the addition 
   of a positive charge. Chem Sci. 2021 Nov 24; 12(45): 15028–15044.

2. Our manuscript.
"""

import argparse
import os
from vmd import *
from rdkit import Chem
import numpy as np
import time
import subprocess

def inarg_sanity(args):
    """Check some of the input cmdline arguments."""
    if args.gmx_command == None:
        if is_tool("gmx"):
            args.gmx_command="gmx"
        elif is_tool("gmx_mpi"):
            args.gmx_command="gmx_mpi"
        else:
            raise Exception("""'gmx' or 'gmx_mpi' not found. Please use the --gmx_command 
                            flag to specify the correct gromacs executable.""")
    else:
        if not is_tool(args.gmx_command):
            raise Exception(f"""'{args.gmx_command}' not found. Please supply the correct
                            gromacs executable.""")
    return args

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
    """"""
    with open(file, 'r') as f:
        inp=f.readlines()
    out=inp[-1].split()
    return float(out[1])+float(out[2])

def paramfile_parser(file):
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

def write_posre(posre, serial, kappa):
    if len(serial)==len(kappa):
        out="""[ position_restraints ]
; atom  type      fx      fy      fz\n"""
        for i, s in enumerate(serial):
            out+=f"{s}     1   {kappa[i]}  {kappa[i]}   {kappa[i]}\n"
        with open(posre, 'w') as f:
            f.write(out)
    return None

def vec2d_angle(v1, v2):
    #angle between two 2D vectors
    v1_norm=v1/np.linalg.norm(v1)
    v2_norm=v2/np.linalg.norm(v2)
    dot=np.dot(v1_norm, v2_norm)
    det=np.linalg.det(np.array([v1_norm, v2_norm]))
    return np.arctan2(det, dot)

def write_frame(filename):
    with open(filename, 'w') as f:
        f.write("0")
    return None

def read_cpt(filename):
    return int(filename[:-4].split("_")[2])

def read_ie(filename):
    dat=np.loadtxt(filename)
    return dat[:,1]

def main(args):
    #read input params
    param_dict=paramfile_parser(args.param_file)
    emtrajid=molecule.new("emout")
    trajid=molecule.new("channel")
    molecule.read(trajid, filename=args.channel_traj, filetype='pdb', waitfor=-1)
    print("Trajectory loaded..")
    nframes=molecule.numframes(trajid)
    print(f"Number of frames: {nframes}")
    fname_prefix=args.channel_traj[:-4]
    pie=[]
    if args.cont==False:
        os.mkdir(f"{fname_prefix}_minimize")
    initframe=0
    if args.cont==True:
        molecule.read(emtrajid, filename=args.trajcnt, filetype='pdb', waitfor=-1)
        initframe=molecule.numframes(emtrajid)
        pie=list(read_ie(args.piefl))
    for frame in range(initframe, nframes):
        molecule.write(trajid, filetype='pdb', filename=f"{fname_prefix}_{frame}.pdb", first=frame, last=frame)
        time.sleep(2)
        lig=atomsel(selection="not protein")
        ligname=lig.resname[0]
        #the idea is to now select the sc of these residues and keep these atoms flexible
        flexsel=atomsel(selection=f"(protein within 4.0 of resname {ligname}) and not hydrogen", molid=trajid)
        flexsel.frame=frame
        flexsel.update()
        nbresidues=list(set(flexsel.residue))
        prot=atomsel(selection="protein and not hydrogen")
        if len(nbresidues)>0:
            nbresstr=""
            for r in nbresidues:
                nbresstr+=str(r)+" "
            nbres_side=atomsel(selection=f"residue {nbresstr} and not type N C O CA and not hydrogen")
            prot.beta=5000
            nbres_side.beta=200
            write_posre(args.input_posre, prot.serial, prot.beta)
        else:
            prot.beta=5000
            write_posre(args.input_posre, prot.serial, prot.beta)
            
        cmd1str=f"{args.gmx_command} editconf -f {fname_prefix}_{frame}.pdb -o {fname_prefix}_{frame}.gro -box 10 10 10".split()
        try:
            cmd1=subprocess.Popen(cmd1str)
            cmd1.wait()
        except:
            write_ie(pie, f"{fname_prefix}_PIE_temp.dat")
            molecule.write(emtrajid, filename=f"{fname_prefix}_mintraj_temp.pdb", filetype="pdb")
        cmd2str=f"{args.gmx_command} grompp -f {args.minim_mdp} -c {fname_prefix}_{frame}.gro -p {args.input_top} -o {fname_prefix}_minimize/{fname_prefix}_em_{frame}.tpr -r {fname_prefix}_{frame}.gro -maxwarn 4".split()
        try:
            cmd2=subprocess.Popen(cmd2str)
            cmd2.wait()
        except:
            write_ie(pie, f"{fname_prefix}_PIE_temp.dat")
            molecule.write(emtrajid, filename=f"{fname_prefix}_mintraj_temp.pdb", filetype="pdb")
        os.system(f"rm {fname_prefix}_{frame}.pdb {fname_prefix}_{frame}.gro")
        os.chdir(f"{fname_prefix}_minimize")
        cmd3str=f"{args.gmx_command} mdrun -nt 16 -s {fname_prefix}_em_{frame}.tpr -deffnm {fname_prefix}_emout_{frame} -v".split()
        try:
            cmd3=subprocess.Popen(cmd3str)
            cmd3.wait()
        except:
            write_ie(pie, f"../{fname_prefix}_PIE_temp.dat")
            molecule.write(emtrajid, filename=f"../{fname_prefix}_mintraj_temp.pdb", filetype="pdb")
        
        molecule.read(emtrajid, filename=f"{fname_prefix}_emout_{frame}.gro", filetype='gro', waitfor=-1)
        if frame==0:
            cmd4str=f"{args.gmx_command} grompp -f ../pie.mdp -c {fname_prefix}_emout_{frame}.gro -p ../{args.input_top} -o ../{fname_prefix}_ie.tpr -maxwarn 4".split()
            try:
                cmd4=subprocess.Popen(cmd4str)
                cmd4.wait()
            except:
                write_ie(pie, f"../{fname_prefix}_PIE_temp.dat")
                molecule.write(emtrajid, filename=f"../{fname_prefix}_mintraj_temp.pdb", filetype="pdb")
        cmd5str=f"{args.gmx_command} mdrun -nt 16 -deffnm {fname_prefix}_ie -rerun {fname_prefix}_emout_{frame}.gro -nb cpu -s ../{fname_prefix}_ie.tpr"
        try:
            cmd5=os.system(cmd5str)
        except:
            write_ie(pie, f"../{fname_prefix}_PIE_temp.dat")
            molecule.write(emtrajid, filename=f"../{fname_prefix}_mintraj_temp.pdb", filetype="pdb")
        time.sleep(1)
        try:
            os.system(f"printf '16 17' | {args.gmx_command} energy -f {fname_prefix}_ie.edr -o {fname_prefix}_ener_{frame}.xvg")
        except:
            write_ie(pie, f"../{fname_prefix}_PIE_temp.dat")
            molecule.write(emtrajid, filename=f"../{fname_prefix}_mintraj_temp.pdb", filetype="pdb")
        pie.append(get_ie(f"{fname_prefix}_ener_{frame}.xvg"))
        
        cmd7str=f"rm {fname_prefix}_em* {fname_prefix}_ie.edr {fname_prefix}_ie.log {fname_prefix}_ener_{frame}.xvg"
        try:
            cmd7=subprocess.Popen(cmd7str, shell=True)
        except:
            write_ie(pie, f"../{fname_prefix}_PIE_temp.dat")
            molecule.write(emtrajid, filename=f"../{fname_prefix}_mintraj_temp.pdb", filetype="pdb")
        os.chdir("../")
        print(f"Processed frame {frame}...")
    molecule.write(emtrajid, filename=f"{fname_prefix}_minimize/{fname_prefix}_mintraj.pdb", filetype="pdb")
    write_ie(pie, f"{fname_prefix}_minimize/{fname_prefix}_PIE.dat")
    print(f">> Wrote protein interaction energy and minimized trajectory for channel solute configurations to file.")
    molecule.delete(trajid)
    print("""######## Determine the translation and rotation angles of the solute #########""")
    zpos=[]
    theta_rad=[]
    phi_rad=[]
    for frame in range(nframes):
        headsel=atomsel(selection=f"not protein and type {param_dict['HEAD_ATM']}", molid=emtrajid)
        headsel.frame=frame
        headsel.update()
        tailsel=atomsel(selection=f"not protein and type {param_dict['TAIL_ATM']}", molid=emtrajid)
        tailsel.frame=frame
        tailsel.update()
        elfhead=atomsel(selection=f"protein and resid {param_dict['ELF_HEAD_RESID']} and type CA ", molid=emtrajid)
        elfhead.frame=frame
        elfhead.update()
        elftail=atomsel(selection=f"protein and resid {param_dict['ELF_TAIL_RESID']} and type CA ", molid=emtrajid)
        elftail.frame=frame
        elftail.update()
        allsel=atomsel(selection="not protein", molid=emtrajid)
        drugvecnorm=np.subtract(headsel.center(), tailsel.center())/np.linalg.norm(np.subtract(headsel.center(), tailsel.center()))
        elfvecnorm=np.subtract(elfhead.center(), elftail.center())/np.linalg.norm(np.subtract(elfhead.center(), elftail.center()))
        theta=np.arccos(drugvecnorm[2])
        theta_rad.append(theta*180/np.pi)
        #### The calculation of azimuthal seems to be problematic 

        phi=vec2d_angle(elfvecnorm[:-1], drugvecnorm[:-1])
        if phi<0:
            phi=phi+2*np.pi
        phi_rad.append(phi*180/np.pi)
        allsel.frame=frame
        allsel.update()
        zpos.append(allsel.center()[2])
    
    write_transrot(zpos, theta_rad, phi_rad, f"{fname_prefix}_minimize/{fname_prefix}_rottrans.dat")
    print(f">> Wrote translation and rotation values for channel solute configurations to output file.")
    molecule.delete(emtrajid)

if __name__=="__main__":
    #setup input commandline arguments
    parser=argparse.ArgumentParser(prog='MCPS.py', 
                                   description="The script implements the MCPS algorithm.")
    parser.add_argument("-p", "--param_file", action='store', 
                        default='params.dat', type=str, 
                        help="Input parameter files.")
    parser.add_argument("-c", "--channel_traj", action='store', 
                        default='channel_drug.pdb', type=str, 
                        help="Input pdb traj file containing the channel and the drug molecule configurations.")
    parser.add_argument("-t", "--input_top", action='store', 
                        default='topol.top', type=str, 
                        help="Topology file to be used for the minimization.")
    parser.add_argument("-pr", "--input_posre", action='store', 
                        default='posre_protein.itp', type=str, 
                        help="Position restraints file for the protein.")
    parser.add_argument("-m", "--minim_mdp", action='store', 
                        default='em.mdp', type=str, 
                        help="Gromacs mdp file for energy minimization.")
    parser.add_argument("-gmxc", "--gmx_command", action='store', 
                        default=None, type=str, 
                        help="Gromacs executable name. The script looks for gmx or gmx_mpi.")
    parser.add_argument("-ct", "--cont", action='store', type=bool, 
                        default=False, 
                        help="True if a continuation is requested.")
    parser.add_argument("-trj", "--trajcnt", type=str,
                        default=None, help="The traj file for continuation.")
    parser.add_argument("-pie", "--piefl", type=str, default=None, 
                         help="PIE temp output file for continuation")
    inargs=inarg_sanity(parser.parse_args())
    main(inargs)
