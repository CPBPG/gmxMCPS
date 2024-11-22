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

import vmd
import glob
import sys
import argparse
import numpy as np
import os

def create_channel_selections(nresidue, nchannels):
    """
    Creates selections for individual channel monomers.
    
    Parameters:
        nresidue (int): Number of residues per monomer
        nchannels (int): Number of channels
    
    Returns:
        list: A list of strings containing the selections.
    """
    selection_list=[]
    for i in range(nchannels):
        selection_list.append(f"protein and residue {i*nresidue} to {((i+1)*nresidue)-1}")
    return selection_list

def assemble_components(inplist, outfile="full.pdb", fileorder=["protein", "lipid", "water", "ions"] ):
    """
    Combines the components provided as *args into a combined file. 
    """
    for f in inplist:
        cmd=f"grep 'ATOM' {f} >> combined.pdb"
        os.system(cmd)
    molid=vmd.molecule.new("combined")
    vmd.molecule.read(molid, filename="combined.pdb", filetype="pdb")
    for i in range(len(fileorder)):
        s=vmd.atomsel(selection=fileorder[i], molid=molid)
        s.write(filetype='pdb', filename=f"part_{i}.pdb")
        
    os.system("rm combined.pdb")
    for i in range(len(fileorder)):
        cmd=f"grep 'ATOM' part_{i}.pdb >> combined.pdb"
        os.system(cmd)
    molid=vmd.molecule.new("reodered")
    vmd.molecule.read(molid, filename="combined.pdb", filetype="pdb")
    fullsys=vmd.atomsel(selection="all", molid=molid)
    fullsys.write(filetype="pdb", filename=outfile)
    os.system("rm combined.pdb")
    return None

def paramfile_parser(paramfile):
    paramtable={"NUMB": 0,
            "U_POS": None,
            "BOXDIM": None,
            "RESNAME": None
            }
    
    def pos_process(field):
        if field.startswith("{"):
            poslist=field[1:-2].split(",")
            out=[float(i) for i in poslist]
        elif field.startswith("["):
            ran=field[1:-2].split(",")
            if len(ran)==3:
                out=np.linspace(float(ran[0]), float(ran[1]), int(ran[2]))
            else:
                raise ValueError("The square bracket format expects three values for {begin}, {end} and {num} (see numpy.linspace).")
        else:
            raise ValueError("Use either {} or [] format to specify the umbrella positions.")
        return list(out)
    
    with open(paramfile, 'r') as inpf:
        params=inpf.readlines()
    
    for line in params:
        if line.startswith("#") or line.startswith("\n"):
            pass
        else:
            argflags=line.split("=")
            if argflags[0]=="NUMB":
                print("WARNING: THE NUMB VALUE SHOULD MATCH THE U_POS LENGTH!!")
                paramtable[argflags[0]]=int(argflags[1])
            elif argflags[0]=="U_POS":
                paramtable[argflags[0]]=pos_process(argflags[1])
            elif argflags[0]=="BOXDIM":
                print("WARNING: MAKE SURE THAT THE BOXDIM VALUES ARE SPECIFIED IN ANGSTROMS!!")
                paramtable[argflags[0]]=argflags[1][:-1].split(",")
            elif argflags[0]=="RESNAME":
                paramtable[argflags[0]]=argflags[1].strip()
            else:
                raise Exception(f"Flag {argflags[0]} seems to be wrong.")
    return paramtable

def insert_boxdim(pdbfile, boxdim):
    new_dim_str=f"CRYST1    {boxdim[0]}    {boxdim[1]}    {boxdim[2]}  90.00  90.00  90.00 P 1           1\n"
    with open(pdbfile, "r") as f:
        inp=f.readlines()
    for i, line in enumerate(inp):
        if "CRYST1" in line:
            inp[i]=new_dim_str
            break
    with open(pdbfile, "w") as f:
        f.write("".join(inp))
    return None

def circular_shift(input_list, x):
    """
    Circularly shift a list by x positions.

    Parameters:
    - input_list: The input list to be shifted.
    - x: The number of positions to shift the list.

    Returns:
    A new list circularly shifted by x positions.
    """
    length = len(input_list)

    # Ensure x is within the range of the list length
    x = x % length

    # Perform circular shift using list slicing
    shifted_list = input_list[-x:] + input_list[:-x]

    return shifted_list

def main(args):
    """
    Processes the input trajectory files obtained from the MCPS analysis, 
    and merges the antibiotic-channel configurations in the trajectory with the
    full system provided as <refpdb>. The code takes as additional inputs, a 
    paramfile containing info on the number of umbrella windows, the 
    corresponding umbrella mean positions, simulations box dimensions and
    the number of asymmetric channels units that are present in the refpdb file.
    """
    traj_files=args.trajlist
    ntraj=len(traj_files)
    refpdb=args.refpdb
    nchannels=args.nchannels
    params=paramfile_parser(args.paramfile)
    NUMB=params["NUMB"]
    UPOS=params["U_POS"]
    BOXDIM=params["BOXDIM"]
    RESNAME=params["RESNAME"]
    verbose=args.verbose
    staggered_placement=args.staggered_placement
    staggered_shift=args.staggered_shift
    
    if verbose:
        print("Following inputs parameters will be used:")
        print(f"NUMB: {NUMB}")
        print(f"U_POS: {UPOS}")
        print(f"BOXDIM: {BOXDIM}")
        print(f"RESNAME: {RESNAME}")
    if ntraj!=nchannels:
        print(f"Number of input transition traj files ({ntraj}) does not match the input value for the flag --nchannels ({nchannels}). ")
        if ntraj>nchannels:
            raise Exception("Number of input transition trajfiles cannot be greater than the value of nchannels.")
        elif ntraj!=1:
            raise Exception(f"Only a single  or {nchannels} transition trajectory files (= nchannels) can be provided as input.")
        else:
            print(f">> Initiating TASS setup using {traj_files[0]}. Note that pore number {args.poreid} will be used. To change this, rerun using flag --poreid.")
    
        
    
    refid=vmd.molecule.new("reference")
    vmd.molecule.read(refid, filename=refpdb, filetype="pdb")
    
    #determine the residue number for individual channel monomers automatically
    protein=vmd.atomsel(selection="protein", molid=refid)
    nresidue=int((np.max(protein.residue)+1)/nchannels)  #+1 because residue count starts at 0
    #create VMD selection for each ref channel
    channel_ref_selections=[vmd.atomsel(selection=i, molid=refid) for i in create_channel_selections(nresidue, nchannels)]
    #write out the membrane and water components
    channel_memb_selection=vmd.atomsel(selection="lipid or water or ions", molid=refid)
    channel_memb_selection.write(filetype="pdb", filename="memb_water_ions.pdb")
    if ntraj==1:
        for i in range(nchannels):
            if i!=args.poreid-1:
                monomersel=channel_ref_selections[i]
                monomersel.write(filetype="pdb", filename=f"monomer{i}.pdb")
    
    traj_ids=[]
    #read the trajectories, waitfor=-1 to read all frames.
    for i in range(ntraj):
        trajid=vmd.molecule.new(f"traj{i+1}")
        vmd.molecule.read(trajid, filename=traj_files[i], filetype="pdb", waitfor=-1)
        traj_ids.append(trajid)
    
    
    trajlengths=[vmd.molecule.numframes(idx) for idx in traj_ids]
    
    #calculate the z-position value for each frame
    traj_zpos=[]
    for i in range(ntraj):
        protsel=vmd.atomsel(selection="protein and type CA", molid=traj_ids[i])
        solsel=vmd.atomsel(selection=f"resname {RESNAME}", molid=traj_ids[i])
        zpos=[]
        for frameid in range(trajlengths[i]):
            protsel.frame=frameid
            solsel.frame=frameid
            zpos.append(solsel.center()[2]-protsel.center()[2])
        traj_zpos.append(np.array(zpos, dtype=np.float32))
    
    if verbose:
        print("Calculated z positions for the trajectories:\n", traj_zpos)
    frames=[]
    #here we select which frames to process for each trajectory
    for i in range(ntraj):
        framedict={}
        traj_z=traj_zpos[i]/10
        for pos in UPOS:
            diff=abs(traj_z-pos)
            framedict[pos]=np.where(diff==np.min(diff))[0][0]
        frames.append(framedict)
    
    
    #here we finally combine the frames and setup the inputs configurations for simulations.
    if staggered_placement==True and ntraj>1:   #logic for the staggered placement
        print("""NOTE: Staggered Placement is set. This means that the solute 
              molecule will be set with different umbrella means for different channels 
              within the same simulation. The umbrella window folders will be numbered serially
              starting from 0. The code will also write out the data into a text
              file that can be used to correctly set up the biasing procedure.""")
        
        print(">> A staggered shift of {staggered_shift} positions will be used.")
        #here we generate a circularly shifted list of umbrella positions 
        # for the solute in case of all additional channels.
        UPOS_SD={}
        UPOS_SD[0]=UPOS
        for i in range(1, nchannels):
            UPOS_SD[i]=circular_shift(UPOS, staggered_shift*i)
            
        #now we generate the configurations
        for uindex in range(NUMB):
            if verbose:
                print(f"Setting up the system for umbrella: {uindex}")
            tmpinpfiles=[]
            for i in range(nchannels):
                trajprotsel=vmd.atomsel(selection="protein", molid=traj_ids[i])
                trajprotsel.frame=int(frames[i][UPOS_SD[i][uindex]])
                if verbose:
                    print(f"Monomer{i}: frame {trajprotsel.frame}")
                trajprotsel.update()
                transform_matrix=trajprotsel.fit(selection=channel_ref_selections[i])
                trajmovesel=vmd.atomsel(selection="all", molid=traj_ids[i])
                trajmovesel.frame=int(frames[i][UPOS_SD[i][uindex]])
                trajmovesel.update()
                trajmovesel.move(transform_matrix)
                trajmovesel.write(filename=f"monomer{i}.pdb", filetype="pdb")
                tmpinpfiles.append(f"monomer{i}.pdb")
            tmpinpfiles.append("memb_water_ions.pdb")
            assemble_components(tmpinpfiles, outfile=f"umb_{uindex}.pdb", 
                             fileorder=["protein", "lipids", "ions", "water", f"resname {RESNAME}"])       
            os.system("rm monomer?.pdb")
            insert_boxdim(f"umb_{uindex}.pdb", BOXDIM)
            time.sleep(5)
            if verbose:
                os.system(f"gmx editconf -f umb_{uindex}.pdb -o input.gro")
            else:
                os.system(f"gmx editconf -f umb_{uindex}.pdb -o input.gro &>> /dev/null")
            os.mkdir(f"U_{uindex}")
            os.system(f"mv umb_{uindex}.pdb input.gro U_{uindex}")
    else:
        for pos in UPOS:
            if verbose:
                print(f"Setting up the system for position: {pos}")
            tmpinplist=[]
            for i in range(nchannels):
                if ntraj==nchannels:
                    trajprotsel=vmd.atomsel(selection="protein", molid=traj_ids[i])
                    trajprotsel.frame=int(frames[i][pos])
                    if verbose:
                        print(f"Monomer{i}: frame {trajprotsel.frame}")
                    trajprotsel.update()
                    transform_matrix=trajprotsel.fit(selection=channel_ref_selections[i])
                    trajmovesel=vmd.atomsel(selection="all", molid=traj_ids[i])
                    trajmovesel.frame=int(frames[i][pos])
                    trajmovesel.update()
                    trajmovesel.move(transform_matrix)
                    trajmovesel.write(filename=f"monomer{i}.pdb", filetype="pdb")
                    tmpinplist.append(f"monomer{i}.pdb")
                else:
                    if args.poreid-1==i:
                        trajprotsel=vmd.atomsel(selection="protein", molid=traj_ids[0])
                        trajprotsel.frame=int(frames[0][pos])
                        if verbose:
                            print(f"Monomer{i}: frame {trajprotsel.frame}")
                        trajprotsel.update()
                        transform_matrix=trajprotsel.fit(selection=channel_ref_selections[i])
                        trajmovesel=vmd.atomsel(selection="all", molid=traj_ids[0])
                        trajmovesel.frame=int(frames[0][pos])
                        trajmovesel.update()
                        trajmovesel.move(transform_matrix)
                        trajmovesel.write(filename=f"monomer{i}.pdb", filetype="pdb")
                        tmpinplist.append(f"monomer{i}.pdb")
                    else:
                        tmpinplist.append(f"monomer{i}.pdb")
                        
            tmpinplist.append("memb_water_ions.pdb")
            print(tmpinplist)
            assemble_components(tmpinplist, outfile=f"umb_{round(pos, ndigits=2)}.pdb", 
                                fileorder=["protein", "lipids", "ions", "water", f"resname {RESNAME}"])
                
            if ntraj==1:
                os.system(f"rm monomer{args.poreid-1}.pdb")
            else:
                os.system("rm monomer?.pdb")
            insert_boxdim(f"umb_{round(pos, ndigits=2)}.pdb", BOXDIM)
            time.sleep(5)
            if verbose:
                os.system(f"gmx editconf -f umb_{round(pos, ndigits=2)}.pdb -o input.gro")
            else:
                os.system(f"gmx editconf -f umb_{round(pos, ndigits=2)}.pdb -o input.gro &>> /dev/null")
            
            os.mkdir(f"U_{round(pos, ndigits=2)}")
            os.system(f"mv umb_{round(pos, ndigits=2)}.pdb input.gro U_{round(pos, ndigits=2)}")
    os.system("rm monomer?.pdb part_*.pdb")
    
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Processes the transition trajectory files to setup TASS")
    parser.add_argument("-tl","--trajlist", nargs='+', help="A sequence of traj files to be processed.", required=True)
    parser.add_argument("-rf", "--refpdb", help="The reference file containing the full pore with the simulation box.")
    parser.add_argument("-N", "--nchannels", help="Number of channels in the refpdb file.", default=1, type=int)
    parser.add_argument("-id", "--poreid", type=int, default=1, help="""If a single transition traj file supplied, the specify the 
                        pore id. This is basically used to specify the specific pore. Default value is 1, which means the first pore
                        in order of appearance in refpdb will be used.""")
    parser.add_argument("-p", "--paramfile", help="Parameter file", default="params.dat")
    parser.add_argument("-SP", "--staggered_placement", help="Position the drugs -SH window means apart in adjacent channel.", action="store_true")
    parser.add_argument("-SH", "--staggered_shift", default=8, type=int, help="Shift used for the staggered placement of molecules in adjacent channels.")
    parser.add_argument("-v", "--verbose", help="Be loud.", action="store_true")
    args = parser.parse_args()
    main(args)








    
    




        
