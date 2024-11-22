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
import numpy as np
from vmd import *

def read_nframes(string):
    if string.isnumeric():
        return int(string)
    else:
        with open(string, 'r') as f:
            txt=f.readlines()
        return int(txt[0].strip())

#preliminaries
def main(args):
    channel_traj=args.trajfile
    nf=read_nframes(args.nframes)
    nchunks=args.nchunks
    
    os.mkdir("data_chunks")
    print("Collecting info:")
    fs= round((os.stat(channel_traj).st_size)/(1024*1024*1024))
    print(f"File size of the input traj is {fs} GB.")
    
    firstframe=np.linspace(0, nf-(nf%nchunks), nchunks+1)
    
    
    for i in range(len(firstframe)-1):
        trajid=molecule.new("channel")
        molecule.read(trajid, filename=channel_traj, filetype='pdb', first=int(firstframe[i]), last=int(firstframe[i+1]-1),  waitfor=-1)
        molecule.write(trajid, filename=f"data_chunks/chk_{i}.pdb", filetype="pdb")
        molecule.delete(trajid)
        print(f">>Written chunk {i}.\n")
    return None

if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Helper script to divide the full trajectory into chunks.")
    parser.add_argument("-t", "--trajfile", action='store', 
                        default='system.pdb', type=str, 
                        help="Trajectory file in pdb format")
    parser.add_argument("-nf", "--nframes", action='store', 
                        type=str, 
                        help="Number of frames in the input file.")
    parser.add_argument("-nc", "--nchunks", action='store', 
                        default=10, type=int, 
                        help="Number of chunks")
    
    inargs=parser.parse_args()
    main(inargs)
