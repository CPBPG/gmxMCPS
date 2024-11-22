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
import numpy as np
import math as m
import multiprocessing as mp
from multiprocessing import Pool, Process, Manager
from joblib import Parallel, delayed
from datetime import datetime
import random
import csv
import glob
import os

def standardize_ie(energy_data):
    """Standardize the energy values, mean shifted to zero."""
    return (energy_data-energy_data.mean())/energy_data.std()

def write_culled_frames(data, file):
    with open(file, 'w') as f:
        out=[str(i) for i in data]
        f.write(" ".join(out)+"\n")

def next_available_steps(data,zbin_frames,accept_boolarr,curr_frame,inc_ind,az_ind,inc_step,az_step,z_idx):
    """Select the next available steps for a MC search based 
       on the provided step size in the inclination and azimuthal space."""
    selected=[]
    selected_zbin={}
    curr_inc=data[curr_frame,inc_ind]
    curr_az=data[curr_frame,az_ind]
    #get the inclination and azimuthal ranges available; based in the last position in the MC search 
    inc_max=curr_inc+inc_step
    inc_min=curr_inc-inc_step
    az_max=curr_az+az_step
    az_min=curr_az-az_step
    #apply PBC to azimuthal value
    az_max_pbc=az_max-360.0
    az_min_pbc=az_min+360.0
    #selection of neighbouring frames
    #get info for frames in the present z bin to select ones that fall in the selection criteria
    for idx in [z_idx, z_idx+1]:
        zfr_all=np.array(zbin_frames[idx])
        next_accept_bool=accept_boolarr[zfr_all]
        zfr_avail=zfr_all[next_accept_bool==0]
        #filter for frames that are within inclination range
        zfr_inc=data[zfr_avail,inc_ind]
        zfr_filter=zfr_avail[np.where((zfr_inc>inc_min) & (zfr_inc<inc_max))]
        #filter for frames within the azimuthal range
        zfr_filter_az=data[zfr_filter,az_ind]
        if az_min<0:
            next_zfr=zfr_filter[np.where(((zfr_filter_az>az_min_pbc) & (zfr_filter_az<360)) | ((zfr_filter_az<az_max) & (zfr_filter_az>0)) )]
        elif az_min>360:
            next_zfr=zfr_filter[np.where(((zfr_filter_az<az_max_pbc) & (zfr_filter_az>0)) | ((zfr_filter_az>az_min) & (zfr_filter_az<360)) )]
        else:
            next_zfr=zfr_filter[np.where((zfr_filter_az>az_min) & (zfr_filter_az<az_max))]
        for i  in next_zfr:
            selected.append(i)
            selected_zbin[i]=idx
    return selected, selected_zbin

def Transition_Search(data,z_list,z_nbins,inc_step,az_step,z_ind,inc_ind,az_ind,energy_ind,iteration,q):
    #print(f"TS search iteration {iteration} initiated ")
    nframes=len(data)
    accept_boolarr=np.zeros(nframes)
    stdenergy=standardize_ie(data[:,energy_ind])
    #initialize some variables
    z_idx=0
    accept_frames=[]
    accept_energy=[]
    random.seed(iteration)
    init_fr=random.choice(z_list[z_idx])
    init_ie=stdenergy[init_fr]
    accept_frames.append(init_fr)
    accept_energy.append(init_ie)
    step_num=0
    #start the search procedure 
    while z_idx<z_nbins-1:
        #set the last accepted frame number and corresponding energy
        curr_frame=accept_frames[step_num]
        curr_energy=accept_energy[step_num]
        #get the list of frames available for a possible Metropolis MC move
        selected_fr, selected_fr_zbin=next_available_steps(data,z_list,accept_boolarr,curr_frame,inc_ind,az_ind,inc_step,az_step,z_idx)
        selected_fr_num=len(selected_fr)
        
        if selected_fr_num==0: #terminate search if nothing found
            return 0
            #q.put([0])
        
        accepted=0
        rejected=0
        # Metropolis MC search until acceptance
        while accepted==0:
            if rejected>=selected_fr_num:
                #print(">>>> Done trying on this iteration.")
                return 0
                #q.put([0])
            #random frame
            fr_trial=random.choice(selected_fr)
            #print("Tried:", fr_trial)
            ie_trial=stdenergy[fr_trial]
            mc_criteria=np.exp(curr_energy-ie_trial)
            #print(f"Current: {curr_energy}, Trial: {ie_trial}")
            if mc_criteria>=1.0:  #accepted if criteria >= 1.0
                accepted+=1
                accept_frames.append(fr_trial)
                accept_energy.append(ie_trial)
                step_num+=1
                accept_boolarr[fr_trial]=1
                z_idx=selected_fr_zbin[fr_trial]
            else:
                rand=np.random.uniform(0,1)
                if mc_criteria>rand:
                    accepted+=1
                    accept_frames.append(fr_trial)
                    accept_energy.append(ie_trial)
                    step_num+=1
                    accept_boolarr[fr_trial]=1
                    z_idx=selected_fr_zbin[fr_trial]
                else:
                    rejected+=1
                    selected_fr=np.delete(selected_fr, np.where(selected_fr==fr_trial))
                    accept_boolarr[fr_trial]=1
    out=[str(ac) for ac in accept_frames]
    q.put(f"Iteration {iteration}: "+",".join(out)+"\n")
    
def check_ie(data, ie_ind):
    """Checks the dataset for unusually high energies. 
       Returns the cleaned data and the indices of the high
       energy frames."""
    ie=data[:,ie_ind]
    ielow=ie<1000
    data_low=data[ielow,:]
    iehigh=ie>=1000
    high_idx=[]
    for i, x in enumerate(iehigh):
        if x==True:
            high_idx.append(i)
    return data_low, high_idx

def save_to_file(q, file):
    with open(file, 'w') as f:
        while True:
            val=q.get()
            if val is None: break
            f.write(val)
        
def main(args):
    #preliminaries reading in the slowcoord files and IE_files for a single channel.
    rtfiles=sorted(glob.glob("chk_*_minimize/*_rottrans.dat"), key=lambda x: int(x.split('_')[-2]))
    iefiles=sorted(glob.glob("chk_*_minimize/*_PIE.dat"), key=lambda x: int(x.split('_')[-2]))
    if len(rtfiles)==len(iefiles):
        ####### log file to indicate errors and time to run algorithm
        output_file=f'transition_search.dat'
        log_file=f'log.log'
        z_ind=1 # index for z
        inc_ind=2 # index for inclination
        az_ind=3 # index for azimuthal
        energy_ind=4 # index for energy
        for i in range(len(iefiles)):
            #read the input files
            data_rt=np.loadtxt(rtfiles[i])
            data_ie=np.loadtxt(iefiles[i], usecols=[1])
            
            fulldat=np.hstack([data_rt, data_ie.reshape(len(data_ie), 1)])
            
            fulldat_cull, culled_idx = check_ie(fulldat, energy_ind)
            #write culled frame numbers (to be used to filter out the trajectory data later).
            write_culled_frames(culled_idx, f"culled_frames_{i}.dat")
            # concatenate the full dataset from individual rottrans and pie files
            if i==0:
                cumfulldat=fulldat_cull
            else:
                cumfulldat=np.vstack([cumfulldat, fulldat_cull])

        max_z=int(np.max(cumfulldat[:,z_ind])) #max z to start
        min_z=int(np.min(cumfulldat[:,z_ind])) # min z to end search
        z_step=1 ### step size in z. 
        inc_step=18 ### step size in inclination. 
        az_step=18 ### step size in azimuthal. 
        num=args.niter
        #create a dict with each bin position as a key mapped 
        #to a corresponding list of frames in the bin as the item
        print("Creating z bins")

        z_num=int((max_z-min_z)/z_step)
        z_list={}
        for j in range(z_num):
            z_i=max_z-z_step*j
            z_list[j]=[]
            for f in range(len(cumfulldat)):
                if (cumfulldat[f][z_ind] < (z_i) and cumfulldat[f][z_ind]>=(z_i-z_step)):
                    z_list[j].append(f)
                    
        print(f"Starting MCPS.")
        
        with open(log_file,'w') as log:
        
            try: #create a multiprocessing instance
                proc=mp.cpu_count()
                d = datetime.now()
                starttime = datetime.now()
                log.write(f"Began process:{starttime} \n")
                m=Manager()
                q=m.Queue() #queue to catch the outputs from each parallel process
                p = Process(target=save_to_file, args=(q, output_file)) #outputs are sent for writing to disk
                p.start()
                Parallel(n_jobs=proc)(delayed(Transition_Search)(cumfulldat,z_list,z_num,inc_step,az_step,z_ind,inc_ind,az_ind,energy_ind,iteration,q) for iteration in range(num))
                q.put(None)
                p.join()
                endtime = datetime.now()
                log.write(f"Finished process successfully {endtime}\n")
            except:
                log.write(f"Job was terminated with an error.")
    return None



if __name__=="__main__":
     #setup input commandline arguments
    parser=argparse.ArgumentParser(prog='runMCPS.py', 
                                   description="The script implements the MCPS algorithm.")
    parser.add_argument("-n", "--niter", action='store', 
                        default=10, type=int, 
                        help="Number of MCPS search iterations to perform.")
    inargs=parser.parse_args()
    main(inargs)
