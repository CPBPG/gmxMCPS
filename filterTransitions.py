#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 15:59:24 2023

@author: aacharya
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import vmd
import argparse





def main(args):
    #These values specify the inclination range that is used to select paths
    # and classify then into path I and path II. So, paths with mean inclination
    #greater than the MEAN_INC_HIGH value are classified as pathI and those with
    #value lesser than MEAN_INC_LOW are classified as path II.
    #Although this need not be changed by the user for most use cases, if the
    #code stops with a "No paths selected" error, then reducing the range
    #would be helpful.
    MEAN_INC_HIGH=140
    MEAN_INC_LOW=40
    #This is used to specify the range that corresponds to the constriction region. Influences the calculation of the
    #average interaction energy and the average inclination value for each transition paths at the CR.
    CR_CUT_LOW=40
    CR_CUT_HIGH=55
    #This is the number of transitions to load from the input transition_search.dat.
    #You can increase these values if the code does not produce enough output paths.
    #Normally the standard value suffices.
    NLOAD=args.nload
    #########################################################################
    #HARDCODED filenames - no need to change!
    ts_paths="transition_search.dat"
    rtfiles=sorted(glob.glob("chk_*_minimize/*_rottrans.dat"), key=lambda x: int(x.split('_')[-2]))
    iefiles=sorted(glob.glob("chk_*_minimize/*_PIE.dat"), key=lambda x: int(x.split('_')[-2]))
    cullfr_files=sorted(glob.glob("culled_frames_*.dat"), key=lambda x: int(x.split('_')[-1][:-4]))
    trajfiles=sorted(glob.glob("chk_*_minimize/*_mintraj.pdb"), key=lambda x: int(x.split('_')[-2]))
    npaths_write= args.npaths ### number of top paths to write out per cluster
    
    #load the data from input dat files 
    for i in range(len(iefiles)):
        #read the input files
        data_rt=np.loadtxt(rtfiles[i])
        data_ie=np.loadtxt(iefiles[i], usecols=[1])
        data_cull=np.array(np.loadtxt(cullfr_files[i]), dtype='int')
        #need to cull the dataset to remove those frames with very high energies
        cull_rt=np.delete(data_rt, data_cull, axis=0)
        cull_ie=np.delete(data_ie, data_cull, axis=0)
        fulldat=np.hstack([cull_rt, cull_ie.reshape(len(cull_ie), 1)])
        #write culled frame numbers (to be used to filter out the trajectory data later).
        # concatenate the full dataset from individual rottrans and pie files
        if i==0:
            cumfulldat=fulldat
        else:
            cumfulldat=np.vstack([cumfulldat, fulldat])
    
    
    
    ## Index for each of the rows in the cumfulldat 
    z_index = 1
    inc_index = 2
    az_index = 3
    en_index = 4
    
    ### en_std and en_mean used to standardize energy.
    en_std = np.std(cumfulldat[:,en_index])
    en_mean = np.mean(cumfulldat[:,en_index])
    
    #load transitions calculated from file
    transitions=[]
    with open(ts_paths, 'r') as tf:
        for l in range(NLOAD):
            line=tf.readline()
            arr=np.fromstring(line.split(":")[1], dtype=int, sep=',')
            transitions.append(arr)
    
    #get corresponding values for each transition
    tr_z={}
    tr_en={}
    tr_inc={}
    tr_az={}
    
    for i in range(NLOAD):
        tr_z[i]=cumfulldat[transitions[i],1]
        tr_en[i]=cumfulldat[transitions[i],4]
        tr_inc[i]=cumfulldat[transitions[i],2]
        tr_az[i]=cumfulldat[transitions[i],3]
        
    #sorted indices for transitions with low mean energy (only for the values in the constriction region)
    #the idea is that since the constriction region poses an entropic barrier, pathways 
    #with stronger interactions will be most feasible
    mean_std_energy=[]
    for i in range(NLOAD):
        men=np.mean( ( ( tr_en[i][np.where((tr_z[i]>CR_CUT_LOW)&(tr_z[i]<CR_CUT_HIGH))] )-en_mean )/en_std)
        mean_std_energy.append(men)
    
    culledkeytop=sorted(tr_en.keys(), key=lambda x: mean_std_energy[x])    
    #we need to group the selected transitions
    #we do it based on the inclination value which determines the orientation of the molecule at
    #the constriction zone
    select_tr_inc={}
    #for the top lowest energy paths, select configurations at the CR
    for i in  culledkeytop:
        select_tr_inc[i]=tr_inc[i][np.where((tr_z[i]>40)&(tr_z[i]<55))]
    
    selected_transitions_pathI=[]
    selected_transitions_pathII=[]
    
    for i in select_tr_inc.keys():
        meaninc=np.mean(select_tr_inc[i])
        if meaninc > MEAN_INC_HIGH: 
            selected_transitions_pathI.append(i)
        elif meaninc < MEAN_INC_LOW:
            selected_transitions_pathII.append(i)
            
    if len(selected_transitions_pathI)==0 or len(selected_transitions_pathII)==0:
        raise Exception("No paths were selected based on the cutoff values used for Mean Inclination.")
    elif len(selected_transitions_pathI)>npaths_write and len(selected_transitions_pathII)>npaths_write:
        selected_transitions_pathI=selected_transitions_pathI[:npaths_write]
        selected_transitions_pathII=selected_transitions_pathII[:npaths_write]
    else:
        print("Number of filtered transition paths for Path I: ", len(selected_transitions_pathI))
        print("Number of filtered transition paths for Path II: ", len(selected_transitions_pathII))
        raise Exception("The script could not find transition paths greater than the default npath values for at least one of the clusters. Try reducing the --npaths value or relex the MEAN_INC_HIGH and MEAN_INC_LOW values. Increasing the --nload value may also be useful.")

    #here we basically collect the frame numbers that need to be extracted from the trajectory files.
    out_transition_data_I=[transitions[i] for i in selected_transitions_pathI]
    out_transition_data_II=[transitions[i] for i in selected_transitions_pathII]
    outframes=[]
    for i in out_transition_data_I:
        outframes+=list(i)
    for i in out_transition_data_II:
        outframes+=list(i)
    outframes=np.array(sorted(set(outframes)))
    
    #now we do the actual extraction
    #nframes track the number of frames in each trajectory
    
    nframes=[0]
    
    for i in range(len(trajfiles)):
        trajid=vmd.molecule.new(f"chunk_{i}")
        vmd.molecule.read(trajid, filename=trajfiles[i], filetype='pdb', waitfor=-1) 
        data_cull=np.array(np.loadtxt(cullfr_files[i]), dtype='int') #remove data points with high energies
        data_cull=data_cull - np.arange(len(data_cull))
        for dat in data_cull:
            vmd.molecule.delframe(trajid, first=dat, last=dat)
        nframes.append(vmd.molecule.numframes(trajid))   #get the total number of frames
        #extracting the outframes that would be written out for the given trajectory file
        to_write_serial=outframes[np.where((outframes>=np.sum(nframes[:i+1]))&(outframes<np.sum(nframes[:i+2])))]  
        to_write_traj=to_write_serial-np.sum(nframes[:i+1]) #adjusting for the frame numbers in the previous trajectory
        for serialidx, framenumber in enumerate(to_write_traj):
            vmd.molecule.write(trajid, filename=f"frameout_{to_write_serial[serialidx]}.pdb", filetype="pdb", first=framenumber , last=framenumber)
        vmd.molecule.delete(trajid)
    
    #create the trajectory for each of the transitions that are specified by out_transition_data
    for i in range(len(out_transition_data_I)):
        transition_serial=selected_transitions_pathI[i]
        transition_serial_zpos=tr_z[transition_serial]
        transition_frames=out_transition_data_I[i]
        trajid=vmd.molecule.new("outtraj")
        for frame in transition_frames:
            if os.path.exists(f"frameout_{frame}.pdb"):
                vmd.molecule.read(trajid, filename=f"frameout_{frame}.pdb", filetype="pdb")
            else:
                raise Exception(f"Missing: frameout_{frame}.pdb")
        if vmd.molecule.numframes(trajid)==len(transition_serial_zpos):
            vmd.molecule.write(trajid, filename=f"pathI_transition_{i+1}_{transition_serial}.pdb", filetype="pdb", first=0, last=-1)
            print(f"Written trajectory for transition {transition_serial}.")
            np.save(f"pathI_transition_zpos_{transition_serial}.npy", arr=transition_serial_zpos)
            print(f"Written z-positions for transition {transition_serial}.")
        else:
            raise Exception(f"The number of data points in traj_zpositions and the frames in output traj do not match.")
    
    for i in range(len(out_transition_data_II)):
        transition_serial=selected_transitions_pathII[i]
        transition_serial_zpos=tr_z[transition_serial]
        transition_frames=out_transition_data_II[i]
        trajid=vmd.molecule.new("outtraj")
        for frame in transition_frames:
            if os.path.exists(f"frameout_{frame}.pdb"):
                vmd.molecule.read(trajid, filename=f"frameout_{frame}.pdb", filetype="pdb")
            else:
                raise Exception(f"Missing: frameout_{frame}.pdb")
        if vmd.molecule.numframes(trajid)==len(transition_serial_zpos):
            vmd.molecule.write(trajid, filename=f"pathII_transition_{i+1}_{transition_serial}.pdb", filetype="pdb", first=0, last=-1)
            print(f"Written trajectory for transition {transition_serial}.")
            np.save(f"pathII_transition_zpos_{transition_serial}.npy", arr=transition_serial_zpos)
            print(f"Written z-positions for transition {transition_serial}.")
        else:
            raise Exception(f"The number of data points in traj_zpositions and the frames in output traj do not match.")



if __name__=="__main__":
     #setup input commandline arguments
    parser=argparse.ArgumentParser(prog='FilterTransitions.py', 
                                   description="""Script to collect the information on the calculated paths
                                   and filter the lowest energy paths into pathI and pathII based the
                                   inclination value observed at the constriction region. Finally, the trajectory
                                   for the selected paths are written out to the disk.""")
    parser.add_argument("-np", "--npaths", action='store', 
                        default=5, type=int, 
                        help="Number of paths to write out per cluster.")
    parser.add_argument("-nl", "--nload", action='store', 
                        default=50000, type=int, 
                        help="Number of paths to load from the transition_search data file.")
    inargs=parser.parse_args()
    main(inargs)



