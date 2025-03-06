# gmxMCPS
A python workflow for setting up a TASS simulation for solute permeation through a pore. The code provided here is useful for generating inputs for any enhanced sampling method that uses a windowing scheme.


## Requirements
1. Python 3.6 - 3.8
2. rdkit (for molecular operations)
3. vmd-python (for molecular visualization and selection)
4. A GROMACS installation (for energy minimization and extraction of pore-drug interaction energy)

## General Workflow

1. Run prepINP.py to generate all possible configurations of a solute (drug molecule) within one or more channels of a protein. Configurations that result in clashes or ring piercings are automatically removed, and the valid configurations saved to the disc as a single PDB file per monomer.

2. Next run the helper script chunk_inp.py which divides the full trajectory data into chunks that can be processed parallely in the next step.

3. For each of the trajectory chunks created, we perform an energy minimization using the runEM.py script. The script writes out the minimized configurations as a PDB file and .dat files with the corresponding z position, inclination angle, azimuthal angle, and interaction energy. 

4. Taking the .dat files as input, the runMCPS.py script performs the Monte-Carlo pathway search algorithm to identify a user-specified number of paths.

5. Subsequently, the filterTransitions.py script classifies the calculated pathways into pathI and pathII type transitions and writes coordinate data for the top path to a PDB file.

6. Finally, run the setupInputs.py on the user-selected pathway dataset to generate the inputs for the windows required for TASS simulations.


For details refer to DETAILS.pdf.


## References
If you use this script, please cite:

Haloi N. et al.  Rationalizing the generation of broad-spectrum antibiotics with the addition of a positive charge. Chem Sci. 2021 Nov 24; 12(45): 15028–15044.
Acharya A., Kleinekathoefer U. Improved Free-Energy Estimates for the Permeation of Bulky Antibiotic Molecules through Porin Channels Using Temperature3 Accelerated Sliced Sampling. J. Chem. Theory. Comput. 


## Author
Abhishek Acharya,
Postdoctoral Fellow,
School of Science, Constructor University Bremen,
Email: abhi117acharya@gmail.com
 



