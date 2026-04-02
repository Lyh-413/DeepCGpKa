# DeepCGpKa
## Project Overview
This project is associated with the research paper Modeling pH dependent protein dynamics by integrating coarse-grained molecular simulation and deep neural network model, which is currently under review at the Journal of Chemical Theory and Computation (JCTC).
The core goal of this project is to realize pKa prediction and pH-dependent dynamics modeling based on coarse-grained structures.
## File Details
The folders oneb, twob, and oneb+ respectively contain one-bead and two-bead models with pKa shift values ranging from -2 to 2, as well as a one-bead model with additional angle features. The twob-all folder contains two-bead models with all pKa shift values. In each folder, cphmd.py and exp67s.py are scripts for constructing input data modeling for the cphmd and exp67s datasets under the corresponding model representations. DCGPKA-exp67s.py and DCGPKA-CpHMD.py are scripts for pKa prediction on the corresponding datasets. 

The data files used, val_n27.csv and test_n69_undersample.csv, are from the paper:
Cai, H.; Li, M.; Lin, Y.-R.; Chen, W.; Wang, S.; Takada, S. Protein pKa Prediction with Machine Learning; ACS Omega, 2021, 6, 34823–34831
doi: 10.1021/acsomega.1c05440
Here we do not redistribute the raw data files.​

In the twob folder within oneb, there are also unfold.py and DCGPKA-unfold.py scripts for input data modeling and pKa prediction for the unfolded protein dataset. Additionally, for the oneb dataset, there is an approach to coarse-grained kinetic modeling of the CagL system under different pH conditions, which can be implemented by modifying the pH value in tf_programme.py and running it.
