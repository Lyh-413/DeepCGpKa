# DeepCGpKa
## Project Overview
This project corresponds to the research paper Modeling pH dependent protein dynamics by integrating coarse-grained molecular simulation and deep neural network model, which is currently under review at The Journal of Chemical Theory and Computation (JCTC).

The core goal is to achieve pKa prediction and pH-dependent dynamic modeling based on coarse-grained protein structures.
## File Details
- The folders `oneb`, `twob`, and `oneb+` contain implementations for the **one-bead model**, **two-bead model**, and **one-bead model with additional angle features**, respectively. The pKa shift range used in these models is from -2 to 2.
- The `twob-all` folder contains the **two-bead model** using the full range of pKa shift values.

In each folder:
- `cphmd.py` and `exp67s.py` are scripts for **input data construction** for the CpHMD and Exp67s datasets under the corresponding coarse-grained representation.
- `DCGPKA-exp67s.py` and `DCGPKA-CpHMD.py` are the main scripts for **pKa prediction** on the corresponding datasets.

Additional files:
- Inside `oneb/twob/`, `unfold.py` and `DCGPKA-unfold.py` are for **input processing and pKa prediction** on the unfolded protein dataset.
- For the `oneb` model, a **coarse-grained molecular dynamics (MD) modeling approach** for the CagL system under different pH conditions is provided. It can be run by modifying the pH value in `tf_programme.py` and executing the script.
## Data Source
The data files used, val_n27.csv and test_n69_undersample.csv, are from the paper:

Cai, H.; Li, M.; Lin, Y.-R.; Chen, W.; Wang, S.; Takada, S. Protein pKa Prediction with Machine Learning; ACS Omega, 2021, 6, 34823–34831, doi: 10.1021/acsomega.1c05440.

