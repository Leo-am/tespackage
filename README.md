# tespackage

![alt text](https://readthedocs.org/projects/pip/badge/?version=latest&style=plastic) 

Python package responsible to establish communication and control the setting of the hardware processor for the transition-edge sensors at the Quantum Technology Lab (QTLab), University of Queensland.

# Usage

After downloading the files in this package to your computer, use the Jupyter notebooks on the folder 'Jupyter Notebooks' to use the package for: 
- Configure the multi-channel analyser (MCA);
- Configure the measurements made by the hardware processor on the TES pulses;
- Calibrate the TES detectors;
- Transform area measurements in photon number information.

To use the package in your own programs/notebooks, import it using 

<code> import tes </code>

# Installation

We recommend that you create a virtual enviroment using conda to work and develop this package.

After downloading the contents of this repository to your machine, you can install the package using the terminal. 

First open the folder where you installed the contents of this repository. 

Then, install the package using:

<code> python setup.py install </code>

# Requirements

The tes package requires: 

- numpy
- scipy
- matplotlib
- numba
- yaml
- pyserial

