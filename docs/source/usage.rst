Usage
=====

After downloading the files in this package to your computer, use the Jupyter notebooks on the folder 'Jupyter Notebooks' to use the package for: 

* Configure the multi-channel analyser (MCA) (*MCA_and_Measurements.ipynb*);
* Configure the measurements made by the hardware processor on the TES pulses (*MCA_and_Measurements.ipynb*);
* Calibrate the TES detectors (*TES Calibration.ipynb*);
* Transform area measurements in photon number information (*TES Counting Photons.ipynb*);
* Plot TES traces obtained with the hardware processor (*TES Trace Generator.ipynb*)

To use the package in your own programs/notebooks, import it using 

.. code-block:: console

   import tes


.. _installation:

Installation
------------

We recommend that you create a virtual enviroment using conda to work and develop this package.

After downloading the contents of this repository to your machine, you can install the package using the terminal. 

First open the folder where you installed the contents of this repository. 

Then, install the package using:

.. code-block:: console

   python setup.py install

Requirements
------------

The tes package requires: 

- numpy
- scipy
- matplotlib
- numba
- yaml
- pyserial
