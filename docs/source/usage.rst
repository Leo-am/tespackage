Usage
=====

After downloading the files in this package to your computer, use the Jupyter notebooks on the folder 'Jupyter Notebooks' to use the package for: 

* Configure the multi-channel analyser (MCA) (_MCA_and_Measurements.ipynb_);
* Configure the measurements made by the hardware processor on the TES pulses (_MCA_and_Measurements.ipynb_);
* Calibrate the TES detectors (_TES Calibration.ipynb);
* Transform area measurements in photon number information (_TES Counting Photons.ipynb_);
* Plot TES traces obtained with the hardware processor (_TES Trace Generator.ipynb_)

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
