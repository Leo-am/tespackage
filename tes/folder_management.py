"""
Module with convenient functions for folder management.

Contains
1) find_folders
2) manage_folders

"""
import os
from pathlib import Path


def find_folders(master_folder):
    """
    Search for folders inside the given address.

    Parameters
    ----------
    master_folder : string
        address of the master folder on local computer.

    Raises
    ------
    AttributeError:
        Raises error when no folder can be found.

    Returns
    -------
    measurements : list
        list with the address of folders inside master_folder.
    """
    master_string = master_folder
    folder_master = Path(master_string)

    if not folder_master.exists():
        raise AttributeError("Folder could not be found.")

    # initialise lists to account for measurement folders and registers

    measurements = []

    # detect the measurement folders where each of the data runnings
    # are stored

    for meas_folder in (folder_master).glob('*'):
        if meas_folder.is_dir():
            measurements.append(meas_folder)
    if len(measurements) == 0:
        print("No folders were found.")
    else:
        print("In the selected path {} measurement folders were found."
              .format(len(measurements)))
        print("Choose the folder to be analysed using its index.")
        for counter, path in enumerate(measurements):
            print("Option {}: \n {}".format(counter, path))

    return measurements


def manage_folders(index, measurement_folders):
    """
    Create a folder structure for a given measurement set.

    Structure as follows:
    Measurement Folder ------> Analysis Folder
                       |-----> Figures Folder

    Parameters
    ----------
    index : integer
        Select the measurement folder to be analysed
    measurements: list
        contains all addresses of measurement folders

    Raises
    ------
    AttributeError:
        Raises error when no folder can be found.

    Return
    ------
    tuple with:
        datapath : path
            address of the measurement folder
        folder_analysis : path
            address of the analysis folder
        folder_figures : path
            address of the figures folder
    """
    measurement_index = index
    datapath = measurement_folders[measurement_index]
    string_path = str(datapath)
    folder_analysis = Path(string_path + '/Analysis')
    folder_figures = Path(string_path + '/Figures')

    if datapath.exists():
        if not folder_analysis.exists():
            os.makedirs(folder_analysis)
        if not folder_figures.exists():
            os.makedirs(folder_figures)
    else:
        raise AttributeError('No data found.')

    print("The analysed data is stored in: \n Option {}: {}"
          .format(index, datapath))
    print("The figures generated will be saved in: \n {}"
          .format(folder_figures))

    return (datapath, folder_analysis, folder_figures)
