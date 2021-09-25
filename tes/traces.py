"""Module to be used with jupyter notebook TES Trace Generator.

"""

# from pathlib import Path

# import numpy as np
# import matplotlib.pyplot as plt

# from tes.data import CaptureData
# from tes.registers import load


def extract_data(datapath, measurements):
    """
    Extract data and registers for a given address.

    Parameters
    ----------
    datapath : str
        String containing the measurement folder address.
    measurements : list of paths
        List with the paths of all measurement folders.

    Returns
    -------
    data : tes.data.CaptureData
        Contains the data to be plotted.
    registers : TYPE
        Contains the information required to plot timing information
        properly.

    """
    registers = []

    # ############# EXTRACTING DATA ###############

    np_load_old = np.load  # save np.load
    # modify the default parameters of np.load to avoid error
    # when using CaptureData
    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

    data = CaptureData(datapath)

    np.load = np_load_old  # restore np.load for future normal usage

    # ############## CHECKING THE REGISTERS FILE ########

    for measurement_folder in measurements:
        if Path(str(measurement_folder)+'_reg.yml').exists():
            registers.append(load(str(measurement_folder)+'_reg.yml'))
            print('Registers were found for:', measurement_folder)
        else:
            registers.append(None)
            print('No register was found for', measurement_folder)

    create_statistics_file(measurements)

    print('Data succesfully extracted.')

    return (data, registers)


def create_statistics_file(measurement_folders):
    """
    Create file with measurements statistics in measurement folder.

    Parameters
    ----------
    measurement_folder : list of paths
        list containing the paths of the folders where the measurements
        are stored and where the stats files will be created.

    Returns
    -------
        None


    """
    # statistics of a measurement
    # save the number of: ticks, events, traces, mca, frames, dropped
    # frames, bad frames, res frames

    stat_dt = np.dtype([
        ('ticks', 'u8'), ('events', 'u8'), ('traces', 'u8'), ('mca', 'u8'),
        ('frames', 'u8'), ('dropped', 'u8'), ('bad', 'u8'), ('res', 'u8')
    ])

    capture_stats = np.zeros(len(measurement_folders), dtype=stat_dt)

    for counter, path in enumerate(measurement_folders):
        stat_file = path/'stat'
        if stat_file.exists():
            capture_stats[counter] = np.memmap(stat_file, dtype=stat_dt)[0]


def plot_traces(
    data,
    ax,
    number_traces,
    trace_length,
    time_register,
    choose_trace,
):
    """
    Plot traces using collected using the FPGA.

    Can plot:
        - a single trace (number_traces = 1, slope = False,
                          details = False)
        - many traces (number_traces > 1)

    Parameters
    ----------
    data : tes.data.CaptureData
        Contains the data to be plotted.
    number_traces : int
        The number of different TES pulses to be plotted.
    trace_length : int
        The number of points in each TES pulse.
    time_register : dict
        Contains the time information for the pulses.
    choose_trace : int
        Choose a single TES pulse to plot.
        (if number_traces == 1)
    ax : matplotlib.axis
        Figure axis where traces will be plotted.

    Returns
    -------
    None.

    """
    FONTSIZE = 18

    # FONT SIZES
    plt.rc("font", size=FONTSIZE)
    plt.rc("axes", titlesize=FONTSIZE)
    plt.rc("axes", labelsize=FONTSIZE)
    plt.rc("xtick", labelsize=FONTSIZE)
    plt.rc("ytick", labelsize=FONTSIZE)
    plt.rc("legend", fontsize=FONTSIZE)

    if data.has_traces:

        # ############ PLOTTING #################

        time_array = np.arange(len(data.samples[0]))

        if time_register is not None:
            trace_pre = time_register["event"]["trace_pre"][0]
            stride = time_register["event"]["trace_stride"][0]
            # a voltage average measurement is made every 4ns
            time = 4 * (time_array[:trace_length] * (stride + 1) - trace_pre)
        else:
            print("Timescale will not be plotted properly.")
            time = time_array

        if number_traces == 1:
            # plot single trace
            ax.plot(
                    time[:trace_length],
                    data.samples[choose_trace][:trace_length],
                    lw=1
                    )
        else:
            for sample in data.samples[:number_traces]:
                ax.plot(time[:trace_length], sample[:trace_length], lw=0.5)

        [ax.spines[axis].set_linewidth(1.5)
         for axis in ["top", "bottom", "left", "right"]
         ]

        exp_x = correct_xticks(ax)
        exp_y = correct_yticks(ax)

        ax.set_xlabel(r'Time ($\times 10^{}$ ns)'.format(exp_x))
        ax.set_ylabel(r'Voltage ($\times 10^{}$ arb. un.)'.format(exp_y))

        [line.set_markersize(5) for line in ax.yaxis.get_ticklines()]
        [line.set_markeredgewidth(3) for line in ax.yaxis.get_ticklines()]

        [line.set_markersize(5) for line in ax.xaxis.get_ticklines()]
        [line.set_markeredgewidth(3) for line in ax.xaxis.get_ticklines()]


def correct_xticks(ax):
    """
    Properly edit the xticks for a matplotlib plot.

    Parameters
    ----------
        ax : axis object
            Axis with x-axis to be edited.

    Returns
    -------
        ax : axis object
            Edited axis object.
        expx : int
            Integer to be added to the x-axis label.
    """
    xtvalues = ax.get_xticks()
    expx = int(np.floor(np.log10(xtvalues[-1])))
    tick_values = xtvalues / 10 ** expx
    tick_labels = [r"${:1.1f}$".format(tick) for tick in tick_values]
    ax.set_xticks(xtvalues)
    ax.set_xticklabels(tick_labels)
    return expx


def correct_yticks(ax):
    """
    Properly edit the yticks for a matplotlib plot.

    Parameters
    ----------
        ax : axis object
            Axis with x-axis to be edited.

    Returns
    -------
        ax : axis object
            Edited axis object.
        expy : int
            Integer to be added to the x-axis label.
    """
    ytvalues = ax.get_yticks()
    expy = int(np.floor(np.log10(ytvalues[-1])))
    tick_values = ytvalues / 10 ** expy
    tick_labels = [r"${:1.1f}$".format(tick) for tick in tick_values]
    ax.set_yticks(ytvalues)
    ax.set_yticklabels(tick_labels)
    return expy
