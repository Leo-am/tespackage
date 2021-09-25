"""
Module with functions to control the MCA.

Should be used together with the Jupyter Notebook
"MCA_and_Measurements.ipynb".

"""
from time import sleep

import numpy as np
import bokeh
from bokeh.plotting import figure

from tes.data import read_mca


def configure_channels(registers, adc_channel, proc_channel, invert):
    """
    Configure connections to ADC channels.

    Connects the selected ADC channel (check in the lab to which
    ADC channel you connected the TES cables) to the selected
    processing channel (digital channel to which you will refer
    in the next steps of the notebook).

    Parameters
    ----------
    registers : dict
        Dictionary that stores the values of all registers that control
        the measurements performed by the FPGA.

    adc_channel : int
        Determine which adc channel is being used.
        May assume values 0-7.

    proc_channel : int
        Determine which processing channel is being used.
        May assume values 0-1.

    invert : bool
        Determines if the signal polarisation should be inverted.
        True inverts, False does not invert.

    Returns
    -------
        None.
    """
    registers.adc[adc_channel].enable = True
    registers.channel[proc_channel].adc_select = adc_channel
    registers.channel[proc_channel].invert = invert

    print("You are using the adc channel: ", adc_channel)
    print("The processing channel connected to it is: ", proc_channel)
    if invert:
        print("The polarisation for channel", proc_channel, "has been inverted.")
    else:
        print("The polarisation for channel", proc_channel, "has not been inverted.")


def baseline_offset(registers, time, channel, bin_width, baseline_offset):
    """
    Determine the detection baseline for TES detections.

    Change the values of baseline_offset until you see a big peak
    in the  histogram. When the peak appears, put its centre at
    x = 0 by choosing an  appropriate baseline_offset.

    Parameters
    ----------
    registers : dict
        Dictionary that stores the values of all registers that
        control the measurements performed by the FPGA.

    time : int
        Time (in seconds) the histogram should be accumulated over.

    channel : int
        Channel to be analysed.

    bin_width : int
        2^(bin_width) will be the width of the histogram bin

    baseline_offset : int
        Determine the baseline offset.

    Returns
    -------
    (fig,) : tuple
        Figure with the plotted histogram.

    """
    registers.baseline[channel].offset = baseline_offset

    ########## Registers determined by the user ##########

    registers.mca.channel = channel
    registers.mca.ticks = time / (registers.tick_period * 4e-9)
    registers.mca.bin_n = bin_width

    ########## Automatically defined ##########

    registers.mca.value = "f"  # the TES filtered sequence
    registers.mca.trigger = 1  # always true
    registers.mca.qualifier = 1  # always true
    registers.mca.lowest_value = -5000

    ########## Updating MCA values ##########

    registers.mca.update
    sleep(1)

    ########## Reading the distribution ##########

    distributions = read_mca(1)

    histogram = distributions[0]

    ########## Plotting the distributions ##########

    fig = figure(
        title="Baseline Offset",
        title_location="left",
        tools="box_select, box_zoom, reset",
    )

    x = (
        histogram.lowest_value
        + np.arange(histogram.last_bin - 1) * histogram.bin_width
        + histogram.bin_width / 2
    )

    # plot config

    fig.yaxis.axis_label = "Counts"
    fig.xaxis.axis_label = "Filtered TES sequence"
    fig.y_range.start = 0
    fig.y_range.follow = "end"
    fig.vbar(
        x=x,
        bottom=0,
        width=histogram.bin_width,
        top=histogram.bins,
        fill_alpha=0.4,
        line_alpha=1.0,
    )

    return (fig,)


def pulse_threshold(registers, time, channel, bin_width, p_thres):
    """
    Determine the pulse threshold.

    Parameters
    ----------
    registers : dict
        Dictionary that stores the values of all registers that control
        the measurements performed by the FPGA.

    time : int
        Time (in seconds) the histogram should be accumulated over.

    channel : int
        Channel to be analysed.

    bin_width : int
        2^(bin_width) will be the width of the histogram bin.

    pulse_threshold : int
        Determine the pulse threshold.

    Returns
    -------
    (fig,) : tuple
        Figure with the plotted histogram.

    """
    registers.event[channel].pulse_threshold = p_thres

    ########## Registers determined by the user ##########

    registers.mca.channel = channel
    registers.mca.ticks = time / (registers.tick_period * 4e-9)
    registers.mca.bin_n = bin_width

    ########## Automatically defined ##########

    registers.mca.value = 3  # f extreme value between zero crossings
    registers.mca.trigger = 5
    registers.mca.qualifier = 1
    registers.mca.lowest_value = 1

    ########## Updating MCA values ##########

    registers.mca.update
    sleep(1)

    ########## Reading the distribution ##########

    distributions = read_mca(1)

    histogram = distributions[0]

    ########## Plotting the distributions ##########

    fig = figure(
        title="Pulse Threshold",
        title_location="left",
        tools="box_select, box_zoom, reset",
    )

    x = (
        histogram.lowest_value
        + np.arange(histogram.last_bin - 1) * histogram.bin_width
        + histogram.bin_width / 2
    )

    # plot config

    fig.yaxis.axis_label = "Counts"
    fig.x_range.start = 0
    fig.x_range.end = 2e4
    fig.y_range.start = 0
    fig.y_range.follow = "end"
    fig.vbar(
        x=x,
        bottom=0,
        width=histogram.bin_width,
        top=histogram.bins,
        fill_alpha=0.4,
        line_alpha=1.0,
    )

    return (fig,)


def slope_threshold(registers, time, channel, bin_width, s_thres):
    """
    Determine the slope threshold.

    Parameters
    ----------
    registers : dict
        Dictionary that stores the values of all registers that control
        the measurements performed by the FPGA.

    time : int
        Time (in seconds) the histogram should be accumulated over.

    channel : in
        Channel to be analysed.

    bin_width : int
        2^(bin_width) will be the width of the histogram bin.

    s_thres : int
        Determine the pulse threshold.

    Returns
    -------
    (fig,) : tuple
        Figure with the plotted histogram.

    """
    registers.event[channel].slope_threshold = s_thres

    ########## Parameters defined by the user ##########

    registers.mca.channel = channel
    registers.mca.ticks = time / (registers.tick_period * 4e-9)
    registers.mca.bin_n = bin_width

    ########## Automatically defined ##########

    registers.mca.value = 6  # s extreme value between zero crossings
    registers.mca.trigger = 6
    registers.mca.qualifier = 1
    registers.mca.lowest_value = 1

    ########## Updating MCA ##########

    registers.mca.update
    sleep(1)

    ########## reading the distribution ##########

    distributions = read_mca(1)  # reads the mca distributions

    histogram = distributions[0]

    ########## Plotting ##########

    fig = figure(
        title="Slope Threshold",
        title_location="left",
        tools="box_select, box_zoom, reset",
    )

    x = (
        histogram.lowest_value
        + np.arange(histogram.last_bin - 1) * histogram.bin_width
        + histogram.bin_width / 2
    )

    # plot config

    fig.yaxis.axis_label = "Counts"
    fig.y_range.start = 0
    fig.y_range.follow = "end"
    fig.y_range.end = 5e4
    fig.x_range.start = 0
    fig.x_range.end = 6e3
    fig.vbar(
        x=x,
        bottom=0,
        width=histogram.bin_width,
        top=histogram.bins,
        fill_alpha=0.4,
        line_alpha=1.0,
    )

    return (fig,)


def area_histogram(
    registers, time, channel, bin_width, p_thres, s_thres, xrange, yrange
):
    """
    Plot area histogram given the thresholds chosen.

    Always run this function to realise a sanity test for the selected
    values for baseline offset, pulse threshold and slope
    threshold.
    You should be able to see clear distinct peaks indicating that
    the current configuration of the TES can discriminate number of
    photons.

    Parameters
    ----------
    registers : dict
        Dictionary that stores the values of all registers that control
        the measurements performed by the FPGA.

    time : int
        Time (in seconds) the histogram should be accumulated over.

    channel : int
        Channel to be analysed.

    bin_width : int
        2^(bin_width) will be the width of the histogram bin.

    s_thres : int
        Determine the pulse threshold.

    Returns
    -------
    (fig,) : tuple
        Figure with the plotted histogram.
    """
    ########## Parameters defined by the user ##########

    registers.mca.channel = channel
    registers.mca.ticks = time / (registers.tick_period * 4e-9)
    registers.mca.bin_n = bin_width

    ########## Automatically defined ##########

    registers.mca.value = 7
    registers.mca.trigger = 3
    registers.mca.qualifier = 1
    registers.mca.lowest_value = -20 * 64

    ########## Detection Thresholds ##########

    registers.event[channel].slope_threshold = s_thres
    registers.event[channel].pulse_threshold = p_thres
    registers.event[channel].area_threshold = 0

    ########## Updating MCA ##########

    registers.mca.update
    sleep(1)

    ########## reading the distribution ##########

    distributions = read_mca(1)  # reads the mca distributions

    histogram = distributions[0]

    ########## Plotting ##########

    fig = figure(
        title="Areas", title_location="left", tools="box_select, box_zoom, reset"
    )

    x = (
        histogram.lowest_value
        + np.arange(histogram.last_bin - 1) * histogram.bin_width
        + histogram.bin_width / 2
    )

    ########## Plotting ##########

    fig.yaxis.axis_label = "Counts"
    fig.y_range.start = 0
    fig.y_range.follow = "end"  #
    fig.y_range.end = yrange
    fig.x_range.start = 0
    fig.x_range.end = xrange
    fig.vbar(
        x=x,
        bottom=0,
        width=histogram.bin_width,
        top=histogram.bins,
        fill_alpha=0.4,
        line_alpha=1.0,
    )

    return (fig,)
