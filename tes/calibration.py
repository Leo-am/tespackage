#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions to calibrate the TES detector.

Author: Leonardo Assis Morais
"""

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import patches
# from scipy import signal
# from scipy.optimize import brentq

# import tes


def maxima(function, thresh=10):
    """
    Find local maxima of a function f.

    Uses rising zero crossings of the gradient of f to find function
    local maxima.

    Parameters
    ----------
    function : ndarray
        Function f to be analysed.
    thresh : int
        Only return a maxima if f[maxima] > thresh.

    Returns
    -------
        : array
        Array with maxima points of f.
    """
    grad = np.gradient(function)
    pos = grad > 0
    xings = (pos[:-1] & ~pos[1:]).nonzero()[0]
    return xings[np.where(function[xings] > thresh)]


def plot_guess(ax, hist, smooth_hist, bin_centres, max_i):
    """
    Plot the histogram with an educated guess for its fitting.

    Visual check if the maxima positions determined by the
    guess_thresholds are reasonable.

    Requires guess_thresholds.

    Parameters
    ----------
        ax : figure axis
            Figure axis of the figure to be plotted.
        hist : np.darray
            Histogram obtained from the guess_thresholds function.
        smooth_hist :
            Smoothed histogram obtained from the guess_thresholds
            function.
        bin_centres :
            Centre of the bins for the histogram. Obtained from the
            guess_thresholds function.
        max_i : list
            List with the position of the maxima points for the
            peaks in histogram. Obtained from the guess_thresholds
            function.

    Returns
    -------
        ax : figure axis
            Axis for the figure with the histogram, smoothed histogram
            and maxima points plotted.
    """
    # ################ FONT SIZES ################
    FONTSIZE = 20
    plt.rc("font", size=FONTSIZE)
    plt.rc("axes", titlesize=FONTSIZE)
    plt.rc("axes", labelsize=FONTSIZE)
    plt.rc("xtick", labelsize=FONTSIZE)
    plt.rc("ytick", labelsize=FONTSIZE)
    plt.rc("legend", fontsize=FONTSIZE)

    # ################ PLOTTING THE HISTOGRAM ################

    # histogram
    ax.plot(
        bin_centres[hist.nonzero()[0]],
        hist[hist.nonzero()[0]],
        "s",
        color='gray',
        markersize=5,
        markerfacecolor="none",
        mew=0.5,
        label="data",
    )
    # smoothed histogram
    ax.plot(bin_centres, smooth_hist, lw=1, label="First guess", color='k')
    # maxima points
    ax.plot(
        bin_centres[max_i],
        smooth_hist[max_i],
        "o",
        color='red',
        label="peaks",
        markerfacecolor="red",
        mew=1,
        markersize=8
    )

    expx = tes.traces.correct_xticks(ax)
    ax.set_ylabel(r"Counts")
    ax.set_xlabel(r"TES Pulse Areas (Arb. Un.) $(\times 10^{{ {:.0f} }})$"
                  .format(expx))
    ax.set_yscale("log")
    ax.legend()

    return ax


def guess_histogram(areas, bin_number, win_length, minimum, max_area):
    """
    Create first guess for histogram fitting.

    Using a Hann function to create an approximate fit for the
    data, estimates the position for the centre of the different
    peaks in the area histogram.

    WARNING:
    The Hann window fitting must be checked to see if
    it is reasonable using the function plot_guess.

    Requires
    --------
        maxima

    Parameters
    ----------
    bin_number : int
        Number of bins used in histogram.
    win_length : int
        Length of the Hann window used in the fit.
    minimum : int
        Minimum value for the function maxima to consider
        a given number of counts as a peak.
    max_area : float
        Areas > max_area are considered invalid counts and
        removed from the analysis. Used to remove event with
        anomalous areas.

    Returns
    -------
        counts : np.darray
            Array with counts for each histogram bin.
        smooth_hist : np.darray
            Array with the continuous fit used to determine
            the positions of the maxima in the data.
        bin_centre : np.darray
            Array with the positions of the histograms bin centres.
            Used for plotting the histogram with correct values
            for areas in the x-axis.
        max_i : list
            List with the position of the maxima points.
    """
    data = areas[areas < max_area]  # remove events with anomalous areas.
    print("The number of events in the analysed data is:", len(data))

    counts, edges = np.histogram(data, bin_number)
    bin_width = edges[1] - edges[0]
    bin_centre = edges[:-1] + bin_width / 2

    # smoothed histogram
    win = signal.hann(win_length)
    smooth_hist = signal.convolve(counts, win, mode="same") / sum(win)
    # loc for distributions
    max_i = maxima(smooth_hist, thresh=minimum)
    max_list = [0.0] + list(bin_centre[max_i])

    return (counts, smooth_hist, bin_centre, max_i, max_list)


def area_histogram(max_area, bin_number, areas):
    """
    Create an area histogram.

    Parameters
    ----------
    max_area : float
        Maximum area allowed (remove extremely large areas not due to
                              photon detections).
    bin_number : int
        Number of histogram bins.
    areas : np.ndarray
        Data with areas of TES pulses.

    Returns
    -------
    bin_centre : np.darray
        Array with the positions of the histograms bin centres.
        Used for plotting the histogram with correct values
        for areas in the x-axis.
    counts : np.ndarray
        Histogram counts.
    error : np.ndarray
        Standard deviation for counts
        (poissonian distribution assumed: std. = sqrt(counts).
    bin_width : float
        Bin width for plotting.

    """
    bin_size = max_area / bin_number
    data = areas[areas < max_area]

    # printing analysis information
    print("The number of points in the analysed data is:", len(data), ".")
    print("We analyse data up to:", max_area, "area units.")
    print("The number of bins in the analysed data is:", bin_number, ".")
    print("The size of each bin is:", bin_size, ".")

    # constructing the histogram
    counts, edges = np.histogram(data, bin_number)
    bin_width = edges[1] - edges[0]
    bin_centre = edges[:-1] + bin_width / 2

    # for bins with 0 counts set error to 1
    # otherwise the error would be 0 and the
    # fitting algorithm will break
    error = np.zeros(len(bin_centre))
    for idx in range(0, len(counts)):
        if counts[idx] != 0:
            error[idx] = np.sqrt(counts[idx])
        else:
            error[idx] = 1.0

    return bin_centre, counts, error, bin_width


def residual_gauss(params, i_var, data, eps_data, max_idx):
    """
    Gaussian model to fit histogram from TES detections.

    Model composed of a sum of N gaussian distributions, to be used
    with the lmfit package to fit histograms from TES detections.
    Returns the residual between data and model, divided by the
    error in the data.

    Parameters
    ----------
    params : lmfit object
        List of parameters to be used in the fit.
    i_var :
        Independent variable to be used in the fit.
    data : np.ndarray
        Data to be fitted.
    eps_data : np.ndarray
        Standard deviation of each data point.
    max_idx : int
        Number of gaussian distributions used in the model.

    Returns
    -------
        residual : np.ndarray
            Array with the different between data and model
            divided by the error in data.
    """
    # the gaussian in the fit is calculate using:
    # amplitude*1/\sqrt{2*pi*scale}*exp[-(x-loc)^2/(2*scale^2)]

    scale = np.zeros(max_idx)  # scale for gaussian fit
    loc = np.zeros(max_idx)  # centre of gaussian distribution
    amp = np.zeros(max_idx)  # amplitude for gaussian fit
    ind_model = np.zeros((max_idx, len(i_var)))

    for idx in range(1, max_idx):
        scale[idx] = params[r"scale{}".format(idx)]
        loc[idx] = params[r"loc{}".format(idx)]
        amp[idx] = params[r"amp{}".format(idx)]
        ind_model[idx] = (
            amp[idx]
            * 1
            / (np.sqrt(2 * np.pi) * scale[idx])
            * np.exp(-((i_var - loc[idx]) ** 2) / (2 * scale[idx] ** 2))
        )

    model = sum(ind_model)

    return (data - model) / eps_data


def plot_area(ax, bin_centre, counts, error, model, plot_steps):
    """
    Require area_histogram to be run before.

    Parameters
    ----------
    ax : figure axis
        Figure axis of the figure to be plotted.
    bin_centre : np.darray
        Array with the positions of the histograms bin centres.
        Used for plotting the histogram with correct values
        for areas in the x-axis.
    counts : np.ndarray
        Histogram counts.
    error : np.ndarray
        Standard deviation for counts
    model : np.ndarray
        Model generated with gaussian_model
    plot_steps : int
        Number of histogram points to be skipped when plotting.

    Returns
    -------
    None.

    Requires
    --------
    area_histogram

    """
    FONT_SIZE = 20
    plt.rc("font", size=FONT_SIZE)
    plt.rc("axes", titlesize=FONT_SIZE)
    plt.rc("axes", labelsize=FONT_SIZE)
    plt.rc("xtick", labelsize=FONT_SIZE)
    plt.rc("ytick", labelsize=FONT_SIZE)
    plt.rc("legend", fontsize=FONT_SIZE)

    ax.errorbar(
        bin_centre[::plot_steps],
        counts[::plot_steps],
        error[::plot_steps],
        fmt="o",
        color="k",
        zorder=1,
        markerfacecolor="k",
        markersize=5,
        markeredgewidth=2,
    )
    ax.plot(bin_centre, model, "--r", zorder=2, lw=2.5)
    expx = tes.traces.correct_xticks(ax)
    ax.set_xlabel(
        r"Pulse {} $(\times 10^{{ {:.0f} }})$ (arb. un.)".format("Area", expx)
    )

    # y axis
    ax.set_ylabel(r"Counts")
    ax.set_yscale("log")

    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(1)

    ax.minorticks_off()
    [line.set_markersize(5) for line in ax.yaxis.get_ticklines()]
    [line.set_markeredgewidth(1) for line in ax.yaxis.get_ticklines()]

    [line.set_markersize(5) for line in ax.xaxis.get_ticklines()]
    [line.set_markeredgewidth(1) for line in ax.xaxis.get_ticklines()]

    ax.set_ylim(0.9e0, 1e4)
    ax.set_xlim(-0.15e6, 8e6)


def gaussian_model(fitting, counts, bin_centre, max_idx):
    """
    Implement mixture model comprised of gaussians.

    Parameters
    ----------
    fitting : class lmfit.minimizer.MinimizerResult
        Result from fitting using lmfit.
    counts : np.ndarray
        Histogram counts.
    bin_centre : np.darray
        Array with the positions of the histograms bin centres.
        Used for plotting the histogram with correct values
        for areas in the x-axis.
    max_idx : int
        Number of gaussian distributions used in the model.

    Returns
    -------
    model : np.ndarray
        Model of composed of sum of gaussians using the results 
        obtained from the fitting with lmfit.

    """
    scale = np.zeros(max_idx)
    loc = np.zeros(max_idx)
    amp = np.zeros(max_idx)
    ind_model = np.zeros((max_idx, len(counts)))

    for idx in range(1, max_idx):
        scale[idx] = fitting.params[r"scale{}".format(idx)]
        loc[idx] = fitting.params[r"loc{}".format(idx)]
        amp[idx] = fitting.params[r"amp{}".format(idx)]
        ind_model[idx] = (
            amp[idx]
            * 1
            / (np.sqrt(2 * np.pi) * scale[idx])
            * np.exp(-((bin_centre - loc[idx]) ** 2) / (2 * scale[idx] ** 2))
        )

    model = sum(ind_model)
    return model


def plot_histogram(ax, data, bin_number, measurement):
    """
    Plot histogram of TES characteristics.

    Parameters
    ----------
    ax : figure axis
        Figure axis of the figure to be plotted.
    data : np.ndarray
        Data extracted from TES (height, area, length, maximum slope).
    bin_number : int
        Number of histogram bins.
    measurement : str
        Type of measurement plotted.

    Returns
    -------
    None.

    """
    FONTSIZE = 20

    plt.rc('font', size = FONTSIZE)
    plt.rc('axes', titlesize = FONTSIZE)
    plt.rc('axes', labelsize = FONTSIZE)
    plt.rc('xtick', labelsize = FONTSIZE)
    plt.rc('ytick', labelsize = FONTSIZE)
    plt.rc('legend', fontsize = FONTSIZE)

    # constructing the histogram from data
    hist, edges = np.histogram(data, bin_number)
    bin_width = edges[1] - edges[0]
    bin_centre = edges[:-1] + bin_width/2

    # plotting the histogram
    ax.plot(bin_centre, hist,
            's', markersize=4, markerfacecolor='k', mew=0.5, color='k')

    exp_x = tes.traces.correct_xticks(ax)
    exp_y = tes.traces.correct_yticks(ax)

    ax.set_xlabel(r'Pulse {} $(\times 10^{})$ (arb. un.)'
                  .format(measurement, {exp_x}))
    ax.set_ylabel(r'Counts $(\times 10^{})$'.format({exp_y}))
    

def find_thresholds(gauss_fit, maxima_list, bin_centre, counts, const):
    """
    Find the counting thresholds given a gaussian fit.

    Parameters
    ----------
    gauss_fit : lmfit fit
         Result of a least square minimisation using lmfit.
    maxima_list : np.ndarray
        Array with the positions of maxima points.
    bin_centre : np.ndarray
        Array with centre bin positions for histogram.
    counts : np.ndarray
        List with counts for histogram.
    const : float
        Value used for scipy.optimise.brentq. 
        See 'Notes' below for more information.

    Returns
    -------
    dist : list
        List with the normalised distributions.
    new_thresh_list :
        List of counting thresholds.

    Requires
    --------
        scipy.optimize.brentq
        
    Notes
    -----
        Sometimes this function breaks if brentq function cannot find
        roots. If that is the case, slowly increase the value of the
        variable 'const'.
    """

    MAX_IDX = len(maxima_list)

    # fitting parameters
    scale = np.zeros(MAX_IDX)
    loc = np.zeros(MAX_IDX)
    amp = np.zeros(MAX_IDX)
    dist = np.zeros((MAX_IDX, len(counts)))

    max_index = np.zeros(MAX_IDX)

    # errors in fitting parameters
    err_scale = np.zeros(MAX_IDX)
    err_loc = np.zeros(MAX_IDX)

    # errors in threshold positioning
    error_plus = np.zeros((MAX_IDX, len(counts)))
    error_minus = np.zeros((MAX_IDX, len(counts)))

    for idx in range(1, MAX_IDX):
        scale[idx] = gauss_fit.params[r"scale{}".format(idx)]
        loc[idx] = gauss_fit.params[r"loc{}".format(idx)]
        amp[idx] = gauss_fit.params[r"amp{}".format(idx)]
        dist[idx] = (
            1
            / (np.sqrt(2 * np.pi) * scale[idx])
            * np.exp(-((bin_centre - loc[idx]) ** 2) / (2 * scale[idx] ** 2))
        )

        max_index[idx] = np.where(dist[idx] == np.amax(dist[idx]))[0][0]
        err_scale[idx] = gauss_fit.params[r"scale{}".format(idx)].stderr
        err_loc[idx] = gauss_fit.params[r"loc{}".format(idx)].stderr
        error_plus[idx] = (
            1
            / (np.sqrt(2 * np.pi) * (scale[idx] + err_scale[idx]))
            * np.exp(
                -((bin_centre - (loc[idx] + err_loc[idx])) ** 2)
                / (2 * (scale[idx] + err_scale[idx]) ** 2)
            )
        )
        error_minus[idx] = (
            1
            / (np.sqrt(2 * np.pi) * (scale[idx] - err_scale[idx]))
            * np.exp(
                -((bin_centre - (loc[idx] - err_loc[idx])) ** 2)
                / (2 * (scale[idx] - err_scale[idx]) ** 2)
            )
        )

    # variables for thresholds
    # const = 3e4  # used to guarantee input functions of brentq have opposite
    # signs when calculating counting threshold positions
    thresh_guess = np.zeros(MAX_IDX)
    thresh_list = np.zeros(MAX_IDX)
    thresh_plus = np.zeros(MAX_IDX)
    thresh_minus = np.zeros(MAX_IDX)

    def brentq_aux(x):
        """
        Auxiliary function used in scipy.optimise.brentq to find
        the threshold positions.

        Must be used as the first argument for scipy.optimize.brentq.
        """
        term1 = (
            1
            / (np.sqrt(2 * np.pi) * scale[idx])
            * np.exp(-((x - loc[idx]) ** 2) / (2 * scale[idx] ** 2))
        )
        term2 = (
            1
            / (np.sqrt(2 * np.pi) * scale[idx + 1])
            * np.exp(-((x - loc[idx + 1]) ** 2) / (2 * scale[idx + 1] ** 2))
        )
        return term1 - term2

    # determining the threshold positions

    # initial guess for thresholds
    for idx in range(1, MAX_IDX - 1):
        thresh_guess[idx] = (
            bin_centre[int(max_index[idx + 1])] - bin_centre[int(max_index[idx])]
        ) / 2 + bin_centre[int(max_index[idx])]

    # finding thresholds
    for idx in range(1, MAX_IDX - 1):
        thresh_list[idx] = brentq(
            brentq_aux, thresh_guess[idx] - const, thresh_guess[idx] + const
        )

    # adding last threshold position
    max_ind = np.where(dist[MAX_IDX - 1] == np.max(dist[MAX_IDX - 1]))
    thresh_list[-1] = thresh_list[-2] + (bin_centre[max_ind] - thresh_list[-2]) * 2
    # determining the error + in thresholds

    for idx in range(1, MAX_IDX - 1):
        thresh_plus[idx] = brentq(
            brentq_aux, thresh_list[idx] - const, thresh_list[idx] + const
        )
        thresh_minus[idx] = brentq(
            brentq_aux, thresh_list[idx] - const, thresh_list[idx] + const
        )

    return dist, thresh_list


def plot_normalised(ax, max_i, bin_centre, dist, thresholds):
    """
    Plot the graph with the normalised distributions.

    Parameters
    ----------
    ax : figure axis
        Figure axis of the figure to be plotted.
    max_i : np.ndarray
        Number of distributions to be plotted.
    bin_centre : np.ndarray
        x-axis positions for histogram data.
    dist : np.ndarray
        Distributions to be plotted.
    thresholds : np.ndarray
        Counting thresholds positions to be plotted.

    Returns
    -------
    None.

    """
    MAX_PLOT = len(max_i)  # distributions to be plotted + 1
    LAST_THRESH = 8.1e6  # position of last threshold plotted
    NOTE_HEIGHT = 1.025e-5  # height of annotated numbers
    COLOR_NUM = 10  # must be 10 or less
    FONT_SIZE = 18
    LAST_DIST = 0.5e6

    plt.rc('font', size=FONT_SIZE)
    plt.rc('axes', titlesize=FONT_SIZE)
    plt.rc('axes', labelsize=FONT_SIZE)
    plt.rc('xtick', labelsize=FONT_SIZE)
    plt.rc('ytick', labelsize=FONT_SIZE)
    plt.rc('legend', fontsize=FONT_SIZE)

    # plotting distributions
    [ax.plot(bin_centre, dist[idx]) for idx in range(1, MAX_PLOT)]

    # to handle colour of the graph
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    shade = 0.2

    # ####### Regions for counting photons ############
    for tres in range(MAX_PLOT-1):
        col = colors[tres % COLOR_NUM]
        # threshold position and coloring intervals
        vert_span = ax.axvspan(thresholds[tres], thresholds[tres+1],
                               alpha=shade, color=col)

        # numbers to identify distributions
        ax.text(bin_centre[np.argmax(dist[tres+1])], NOTE_HEIGHT, r'{}'.
                format(tres+1), ha='center', va='center',
                fontsize=FONT_SIZE, color=col)

    # last threshold
    vert_span = ax.axvspan(thresholds[tres+1], LAST_THRESH, alpha=shade,
                           color=colors[(MAX_PLOT-1) % COLOR_NUM])
    # last coloring interval
    vert_color = patches.Patch(facecolor=colors[(MAX_PLOT-1) % COLOR_NUM],
                               edgecolor=colors[(MAX_PLOT-1) % COLOR_NUM])
    # last number
    ax.text(bin_centre[np.argmax(dist[MAX_PLOT-1])] + LAST_DIST,
            NOTE_HEIGHT, r'{}+'.format(MAX_PLOT), ha='center',
            va='center', fontsize=FONT_SIZE,
            color=colors[(MAX_PLOT-1) % COLOR_NUM])

    # axis ticks
    expy = tes.traces.correct_yticks(ax)
    expx = tes.traces.correct_xticks(ax)

    ax.set_ylabel(r'Normalised Distributions $(\times 10^{{ {} }})$'
                  .format(expy))
    ax.set_xlabel(r'Pulse Area $(\times 10^{{ {} }})$ (arb. un.)'.format(expx))
