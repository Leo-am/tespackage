"""
Module to assist in counting photons routine.

1) coincidence
3) window
"""

from collections import namedtuple

import numpy as np
import csv
from numpy import (logical_and as and_l, logical_not as not_l)
from numba import jit


def get_thresholds(calibration_file):
    """
    Load the threshold from a calibration file.

    Calibration file must be generated with "TES_Calibration.ipynb".

    Parameters
    ----------
    calibration_file : str
        Calibration file path.

    Returns
    -------
    thresholds : np.array
        Array with the threshold positions.

    """
    read_threshold = []

    with open(calibration_file, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        header = next(csv_reader)
        data_set = header
        if header is not None:
            for row in csv_reader:
                read_threshold.append(row)
    print("Data used for calibration: \n {}".format(data_set[0]))
    thresholds = np.array([float(idx) for idx in read_threshold[1]])
    return thresholds


def coincidence(abs_time, mask, low, high):
    """
    Find coincidences between two channels.

    After finding a coincidence, return indices of coincident events.

    Parameters
    ----------
    abs_time : ndarray
        Array where each entry corresponds to the time which the
        detection was performed with respect to an initial time t0.
        This array can be constructed using
        np.cumsum(CaptureData(datapath).times).
    mask : ndarray bool
        Identifies the channels in abs_time.
        For abs_times where mask is False, time t is coincident if
        abs_time+low <= t <= abs_time+high.
        If the mask is True t is coincident if
        abs_time-high <= t <= abs_time-low.
    low : float
        The starting of the coincidence window.
    high : float
        The ending side of the coincidence window.

    Returns
    -------
    coincidences(coinc, coinc_mask) : tuple
        coinc : ndarray
            Indexes of the coincident event in the other channel.
            When more than one event is found in the window the negated
            value of the first index is entered.
        coinc_mask : ndarray bool
            Indicates where exactly one event was found in the window.

    Requires:
    ---------
        tes.counts.window
    """
    coinc = np.zeros(len(abs_time), np.int32)
    coinc_mask = np.zeros(len(abs_time), np.bool)

    for time_ind in range(len(abs_time)):

        if time_ind % 10000000 == 0:
            print(time_ind)

        # if the time is not associated with a detection event enter this if
        if not mask[time_ind]:

            low_index, high_index = window(time_ind, abs_time, low, high)
            offset = low_index

        else:
            low_index, high_index = window(time_ind, abs_time, -high, -low)
            offset = high_index

        coinc_length = high_index - low_index
        if high_index and low_index:

            if coinc_length == 1:
                coinc[time_ind] = time_ind + offset
                coinc_mask[time_ind] = True

            elif coinc_length > 1:
                coinc[time_ind] = -(time_ind + offset)
                coinc_mask[time_ind] = False

    return coinc, coinc_mask


@jit
def window(ind, abs_time, low, high):
    """
    Find coincident events in the given window.

    Get offsets from the current index ind in abs_time that are in the
    relative time window defined by low and high.

    Parameters
    ----------
    ind : int
        Current index for abs_time.

    abs_time : ndarray
        Absolute times.

    low : float
        Starting point of the relative window.

    high : float
        Ending point of the relative window.

    Returns
    -------
    (low_index, high_index) : tuple
        coincident times are abs_time[low_index:ind+high_index]
    """
    length = len(abs_time)
    now = abs_time[ind]
    low_index = 0  # low index offset
    high_index = 0  # high index offset

    low_t = now + low
    high_t = now + high

    if low_t < now:
        # before and: guarantees the index will be always >= 0
        # after and:
        while (ind + low_index >= 0) and abs_time[ind + low_index] >= low_t:
            low_index -= 1

    else:
        # 1term: guarantees the index will never exceed array length
        # 2term: check if the absolute_time[index + low_index] is
        # smaller than low_t
        # if it is, low_index is increased by one
        # if it is not, finish while loop and low_index is returned
        # this low index will be the last index for abs_time where
        # it is still smaller than low_t
        while (low_index + ind < length) and abs_time[ind + low_index] < low_t:
            low_index += 1

    if high_t < now:
        while (ind + high_index >= 0) and abs_time[ind + high_index] > high_t:
            high_index -= 1

    else:
        # 1term: guarantees the index will never exceed array length
        # 2term: check if the absolute_time[index + high_index]
        # is smaller than high_t
        # if it is, high_index is increased by one
        # if it is not, finish while loop and high_index is returned
        # this high index will be the last index for abs_time where
        # it is still smaller than _t
        while ((high_index + ind < length) and
               abs_time[ind + high_index] <= high_t):
            high_index += 1
    # offset indexes marking abs_times in the window
    return low_index, high_index


Counts = namedtuple('Counts', 'count uncorrelated vacuum')


def counting_photons(data, thresh_list, vacuum=False,
                     coinc_mask=None, herald_mask=None):
    """
    Convert TES pulse area in photon-number.

    Given an TES pulse area data set and the calibration (list with the
    counting thresholds, passed through thresh_list), returns
    the number of counts for each Fock state.

    If the mask with the heralding source is given, also includes
    the number of vacuum counts.

    Parameters
    ----------
    data : np.ndarray
        Measured data.

    thresh_list : np.ndarray
        List with the counting thresholds obtained after detector
        calibration.

    vacuum : bool
        If true, it will use the herald_mask to calculate vacuum
        counts

    coinc_mask : ndarray, bool
        Mask indicating coincidence counts.

    herald_mask : ndarray, bool
        Mask indicating which events are in the heralding channel.

    Returns
    -------
    Counts(count, uncorrelated, vacuum) : named tuple
        count : ndarray
            Counts for each photon number
        uncorrelated :
            Counts events not correlated with herald.
            Only calculated when coinc_mask and herald_mask are
            provided.
        vacuum :
            indicates if count[0] contains vacuum counts
    """
    if vacuum and (coinc_mask is None or herald_mask is None):
        raise AttributeError('Neither coinc_mask or herald_mask cannot'
                             'be None when vacuum = True.'
                             )

    if (
        (herald_mask is not None and coinc_mask is None) or
        (herald_mask is None and coinc_mask is not None)
       ):
        raise AttributeError('coinc_mask or herald_mask both be'
                             'specified or both None.'
                             )

    # data events are the ones which are not in the heralding channel
    if herald_mask is not None:
        data_mask = not_l(herald_mask)
        uncorrelated = []

    else:
        data_mask = None
        uncorrelated = None

    counts = []

    for thresh in range(1, len(thresh_list)):
        # and_l gives a boolean array with true (1) values where
        # condition is satisfied .nonzero() gives the indexes of
        # position with value true .nonzero()[0] gives the first
        # value of the position here, we are interested just in
        # the number of counts, not when these counts happened
        # therefore, len(and_l (.nonzero()[0]) ) gives us the
        # answer
        counts.append(
            # determine the number of counts for a given # photons by
            # calculating the length of a list that stores the
            # positions of data which are in between two thresholds
            len(
                # return a boolean list: 1 when condition are satisfied
                # 0 when they are not
                and_l(
                    # returns a list with the positions of vector data
                    # which are inside two adjacent thresholds
                    (data > thresh_list[thresh - 1]),
                    (data <= thresh_list[thresh])
                ).nonzero()[0]  # checking the number of nonzero
                                # elements to count events
            )
        )

        # gives the number of uncorrelated counts if the herald mask
        # is given
        if uncorrelated is not None:
            # coinc_mask[data_mask] selects the position in coinc_mask
            # which are related to the data being analysed
            uncorrelated.append(
                len(
                    and_l(

                        data[not_l(coinc_mask)[data_mask]] >
                        thresh_list[thresh-1],
                        data[not_l(coinc_mask)[data_mask]] <=
                        thresh_list[thresh]
                    ).nonzero()[0]
                )
            )
    print(uncorrelated[0])
    # counts above the last threshold
    counts.append(len((data > thresh_list[-1]).nonzero()[0]))
    if uncorrelated is not None:
        uncorrelated.append(
            len(
                (data[not_l(coinc_mask)[data_mask]] >
                    thresh_list[-1]).nonzero()[0]
            )
        )

    # counting vacuum
    # uncorrelated heralds + correlated data <= the first threshold
    if vacuum:
        # counts the number of uncorrelated heralds
        uncorrelated_heralds = len(
            (not_l(coinc_mask)[herald_mask]).nonzero()[0]
        )
        counts.insert(0, 0)
        uncorrelated.insert(0, uncorrelated_heralds)

    if uncorrelated is not None:
        uncorrelated = np.array(uncorrelated)

    print('The number of uncorrelated counts is', uncorrelated)
    print('The total number of counts is', counts)
    return Counts(np.array(counts), uncorrelated, vacuum)
