"""
Module used to perform measurements with the TES.

"""

import yaml

from tes.registers import Registers
from tes.data import capture


def trace_drive(time, channel, p_thres, s_thres,
                baseline_sub, datapath, filename):
    """
    Record TES traces.

    Parameters
    ----------
    time : int
        Time in seconds to take measurements.

    channel : int
        Processing channel chosen to take measurements.

    p_thres : int
        Pulse threshold chosen using the MCA.

    s_thres : int
        Slope threshold chosen using the MCA.

    base_sub : bool
        If True, the baseline correction will be activated.
        It can automatically update the baseline level in
        the case where it changes.

    datapath : str
        Folder where the registers will be saved.

    filename : str
        Name of the file where the registers will be saved.

    Returns
    -------
    None

    Notes
    -----
        Keep your trace measurements up to 1 minute.
    """
    r = Registers('tcp://smp-loophole.instrument.net.uq.edu.au:10001')
    time_measurement = time/(r.tick_period*4e-9)
    print(time_measurement)
    r.baseline[channel].subtraction = baseline_sub

    # disabling all event registers
    r.event.enable = False

    # trace settings
    r.event[channel].packet = 'trace'
    r.event[channel].trace = 0
    r.event[channel].trace_type = 'single'
    r.event[channel].trace_sequence = 0
    r.event[channel].trace_stride = 5
    r.event[channel].trace_pre = 512
    r.event[channel].trace_length = 2048

    # event settings
    r.event[channel].timing = 0
    r.event[channel].max_rises = 1
    r.event[channel].height = 'peak'
    r.event[channel].pulse_threshold = p_thres
    r.event[channel].slope_threshold = s_thres
    r.event[channel].area_threshold = 0

    # inform user about baseline subtraction
    if r.baseline[channel].subtraction:
        print('bl on')
        measurement = (
            'trace-pulse_BL-{!r}-{!r}-'
            .format(r.event[channel].timing, r.event[channel].height)
        )
    else:
        print('bl off')
        measurement = (
            'trace-pulse-{!r}-{!r}-'
            .format(r.event[channel].timing, r.event[channel].height)
        )

    # Making measurements
    r.event.enable = True
    # limit to 1 min as no dropped frames
    c = capture(filename, measurement, ticks=time_measurement)
    r.event.enable = False

    # saving registers
    fname = '{}{}/{}_reg.yml'.format(datapath, filename, measurement)
    f = open(fname, 'w+')
    f.write(yaml.dump(dict(r.all)))
    f.close()

    # checking what has been done
    print(measurement)
    print(c)


def pulse_drive(time, channel, p_thres, s_thres,
                baseline_sub, datapath, filename):
    """
    Perform measurements over TES traces.

    Measure the following characteristics of TES traces:
        1) Length
        2) Area
        3) Maximum slope or height
        4) Rise Time

    Parameters
    ----------
    time : int
        Time in seconds to take measurements.

    channel : int
        Processing channel chosen to take measurements.

    p_thres : int
        Pulse threshold chosen using the MCA.

    s_thres : int
        Slope threshold chosen using the MCA.

    base_sub : bool
        If True, the baseline correction will be activated.
        It can automatically update the baseline level in
        the case where it changes.

    datapath : str
        Folder where the registers will be saved.

    filename : str
        Name of the file where the registers will be saved.

    Returns
    -------
    None
    """
    r = Registers('tcp://smp-loophole.instrument.net.uq.edu.au:10001')
    time_measurement = time/(r.tick_period*4e-9)
    print(time_measurement)
    r.baseline[channel].subtraction = baseline_sub

    # disabling all event registers
    r.event.enable = False

    # trace settings
    r.event[channel].packet = 'pulse'

    # event settings
    r.event[channel].timing = 0
    r.event[channel].max_rises = 1
    r.event[channel].height = 'peak'
    r.event[channel].pulse_threshold = p_thres
    r.event[channel].slope_threshold = s_thres
    r.event[channel].area_threshold = 0

    # inform user about baseline subtraction

    if r.baseline[channel].subtraction:
        print('bl on')
        measurement = (
            '-pulse_BL-{!r}-{!r}-test3'
            .format(r.event[channel].timing, r.event[channel].height)
        )

    else:
        print('bl off')
        measurement = (
            '-pulse-{!r}-{!r}-test3'
            .format(r.event[channel].timing, r.event[channel].height)
        )

    # Making measurements
    r.event.enable = True
    c = capture(filename, measurement, ticks=time_measurement)
    r.event.enable = False

    # saving registers
    fname = '{}{}/{}_reg.yml'.format(datapath, filename, measurement)
    f = open(fname, 'w+')
    f.write(yaml.dump(dict(r.all)))
    f.close()

    # checking what has been done

    print(measurement)
    print(c)
