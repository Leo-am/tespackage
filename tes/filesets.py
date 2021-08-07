"""
# dict containing tuples (filename, dtype, is_list, is_sliceable) representing a set of testbench output files
# to be read to create a simulation.data Data class
#
# The dict key is the name to give the resulting attribute in the Data class instance
# The file is read using numpy.fromfile() with the given dtype
# If is_list is True then all files of the form filenmameX are read, where X is a digit indicating the channel
# the created attribute is a list of values.
# is_slicable boolean indicates that the attribute should be included when creating a Data.Slice object.
# When the dtype includes a field labeled index, the slice will contain the values where the index field is
# in the slice bounds rather than the traditional start:stop range
"""

import numpy as np
from tes.data import File


meas_dt = np.dtype([("index", np.int32), ("area", np.int32), ("extrema", np.int32)])
index_dt = np.dtype([("index", np.uint32)])
stream_dt = np.dtype([("data", ">u8"), ("last", np.bool)])
error_dt = np.dtype([("index", np.uint32), ("flags", np.uint8)])
traces_dt = np.dtype(
    [
        ("input", np.int16),
        ("raw", np.int16),
        ("filtered", np.int16),
        ("slope", np.int16),
    ]
)

measurement_overflow_TB = {
    "trace": File(
        "traces",
        np.dtype([("raw", np.int16), ("filtered", np.int16), ("slope", np.int16)]),
        True,
        True,
    ),
    "raw": File(
        "raw",
        np.dtype([("index", np.int32), ("area", np.int32), ("extrema", np.int16)]),
        True,
        True,
    ),
    "filtered": File(
        "filtered",
        np.dtype([("index", np.int32), ("area", np.int32), ("extrema", np.int16)]),
        True,
        True,
    ),
    "slope": File(
        "slope",
        np.dtype([("index", np.int32), ("area", np.int32), ("extrema", np.int16)]),
        True,
        True,
    ),
    "pulse": File(
        "pulse",
        np.dtype([("index", np.int32), ("area", np.int32), ("extrema", np.int16)]),
        True,
        True,
    ),
    "pulse_start": File("pulsestart", index_dt, True, True),
    "pulse_stop": File("pulsestop", index_dt, True, True),
    "slope_thresh_xing": File("slopethreshxing", index_dt, True, True),
    "peak": File("peak", index_dt, True, True),
    "heights": File("height", index_dt, True, True),
    "cfd_low": File("cfdlow", index_dt, True, True),
    "cfd_high": File("cfdhigh", index_dt, True, True),
    "trigger": File("trigger", index_dt, True, True),
    "mux_stream": File("muxstream", stream_dt, False, False),
    "event_stream": File("eventstream", stream_dt, True, False),
    "mca_stream": File("mcastream", stream_dt, False, False),
    "ethernet_stream": File("ethernetstream", stream_dt, False, False),
    "cfd_error": File("cfderror", error_dt, False, True),
    "time_overflow": File("timeoverflow", np.int32, False, True),
    "peak_overflow": File("peakoverflow", error_dt, False, True),
    "mux_full": File("muxfull", error_dt, False, True),
    "mux_overflow": File("muxoverflow", error_dt, False, True),
    "framer_overflow": File("frameroverflow", error_dt, False, True),
    "baseline_error": File("baselineerror", error_dt, False, True),
    "settings": File("setting", np.int32, True, False),
    "mca_settings": File("mcasetting", np.int32, False, False),
    "byte_stream": File(
        "bytestream",
        np.dtype([("index", np.uint32), ("data", np.uint8), ("last", np.bool)]),
        False,
        False,
    ),
}
measurement_subsystem_TB = {
    "trace": File("traces", traces_dt, True, True),
    "raw": File("raw", meas_dt, True, True),
    "filtered": File("filtered", meas_dt, True, True),
    "slope": File("slope", meas_dt, True, True),
    "pulse": File("pulse", meas_dt, True, True),
    "pulse_start": File("pulsestart", index_dt, True, True),
    "pulse_start": File("pulsestop", index_dt, True, True),
    "slope_thresh_xing": File("slopethreshxing", index_dt, True, True),
    "peak": File("peak", index_dt, True, True),
    "peak_start": File("peakstart", index_dt, True, True),
    "peak_start": File("eventstart", index_dt, True, True),
    "heights": File("height", index_dt, True, True),
    "cfd_low": File("cfdlow", index_dt, True, True),
    "cfd_high": File("cfdhigh", index_dt, True, True),
    "trigger": File("trigger", index_dt, True, True),
    "mux_stream": File("muxstream", stream_dt, False, False),
    "event_stream": File("eventstream", stream_dt, True, False),
    "mca_stream": File("mcastream", stream_dt, False, False),
    "ethernet_stream": File("ethernetstream", stream_dt, False, False),
    "cfd_error": File("cfderror", error_dt, False, True),
    "time_overflow": File("timeoverflow", np.int32, False, True),
    "peak_overflow": File("peakoverflow", error_dt, False, True),
    "mux_full": File("muxfull", error_dt, False, True),
    "mux_overflow": File("muxoverflow", error_dt, False, True),
    "framer_overflow": File("frameroverflow", error_dt, False, True),
    "baseline_error": File("baselineerror", error_dt, False, True),
    "settings": File("setting", np.int32, True, False),
    "mca_settings": File("mcasetting", np.int32, False, False),
    "byte_stream": File(
        "bytestream",
        np.dtype([("index", np.uint32), ("data", np.uint8), ("last", np.bool)]),
        False,
        False,
    ),
}

measurement_unit_TB = {
    "trace": File("traces", traces_dt, False, True),
    "raw": File("raw", meas_dt, False, True),
    "filtered": File("filtered", meas_dt, False, True),
    "slope": File("slope", meas_dt, False, True),
    "pulse": File("pulse", meas_dt, False, True),
    "pulse_start": File("pulsestart", index_dt, False, True),
    "pulse_stop": File("pulsestop", index_dt, False, True),
    "event_start": File("eventstart", index_dt, False, True),
    "slope_thresh_xing": File("slopethreshxing", index_dt, False, True),
    "peak": File("peak", index_dt, False, True),
    "peak_start": File("peakstart", index_dt, False, True),
    "heights": File("height", index_dt, False, True),
    "cfd_low": File("cfdlow", index_dt, False, True),
    "cfd_high": File("cfdhigh", index_dt, False, True),
    "trigger": File("trigger", index_dt, False, True),
    "event_stream": File("eventstream", stream_dt, False, False),
    "cfd_error": File("cfderror", error_dt, False, True),
    "time_overflow": File("timeoverflow", np.int32, False, True),
    "peak_overflow": File("peakoverflow", error_dt, False, True),
    "settings": File("setting", np.int32, False, False),
}
