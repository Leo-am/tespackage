"""
MCA low level control. 

Classes: 

1) Value
2) Trigger
3) Qualifier
4) Distribution

"""

import numpy as np
from tes.base import VhdlEnum


class Value(VhdlEnum):
    disabled = 0
    f = 1             # the filtered tes sequence
    f_area = 2        # the area of f between  zero crossings
    f_extrema = 3     # the extreme value of f between zero crossings
    s = 4             # the slope of f
    s_area = 5        # the area of s between zero crossings
    s_extrema = 6     # the extreme value of s between zero crossings
    pulse_area = 7    # the area of s above the pulse_threshold setting
    raw = 8           # the raw unfiltered tes sequence
    cfd_high = 9      # the height measurement sequence. the 
                      # measurement is dependent of the height setting.
    pulse_timer = 10  # the time since the rising zero crossing of s at
                      # the start of a pulse
    rise_timer = 11   # the time since the timestamp was applied to a 
                      # rise.
    # in thesis there is an option 12 which is missing here


class Trigger(VhdlEnum):
    disabled = 0
    clock = 1
    pulse_t_pos = 2
    pulse_t_neg = 3
    slope_t_pos = 4
    f_0xing = 5
    s_0xing = 6
    s_0xing_pos = 7
    s_0xing_neg = 8
    cfd_high = 9
    cfd_low = 10
    max_slope = 11


class Qualifier(VhdlEnum):
    disabled = 0
    all = 1
    valid_rise = 2
    above_area = 3
    above = 4
    will_cross = 5
    armed = 6
    will_arm = 7
    rise = 8
    valid_peak1 = 9
    valid_peak2 = 10
    valid_peak3 = 11


header_dt = np.dtype(
    [
        ("size", np.uint16),
        ("last_bin", np.uint16),
        ("lowest_value", np.int32),
        ("reserved", np.uint16),
        ("most_frequent", np.uint16),
        ("flags", np.uint32),
        ("total", np.uint64),
        ("start_time", np.uint64),
        ("stop_time", np.uint64),
    ]
)


class Distribution:
    """
    wrapper for transmitted zmq frame representing a MCA distribution
    """

    def __init__(self, data, buffer=True):
        self.data = data
        self.buffer = buffer
        # these should just be views on the frame
        # not clear if the HDF5 version is copying
        # if buffer:
        self.header = np.frombuffer(self.data[:40], dtype=header_dt)[0]
        self.counts = np.frombuffer(self.data[40:], dtype=np.dtype(np.uint32))
        # else:
        #     self.header = self.data[:40].view(header_dt)[0]
        #     self.counts = self.data[40:].view(np.uint32)

    @property
    def most_frequent(self):
        return self.header["most_frequent"]

    # flags
    @property
    def channel(self):
        return np.bitwise_and(self.header["flags"], 0x00000007)

    @property
    def bin_width(self):
        return 2 ** np.right_shift(np.bitwise_and(self.header["flags"], 0x000000F8), 3)

    @property
    def trigger(self):
        return Trigger(
            np.right_shift(np.bitwise_and(self.header["flags"], 0x00000F00), 8)
        )

    @property
    def value(self):
        return Value(
            np.right_shift(np.bitwise_and(self.header["flags"], 0x0000F000), 12)
        )

    @property
    def qualifier(self):
        return Qualifier(
            np.right_shift(np.bitwise_and(self.header["flags"], 0x000F0000), 16)
        )

    @property
    def bins(self):
        return self.counts[1:-1]

    @property
    def underflow(self):
        return self.counts[0]

    @property
    def overflow(self):
        return self.counts[-1]

    @property
    def lowest_value(self):
        return self.header["lowest_value"]

    @property
    def highest_value(self):
        return self.lowest_value + self.bin_width * (self.last_bin - 1)

    @property
    def total(self):
        return self.header["total"]

    @property
    def start_time(self):
        return self.header["start_time"]

    @property
    def stop_time(self):
        return self.header["stop_time"]

    @property
    def last_bin(self):
        return self.header["last_bin"]

    def __repr__(self):
        return "Distribution: channel:{:} value:{:}, trigger:{:}, qualifier:{:}".format(
            self.channel, str(self.value), str(self.trigger), str(self.qualifier)
        )
