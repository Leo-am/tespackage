"""
Classes:

1) VhdlEnum
2) TraceType
3) Signal
4) Height
5) Timing
6) detection
7) Payload
8) lookup

Function: 
1) pulse_fmt

"""


from enum import Enum
import numpy as np
import logging

_logger = logging.getLogger(__name__)


class VhdlEnum(int, Enum):
    def __str__(self):
        return self.name.replace("_", " ")
        # return self.name

    def __repr__(self):
        return self.name

    def select(self):
        return pow(2, self.value)

    def latex(self):
        return self.name.replace("_", "\\_")


class TraceType(VhdlEnum):
    """
    Value of the trace_type event.trace_type register.
    """

    single = 0
    average = 1
    dot_product = 2
    dot_product_trace = 3


class Signal(VhdlEnum):
    """
    The signal recorded in a trace.
    """

    none = 0
    raw = 1
    f = 2
    s = 3


class Height(VhdlEnum):
    """
    Value of the event.height register.
    """

    peak = 0
    cfd_high = 1
    cfd_height = 2
    max_slope = 3


class Timing(VhdlEnum):
    """
    Value of the event.timing register.
    """

    pulse_threshold = 0
    slope_threshold = 1
    cfd_low = 2
    max_slope = 3


class Detection(VhdlEnum):
    """
    Value of the event.packet register.
    """

    rise = 0
    area = 1
    pulse = 2
    trace = 3


class Payload(VhdlEnum):
    """
    Payload type in capture files.
    """

    rise = 0
    area = 1
    pulse = 2
    single_trace = 3
    average_trace = 4
    dot_product = 5
    dot_product_trace = 6
    tick = 7
    mca = 8
    bad_frame = 9


def lookup(value, enum):
    if isinstance(value, str):
        return enum[value]
    else:
        return enum(value)


tick_fmt = [
    ("period", "u4"),
    ("eflags", "(2,)u1"),
    ("time", "u2"),
    ("timestamp", "u8"),
    ("overflow", "u1"),
    ("error", "u1"),
    ("cfd_error", "u1"),
    ("res2", "u1"),
    ("events_lost", "u4"),
]

header_fmt = [
    ("size", "u2"),
    ("tflags", "(2,)u1"),
    ("eflags", "(2,)u1"),
    ("time", "u2"),
]

pulse_header_fmt = [("area", "u4"), ("pulse_length", "u2"), ("pre_trigger", "u2")]

pulse_rise_fmt = [
    ("height", "i2"),
    ("minimum", "i2"),
    ("rise_time", "u2"),
    ("ptime", "u2"),
]

rise_fmt = [("height", "i2"), ("minimum", "i2"), ("eflags", "(2,)u1"), ("time", "u2")]


def rise2_fmt(n):
    s = "({},)".format(n)
    if n < 1 or n > 4:
        raise RuntimeError("{} out of range 0 to 3".format(n))
    return [(field[0], "({},){}".format(n, field[1])) for field in pulse_rise_fmt]


area_fmt = [("area", "u4"), ("eflags", "(2,)u1"), ("time", "u2")]

dot_product_fmt = [("dot_product", "u8")]

av_trace_fmt = header_fmt + [("multi_pulse", "u4"), ("multi_peak", "u4")]

sample_fmt = [("samples", "(2048,)i2")]


def pulse_fmt(n):
    if n < 1 or n > 4:
        raise RuntimeError("{} out of range 0 to 3".format(n))
    return header_fmt + pulse_header_fmt + [("rise", pulse_rise_fmt, (n,))]


tick_dt = np.dtype(tick_fmt)

# type bit 7 seq error, bit 6 header frame
fidx_dt = np.dtype(
    [
        ("payload", "u8"),
        ("length", "u4"),
        ("event_size", "u2"),
        ("changed", "u1"),
        ("type", "u1"),
    ],
    align=True,
)

tidx_dt = np.dtype([("start", "u4"), ("stop", "u4")], align=True)

payload_ref_dt = np.dtype([("start", "u8"), ("length", "u8")], align=True)

protocol_dt = np.dtype(
    [
        ("frame_seq", "u2"),
        ("protocol_seq", "u2"),
        ("event_size", "u2"),
        ("event_type", "(2,)u1"),
    ],
    align=True,
)
