"""
Classes: 

1) RegisterError
2) Borg
3) Transport_base
4) _ZmqTransport
5) _DirectSerial
6) RegInfo

Functions:

1) _map_str
2) def _field_str
3) _cpu_version
4) _from_onehot
5) _to_onehot
6) _to_cf
7) _from_cf
8) _to_gain(g):
9) _from_gain

"""


import numpy as np
import logging
import zmq
import serial
import serial.tools.list_ports
from collections import OrderedDict
from functools import partial
from tes.base import lookup, Detection, Timing, Height, Signal, TraceType
from tes import mca

# __all__ = ['RegInfo', 'mca_map', ]

_logger = logging.getLogger(__name__)


class RegisterError(AttributeError):
    def __init__(self, non_hex=False, bad_length=False, axi="OKAY"):
        self.non_hex = non_hex
        self.bad_length = bad_length
        self.axi = axi

    def __str__(self):
        return (
            "register access error - non_hex:{} bad_length:{} AXI:{}\n".format(
                self.non_hex, self.bad_length, self.axi
            )
            + "AXI:DECERR indicates a non-existent address\n"
            + "AXI:SLVERR typically indicates an access violation "
            + "eg. writing a read-only address"
        )


# Borg singleton pattern for transports
class _Borg:
    _shared_state = {}

    def __init__(self):
        self.__dict__ = self._shared_state


# make this an ABC
class _TransportBase(_Borg):
    """this will need to be a singleton"""

    def __init__(self):
        super().__init__()

    def open(self, port):
        _logger.debug("opening dummy transport port={}".format(port))

    def close(self):
        pass

    def read(self, address):
        if address == 0x10800000:
            return 0x00000808
        _logger.debug("_dummy_read:address:{:08X}".format(address))
        return 0

    def write(self, value, address):
        _logger.debug("_dummy_write:{:08X} to {:08X}".format(value, address))

    def _readline(self):
        return []

    def _get_response(self):

        l = self._readline()
        if len(l) == 0:
            raise serial.SerialTimeoutException
        _logger.debug("_get_response:{:}".format(l[:-1]))
        resp = int(chr(l[-3]), 16)
        if resp != 0:
            axi_bits = resp & 3
            err = (resp & 0xC) >> 2
            non_hex = (err & 2) != 0
            bad_length = (err & 1) != 0
            if axi_bits == 3:
                axi = "DECERR"
            elif axi_bits == 2:
                axi = "SLVERR"
            elif axi_bits == 0:
                axi = "OKAY"
            else:
                axi = "UNKNOWN"
            raise RegisterError(non_hex, bad_length, axi)
        else:
            if len(l[:-3]) == 0:
                return
            data = np.frombuffer(
                bytearray.fromhex(l[:-3].decode("utf8")), np.uint32
            ).byteswap()[0]
            return data


class _ZmqTransport(_TransportBase):
    def __init__(self):
        super().__init__()
        self.context = None
        self.socket = None

    def open(self, port):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        _logger.info("connecting to server at {:}".format(port))
        self.socket.connect(port)

    def read(self, address):
        _logger.debug("_zmq_read:address:{:08X}".format(address))
        self.socket.send(
            b"00000000" + bytes("{:08X}02\n".format(address), encoding="utf8")
        )
        return self._get_response()

    def write(self, data, address):
        _logger.debug("_zmq_write:{:08X} to {:08X}".format(data, address))
        self.socket.send(
            bytes("{:08X}{:08X}01\n".format(data, address), encoding="utf8")
        )
        return self._get_response()

    def _readline(self):
        return self.socket.recv()


class _DirectSerial(_TransportBase):
    def __init__(self, port):
        super().__init__()
        self.serial = None

    def open(self, port):
        _logger.info("opening serial port on {:}".format(port))
        self.serial = serial.serial_for_url(port, baudrate=115200, timeout=2)

    def close(self):
        self.serial.close()

    def read(self, address):
        self.serial.write(
            b"00000000" + bytes("{:08X}02\n".format(address), encoding="utf8")
        )
        _logger.debug("reading address:{:08X}".format(address))
        return self._get_response()

    def write(self, data, address):
        self.serial.write(
            bytes("{:08X}{:08X}01\n".format(data, address), encoding="utf8")
        )
        _logger.debug("writing {:08X} to {:08X}".format(data, address))
        return self._get_response()


def _map_str(field, address):
    return "0x{:08X} {:}".format(address, _field_str(field))


def _field_str(field):
    return "(0x{:08X},{:})".format(*field) if field else "()"


"""set the transport for Registers"""
_Transport = _ZmqTransport


class RegInfo:
    """Class describing A FPGA register""

    """

    _transport = _Transport()

    def __init__(
        self,
        address,
        field,
        strobe,
        output_transform,
        input_transform,
        loadable,
        name=None,
        doc=None,
    ):
        """

        :param address: 32 bit FPGA register address
        :param field: tuple (32 bit mask, shift)
        :param strobe: boolean indicating a strobe value.
        :param output_transform: called on the register value after
               reading
        :param input_transform: called on value before writing
        :param loadable this value can be loaded from a dict
        :param doc: docstring for this register
        """
        # _logger.debug('RegInfo.__init__')
        self.address = address
        self.field = field  # (mask, shift)
        self.strobe = strobe
        self.output_transform = output_transform
        self.input_transform = input_transform
        self.loadable = loadable
        self.name = name
        self.__doc__ = doc
        # super().__init__()
        # self.read = self._transport.read
        # self.write = self._transport.write

    def get(self, transform=None, indices=None):
        _logger.debug("RegInfo.get:indices {}".format(indices))
        if transform:
            values = []
            for c in indices:
                address, field = transform(self.address, self.field, channel=c)
                values.append(self._get_reg(address, field))
            if len(values) == 1:
                return values[0]
            return values
        else:
            return self._get_reg(self.address, self.field)

    def _get_reg(self, address, field):
        # strobe must not have an empty field tuple
        if self.strobe:
            if not field:
                raise AttributeError("_get_reg:strobe with empty field, bad reg_map")
            self._transport.write(field[0], address)
            return

        data = self._transport.read(address)
        _logger.debug("_get_reg:Reading 0x{:08X} returned 0x{:X}".format(address, data))

        if len(field):
            # data = np.right_shift(np.bitwise_and(data, field[0]), field[1])
            data = (data & field[0]) >> field[1]
            _logger.debug(
                "_get_reg:extract field{:} -> 0x{:X}".format(_field_str(field), data)
            )
        if self.output_transform is not None:
            tdata = self.output_transform(data)
            _logger.debug("_get_reg:transform output 0x{:X} -> {}".format(data, tdata))
            return tdata

        return int(data)

    def set(self, values, transform=None, indices=None):
        _logger.debug("RegInfo.set({})[{}] to {}".format(self.name, indices, values))

        if isinstance(values, dict):
            if not self.loadable:
                _logger.debug("RegInfo.set({}):not loadable".format(self.name))
                return

        if transform:
            if isinstance(values, dict):
                if self.name in values:
                    reg_values = values[self.name]
                else:
                    raise AttributeError(
                        "can't interpret {} as {} values".format(values, self.name)
                    )
            elif isinstance(values, str):  # name of an enum
                reg_values = (values,)
            else:
                reg_values = values

            try:
                len_values = len(reg_values)
            except TypeError:
                len_values = 1
                reg_values = (values,)

            if len_values > len(indices):
                reg_values = [reg_values[i] for i in indices]
            elif len_values != len(indices) and len_values != 1:
                raise IndexError(
                    "Cannot broadcast {:} values to {:} indices".format(
                        len_values, len(indices)
                    )
                )

            _logger.debug(
                "RegInfo.set({})[{}] to {}".format(self.name, indices, reg_values)
            )

            for i in range(len(indices)):
                address, field = transform(self.address, self.field, channel=indices[i])
                if len_values == 1:
                    self._set_reg(reg_values[0], address, field)
                else:
                    self._set_reg(reg_values[i], address, field)
        else:
            self._set_reg(values, self.address, self.field)

    def _set_reg(self, value, address, field):
        # strobe must have a non empty bit_field tuple
        _logger.debug("_set_reg:{} to {}".format(_map_str(field, address), value))
        if self.input_transform:
            tvalue = self.input_transform(value)
            _logger.debug(
                "_set_reg:input_transform {} -> 0x{:08X}".format(value, tvalue)
            )
            value = tvalue

        if self.strobe:
            if not field:
                raise AttributeError("strobe with empty field, bad reg_map")
            self._transport.write(field[0], address)
            return

        if field:
            old_value = self._transport.read(address)

            new_value = (old_value & ~field[0]) | ((int(value) << field[1]) & field[0])
            _logger.debug(
                "_set_reg:0x{:08X} from 0x{:08X} to 0x{:08X}".format(
                    address, old_value, new_value
                )
            )
        else:
            new_value = int(value)
            _logger.debug("_set_reg:0x{:X} to 0x{:08X}".format(address, new_value))

        written = self._transport.write(new_value, address)

        if written is not None:
            _logger.debug(
                "_set_reg:response from write to 0x{:08X} is 0x{:08X}".format(
                    address, written
                )
            )
        else:
            _logger.debug("_set_reg:no response writing to 0x{:08X}".format(address))


# IO transforms
def _cpu_version(data):
    y = 2016 + ((data & 0xF0000000) >> 30)
    m = (data & 0x0F000000) >> 24
    d = (data & 0x00FF0000) >> 16
    h = (data & 0x0000FF00) >> 8
    mi = data & 0x000000FF
    return "{:04d}-{:02d}-{:02d} {:02d}:{:02d}".format(y, m, d, h, mi)


def _from_onehot(value):
    if value == 0:
        return 0

    b = np.log2(value)
    # print(value, b, np.modf(b))

    if np.modf(b)[0] != 0:
        raise ArithmeticError("{:} is not one-hot".format(b))
    return int(b)


def _to_onehot(value, bits=8):
    v = int(value)
    if v > bits - 1:
        raise AttributeError("value must be <= {:}".format(bits - 1))
    return 2 ** v


def _to_cf(cf):
    if (cf < 0) or (cf > 1):
        raise AttributeError("Constant fraction must be between 0 and 0.5")
    cfi = int(cf * 2 ** 17)
    if abs(cf - (cfi / 2 ** 17)) > abs(cf - ((cfi + 1) / 2 ** 17)):
        return cfi + 1
    return cfi


def _from_cf(value):
    return value / 2 ** 17


def _to_gain(g):
    if (g < 0) or (g > 6):
        raise AttributeError("Gain must be between 0 and 6.5")
    gi = int(g * 2)
    if abs(g - (gi / 2)) > abs(g - ((gi + 1) / 2)):
        return gi + 1
    return gi


def _from_gain(value):
    return float(value / 2)


event_lookup = partial(lookup, enum=Detection)
height_lookup = partial(lookup, enum=Height)
timing_lookup = partial(lookup, enum=Timing)
value_lookup = partial(lookup, enum=mca.Value)
trigger_lookup = partial(lookup, enum=mca.Trigger)
qualifier_lookup = partial(lookup, enum=mca.Qualifier)
signal_lookup = partial(lookup, enum=Signal)
trace_type_lookup = partial(lookup, enum=TraceType)

global_map = OrderedDict(
    [
        (
            "hdl_version",
            RegInfo(
                0x10000001,
                (),
                False,
                lambda x: "sha-1 {:07X}".format(x),
                None,
                False,
                doc="""
        Read only, unsigned:24 bit
        The short SHA-1 for the commit of the HDL code in the git repository.

        type:str
        """,
            ),
        ),
        (
            "cpu_version",
            RegInfo(
                0x10000000,
                (),
                False,
                _cpu_version,
                None,
                False,
                doc="""
        Read only, unsigned:32 bit
        Date and time central cpu code was assembled.

        The format is YMDDHHmm where each character represents a 4 bit nibble.
        Y: Year mod 16
        M: Month
        DD: Day
        HH: Zulu time hour
        mm: Minute

        type:str "day-month-year hour:minute"
        """,
            ),
        ),
        (
            "channel_count",
            RegInfo(
                0x10800000,
                (0x000000FF, 0),
                False,
                None,
                None,
                False,
                doc="""
        Read only, unsigned:8 bit
        Number of processing channels in the instantiated in the FPGA.

        type:int
        """,
            ),
        ),
        (
            "adc_chips",
            RegInfo(
                0x10800000,
                (0x0000FF00, 8),
                False,
                None,
                None,
                False,
                doc="""
        Read only unsigned:8 bit
        Number of dual channel ADC chips on the FMC card attached to the FPGA.

        type:int
        """,
            ),
        ),
        (
            "adc_enable",
            RegInfo(
                0x10000080,
                (0x000000FF, 0),
                False,
                None,
                None,
                True,
                doc="""
        unsigned:8 bit
        Enables the ADC channels corresponding to the set bits,
        disabled channels are in low power mode.
        See the adc.enable register to enable on a per channel basis.

        type:int
        """,
            ),
        ),
        (
            "event_enable",
            RegInfo(
                0x10000100,
                (0x000000FF, 0),
                False,
                None,
                None,
                True,
                doc="""
        unsigned:8 bit
        Enables event transmission for the channels corresponding to the
        set bits.

        See the event.enable register to enable on a per channel basis.

        type:int
        """,
            ),
        ),
        (
            "tick_period",
            RegInfo(
                0x10000020,
                (),
                False,
                None,
                None,
                True,
                doc="""
        unsigned:32 bit
        The time (4ns clock pulses) between tick events.

        type:int
        """,
            ),
        ),
        (
            "tick_latency",
            RegInfo(
                0x10000040,
                (),
                False,
                None,
                None,
                True,
                doc="""
        unsigned:32 bit
        Maximum time (4ns clock pulses) to wait after a tick event before
        flushing the buffer until another tick event is found.

        type:int
        """,
            ),
        ),
        (
            "window",
            RegInfo(
                0x10000400,
                (),
                False,
                None,
                None,
                True,
                doc="""
        1 bit
        Time window (4ns clock pulses) for determining the new_window bit in
        event_flags.

        type:bool
        """,
            ),
        ),
        (
            "mtu",
            RegInfo(
                0x10000010,
                (),
                False,
                lambda x: int(x * 8),
                lambda x: x // 8,
                True,
                doc="""
         unsigned 16 bit: (maximum number of 8 byte words)
        Maximum size of transmitted Ethernet frames. The FPGA register holds the
        MTU size in 8 byte words, the value returned will always be a multiple 
        of 8 and the value sent to the register is floor(mtu/8)

        type:int
        """,
            ),
        ),
        (
            "ad9510_status",
            RegInfo(
                0x10800000,
                (0x00040000, 18),
                False,
                bool,
                None,
                False,
                doc="""
        1 bit
        The status pin on the AD9510 clock generator chip on the FMC digitiser
        card (see the AD9510 data sheet).

        type:bool
        """,
            ),
        ),
        (
            "vco_power_en",
            RegInfo(
                0x10000200,
                (0x00000002, 1),
                False,
                bool,
                None,
                False,
                doc="""
        1 bit
        AD9510 clock generator chip VCO power enable pin. (see data sheet).

        type:bool
        """,
            ),
        ),
        (
            "fmc",
            RegInfo(
                0x10800000,
                (0x00010000, 16),
                False,
                bool,
                None,
                False,
                doc="""
        Read only, 1 bit
        A FMC digitiser card is connected to the FPGA.

        type:bool
        """,
            ),
        ),
        (
            "fmc_power",
            RegInfo(
                0x10800000,
                (0x00020000, 17),
                False,
                bool,
                None,
                False,
                doc="""
        Read only, 1 bit
        The FMC card is present and powered up.

        type:bool
        """,
            ),
        ),
        (
            "fmc_internal_clock",
            RegInfo(
                0x10000200,
                (0x00000001, 0),
                False,
                bool,
                None,
                False,
                doc="""
        1 bit
        The state of internal clock pin on the FMC108. (see data sheet)

        type:bool
        """,
            ),
        ),
        (
            "mmcm_locked",
            RegInfo(
                0x10800000,
                (0x00080000, 19),
                False,
                bool,
                None,
                False,
                doc="""
        read only, 1 bit
        The FPGA clock management tile is locked to FMC clock from the AD9510.

        type:bool
        """,
            ),
        ),
        (
            "iodelay_ready",
            RegInfo(
                0x10800000,
                (0x00100000, 20),
                False,
                bool,
                None,
                False,
                doc="""
        Read only, 1 bit
        The FPGA iodelay controller for inputs from ADCs is initialised.

        type:bool
        """,
            ),
        ),
    ]
)

mca_map = OrderedDict(
    [
        (
            "lowest_value",
            RegInfo(
                0x10000004,
                (),
                False,
                lambda x: int(np.int32(x)),
                lambda x: int(np.uint32(x)),
                True,
                doc="""
        signed:32 bit
        Values < lowest value are placed in the underflow bin.

        type:int
        """,
            ),
        ),
        (
            "ticks",
            RegInfo(
                0x10000008,
                (),
                False,
                None,
                None,
                True,
                doc="""
        unsigned:32 bit
        The number of tick_periods to accumulate statistics over.

        type:int
        """,
            ),
        ),
        (
            "value",
            RegInfo(
                0x10000002,
                (0x0000000F, 0),
                False,
                mca.Value,
                value_lookup,
                True,
                doc="""
        unsigned:4 bit
        The value to collect statistics for.

        type:tes.mca.Value, enum name or value can be used as input.
        """,
            ),
        ),
        (
            "trigger",
            RegInfo(
                0x10000002,
                (0x000000F0, 4),
                False,
                mca.Trigger,
                trigger_lookup,
                True,
                doc="""
        unsigned:4 bit
        Values are only included when the selected trigger is True.

        type:tes.mca.Trigger, enum name or value can be used as input.
        """,
            ),
        ),
        (
            "channel",
            RegInfo(
                0x10000002,
                (0x00000700, 8),
                False,
                None,
                None,
                True,
                doc="""
        unsigned:3 bit
        The channel to capture statistics from.

        type:int
        """,
            ),
        ),
        (
            "bin_n",
            RegInfo(
                0x10000002,
                (0x0000F800, 11),
                False,
                None,
                None,
                True,
                doc="""
        unsigned:5 bit
        The width of histogram bins is 2**bin_n.

        type:int
        """,
            ),
        ),
        (
            "last_bin",
            RegInfo(
                0x10000002,
                (0x3FFF0000, 16),
                False,
                None,
                None,
                True,
                doc="""
        unsigned:14 bit
        The bin used for overflows and the last bin in the histogram.

        type:int
        """,
            ),
        ),
        (
            "update",
            RegInfo(
                0x10002000,
                (0x00000002, 0),
                True,
                None,
                None,
                False,
                doc="""
        strobe
        Update the MCA settings on next tick if the buffer is free.

        type:bool
        """,
            ),
        ),
        (
            "update_on_completion",
            RegInfo(
                0x10002000,
                (0x00000001, 0),
                True,
                None,
                None,
                False,
                doc="""
        strobe
        Update the MCA after the current buffer has completed its ticks and
        the next buffer is free.

        type:bool
        """,
            ),
        ),
        (
            "qualifier",
            RegInfo(
                0x10000800,
                (0x0000000F, 0),
                False,
                mca.Qualifier,
                qualifier_lookup,
                True,
                doc="""
        unsigned:4 bit
        Qualifier for trigger, this must also be true for the value to be
        included in the histogram.

        type:tes.mca.Qualifier, enum name or value can be used as input.
        """,
            ),
        ),
    ]
)

adc_map = OrderedDict(
    [
        (
            "enable",
            RegInfo(
                0x10000080,
                (),
                False,
                bool,
                None,
                True,
                doc="""
        1 bit
        Sets or clears the bit corresponding to this channel in the adc_enable
        register. When False the ADC is put in a low-power mode.

        type:bool
        """,
            ),
        ),
        (
            "pattern",
            RegInfo(
                0x20000062,
                (0x00000007, 0),
                False,
                None,
                None,
                False,
                doc="""
        unsigned:3 bit
        Set the type of test pattern for the corresponding ADC chip.
        See the ADS62P49 data sheet.

        type:np.uint8
        """,
            ),
        ),
        (
            "pattern_low",
            RegInfo(
                0x20000051,
                (0x000000FF, 0),
                False,
                None,
                None,
                False,
                doc="""
        unsigned:8 bit
        The low byte of the of the custom test pattern.
        See the ADS62P49 data sheet.

        type:int
        """,
            ),
        ),
        (
            "pattern_high",
            RegInfo(
                0x20000052,
                (0x0000003F, 0),
                False,
                None,
                None,
                False,
                doc="""
        unsigned:6 bit
        The upper 6 bits of the custom test pattern.
        See the ADS62P49 data sheet.

        type:int
        """,
            ),
        ),
        (
            "gain",
            RegInfo(
                0x20000055,
                (0x000000F0, 4),
                False,
                _from_gain,
                _to_gain,
                True,
                doc="""
        unsigned:4 bit
        The gain of the ADC input stage, 0 to 6 dB in 0.5 dB steps.
        See the ADS62P49 data sheet.

        type:float
        """,
            ),
        ),
    ]
)

channel_map = OrderedDict(
    [
        (
            "cpu_version",
            RegInfo(
                0x00000000,
                (),
                False,
                _cpu_version,
                None,
                False,
                doc="""
        Read only, unsigned:32 bit
        Date and time the channel cpu code was assembled.

        Same format as the main_cpu register.

        type:str 'day-month-year hour:minute'
        """,
            ),
        ),
        (
            "adc_select",
            RegInfo(
                0x00000800,
                (0x0000000F, 0),
                False,
                _from_onehot,
                _to_onehot,
                True,
                doc="""
        unsigned:4 bit
        Select the ADC output that this channel processes.

        type:int
        """,
            ),
        ),
        (
            "invert",
            RegInfo(
                0x00000800,
                (0x00000100, 8),
                False,
                bool,
                None,
                True,
                doc="""
        1 bit
        Multiply the selected ADC output by -1 before processing.

        type:bool
        """,
            ),
        ),
        (
            "delay",
            RegInfo(
                0x00000020,
                (0x000003FF, 0),
                False,
                None,
                None,
                True,
                doc="""
        unsigned:10 bit
        The time (4ns clock pulses) the selected ADC output is delayed before
        processing.

        type:int
        """,
            ),
        ),
    ]
)

baseline_map = OrderedDict(
    [
        (
            "offset",
            RegInfo(
                0x00000040,
                (0xFFFFFFFF, 0),
                False,
                lambda x: int(np.uint32(x)),
                None,
                True,
                doc="""
        signed:16 bit
        This values is subtracted from the ADC signal to correct DC offset.

        type:np.int16
        """,
            ),
        ),
        (
            "time_constant",
            RegInfo(
                0x00000080,
                (),
                False,
                None,
                None,
                True,
                doc="""
        unsigned:32 bit
        The time (4ns clock pulses) between baseline MCA buffer swaps.

        type:int
        """,
            ),
        ),
        (
            "threshold",
            RegInfo(
                0x00000100,
                (),
                False,
                np.int32,
                None,
                True,
                doc="""

        ADC values, corrected by the offset register, that are > threshold are
        not included in the baseline estimate.

        type:int
        """,
            ),
        ),
        (
            "count_threshold",
            RegInfo(
                0x00000200,
                (),
                False,
                None,
                None,
                True,
                doc="""
        unsigned:32 bit
        The most frequent bin of the baseline MCA histogram in is only included
        in the baseline estimate when its frequency is greater than
        count_threshold.

        type:int
        """,
            ),
        ),
        (
            "new_only",
            RegInfo(
                0x00000400,
                (0x00000001, 0),
                False,
                bool,
                None,
                True,
                doc="""
        1 bit
        When true the baseline MCAs most frequent bin is included in the
        baseline estimate only when it changes, otherwise it is also included
        when it's frequency changes.

        type:bool
        """,
            ),
        ),
        (
            "dynamic",
            RegInfo(
                0x00000400,
                (0x00000002, 1),
                False,
                bool,
                None,
                True,
                doc="""
        1 bit
        When true the baseline estimate is subtracted from the offset corrected
        ADC signal. Ths effectively turns the baseline correction on and off.

        type:bool
        """,
            ),
        ),
    ]
)

cfd_map = OrderedDict(
    [
        (
            "fraction",
            RegInfo(
                0x00000008,
                (),
                False,
                _from_cf,
                _to_cf,
                True,
                doc="""
        unsigned:0.17 bit
        The constant fraction.

        type:float
        """,
            ),
        ),
        (
            "rel2min",
            RegInfo(
                0x00000008,
                (0x80000000, 31),
                False,
                bool,
                None,
                True,
                doc="""
        1 bit
        When True, the constant fraction thresholds are calculated from
        the difference between the maxima of a rise and the preceding minima. 
        Otherwise CFD threshold are calculated from the value of the maxima.

        type:bool
        """,
            ),
        ),
    ]
)

event_map = OrderedDict(
    [
        (
            "enable",
            RegInfo(
                0x10000100,
                (),
                False,
                bool,
                None,
                False,
                doc="""
        1 bit
        Sets/clears the corresponding bit in the global.event_enable register
        and enables/disables event packet transmission from this channel.

        type:bool
        """,
            ),
        ),
        (
            "packet",
            RegInfo(
                0x00000001,
                (0x00000003, 0),
                False,
                Detection,
                event_lookup,
                True,
                doc="""
        unsigned:2 bit
        The type of event packet generated by this channel.

        type:tes.base.Event, input can be enum name or value.
        """,
            ),
        ),
        (
            "timing",
            RegInfo(
                0x00000001,
                (0x0000000C, 2),
                False,
                Timing,
                timing_lookup,
                True,
                doc="""
        unsigned:2 bit
        The point at which a rise is timestamped.

        type:tes.base.timing, input can be enum name or value.
        """,
            ),
        ),
        (
            "max_rises",
            RegInfo(
                0x00000001,
                (0x000000F0, 4),
                False,
                None,
                None,
                True,
                doc="""
        unsigned:4 bit
        The maximum number of rises that a pulse event can record is
        max_rises+1. This sets the length of pulse events.

        type:int
        """,
            ),
        ),
        (
            "height",
            RegInfo(
                0x00000001,
                (0x00000300, 8),
                False,
                Height,
                height_lookup,
                True,
                doc="""
        unsigned:2 bit
        The value placed in the height field of events.

        type:tes.base.Height, input can be enum name or value.
        """,
            ),
        ),
        (
            "trace",
            RegInfo(
                0x00000001,
                (0x00003000, 12),
                False,
                TraceType,
                trace_type_lookup,
                True,
                doc="""
        unsigned:2 bit
        The type of trace to record, single average or dot_product.

        type:tes.base.TraceType, input can be enum name or value.
        """,
            ),
        ),
        (
            "trace_sequence",
            RegInfo(
                0x00000001,
                (0x00000C00, 10),
                False,
                Signal,
                signal_lookup,
                True,
                doc="""
        unsigned:2 bit
        The signal to trace.

        type:tes.base.Signal, input can be enum name or value.
        """,
            ),
        ),
        (
            "trace_pre",
            RegInfo(
                0x00000400,
                (0xFFFF0000, 16),
                False,
                None,
                None,
                True,
                doc="""
        unsigned:10 bit
        Time (4ns clocks) to record before the timing point. Note, the 
        trace_stride setting effects the number of samples before the timing 
        point and whether the timing point is represented by a sample.

        type:int, 
        """,
            ),
        ),
        (
            "trace_stride",
            RegInfo(
                0x00000001,
                (0x0003C000, 14),
                False,
                None,
                None,
                True,
                doc="""
        unsigned:4 bit
        The trace signal is sampled every trace_stride clocks.

        type:int, 
        """,
            ),
        ),
        (
            "trace_length",
            RegInfo(
                0x00000001,
                (0x0FFC0000, 18),
                False,
                lambda x: int(x * 4),
                lambda x: x // 4,
                True,
                doc="""
        unsigned:10 bit (number of 4 sample chunks)
        The number samples to record for the trace. 
        
        Note: The FPGA register holds the number of 4 sample chunks to 
        record, so the value returned will always be a multiple of 4, and the 
        value sent to the register is floor(trace_length/4).

        type:int, 
        """,
            ),
        ),
        (
            "pulse_threshold",
            RegInfo(
                0x00000002,
                (),
                False,
                None,
                None,
                True,
                doc="""
        unsigned:16 bit
        For a rise to be produce an event its maxima must be
        greater than or equal to the pulse_threshold register.

        type:int
        """,
            ),
        ),
        (
            "slope_threshold",
            RegInfo(
                0x00000004,
                (),
                False,
                None,
                None,
                True,
                doc="""
        unsigned:16 bit
        When slope makes a positive going crossing of slope_threshold
        the rise detector is armed and remains armed until the slope
        makes a negative going zero crossing. For a rise to produce an
        event the rise detector must be armed when the rise reaches its maxima.

        type:int
        """,
            ),
        ),
        (
            "area_threshold",
            RegInfo(
                0x00000010,
                (),
                False,
                None,
                None,
                True,
                doc="""
        unsigned:32 bit
        The area of a pulse must be greater than or equal to
        area_threshold to produce an area or pulse event packet.

        type:int
        """,
            ),
        ),
    ]
)
