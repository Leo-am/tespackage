"""
# registers.py created by Geoff Gillett
# modifications by Leonardo Assis
# yaml.load does not work anymore and needs to be fixed
# yaml.load upload to yaml.UnsafeLoad to properly function
# with recent yaml versions
# changes made: 24/07/2020

Functions: 

1) _adc_spi_transform
2) _channel_transform
3) save
4) load

Classes:

1) _RegisterMap
2) _RegisterGroup
3) _GroupIterator
4) Registers
"""




from tes2.base import VhdlEnum
from tes2.maps import (
    _Transport,  _map_str, _to_onehot, mca_map, adc_map, channel_map,
    baseline_map, event_map, global_map, cfd_map
)
import logging
import yaml

_logger = logging.getLogger(__name__)


# address transforms alter the RegInfo address and field so that that they
# address the # register field for required channel
def _adc_spi_transform(address, field, channel=0):
    # map spi to ADCs
    no_b_addresses = [0x20000051, 0x20000052]
    if ((address & 0xFF000000) >> 24) == 0x20:  # it's a SPI address
        # these address are not transformed
        spi_address = address & 0x00000FF
        if channel % 2 and address not in no_b_addresses:
            spi_address += 0x13
        mosi_mask = pow(2, channel // 2) << 8
        spi = 0x20000000 | mosi_mask | spi_address
        _logger.debug(
            (
                '_adc_spi_transform:chip address:0x{:02X} MOSI:0x{:02X} -> ' +
                'SPI address:0x{:08X}'
            ).format(spi_address, mosi_mask, spi)
        )
        taddress, tfield = spi, field
    else:
        taddress, tfield = address, (_to_onehot(channel), channel)

    _logger.debug(
        '_adc_spi_transform:channel={:} {:} -> {:}'
        .format(channel, _map_str(field, address), _map_str(tfield, taddress))
    )
    return taddress, tfield


def _channel_transform(address, field, channel=0):
    if address & 0x10000000:  # its an enable
        taddress, tfield = address, (_to_onehot(channel), channel)
    else:
        taddress, tfield = (address & 0xF0FFFFFF) | (channel << 24), field

    _logger.debug(
        '_channel_transform:channel={:} {:} -> {:}'
        .format(channel, _map_str(field, address), _map_str(tfield, taddress))
    )
    return taddress, tfield


# base classes
class _RegisterMap:
    """base class for a register map"""

    def __init__(self, name, reg_map, doc=None):
        # _logger.debug('_RegisterMap:init')
        for reg, info in reg_map.items():
            info.name = reg
        object.__setattr__(self, 'reg_map', reg_map)
        self.name = name
        self.__doc__ = doc
        _logger.debug('RM({}):init'.format(name))
        self._loading = False

    def __iter__(self):
        _logger.debug('RM:({}).__iter__'.format(self.name))
        for name, register in self.reg_map.items():
            if not register.strobe:
                yield name, register.get()

    def __getattr__(self, attr):
        _logger.debug('RM.__getattr__({})'.format(attr))
        if attr in self.reg_map:
            return self.reg_map[attr].get()
        else:
            raise AttributeError('No register named:{}'.format(attr))

    def __setattr__(self, attr, value):
        _logger.debug('RM.__setattr__:{} to {}'.format(attr, value))
        if attr in self.reg_map:
            # if self.reg_map[attr].loadable:
            self.reg_map[attr].set(value)
            # else:
            #     _logger.debug(
            #         '_RegisterMap.__setattr__:{} not loadable'.format(attr)
            #     )
        else:
            super().__setattr__(attr, value)

    def __get__(self, instance, owner):
        # _logger.debug('_RegisterMap.__get__()')
        return self

    def __set__(self, instance, values):
        _logger.debug('_RegisterMap.__set__()')
        if isinstance(values, dict): # dictionary assignment to map
            _logger.debug(
                '_RegisterMap.__set__({}) to {}'.format(self.name, values)
            )
            for reg, value in values.items():
                if reg in self.reg_map:
                    if self.reg_map[reg].loadable:
                        self.reg_map[reg].set(value)
                else:
                    raise AttributeError('No register named:{}'.format(reg))
        else:
            raise AttributeError('dict required, got {}'.format(type(values)))

    def help(self):
        print(self.__doc__)
        print()
        for reg in self.reg_map:
            print(reg)
            print(self.reg_map[reg].__doc__)

    def __repr__(self):
        return '{:}'.format(dict(self))


class _RegisterGroup(_RegisterMap):
    """base class for a register map with channel indexing"""

    def __init__(self, name, reg_map, channels, transform, doc=None):
        """

        :param channels:  number of channels
        :param transform:
        callable transform(address, field, channel) -> address, field
        transforms address and field to access the register for channel.
        """
        # _logger.debug('_RegisterGroup:init')
        self._size = channels
        super().__init__(name, reg_map, doc)
        self.transform = transform
        #  Used internally for indexing.
        # FIXME issue on start up if static value assigned at Register class
        # creation Does not match the true value fix by making size a property?
        self._indices = (*range(self._size),)

    # @property
    # def _size(self):
    #     return self._size
    #
    # @_size.setter
    # def _size(self, value):
    #     self.__size = value
    #     self._indices = (*range(value),)

    def __getattr__(self, attr):
        # _logger.debug('_RegisterGroup.__getattr__:{:}'.format(attr))
        if attr in self.reg_map:
            # _logger.debug('_RegisterGroup.__getattr__ REGISTER')
            value = self.reg_map[attr].get(
                transform=self.transform, indices=self._indices
            )
            # self._indices = (*range(self._size),)
            return value
        else:
            # self._indices = (*range(self._size),)
            raise AttributeError('No register named:{}'.format(attr))

    def __setattr__(self, attr, value):
        if attr in ['_size', '_indices', 'name']:
            _logger.debug('RG.__setattr__({}) BYPASS'.format(attr))
            object.__setattr__(self, attr, value)
            return
        rg_name = object.__getattribute__(self, 'name')
        _logger.debug(
            'RG:({}).__setattr__:{} to {}'.format(rg_name, attr, value)
        )
        if attr in self.reg_map:
            # _logger.debug('_RegisterGroup.__setattr__ REGISTER')
            # if self.reg_map[attr].loadable:
            if isinstance(value, dict):
                if self.reg_map[attr].loadable:
                    if rg_name in value:
                        self.reg_map[attr].set(
                            value[rg_name], transform=self.transform,
                            indices=self._indices
                        )
                        return
                else:
                    _logger.debug(
                        'RG:({}).__setattr__:{} not loadable'
                        .format(rg_name, attr)
                    )
                    return
            self.reg_map[attr].set(
                value, transform=self.transform, indices=self._indices
            )
        else:
            _logger.debug('RG.setattr -> object.setattr({})'.format(attr))
            object.__setattr__(self, attr, value)

    def __get__(self, instance, owner):
        # _logger.debug('_RegisterGroup.__get__()')
        self._indices = (*range(self._size),)
        return self

    def __set__(self, instance, values):
        _logger.debug('RG:{}.__set__ {}'.format(self.name, values))
        transform = self.transform
        indices = self._indices
        if isinstance(values, dict):  # dictionary assignment to entire group
            if self.name in values:  # group name is dict key
                group_values = values[self.name]
            else:
                group_values = values

            for reg, value in group_values.items():
                if reg in self.reg_map:
                    vals = [value[i] for i in indices]
                    if self.reg_map[reg].loadable:
                        self.reg_map[reg].set(vals, transform, indices)
                    else:
                        _logger.debug(
                            'RG:{} __set__ {} is not loadable'
                            .format(self.name, reg)
                        )
                else:
                    object.__setattr__(self, '_indices', (*range(self._size),))
                    raise AttributeError('No register named:{}'.format(reg))
            object.__setattr__(self, '_indices', (*range(self._size),))

    def _set_indices(self, index):
        if isinstance(index, tuple):
            indices = index
        elif isinstance(index, slice):
            indices = range(*index.indices(self._size))
        elif type(index) is int:
            if index < 0 or index >= self._size:
                raise IndexError
            indices = (index,)
        else:
            raise NotImplementedError(
                'Indexing with {} not implemented'.format(type(index))
            )
        _logger.debug(
            'RegisterGroup:{}._set_indices {} -> {}'
            .format(self.name, index, indices)
        )
        self._indices = indices

    # this is called for r.group[i].reg = value or r.group[i].reg
    def __getitem__(self, index):
        """return iterable containing register channels to read/write"""
        _logger.debug('RG:{}.__getitem__({})'.format(self.name, index))
        self._set_indices(index)
        return self

    # this is called for r.group[i] = value
    def __setitem__(self, index, value):
        _logger.debug('RG:{}.__setitem__({})'.format(self.name, index))
        self._set_indices(index)
        if isinstance(value, dict):  # dictionary assignment to entire group
            _logger.debug(
                'RG:{}.__setitem__({}) to {}'.format(self.name, index, value)
            )
            self.__set__(self, value)
        else:
            raise TypeError('can only assign a dict to a register group')

    def __iter__(self):
        for name, register in self.reg_map.items():
            if not register.strobe:
                yield name, register.get(self.transform, self._indices)

    def __repr__(self):
        return '{:}'.format(dict(self))


class _GroupIterator:
    def __init__(self, root):
        self.root = root

    def generator(self):
        yield 'root', self.root
        for attr_name in dir(self.root):
            attr = getattr(self.root, attr_name)
            if isinstance(attr, _RegisterMap):
                yield attr_name, attr

    def __iter__(self):
        return self.generator()


class Registers(_RegisterMap):
    """
    Client for reading and writing the internal FPGA control registers.

    Registers are arranged in functional groups and accessed through an
    instance of the Registers class. Let r be an instance of Registers
    then r.regname  references the regname register while r.groupname.regname
    the regname register of the  groupname group. Some groups support
    indexing to reference a register for a particular channel. Slicing and
    fancy indexing are supported while ommiting indexing is equvalent  to
    referencing ALL channels.

    For example:
    r.groupname[0].regname refers to the regname register of the groupname group
    for channel 0. While r.groupname.regname refers to the same register for all
    channels. Therefore, r.groupname.regname will return a list containing the
    value of the register for each channel, r.groupname.regname = value will set
    the for all channels to the same value and
    r.groupname.regname = [value0, value1, ..., valuen] will broadcast the list
    of values to the appropriate channel.

    Groups without indexing:
    No groupnme - accesses a general register. See help(Registers).
    mca - accesses the registers controlling the MCA. See help(McaRegisters).

    Groups supporting indexing:
    channel controls input to the processing channels. See help(ChannelRegisters
    event  controls event output. See help(EventRegisters).
    baseline  controls the baseline process. See help(BaselineRegisters).
    cfd  controls the constant fraction process. See help(CfdRegisters).
    adc  controls the ADC chips. See help(AdcRegisters).

    The the number of channels in the adc group is twice the value of the
    general register adc_chips while the number of channels in all other
    groups is the value of the general register channel_count.

    TODO add dict indexing

    """

    mca = _RegisterMap(
        'mca',
        mca_map,
        doc=""" Registers controlling the MCA

        """
    )

    adc = _RegisterGroup(
        'adc',
        adc_map,
        8,
        _adc_spi_transform,
        doc="""
        Registers controlling the ADC chips on the FMC108 digitiser
        card.

        """
    )

    channel = _RegisterGroup(
        'channel',
        channel_map,
        2,
        _channel_transform,
        doc="""
        Registers controlling channel input.

        """
    )

    baseline = _RegisterGroup(
        'baseline',
        baseline_map,
        2,
        _channel_transform,
        doc="""
        Registers controlling baseline correction.

        """
    )
    cfd = _RegisterGroup(
        'cfd',
        cfd_map,
        2,
        _channel_transform,
        doc="""
        Registers controlling the cfd process.

        """
    )
    event = _RegisterGroup(
        'event',
        event_map,
        2,
        _channel_transform,
        doc="""
        Registers controlling event output.

        """
    )

    def __init__(self, port=None):
        super().__init__('root', global_map, self.__doc__)
        self._transport = _Transport()
        self._transport.open(port)

        self.dsp_channels = global_map['channel_count'].get()
        self.adc_channels = global_map['adc_chips'].get() * 2

        self.version = (
            ('HDL:{}\nMain CPU:{}\nChannel CPU:{}\n' +
             'ADC channels:{} Processing channels:{}')
            .format(
                self.hdl_version, self.cpu_version, self.channel.cpu_version,
                self.adc_channels, self.dsp_channels
            )
        )
        _logger.info(self.version)

        self.channel._size = self.dsp_channels
        self.baseline._size = self.dsp_channels
        self.event._size = self.dsp_channels
        self.adc._size = self.adc_channels
        object.__setattr__(self, 'all', self._Iterator(self))

    def __setattr__(self, key, value):
        if key == 'all':
            _logger.debug('Registers.__setattr__ all')
            if not isinstance(value, dict):
                raise AttributeError(
                    'dict required, not {}'.format(type(value))
                )
            for group in value:
                _logger.debug('Registers.__setattr__ load:{}'.format(group))
                if group == 'root':
                    for reg, val in value[group].items():
                        object.__setattr__(self, reg, val)
                else:
                    object.__setattr__(self, group, value[group])
            return
        elif key == 'global':
            _logger.debug('setattr global')
            return
        _logger.debug(
            'Registers.__setattr__({}) -> super().__setattr__'.format(key)
        )
        super().__setattr__(key, value)

    def _group_generator(self):
        yield self.name, self
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, _RegisterMap):
                yield attr_name, attr

    def _all(self):
        for reg, value in self._group_generator():
            yield reg, dict(value)

    # def __iter__(self):
    #     _logger.debug('Registers.__iter__')
    #     return self.all

    # FIXME why not make
    class _Iterator:
        def __init__(self, outer):
            self.outer = outer

        def __iter__(self):
            return self.outer._all()


def save(d, filename):
    # convert enums to str
    for group in d:
        # print(group)
        for reg in d[group]:
            # print('  '+reg)
            values = d[group][reg]
            try:
                l = len(values)
            except TypeError:
                l = 1
            if l > 1:
                if isinstance(values[0], VhdlEnum):
                    d[group][reg] = [str(v) for v in values]
            else:
                if isinstance(values, VhdlEnum):
                    d[group][reg] = str(values)

    with open(filename, 'w') as f:
        f.write(yaml.dump(d))


def load(filename):
    with open(filename, 'r') as f:
        return yaml.load(f, Loader=yaml.UnsafeLoader)
