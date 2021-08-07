"""
Classes:

1) PayloadType
"""


from enum import IntEnum


class PayloadType(IntEnum):
    peak = 0
    area = 1
    pulse = 2
    trace = 3
    tick = 4
    mca = 5

    def __str__(self):
        return self.name.replace("-", " ")
