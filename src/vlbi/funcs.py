#! /usr/bin/env python3
"""This file contains a collection of tools and functions that may be used in a variety
of the different programs in the repository.
"""

import sys
import argparse
import datetime
from matplotlib import pyplot as plt
from enum import IntEnum
from dataclasses import dataclass
from collections import defaultdict
from typing import Optional, Iterable, Generator
from rich import print as rprint
from rich import progress
from rich_argparse import RichHelpFormatter
import numpy as np
from astropy import coordinates as coord
from astropy import units as u
from casatools import msmetadata as msmd
from casatools import table as tb



class Stokes(IntEnum):
    """The Stokes types defined as in the enum class from casacore code.
    This is the definition for the data in Measurement Sets (MS), which is only hardwritten on them.
    """
    Undefined = 0 # Undefined value
    I = 1 # standard stokes parameters  # noqa - type: ignore
    Q = 2
    U = 3
    V = 4
    RR = 5 # circular correlation products
    RL = 6
    LR = 7
    LL = 8
    XX = 9 # linear correlation products
    XY = 10
    YX = 11
    YY = 12
    RX = 13 # mixed correlation products
    RY = 14
    LX = 15
    LY = 16
    XR = 17
    XL = 18
    YR = 19
    YL = 20
    PP = 21 # general quasi-orthogonal correlation products
    PQ = 22
    QP = 23
    QQ = 24
    RCircular = 25 # single dish polarization types
    LCircular = 26
    Linear = 27
    Ptotal = 28 # Polarized intensity ((Q^2+U^2+V^2)^(1/2))
    Plinear = 29 #  Linearly Polarized intensity ((Q^2+U^2)^(1/2))
    PFtotal = 30 # Polarization Fraction (Ptotal/I)
    PFlinear = 31 # linear Polarization Fraction (Plinear/I)
    Pangle = 32 # linear polarization angle (0.5  arctan(U/Q)) (in radians)



def chunkert(counter: int, max_length: int, increment: int) -> Generator[Iterable[int], None, None]:
    """Silly function to select a subset of an interval in
       [counter, counter + increment] : 0 < counter < max_length.

    Yields the tuple (counter, + interval_increment) :
                        interval_increment = min(increment, max_length - counter))
    """
    while counter < max_length:
        this_increment = min(increment, max_length - counter)
        yield (counter, this_increment)
        counter += this_increment



def mjd2date(mjd: float) -> datetime.datetime:
    """Returns the datetime for the given MJD date.
    """
    origin = datetime.datetime(1858, 11, 17)
    return origin + datetime.timedelta(mjd)






