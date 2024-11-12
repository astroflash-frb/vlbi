#!/usr/bin/env python3
"""Provides an overview of the information contained in a MS file.

Usage: msoverview.py [-h] [-v] msdata

Inputs:
    msdata : str     Path to the MS file.

Optional:
    -h  Display the help
    -i  Ignore the MS check to spot which antennas observed (slow).
    -v  Display the version of the program.


Version: 2.2.1
Author: Benito Marcote (marcote@jive.eu)


version 2.2.1 changes (Jul 2023)
- chunkert chunk value changed to 100 (optimal IO speed)
version 2.2 (Nov 2022)
"""
import os
import numpy as np
import pickle
import json
import datetime as dt
import argparse
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
from pyrap import tables as pt
from astropy import units as u
from astropy import coordinates as coord
from rich import progress
import blessed



__version__ = '2.2.2'

def chunkert(f, l, cs):
    while f<l:
        n = min(cs, l-f)
        yield (f, n)
        f = f + n

percent = lambda x, y: (float(x)/float(y))*100.0


class Source(object):
    """Defines a source by name, type (i.e. target, reference, fringefinder, other)
    and if it must be protected or not (password required to get its data).
    """
    @property
    def name(self):
        return self._name

    @property
    def coordinates(self):
        return self._coordinates

    def __init__(self, name: str, coordinates: coord.SkyCoord):
        assert isinstance(name, str), f"The name of the source must be a string (currrently {name})"
        assert isinstance(coordinates, coord.SkyCoord), \
               f"The coordinates of the source must be an astropy.coordinates.SkyCoord (currrently {coordinates})"
        self._name = name
        self._coordinates = coordinates

    def __iter__(self):
        for key in ('name', 'coordinates'):
            yield key, getattr(self, key)

    def json(self):
        """Returns a dict with all attributes of the object.
        I define this method to use instead of .__dict__ as the later only reporst
        the internal variables (e.g. _username instead of username) and I want a better
        human-readable output.
        """
        d = dict()
        for key, val in self.__iter__():
            d[key] = val

        return d


@dataclass
class Antenna:
    name: str
    observed: bool = False
    subbands: tuple = ()


class Antennas(object):
    """List of antennas (Antenna class)
    """
    def __init__(self, antennas=None):
        if antennas is not None:
            self._antennas = antennas[:]
        else:
            self._antennas = []

    def add(self, new_antenna):
        assert isinstance(new_antenna, Antenna)
        self._antennas.append(new_antenna)

    @property
    def names(self):
        return [a.name for a in self._antennas]

    @property
    def observed(self):
        return [a.name for a in self._antennas if a.observed]

    @property
    def subbands(self):
        return [a.subbands for a in self._antennas if a.observed]

    def __len__(self):
        return len(self._antennas)

    def __getitem__(self, key):
        return self._antennas[self.names.index(key)]

    def __delitem__(self, key):
        return self._antennas.remove(self.names.index(key))

    def __iter__(self):
        return self._antennas.__iter__()

    def __reversed__(self):
        return self._antennas[::-1]

    def __contains__(self, key):
        return key in self.names

    def __str__(self):
        return f"Antennas([{','.join(self.names)}])\nObserved: {','.join(self.observed)}\n "

    def json(self):
        """Returns a dict with all attributes of the object.
        I define this method to use instead of .__dict__ as the later only reporst
        the internal variables (e.g. _username instead of username) and I want a better
        human-readable output.
        """
        d = dict()
        for ant in self.__iter__():
            d['Antenna'] = ant.__dict__

        return d


class Subbands(object):
    """Defines the frequency setup of a given observation with the following data:
        - n_subbands :  int
            Number of subbands.
        - channels : int
            Number of channels per subband.
        - frequencies : array-like
            Reference frequency for each channel and subband (NxM array, with N
            number of subbands, and M number of channels per subband).
        - bandwidths : astropy.units.Quantity or float
            Total bandwidth for each subband.
    """
    @property
    def n_subbands(self):
        return self._n_subbands

    @property
    def channels(self):
        return self._channels

    @property
    def central_frequency(self):
        return (self._freqs[0,0] + self._freqs[-1,-1])/2

    @property
    def frequencies(self):
        return self._freqs

    @property
    def bandwidths(self):
        return self._bandwidths

    def __init__(self, chans: int, freqs, bandwidths):
        """Inputs:
            - chans : int
                Number of channels per subband.
            - freqs : array-like
                Reference frequency for each channel and subband (NxM array, M number
                of channels per subband.
            - bandwidths : float or astropy.units.Quantity
                Total bandwidth for each subband. If not units are provided, Hz are assumed.
        """
        self._n_subbands = freqs.shape[0]
        assert isinstance(chans, (int, np.int32, np.int64)), \
            f"Chans {chans} is not an int as expected (found type {type(chans)})."
        assert isinstance(bandwidths, float) or isinstance(bandwidths, u.Quantity), \
            f"Bandiwdth {bandwidths} is not a float or Quantity as expected (found type {type(bandwidths)})."
        assert freqs.shape == (self._n_subbands, chans)
        self._channels = int(chans)
        self._freqs = np.copy(freqs)
        if isinstance(bandwidths, float):
            self._bandwidths = bandwidths*u.Hz
        else:
            self._bandwidths = bandwidths

    def __iter__(self):
        for key in ('n_subbands', 'channels', 'bandwidths', 'frequencies'):
            yield key, getattr(self, key)

    def json(self):
        """Returns a dict with all attributes of the object.
        I define this method to use instead of .__dict__ as the later only reporst
        the internal variables (e.g. _username instead of username) and I want a better
        human-readable output.
        """
        d = dict()
        for key, val in self.__iter__():
            if isinstance(val, u.Quantity):
                d[key] = val.to(u.Hz).value
            elif isinstance(val, np.ndarray):
                d[key] = list(val)
            else:
                d[key] = val

        return d


class Experiment(object):
    """Defines and EVN experiment with all relevant metadata.
    """
    @property
    def msname(self):
        """Name of the path to the MS, in upper case.
        """
        return self._msname

    @property
    def obsdate(self):
        """Epoch at which the EVN experiment was observed (starting date), in datetime.date format.
        """
        return self._obsdate.strftime('%y%m%d')

    @obsdate.setter
    def obsdate(self, obsdate):
        if isinstance(obsdate, str):
            self._obsdate = dt.datetime.strptime(self.obsdate, '%y%m%d')
        elif isinstance(obsdate, dt.date):
            self._obsdate = obsdate
        elif isinstance(obsdate, dt.datetime):
            self._obsdate = obsdate.date()
        else:
            raise TypeError(f"The variable {obsdate} has an unexpected type (none str or date/datetime)")

    @property
    def obsdatetime(self):
        """Epoch at which the EVN experiment was observed (starting date), in datetime format.
        """
        return self._obsdate

    @property
    def timerange(self):
        """Start and end time of the observation in datetime format.
        """
        return self._startime, self._endtime

    @timerange.setter
    def timerange(self, times):
        """Start and end time of the observation in datetime format.
        Input:
            - times : tuple of datetime
                Tupple with (startime, endtime), each of them in datetime format.
        """
        starttime, endtime = times
        assert isinstance(starttime, dt.datetime)
        assert isinstance(endtime, dt.datetime)
        self._startime = starttime
        self._endtime = endtime
        self.obsdate = starttime.date()

    @property
    def sources(self):
        """List of sources observed in the experiment.
        """
        return self._sources

    @sources.setter
    def sources(self, new_sources):
        """List of sources observed in the experiment.
        """
        self._sources = list(new_sources)

    @property
    def antennas(self):
        """List of antennas that were scheduled during the experiment.
        """
        return self._antennas

    @antennas.setter
    def antennas(self, new_antennas):
        isinstance(new_antennas, Antennas)
        self._antennas = new_antennas

    @property
    def freqsetup(self):
        return self._freqsetup

    @property
    def ignore_check(self):
        return self._ignore

    @ignore_check.setter
    def ignore_check(self, ignore):
        self.ignore = ignore

    def __init__(self, msfile, ignore_check: bool = False):
        """Initializes an EVN experiment with the given name.

        Inputs:
        - msfile : str
               The path to the MS file to be read.
        - ignore_check : bool [default = False]
               If True, it will not go through the MS to identify the stations that did not observe or
               the subbands that each antenna observed..
               This process is quite slow so it can be ignored to get a quick output.
               If True, all antennas will be assumed to have observed and have the full bandwidth of the observation.
        """
        assert os.path.isdir(msfile), f"The provided MS file {msfile} could not be found."
        self._msname = msfile
        self._obsdate = None  #TODO: is this computed with setupt?
        # Attributes not known until the MS file is created
        self._startime = None
        self._endtime = None
        self._sources = []
        self._antennas = Antennas()
        self._freqsetup = None
        self._ignore = ignore_check
        self.get_setup_from_ms()

    def get_setup_from_ms(self):
        """Obtains the time range, antennas, sources, and frequencies of the observation
        from all existing passes with MS files and incorporate them into the current object.
        """
        try:
            with pt.table(self.msname, readonly=True, ack=False) as ms:
                with pt.table(ms.getkeyword('ANTENNA'), readonly=True, ack=False) as ms_ant:
                    antenna_col = ms_ant.getcol('NAME')
                    for ant_name in antenna_col:
                        ant = Antenna(name=ant_name, observed=True)
                        self._antennas.add(ant)

                with pt.table(ms.getkeyword('DATA_DESCRIPTION'), readonly=True, ack=False) as ms_spws:
                    spw_names = ms_spws.getcol('SPECTRAL_WINDOW_ID')

                if not self.ignore_check:
                    ant_subband = defaultdict(set)
                    print('\nReading the MS to find the antennas that actually observed...')
                    with progress.Progress() as progress_bar:
                        task = progress_bar.add_task("[yellow]Reading MS...", total=len(ms))
                        for (start, nrow) in chunkert(0, len(ms), 100):
                            ants1 = ms.getcol('ANTENNA1', startrow=start, nrow=nrow)
                            ants2 = ms.getcol('ANTENNA2', startrow=start, nrow=nrow)
                            spws = ms.getcol('DATA_DESC_ID', startrow=start, nrow=nrow)
                            msdata = ms.getcol('DATA', startrow=start, nrow=nrow)

                            for ant_i,antenna_name in enumerate(antenna_col):
                                for spw in spw_names:
                                    cond = np.where((ants1 == ant_i) & (ants2 == ant_i) & (spws == spw))
                                    if not (abs(msdata[cond]) < 1e-5).all():
                                        ant_subband[antenna_name].add(spw)

                            progress_bar.update(task, advance=nrow)

                    for antenna_name in self.antennas.names:
                        self._antennas[antenna_name].subbands = tuple(ant_subband[antenna_name])
                        self._antennas[antenna_name].observed = len(self._antennas[antenna_name].subbands) > 0
                else:
                    for antenna_name in self.antennas.names:
                        self._antennas[antenna_name].subbands = spw_names
                        self._antennas[antenna_name].observed = True

                with pt.table(ms.getkeyword('FIELD'), readonly=True, ack=False) as ms_field:
                    src_names = ms_field.getcol('NAME')
                    src_coords = ms_field.getcol('PHASE_DIR')
                    for a_name, a_coord in zip(src_names, src_coords):
                        self.sources.append(Source(a_name, coord.SkyCoord(*a_coord[0], unit=(u.rad, u.rad))))

                with pt.table(ms.getkeyword('OBSERVATION'), readonly=True, ack=False) as ms_obs:
                    self.timerange = dt.datetime(1858, 11, 17, 0, 0, 2) + \
                         ms_obs.getcol('TIME_RANGE')[0]*dt.timedelta(seconds=1)

                with pt.table(ms.getkeyword('SPECTRAL_WINDOW'), readonly=True, ack=False) as ms_spw:
                    self._freqsetup = Subbands(ms_spw.getcol('NUM_CHAN')[0],
                                                ms_spw.getcol('CHAN_FREQ'),
                                                ms_spw.getcol('TOTAL_BANDWIDTH')[0])
        except RuntimeError:
            print(f"WARNING: {self.msname} not found.")

    def store(self, path=None):
        """Stores the current Experiment into a file in the indicated path. If not provided,
        it will be '.{expname.lower()}.obj' where exp is the name of the experiment.
        """
        if path is not None:
            self._local_copy = path

        with open(self._local_copy, 'wb') as file:
            pickle.dump(self, file)

    def store_json(self, path=None):
        """Stores the current Experiment into a JSON file.
        If path not prvided, it will be '{expname.lower()}.json'.
        """
        raise NotImplementedError
        if path is not None:
            self._local_copy = path

        with open(self._local_copy, 'wb') as file:
            json.dump(self.json(), file, cls=ExpJsonEncoder, indent=4)

    def load(self, path=None):
        """Loads the current Experiment that was stored in a file in the indicated path. If path is None,
        it assumes the standard path of '.{exp}.obj' where exp is the name of the experiment.
        """
        if path is not None:
            self._local_copy = path

        with open(self._local_copy, 'rb') as file:
            obj = pickle.load(file)

        return obj

    def __repr__(self, *args, **kwargs):
        rep = super().__repr__(*args, **kwargs)
        rep.replace("object", f"object ({self.msname})")
        return rep

    def __str__(self):
        return f"<Experiment {self.msname}>"

    def __iter__(self):
        for key in ('msname', 'obsdate', 'obsdatetime', 'timerange', 'sources', 'antennas', 'cwd'):
            yield key, getattr(self, key)

    def json(self):
        """Returns a dict with all attributes of the object.
        I define this method to use instead of .__dict__ as the later only reporst
        the internal variables (e.g. _username instead of username) and I want a better
        human-readable output.
        """
        d = dict()
        for key, val in self.__iter__():
            if hasattr(val, 'json'):
                d[key] = val.json()
            elif isinstance(val, Path):
                d[key] = val.name
            elif isinstance(val, dt.datetime):
                d[key] = val.strftime('%Y-%m-%d')
            elif isinstance(val, dt.date):
                d[key] = val.strftime('%Y-%m-%d')
            elif isinstance(val, list) and (len(val) > 0) and hasattr(val[0], 'json'):
                d[key] = [v.json() for v in val]
            elif isinstance(val, tuple) and (len(val) > 0) and isinstance(val[0], dt.datetime):
                d[key] = [v.strftime('%Y-%m-%d %H:%M:%S') for v in val]
            elif isinstance(val, dict):
                d[key] = {}
                for k, v in val:
                    if hasattr(v, 'json'):
                        d[key][k] = v.json()
                    elif hasattr(v, 'name'):
                        d[key][k] = v.name
                    else:
                        d[key][k] = v
            else:
                d[key] = val

        return d

    def print_blessed(self):
        """Pretty print of the full experiment with all available data.
        """
        term = blessed.Terminal()
        with term.fullscreen(), term.cbreak():
            # s = term.center(term.red_on_bright_black(f"EVN Post-processing of {self.expname.upper()}")) + '\n\n'
            s = term.red_on_bright_black(term.center(term.bold(self.msname)))
            s += f"{term.normal}\n\n{term.normal}"
            s += term.bright_black('Obs date: ') + self.obsdatetime.strftime('%d/%m/%Y')
            s += f" {'-'.join([t.time().strftime('%H:%M') for t in self.timerange])} UTC\n\n"

            s += term.bold_green('SETUP\n')
            # loop over passes
            s += term.bright_black('Central Frequency: ') + f"{self.freqsetup.central_frequency/1e9:0.04} GHz\n"
            s += term.bright_black('Frequency Range: ') + \
                 f"{self.freqsetup.frequencies[0,0]/1e9:0.04}-" \
                 f"{self.freqsetup.frequencies[-1,-1]/1e9:0.04} GHz.\n"
            s += term.bright_black('Bandwidth: ') + \
                 f"{self.freqsetup.n_subbands} x " \
                 f"{self.freqsetup.bandwidths.to(u.MHz).value}-MHz subbands. " \
                 f"{self.freqsetup.channels} channels each.\n\n"

            s += term.bold_green('SOURCES\n')
            for src in self.sources:
                s += f"{src.name}: {term.bright_black(src.coordinates.to_string('hmsdms'))}\n"

            s += '\n'
            s += term.bold_green('ANTENNAS\n')
            s += term.bright_black('Observed antennas: ') + \
                 f"{', '.join([ant.name for ant in self.antennas if ant.observed])}\n"
            missing_ants = [ant.name for ant in self.antennas if not ant.observed]
            s += term.bright_black('Did not observe: ') + \
                 f"{', '.join(missing_ants) if len(missing_ants) > 0 else 'None'}\n\n"

            # In case of antennas not observing the full bandwidth (this may be per correlator pass)
            ss = ""
            for antenna in self.antennas:
                if 0 < len(antenna.subbands) < self.freqsetup.n_subbands:
                    ss += f"    {antenna.name}: {antenna.subbands}\n"

            if ss != "":
                s += term.bright_black('Antennas with smaller bandwidth:\n') + ss

            s_final = term.wrap(s, width=term.width)

            def print_all(ss):
                print(term.clear)
                for a_ss in ss:
                    print(a_ss)

                print(term.move_y(term.height - 3) + \
                      term.center(term.on_bright_black('press any key to continue (or Q to cancel)')).rstrip())
                return term.inkey()#.strip()

            # Fitting the terminal
            i, i_width = 0, term.height - 5
            while i < len(s_final):
                value = print_all(s_final[i:min(i+i_width, len(s_final)+1)])
                if value.lower() == 'q':
                    return False
                elif value.is_sequence and (value.name == 'KEY_UP'):
                    i = max(0, i-i_width)
                else:
                    i += i_width

            return True


def main():
    usage = "%(prog)s [-h] [-v] [-i] <measurement set>"
    description="""Provides an overview of the information contained in a MS file.
    """
    parser = argparse.ArgumentParser(description=description, prog='msoverview.py', usage=usage)
    parser.add_argument('msdata', type=str, help='Path to the MS to file read.')
    parser.add_argument('-i', '--ignore', default=False, action="store_true",
                        help='Ignore checking the MS to spot antennas that did not observe (slow) or ' \
                             'have individual bandwidths.')
    parser.add_argument('-v', '--version', action='version', version='%(prog)s {}'.format(__version__))
    arguments = parser.parse_args()
    mspath = arguments.msdata[:-1] if arguments.msdata[-1]=='/' else arguments.msdata
    ms = Experiment(mspath, ignore_check=arguments.ignore)
    ms.print_blessed()

if __name__ == '__main__':
    main()
