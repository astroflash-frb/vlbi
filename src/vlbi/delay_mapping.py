#! /usr/bin/env python3

import os
import sys
import argparse
from matplotlib import pyplot as plt
from dataclasses import dataclass
from collections import defaultdict
from typing import Optional, Union #, Iterable, Generator
from rich import print as rprint
from rich import progress
from rich_argparse import RichHelpFormatter
import numpy as np
from astropy import coordinates as coord
from astropy import units as u
from astropy.io import ascii
from casatools import msmetadata as msmd
from casatools import table as tb
from vlbi import funcs


_DESCRIPTION = """Runs delay mapping on a calibrated Ms dataset.
It will produce a .html file with all lag plots for the different baselines and subbands/polarizations.
Retuns the a-priori position of the given source and the individual delays calculated from the lag space.

It assumes that the given MS has a single source, and it has already been calibrated.
"""

@dataclass
class Antenna:
    name: str
    subbands: tuple[int]


@dataclass
class MSinfo:
    """Minimal information contained in a MS
    """
    antennas: list[Antenna]
    source: str
    coordinates: coord.SkyCoord
    bandwidth: u.Quantity # per subband
    freq_central: u.Quantity
    channels: int
    subbands: list[int]
    polarizations: list[int]
    polarizations_str: list[str]


def create_project_structure(project_code: str, path: Optional[str] = None):
    """Creates the directory structure used to store the created files.

    Inputs:
        project_code : str
            The code or name for the (main) project directory.
        path : str
            The path where the directory will be created. If not provided,
            it will be in the current working directory ($PWD).

    Returns:
        str: The path to the created project directory.
    """
    project_dir = os.path.join(os.getcwd() if path is None else path, project_code)
    os.makedirs(project_dir, exist_ok=True)

    # Create subdirectories
    os.makedirs(os.path.join(project_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(project_dir, 'results'), exist_ok=True)
    return project_dir


def print_ms(msinfo: MSinfo, saveto: Optional[str] = None, silence: bool = False):
    """Prints in terminal all information relevant to the MS to read.

    Inputs
        msinfo : MSinfo
            The object including the metadata from the MS.
        saveto : str  (default = None)
            If specified, it will save the information into the 'saveto' file name.
        silence : bool  (default False)
            If True, then it does not print anything in terminal. It will only save the file (if specified).

    """
    s = "\n\n"
    s += "[bold]Information from the MS:[/bold]\n"
    s += f"[bold]Source:[/bold] {msinfo.source}  ({msinfo.coordinates.to_string('hmsdms')})\n"
    s += f"[bold]Setup:[/bold] {len(msinfo.subbands)} x {msinfo.bandwidth.to(u.MHz):.0f} subbands, " \
         f"{msinfo.channels} channels each.\n"
    s += f"[bold]Central frequency:[/bold] {msinfo.freq_central.to(u.GHz):.3f}\n"   # type: ignore
    s += f"[bold]Antennas & subbands[/bold] (starting in zero):\n"
    for ant in msinfo.antennas:
        if len(ant.subbands) > 0:
            s += f"    {ant.name}: {' '*(3*(ant.subbands[0]))}{ant.subbands}\n"
        else:
            s += f"    {ant.name}: ---\n"


    if not silence:
        rprint(s + "\n\n")

    if saveto is not None:
        with open(saveto, 'w') as afile:
            afile.write(s.replace('[bold]', '').replace('[/bold]', ''))


def get_ms_metadata(msfile: str, chunks: int = 100):
    """Reads the MS file and extracts the number of subbands, antennas, polarizations,
    and which combinations of them actually contain data.

    Inputs
        msfile : str
            The MS file to read. Should be a MS with calibrated visibilities and a single source.
        chunks : int  (default = 100)
            The chanks of data to read during the MS reading. Different values may affect I/O speed
            but otherwise it does not have any other effect. The default one is usually the optimal
            one for standard EVN-like data sets.
    Returns
        msinfo : MSinfo
            Object with all metadata read from the MS (see MSinfo class for more information).
    """
    m = msmd(msfile)
    if not m.open(msfile):
        rprint(f"[red bold]ERROR: Cannot find the given MS file ({msfile}).[/red bold]")
        sys.exit(1)

    ms: dict = {}
    ms['field'] = set()
    ms['antennas'] = []
    try:
        antennas = m.antennanames()
        ms['subbands'] = list(range(m.nspw()))
        src_names = m.fieldnames()
        nrows = int(m.nrows())
        # To be able  to get the parallel hands, either circular or linear
        corr_order = [funcs.Stokes(i) for i in m.corrtypesforpol(0)]
        corr_pos = []
        try:
            corr_pos.append(corr_order.index(funcs.Stokes.RR))
            corr_pos.append(corr_order.index(funcs.Stokes.LL))
            ms['polarization-basis'] = ['RR', 'LL']
        except ValueError:
            try:
                corr_pos.append(corr_order.index(funcs.Stokes.XX))
                corr_pos.append(corr_order.index(funcs.Stokes.YY))
                ms['polarization-basis'] = ['XX', 'YY']
            except ValueError:
                rprint("[bold red]The associated MS does not have neither circular nor " \
                       "linear-based polarization information[/bold red]")
                sys.exit(1)

        # print(f"{corr_order=}, {m.corrtypesforpol(0)=}, {corr_pos=}")
        ms['polarizations'] = corr_pos
        ms['phase-centers'] = m.phasecenter()
        ms['freq_central'] = u.Quantity(((m.meanfreq(ms['subbands'][-1]) + m.meanfreq(0)) / 2.0), u.Hz).to(u.GHz)
        ms['channels'] = m.nchan(0)
        ms['bandwidth'] = (m.bandwidths()[0]*u.Hz).to(u.MHz)
    finally:
        m.close()


    m = tb(msfile)
    if not m.open(msfile):
        rprint(f"[red bold]ERROR: Cannot find the given MS file ({msfile}).[/red bold]")
        sys.exit(1)

    try:
        ant_subband: dict[str, set[int]] = {ant: set() for ant in antennas}
        rprint('\n[bold]Reading the MS to find which antennas actually observed and which subbands...[/bold]')
        with progress.Progress() as progress_bar:
            task = progress_bar.add_task("[yellow]Reading MS...", total=nrows)
            for (start, nrow) in funcs.chunkert(0, nrows, chunks):
                ants1 = m.getcol('ANTENNA1', startrow=start, nrow=nrow)
                ants2 = m.getcol('ANTENNA2', startrow=start, nrow=nrow)
                field = m.getcol('FIELD_ID', startrow=start, nrow=nrow)
                spws = m.getcol('DATA_DESC_ID', startrow=start, nrow=nrow)
                msdata = m.getcol('DATA', startrow=start, nrow=nrow)
                for src in set(field):
                    ms['field'].add(src)

                for ant_i,antenna_name in enumerate(antennas):
                    for spw in ms['subbands']:
                        cond = np.where(((ants1 == ant_i) | (ants2 == ant_i)) & (spws == spw))
                        if not (abs(msdata[corr_pos][:, :, cond[0]]) < 1e-5).all():
                            ant_subband[antenna_name].add(spw)
                        # testing a much faster check...  But it picks everything
                        # if len(cond[0]) > 0:
                        #     ant_subband[antenna_name].add(spw)

                progress_bar.update(task, advance=nrow)

        for ant in ant_subband:
            ms['antennas'].append(Antenna(name=ant, subbands=tuple(ant_subband[ant]))) # type: ignore

        if len(ms['field']) > 1:
            rprint("[orange bold]More than one source was found in this MS. "
                   "Please make a single-source file.[/orange bold]")
            sys.exit(1)

        ms['field'] = tuple(ms['field'])[0]

        ms['source'] = src_names[ms['field']]  # noqa - type: ignore
        src_coords = ms['phase-centers']  # dummy as phasecenter() only reports one coordinates
        ms['coordinates'] = coord.SkyCoord(ra=src_coords['m0']['value'], dec=src_coords['m1']['value'],
                                        unit=(src_coords['m0']['unit'], src_coords['m1']['unit']),
                                        equinox=src_coords['refer'])
    finally:
        m.close()

    return MSinfo(*[ms[key] for key in ('antennas', 'source', 'coordinates', 'bandwidth', 'freq_central', \
                                        'channels', 'subbands', 'polarizations', 'polarization-basis')])



def delay_snr(phases, weights, freq_resolution: float, padding: int = 8):
    """Computes the lag spectrum for the given data belonging to a single baseline and single polarization

    Inputs
        phases : np.array (shape = Nchannels x Nvisibilities)
            The phases as obtained from the DATA column in the MS. Flagged data should have already been removed.
        weights : (shape = Nvisibilities)
            The weights associated with the previous data.
        freq_resolution : float
            The frequency resolution of each spectral channel in the data, in Hz.

    Returns
        lags : 1D np.array
            Array with the lag numbers.
        lag_spec : 1D np.array
            Amplitudes obtained for each lag given in 'lags'.
        snr_p : float
            Signal-to-noise (SNR) ratio of the peak in the lag spectrum.
        delay_p : float
            Delay at which the the peak in the lag spectrum is found.
        lags_offset_p : float
            Lag offset at which the peak in the lag spectrum is found.
    """
    n_channels = phases.shape[0]
    # Makes a wider window to avoid FFT issues
    if len(phases.shape) == 1:
        # Special case where there is only one visibility
        padded_phases = np.zeros((padding*n_channels,), complex)
        padded_phases[0:n_channels] = phases
    else:
        padded_phases = np.zeros((padding*n_channels, phases.shape[1]), complex)
        padded_phases[0:n_channels,:] = phases

    lag_spec = np.fft.fftshift(np.abs(np.fft.fft(padded_phases)))
    n_pad = len(lag_spec)
    lags = np.arange(n_pad) - n_pad/2
    lags_offset_p = np.argmax(lag_spec) - n_pad/2

    # Calculating SNR
    lag_peak = np.max(np.abs(np.fft.fft(phases)))
    xcount = np.sum(np.abs(phases))
    sum_weights = xcount*weights
    sum_weights2 = sum_weights*weights
    x = lag_peak/xcount*np.pi/2.0
    snr_p = (np.pow(np.tan(x), 1.163) * np.sqrt(sum_weights/np.sqrt(sum_weights2/xcount)))

    # Calculating delay
    delay_p = (lags_offset_p/n_pad) / freq_resolution
    assert all([isinstance(q, float) for q in (snr_p, delay_p, lags_offset_p)]), \
           f"They all should be floats but are {snr_p=}, {delay_p=}, {lags_offset_p=}"
    return lags, lag_spec, snr_p, delay_p, lags_offset_p


def gaussian_func(x, norm: float, x0: float, sigma: float):
    """Returns the values of a Gaussian function centered at x0 with sigma parameter and 'norm'
    normalization.
    """
    return norm*np.exp(-(x-x0)**2/(2*sigma**2))


def plot_lags(data: dict, refant: str, baseline: str, subband: int,
              tosave: Union[str, bool] = False, path: Optional[str] = None):
    """Plots the lag spectrum for the given baseline, and subband.
    Inputs
        data : dict
            Must contain a dict with a keys to be the observed polarizations to plot.
            For each key, it must contain a dictionary with the following keys:
            lag (1D array), lag_spec (1D array), snr_p (float), delay_p (float), lags_offset_p (float).
        refant : str
            Name of the reference antenna
        baseline : str
            Name of the antenna used for the given baseline to 'refant'.
        subband : int
            The number of the plotted subband.
        tosave : str or bool   (default = False)
            Option to save the plot into a file. It can be a boolean (True to save it or False not to do it).
            Or a string, which will then be the filename to use for saving it.
        path : str  (defualt None)
            The path (directory) where to save the plot (if tosave is not False).
            If none, it will be saved in the current directory.
    """
    fig, ax = plt.subplots(1)
    ax.set_title('Lag Spectrum')
    for pol in data:
        ax.plot(data[pol]['lags'], data[pol]['lag_spec'], label=pol.upper())

    ax.annotate("Lag offset: " + " ".join([f"{pol.upper()}: {data[pol]['lags_offset_p']:.2f}" for pol in data]),
                xy=(0.55, 0.8), xycoords='axes fraction', fontsize=10)
    ax.annotate("SNR: " + " ".join([f"{pol.upper()}: {data[pol]['snr_p']:.2f}" for pol in data]),
                xy=(0.55, 0.9), xycoords='axes fraction', fontsize=10)
    ax.annotate("Delay: " + " ".join([f"{pol.upper()}: {data[pol]['delay_p']:.2e}" for pol in data]),
                xy=(0.55, 0.7), xycoords='axes fraction', fontsize=10)
    # ax.set_xlabel(f"")
    ax.legend(loc=2)
    if tosave:
        data2write = {}
        for pol in data:
            data2write[f"Lag_{pol}"] = data[pol]['lags']
            data2write[f"Lag_spec_{pol}"] = data[pol]['lag_spec']

        ascii.write(data2write, f"{'.' if path is None else path}/lag_spectrum_{refant}-{baseline}_SB" \
                        f"{str(subband)}.txt")
        if isinstance(tosave, str):
            fig.savefig(tosave, bbox_inches=0.001, pad_inches='tight')
        else:
            for ext in ('png', 'pdf'):
                # Saving both PDF and PNG so the plots are useful in both PDF format and for the html file
                fig.savefig(f"{'.' if path is None else path}/lag_spectrum_{refant}-{baseline}_SB" \
                            f"{str(subband)}.{ext}", bbox_inches='tight', pad_inches=0.001)


def main(msfile: str, refant: str = 'EF', baselines: Optional[list[str]] = None, snr: int = 7,
         spw: Optional[list[int]] = None, padding: int = 8, verbose: bool = True):
    """Runs delay mapping on a calibrated Ms dataset.
    It will produce a .html file with all lag plots for the different baselines and subbands/polarizations.
    Retuns the a-priori position of the given source and the individual delays calculated from the lag space.

    It assumes that the given MS has a single source, and it has already been calibrated.

    Inputs
        msfile : str
            The single-source calibrated MS file to process.
        refant : str  (default = 'EF')
            The reference antenna (only baselines to the reference antenna will be computed).
        baselines : list[str]   (default = None)
            List of antennas to which compute the delay mapping. All baselines from refant to antennas
            included in 'baselines' will be computed. If None, then all available antennas will be used.
        snr : int  (default = 7)
            The minimum SNR required to consider a significant fringe in the lag space.
        spw : list[int]  (default None)
            List of subbands to consider. If None, then all of them will be considered.
        padding : int  (default = 8)
            Padding factor for the FFT in the lag space.
        verbose : bool  (default True)
            If True, then it will print in terminal the metadata from the read MS. Otherwise it will go silent.
    """
    dm_path = msfile.replace('.ms', '') + '.delay_mapping'
    create_project_structure(dm_path)
    msinfo = get_ms_metadata(msfile)
    print_ms(msinfo, saveto=dm_path + '/results/summary_ms.txt', silence=not verbose)

    # Read all data from MS
    ms = tb(msfile)
    if not ms.open(msfile):
        rprint(f"[red bold]ERROR: Cannot find the given MS file ({msfile}).[/red bold]")
        sys.exit(1)

    uvws = ms.getcol('UVW')
    ants1 = ms.getcol('ANTENNA1')
    ants2 = ms.getcol('ANTENNA2')
    spws = ms.getcol('DATA_DESC_ID')
    phases = np.exp(1J*np.angle(ms.getcol('DATA')))
    weights = ms.getcol('WEIGHT')
    flagged = ms.getcol('FLAG')

    # Zero data where it is flagged
    phases[flagged] = 0
    # print(phases.shape)  # 4, 64, 224

    assert (phases > 0+0j).any(), "All phases in the MS are zero (maybe everything is flagged)?"
    # Separate from baselines and polarizations as a dict
    if baselines is None:
        baselines = [ant.name for ant in msinfo.antennas if ant.name != refant]

    if spw is None:
        spw = msinfo.subbands
    # data = split_data(phases, weights, ants1, ants2, msinfo.polarizations, refant, baselines)

    antenna_names = [ant.name for ant in msinfo.antennas]
    lag_results: dict = {}
    results4file: dict = defaultdict(list)
    for basel in baselines:
        lag_results[basel] = {}
        for a_spw in spw:
            lag_results[basel][a_spw] = {}
            condition = np.where(((ants1 == antenna_names.index(refant)) | \
                                  (ants2 == antenna_names.index(refant))) & \
                                 ((ants1 == antenna_names.index(basel)) | \
                                  (ants2 == antenna_names.index(basel))) &
                                 (spws == a_spw))
            for pol, pol_str in zip(msinfo.polarizations, msinfo.polarizations_str):   # RR, LL
                # if len(phases[pol,:, condition]) > 0:
                if (phases[pol, :, condition] > 0.0).any():
                    results = delay_snr(phases[pol,:,condition].squeeze(), weights[pol,condition].squeeze(),
                                        msinfo.freq_central.to(u.Hz).value, padding=padding)
                    lag_results[basel][a_spw][pol_str] = {key: result for result, key in \
                            zip(results, ('lags', 'lag_spec', 'snr_p', 'delay_p', 'lags_offset_p'))}

                    results4file['antenna1'].append(refant)
                    results4file['antenna2'].append(basel)
                    results4file['spw'].append(a_spw)
                    results4file['polarization'].append(pol_str)
                    results4file['u'].append(np.mean(uvws[0, condition]))
                    results4file['v'].append(np.mean(uvws[1, condition]))
                    results4file['snr'].append(lag_results[basel][a_spw][pol_str]['snr_p'])
                    results4file['delay'].append(lag_results[basel][a_spw][pol_str]['delay_p'])
                    results4file['lag_offset'].append(lag_results[basel][a_spw][pol_str]['lags_offset_p'])

            plot_lags(lag_results[basel][a_spw], refant, basel, a_spw, tosave=True, path=dm_path + '/plots')
    # In a parallel way:
    # Compute lags
    # calculate delay and errors

    ascii.write(results4file, dm_path + '/results/delays.txt', overwrite=True,
                formats={'snr_p': '%3f', 'delay': '%.3e', 'lag_offset': '%n'})
    # create plots
    # create html with all results


def cli():
    """Runs delay mapping on a calibrated Ms dataset.
    It will produce a .html file with all lag plots for the different baselines and subbands/polarizations.
    Retuns the a-priori position of the given source and the individual delays calculated from the lag space.

    It assumes that the given MS has a single source, and it has already been calibrated.
    """
    parser = argparse.ArgumentParser(description=_DESCRIPTION, formatter_class=RichHelpFormatter)
    parser.add_argument("msfile", type=str, help="The single-source calibrated MS file to process.")
    parser.add_argument("-r", "--refant", type=str, default='EF', help="Reference antenna (by default EF).")
    parser.add_argument("-b", "--baselines", nargs='+',type=str, default=None,
                        help="List (space-separated) of baselines to 'refant' that will be considered. "
                        "By default it will consider all.")
    parser.add_argument("--snr", type=int, default=7,
                        help="The minimum SNR required to consider a significant fringe in the lag space."
                        " By default is set to SNR = 7.")
    parser.add_argument("--spw", nargs='+', type=int, default=None,
                        help="List (space-separated) of subbands to consider. "
                        "By default it will consider all.")
    parser.add_argument("-p", "--padding", type=int, default=8,
                        help="Padding factor for the FFT in lag space.")
    parser.add_argument("-v", "--verbose", action="store_true", default=False,
                        help="Show MS information while running.")
    # parser.add_argument("--snr", type=int, default=7,
    #                     help="The minimum SNR required to consider a significant fringe in the lag space.")
    args = parser.parse_args()
    if not os.path.isdir(args.msfile):
        rprint(f"[bold red]\nThe given MS file ({args.msfile}) does not exist or cannot be found.[/bold red]")
        sys.exit(1)

    main(args.msfile, args.refant, args.baselines, args.snr, args.spw, args.padding, verbose=args.verbose)


if __name__ == '__main__':
    cli()

