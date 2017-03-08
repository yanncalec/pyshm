#!/usr/bin/env python

"""Plot and interpret the data or results of analysis.
"""

import sys, os, argparse
import numpy as np
import scipy as sp
import pandas as pd
import json
from pyshm import OSMOS, Tools, Stat
from pyshm.script import static_data_analysis_template, examplestyle, warningstyle, load_result_of_analysis
from pyshm.script import MyEncoder, to_json


import matplotlib
# matplotlib.use("qt5agg")
import matplotlib.pyplot as plt
# import matplotlib.colors as colors
import mpld3
plt.style.use('ggplot')

# class Options:
#     verbose=False  # print message
#     info=False  # only print information about the project
#     subcommand='static'  # type of deconvolution model


def compute_local_statistics(Yerr, mad, mwsize):
    """Compute the local statistics: mean and standard deviation and normalized
    observation.

    """
    Merr0 = {}; Serr0 = {}; Nerr0 = {};

    for loc, yerr in Yerr.items():
        # yerr = Ycpn[loc] - Yprd[loc]
        # moving average and standard deviation
        merr, serr = Stat.local_statistics(yerr, mwsize, mad=mad, causal=False, drop=True)
        nerr = abs(yerr-merr)/serr  # normalized error
        Merr0[loc], Serr0[loc], Nerr0[loc] = merr, serr, nerr

    Merr = pd.DataFrame(Merr0, columns=list(Yerr.keys()), index=Yerr.index)
    Serr = pd.DataFrame(Serr0, columns=list(Yerr.keys()), index=Yerr.index)
    Nerr = pd.DataFrame(Nerr0, columns=list(Yerr.keys()), index=Yerr.index)

    return Merr, Serr, Nerr


__script__ = __doc__

__warning__ = "Warning:" + warningstyle("\n ")

examples = []
examples.append(["%(prog)s -h", "Print this help messages (about common parameters)"])
examples.append(["%(prog)s raw DBDIR/153/Raw.pkl", "Apply preprocessing with default parameters on the project of PID 153 (the project lied in the database directory DBDIR), plot the static data in a subfolder named figures/Static and print messages."])
examples.append(["%(prog)s --sflag -v DBDIR/036", "Apply preprocessing by removing syncrhonisation error on the project of PID 36."])
__example__ = "Some examples of use (change the path seperator '/' to '\\' on Windows platform):" + "".join([examplestyle(x) for x in examples])


def main():
    # usage_msg = '%(prog)s <subcommand> <infile> <outdir> [options]'
    # parser = argparse.ArgumentParser(description=__script__, usage=usage_msg)
    mainparser = argparse.ArgumentParser(description=__script__,
                                         formatter_class=argparse.RawDescriptionHelpFormatter,
                                         epilog=__warning__ + "\n\n" + __example__)#,add_help=False)

    mainparser.add_argument('infile', type=str, help='input data file.')
    # mainparser.add_argument('outdir', nargs='?', type=str, default=None, help='directory where results (figures and data files) will be saved.')
    # mainparser.add_argument('outdir', type=str, default=None, help='directory where results (figures and data files) will be saved.')
    mainparser.add_argument('-v', '--verbose', dest='verbose', action='count', default=0, help='Print messages.')
    mainparser.add_argument('-o', '--overwrite', dest='overwrite', action='store_true', default=False, help='Overwrite on the input file.')

    lstat_opts = mainparser.add_argument_group('Options for local statistics')  # local statistics
    lstat_opts.add_argument('--mad', dest='mad', action='store_true', default=False, help='Use median based estimator (default: use empirical estimator).')
    lstat_opts.add_argument('--mwsize', dest='mwsize', type=int, default=24*10, help='Size of the moving window (default=240).', metavar='integer')
    #     lstat_opts.add_argument('--causal', dest='causal', action='store_true', default=False, help='Use causal window (default: non causal).')

    hurst_opts = mainparser.add_argument_group('Options for the Hurst exponent (trend component only)')  # Hurst exponent
    hurst_opts.add_argument('--hrng', dest='hrng', nargs=2, type=int, default=(0,8), help='Wavelet scale range index for computation of Hurst exponent (default=(0,8).', metavar='integer')
    hurst_opts.add_argument('--hwsize', dest='hwsize', type=int, default=24*10, help='Size of the moving window for computation of Hurst exponent (default=240).', metavar='integer')

    options = mainparser.parse_args()

    # check the input data file
    if not os.path.isfile(options.infile):
        raise FileNotFoundError(options.infile)

    # if not os.path.isdir(options.outdir):
    #     raise FileNotFoundError(options.outdir)

    idx = options.infile.rfind(os.path.sep, 0)
    outdir = options.infile[:idx]
        # print(options.outdir)
        # raise SystemExit
        # outdir0 = os.path.join(options.infile[:idx0], 'Outputs', options.infile[idx1+1:idx2])

    # subfolder for output
    # outdir = os.path.join(options.outdir, 'data') #{}/_[{}_{}]'.format(options.subcommand.upper(), options.component.upper()))

    Res = load_result_of_analysis(options.infile)

    if "func_name" not in Res:
        raise KeyError("{}: Not a valid file of analysis".format(options.infile))

    # idx0 = options.infile.rfind(os.path.sep, 0)
    # idx1 = options.infile.find('.', idx0)
    # figdir = os.path.join(options.outdir, options.infile[idx0+1:idx1])
    # figdir = options.outdir
    # try:
    #     os.makedirs(figdir)
    # except:
    #     pass

    Yprd = Res['Yprd']  # predictions
    Locations = list(Yprd.keys())  # list of sensors
    # Xcpn = Res['Xcpn'][Locations]  # Res['Xcpn'] contains temperature data of all sensors
    Ycpn = Res['Ycpn'][Locations]  # Res['Ycpn'] contains elongation data of all sensors, extract only those having some prediction results
    Yerr = Ycpn - Yprd  # residual
    # Midx = Res['Midx']
    algo_options = dict(Res['algo_options'])  # options of parameters
    component = algo_options['component']  # name of the component being analyzed
    Emptydf = pd.DataFrame({loc: None for loc in Locations}, index=Yprd.index)  # an empty dataframe

    # Local statistics
    if options.verbose:
        print('Computing local statistics...')
    Merr, Serr, Nerr = compute_local_statistics(Yerr, options.mad, options.mwsize)

    # Hurst exponents and index ranges of instability period: defined only for trend or all component
    if component.upper() in ['TREND', 'ALL']:
        if options.verbose:
            print('Computing the Hurst exponent...')

        Hexp0 = {}
        if component.upper() == "ALL":
            mYerr = Yerr.rolling(window=24, min_periods=1, center=True).mean() # .median() #.bfill()
        else:
            mYerr = Yerr

        for loc, yerr in mYerr.items():
            # yerr = yerr0.copy(); yerr.loc[Midx[loc]] = np.nan
            hexp, *_ = Stat.Hurst(np.asarray(yerr), options.mwsize, sclrng=options.hrng, wvlname="haar")  # Hurst exponent
            # hexp[Midx[loc]] = np.nan
            # hblk[Midx[loc]] = np.nan
            # Hexp0[loc], Hblk[loc] = hexp, hblk
            Hexp0[loc] = hexp

        Hexp = pd.DataFrame(Hexp0, index=Yerr.index)
    else:
        Hexp = Emptydf.copy()

    outfile = options.infile if options.overwrite else os.path.join(outdir, 'Analysis') + '.json'
    Res.update({'Yerr':Yerr, 'Merr':Merr, 'Serr':Serr, 'Nerr':Nerr, 'Hexp':Hexp})
    # resdic = {'Xcpn':Xcpn, 'Ycpn':Ycpn, 'Yprd':Yprd, 'Merr':Merr, 'Serr':Serr, 'Nerr':Nerr, 'Hexp':Hexp, 'Midx':Midx}

    resjson = to_json(Res, verbose=options.verbose)
    with open(outfile, 'w') as fp:
        json.dump(resjson, fp, cls=MyEncoder)

    if options.verbose:
        print('Results saved in {}'.format(outfile))


if __name__ == "__main__":
    main()
