#!/usr/bin/env python

"""Post-process of the results of analysis.
"""

import sys, os, argparse
import numpy as np
import scipy as sp
import pandas as pd
import pickle, json
from pyshm import OSMOS, Tools, Stat, Models
from pyshm.script import static_data_analysis_template, examplestyle, warningstyle, load_result_of_analysis, MyEncoder, to_json
from joblib import Parallel, delayed

# import matplotlib
# # matplotlib.use("qt5agg")
# import matplotlib.pyplot as plt
# # import matplotlib.colors as colors
# import mpld3
# plt.style.use('ggplot')


def Hurstfunc(loc, X, mwsize, hrng):
    Y, *_ = Stat.Hurst(np.asarray(X), mwsize, sclrng=hrng, wvlname="haar")  # Hurst exponent
    return {loc: Y}

def compute_local_statistics(Yerr, mad, mwsize):
    """Compute the local statistics: mean and standard deviation and normalized
    observation.

    """
    Merr0 = {}; Serr0 = {}; Nerr0 = {}

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

# __warning__ = "Warning:" + warningstyle("\n ")

examples = []
examples.append(["%(prog)s OUTDIR/153/../Results.json -o", "Apply analysis with default parameters on the project of PID 153 (the project lied in the database directory DBDIR), plot the static data in a subfolder named figures/Static and print messages."])
__example__ = "Some examples of use (change the path seperator '/' to '\\' on Windows platform):" + "".join([examplestyle(x) for x in examples])


def main():
    # usage_msg = '%(prog)s <subcommand> <infile> <outdir> [options]'
    # parser = argparse.ArgumentParser(description=__script__, usage=usage_msg)
    mainparser = argparse.ArgumentParser(description=__script__,
                                         formatter_class=argparse.RawDescriptionHelpFormatter,
                                         epilog="\n\n" + __example__)#,add_help=False)
                                        #  epilog=__warning__ + "\n\n" + __example__)#,add_help=False)

    mainparser.add_argument('infile', type=str, help='input data file.')
    # mainparser.add_argument('outdir', nargs='?', type=str, default=None, help='directory where results (figures and data files) will be saved.')
    # mainparser.add_argument('outdir', type=str, default=None, help='directory where results (figures and data files) will be saved.')
    mainparser.add_argument('-v', '--verbose', dest='verbose', action='count', default=0, help='print messages.')
    mainparser.add_argument('-o', '--overwrite', dest='overwrite', action='store_true', default=False, help='overwrite on the input file.')

    proj_opts = mainparser.add_argument_group('Options for subspace projection')  # projection
    proj_opts.add_argument('--cdim', dest='cdim', type=int, default=None, help='dimension of the subspace (default=None), no subspace projection if set to 0, if given vthresh will be ignored ', metavar='integer')
    proj_opts.add_argument('--vthresh', dest='vthresh', type=float, default=0.9, help='relative threshold for subspace projection (default=0.9).', metavar='float')
    proj_opts.add_argument("--corrflag", dest="corrflag", action="store_true", default=False, help="use correlation matrix in subspace projection.")
    # proj_opts.add_argument('--drophead', dest='drophead', type=int, default=3*24*30, help='drop the begining of the input data in the computation of subspace projection (default=3*24*30).', metavar='integer')
    proj_opts.add_argument("--sidx", dest="sidx", type=int, default=0, help="starting time index (an integer) of the training data (default=0).", metavar="integer")
    proj_opts.add_argument("--Ntrn", dest="Ntrn", type=int, default=None, help="length of the training data (default=None, use all available data).", metavar="integer")

    lstat_opts = mainparser.add_argument_group('Options for local statistics')  # local statistics
    lstat_opts.add_argument('--mad', dest='mad', action='store_true', default=False, help='use median based estimator (default: use empirical estimator).')
    lstat_opts.add_argument('--mwsize', dest='mwsize', type=int, default=24*10, help='size of the moving window (default=240).', metavar='integer')
    lstat_opts.add_argument('--causal', dest='causal', action='store_true', default=False, help='use causal window (default: non causal).')

    hurst_opts = mainparser.add_argument_group('Options for the Hurst exponent (trend component only)')  # Hurst exponent
    hurst_opts.add_argument('--hrng', dest='hrng', nargs=2, type=int, default=(0,8), help='wavelet scale range index for computation of Hurst exponent (default=(0,8).', metavar='integer')
    hurst_opts.add_argument('--hwsize', dest='hwsize', type=int, default=24*10, help='size of the moving window for computation of Hurst exponent (default=240).', metavar='integer')

    options = mainparser.parse_args()

    # check the input data file
    if not os.path.isfile(options.infile):
        raise FileNotFoundError(options.infile)

    # if not os.path.isdir(options.outdir):
    #     raise FileNotFoundError(options.outdir)

    idx = options.infile.rfind(os.path.sep, 0)
    outdir = options.infile[:idx]

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
    Tidx = Yprd.index  # timestamp index
    alocs = list(Yprd.keys())  # list of active sensors
    # Xcpn = Res['Xcpn'][alocs]  # Res['Xcpn'] contains temperature data of all sensors
    Ycpn = Res['Ycpn'][alocs]  # Res['Ycpn'] contains elongation data of all sensors, extract only those having some prediction results
    Yerr = Res['Yerr']  # residual
    Midx = Res['Midx']  # indexes of missing data

    algo_options = dict(Res['algo_options'])  # options of parameters
    component = algo_options['component']  # name of the component being analyzed
    Emptydf = pd.DataFrame({loc: None for loc in alocs}, index=Tidx)  # an empty dataframe

    # projection of error onto a subspace
    if options.cdim is not None:
        options.cdim = min(len(alocs)-1, options.cdim)  # special case
    Yerr0 = np.asarray(Yerr).T
    # throw away the first (two) months
    Yssp0, (U,S), options.cdim = Models.ssproj(Yerr0, cdim=options.cdim, vthresh=options.vthresh, corrflag=options.corrflag, sidx=options.sidx, Ntrn=options.Ntrn)
    Yssp = pd.DataFrame(Yssp0.T, columns=alocs, index=Tidx)
    Yerp0 = Yerr0 - Yssp0  # error of projection, which will be used by analysis
    Yerp = pd.DataFrame(Yerp0.T, columns=alocs, index=Tidx)
    Scof = (U @ np.diag(np.sqrt(S/S[0])))  # Scof[:,:3] are the first 3 PCA coefficients

    # Local statistics
    if options.verbose:
        print('Computing local statistics...')
    Merr, Serr, Nerr = compute_local_statistics(Yerr, options.mad, options.mwsize)
    Merp, Serp, Nerp = compute_local_statistics(Yerp, options.mad, options.mwsize)

    # Hurst exponents and index ranges of instability period: defined only for trend or all component
    if component.upper() in ['TREND', 'ALL']:
        if options.verbose:
            print('Computing the Hurst exponent...')

        if component.upper() == "ALL":
            # Hurst exponent is computed for the local mean of the ALL component (in order to reduce the impact of the daily cycle)
            mYerp = Yerp.rolling(window=24, min_periods=1, center=True).mean() # .median() #.bfill()
        else:
            mYerp = Yerp  # no daily cycle in the TREND component

        Hexp0 = {}
        # parallel version:
        # htoto = Parallel(n_jobs=4)(delayed(Hurstfunc)(loc, mYerp[loc].diff(24*2), options.hwsize, options.hrng) for loc in alocs)
        htoto = Parallel(n_jobs=4)(delayed(Hurstfunc)(loc, mYerp[loc], options.hwsize, options.hrng) for loc in alocs)
        for h in htoto:
            Hexp0.update(h)
        # sequential version:
        # for loc in alocs:
        #     # yerr = yerr0.copy(); yerr.loc[Midx[loc]] = np.nan
        #     Hexp0[loc], *_ = Stat.Hurst(np.asarray(mYerp[loc]), options.mwsize, sclrng=options.hrng, wvlname="haar")  # Hurst exponent
        Hexp = pd.DataFrame(Hexp0, index=Tidx)
    else:
        Hexp = Emptydf.copy()

    outfile = options.infile if options.overwrite else os.path.join(outdir, 'Analysis') + '.pkl'
    Res.update({'Yssp':Yssp, 'cdim':options.cdim, 'Yerp':Yerp, 'Scof':Scof, 'Merp':Merp, 'Serp':Serp, 'Nerp':Nerp, 'Merr':Merr, 'Serr':Serr, 'Nerr':Nerr, 'Hexp':Hexp})

    with open(outfile, 'wb') as fp:
        pickle.dump(Res, fp)

    if options.verbose:
        print('Results saved in\n{}'.format(outfile))


if __name__ == "__main__":
    main()
