#!/usr/bin/env python

"""Analysis of static data using the simple linear model with estimation of thermal delay.
"""

from pyshm.script import static_data_analysis_template

class Options:
    verbose=False  # print message
    info=False  # only print information about the project
    mwmethod='mean'  # method for computation of trend component
    mwsize=24  # size of moving window for computation of trend component
    kzord=1  # order of KZ filter
    component='Trend'  # name of the component for analysis, ['All', 'Seasonal', 'Trend']
    timerange=[None,None]  # beginning of the data set, a string
    # constflag=False


@static_data_analysis_template
def Thermal_static_data(options, Xcpn, Ycpn):
    """
    Args:
        infile (str): name of pickle file containing the preprocessed static data
        outdir0 (str): name of the directory for the output
        options (Options): instance containing the fields of the class Options
    Return:
        a dictionary containing the following fields:
        Yprd: final prediction from inputs
        Aprd: contribution of the first group of inputs
        Bprd: contribution of the second group of inputs, if exists
        Yerr: error of prediction
        Mxd: objects of deconvolution model
    """

    from pyshm import Models, Stat, Tools
    import numpy as np
    import pandas as pd

    D0 = {}  # thermal delay
    C0 = {}  # correlation
    K0 = {}  # slope or thermal expansion coefficient
    B0 = {}  # intercept
    D1 = {}  # thermal delay
    C1 = {}  # correlation
    K1 = {}  # slope or thermal expansion coefficient
    B1 = {}  # intercept
    Yprd0 = {}  # final prediction from inputs
    Yerr0 = {}  # error of prediction

    Tidx = Xcpn.index  # time index
    Locations = list(Xcpn.keys())

    if options.verbose:
        print('Analysis of the \'{}\' component...'.format(options.component.upper()))

    for aloc in Locations:
        if options.verbose:
            print('   Processing the location {}...'.format(aloc))

        # Data of observations, 1d arrays
        Xobs = np.asarray(Xcpn[aloc].diff())
        Yobs = np.asarray(Ycpn[aloc].diff())

        # Estimation of optimal delay
        D0[aloc], C0[aloc], K0[aloc], B0[aloc] = Stat.mw_linear_regression_with_delay(Yobs, Xobs, wsize=options.wsize, dlrange=options.dlrange)

        # Smoothing of estimation
        toto = Tools.LU_filter(D0[aloc], wsize=options.luwsize) # wsize ranges from 3*24 to 10*24
        toto[np.isnan(toto)]=0
        D1[aloc] = np.int32(toto)

        # Second regression with the smoothed estimation
        D1[aloc], C1[aloc], K1[aloc], B1[aloc] = Stat.mw_linear_regression_with_delay(Yobs, Xobs, D0=D0[aloc], wsize=options.wsize, dlrange=options.dlrange)

        # Prediction and error
        Yprd0[aloc] = K1[aloc] * Xobs + B1[aloc]
        Yerr0[aloc] = Yobs - Yprd0[aloc]

    # D_raw = pd.DataFrame(D0, columns=Ycpn.columns, index=Tidx)
    # C_raw = pd.DataFrame(C0, columns=Ycpn.columns, index=Tidx)
    D = pd.DataFrame(D1, columns=Ycpn.columns, index=Tidx)
    C = pd.DataFrame(C1, columns=Ycpn.columns, index=Tidx)
    K = pd.DataFrame(K1, columns=Ycpn.columns, index=Tidx)
    B = pd.DataFrame(B1, columns=Ycpn.columns, index=Tidx)
    Yprd = pd.DataFrame(Yprd0, columns=Ycpn.columns, index=Tidx)
    Yerr = pd.DataFrame(Yerr0, columns=Ycpn.columns, index=Tidx)

    return {'Delay':D, 'Corr':C, 'Slope':K, 'Intercept':B, 'Yprd':Yprd, 'Yerr':Yerr}

# __all__ = ['Thermal_static_data', 'Options']

__script__ = __doc__


import sys, os, argparse

def main():
    usage_msg = '%(prog)s [options] <infile> [outdir]'

    parser = argparse.ArgumentParser(description=__script__, usage=usage_msg)

    parser.add_argument('infile', type=str, help='preprocessed data file containing all sensors of one project (see the script Preprocessing_of_data.py)')
    parser.add_argument('outdir', nargs='?', type=str, default=None, help='directory where results (figures and data files) are saved.')

    sensor_opts = parser.add_argument_group('Sensor options')
    sensor_opts.add_argument('--component', dest='component', type=str, default='Trend', help='Type of component of data for analysis: Trend (default), Seasonal, All.', metavar='string')

    wdata_opts = parser.add_argument_group('Data truncation options')
    wdata_opts.add_argument('--timerange', dest='timerange', nargs=2, type=str, default=[None,None], help='Starting and ending timestamp index (default=the whole data set).', metavar='YYYY-MM-DD')

    ddata_opts = parser.add_argument_group('Component decomposition options')
    ddata_opts.add_argument('--mwmethod', dest='mwmethod', type=str, default='mean', help='Type of moving window mean estimator for decomposition of component: mean (default), median.', metavar='string')
    ddata_opts.add_argument('--mwsize', dest='mwsize', type=int, default=24, help='Length of the moving window (default=24).', metavar='integer')
    ddata_opts.add_argument('--kzord', dest='kzord', type=int, default=1, help='Order of Kolmogorov-Zurbenko filter (default=1).', metavar='integer')

    model_opts = parser.add_argument_group('Model options')
    model_opts.add_argument('--wsize', dest='wsize', type=int, default=24*10, help='Length of the moving window for analysis of thermal coefficients (default=24*10).', metavar='integer')
    model_opts.add_argument('--luwsize', dest='luwsize', type=int, default=24*5, help='Length of the smoothing window for estimation of thermal delay (default=24*10).', metavar='integer')
    # model_opts.add_argument('--dlrange', dest='dlrange', type=tuple, default=(-6,6), help='Range of search for the optimal delay, default=(-6,6)', metavar='[integer, integer]')
    model_opts.add_argument('--dlrange', dest='dlrange', type=lambda s: [int(x) for x in s.split(',')], default=[-6,6], help='Range of search for the optimal delay, default=[-6,6]', metavar='\"integer,  integer\"')

    parser.add_argument('-v', '--verbose', dest='verbose', action='count', default=0, help='Print message.')
    parser.add_argument('--info', dest='info', action='store_true', default=False, help='Print only information about the project.')

    options = parser.parse_args()

    if not os.path.isfile(options.infile):
        raise FileNotFoundError(options.infile)
    # output directory
    if options.outdir is None:
        idx2 = options.infile.rfind(os.path.sep, 0)
        idx1 = options.infile.rfind(os.path.sep, 0, idx2)
        idx0 = options.infile.rfind(os.path.sep, 0, idx1)
        outdir0 = os.path.join(options.infile[:idx0], 'Outputs', options.infile[idx1+1:idx2])
    else:
        outdir0 = options.outdir

    func_name = __name__[__name__.rfind('.')+1:]
    outdir = os.path.join(outdir0,'{}_[{}_wsize={}_dlrange={}]'.format(func_name, options.component.upper(), options.wsize, tuple(options.dlrange)))
    try:
        os.makedirs(outdir)
    except OSError:
        pass
    outfile = os.path.join(outdir, 'Results.pkl')

    _ = Thermal_static_data(options.infile, outfile, options)


if __name__ == "__main__":
    main()
