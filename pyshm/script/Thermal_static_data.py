#!/usr/bin/env python

"""Analysis of static data using the simple linear model with estimation of thermal delay.
"""

import sys, os, argparse
from pyshm.script import static_data_analysis_template, examplestyle, warningstyle


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
def Thermal_static_data(Xcpn, Ycpn, options):
    """Wrapper function of _Thermal_static_data for decorator functional.

    Args:
        infile (str): name of pickle file containing the preprocessed static data
        outdir0 (str): name of the directory for the output
        options (Options): instance containing the fields of the class Options
    Return:
        same as _Deconv_static_data.
    """
    return _Thermal_static_data(Xcpn, Ycpn, options)


def _Thermal_static_data(Xcpn, Ycpn, options):
    """
    Args:
        infile (str): name of pickle file containing the preprocessed static data
        outdir0 (str): name of the directory for the output
        options (Options): instance containing the fields of the class Options
    Return:
        a dictionary containing the following fields of estimation:
        'Delay': thermal delay
        'Correlation': correlation between temperature and elongation after compensation of delay
        'Slope' and 'Intercept': thermal coefficient
        'Yprd': prediction
        'Yerr': residual
    """

    from pyshm import Stat, Tools
    import numpy as np
    import pandas as pd

    # Dictionaries for the first estimation
    # D0 = {}  # thermal delay
    # C0 = {}  # correlation between temperature and elongation
    # K0 = {}  # slope or thermal expansion coefficient
    # B0 = {}  # intercept

    # Dictionaries estimation
    D1 = {}  # thermal delay
    C1 = {}  # correlation
    K1 = {}  # slope or thermal expansion coefficient
    B1 = {}  # intercept

    Yprd0 = {}  # final prediction from inputs
    Yerr0 = {}  # error of prediction

    Tidx = Xcpn.index  # time index
    Locations = list(Xcpn.keys())

    if options.alocs is None:  # sensors to be analyzed
        options.alocs = Locations.copy()  # is not given use all sensors

    if options.verbose:
        print('Analysis of thermal properties of the \'{}\' component...'.format(options.component.upper()))

    for aloc in options.alocs:
        if options.verbose:
            print('\tProcessing the location {}...'.format(aloc))

        # Data of observations, 1d arrays
        Xobs = np.asarray(Xcpn[aloc])
        Yobs = np.asarray(Ycpn[aloc])
        Xvar = np.asarray(Xcpn[aloc].diff())
        Yvar = np.asarray(Ycpn[aloc].diff())

        # Estimation of optimal delay
        # D0[aloc], C0[aloc], K0[aloc], B0[aloc] = Stat.mw_linear_regression_with_delay(Yvar, Xvar, D0=None, wsize=options.wsize, dlrange=options.dlrange)
        toto, *_ = Stat.mw_linear_regression_with_delay(Yvar, Xvar, D0=None, wsize=options.wsize, dlrange=options.dlrange)

        # Smoothing of estimation
        Ds = np.int32(Tools.LU_filter(toto, wsize=options.luwsize))  # smoothed estimation of delay
        Ds[np.isnan(Ds)] = 0  # convert nan to 0
        Ds[Ds>options.dlrange[1]] = 0
        Ds[Ds<options.dlrange[0]] = 0
        Dv = np.ones(len(Xobs), dtype=int)*np.int(np.mean(Ds))  # use the mean value as delay

        # Second regression with the smoothed delay
        _, C1[aloc], K1[aloc], B1[aloc] = Stat.mw_linear_regression_with_delay(Yvar, Xvar, D0=Dv, wsize=options.wsize, dlrange=options.dlrange)
        D1[aloc] = Ds.copy()
        # print(np.sum(~np.isnan(D0[aloc])))
        # print(D1[aloc])

        # Prediction and error
        # didx = np.int32(np.minimum(len(Xobs)-1, np.maximum(0, np.arange(len(Xobs)) + D1[aloc])))
        didx = np.int32(np.minimum(len(Xobs)-1, np.maximum(0, np.arange(len(Xobs)) + Dv)))
        Xobs_delayed = Xobs[didx]
        if options.const:
            Yprd0[aloc] = K1[aloc] * Xobs_delayed + B1[aloc] * np.arange(len(Xobs))
        else:
            Yprd0[aloc] = K1[aloc] * Xobs_delayed
        Yerr0[aloc] = Yobs - Yprd0[aloc]

    # Convert to pandas format
    # D_raw = pd.DataFrame(D0, columns=Ycpn.columns, index=Tidx)
    # C_raw = pd.DataFrame(C0, columns=Ycpn.columns, index=Tidx)
    D = pd.DataFrame(D1, columns=Ycpn.columns, index=Tidx)
    C = pd.DataFrame(C1, columns=Ycpn.columns, index=Tidx)
    K = pd.DataFrame(K1, columns=Ycpn.columns, index=Tidx)
    B = pd.DataFrame(B1, columns=Ycpn.columns, index=Tidx)
    Yprd = pd.DataFrame(Yprd0, columns=Ycpn.columns, index=Tidx)
    # Yerr = pd.DataFrame(Yerr0, columns=Ycpn.columns, index=Tidx)

    return {'Delay':D, 'Correlation':C, 'Slope':K, 'Intercept':B, 'Yprd':Yprd} #, 'Yerr':Yerr}

# __all__ = ['Thermal_static_data', 'Options']

__script__ = __doc__

__warning__ = "Warning:" + warningstyle("\n This script can be applied only on data preprocessed by the script osmos_preprocessing (the data file is typically named Preprocessed_static.pkl). Two distinct models (static and dynamic) are implemented and are accessible via the corresponding subcommand.")

examples = []
examples.append(["%(prog)s -h", "Print this help messages (about common parameters)"])
examples.append(["%(prog)s static -h", "Print help messages about the static model"])
examples.append(["%(prog)s dynamic -h", "Print help messages about the dynamic model"])
examples.append(["%(prog)s static DBDIR/153 OUTDIR/153 --alocs=754 --time0 2016-03-01 --time1 2016-08-01 -vv", "Apply the static model on the preprocessed data of the project of PID 153 for the period from 2016-03-01 to 2016-08-01, and save the results in the directory named OUTDIR/153. Process only the sensor of location 754 (--alocs=754), use the temperature of the same sensor to explain the elongation data (scalar model). Print supplementary messages."])
examples.append(["%(prog)s static DBDIR/153 OUTDIR/153 --alocs=754,755 --ylocs=0 -v", "Process the sensors of location 754 and 755, for each of them use the temperature of both to explain the elongation data (vectorial model)."])
examples.append(["%(prog)s static DBDIR/153 OUTDIR/153 --ylocs=0 -v", "Process all sensors, for each of them use the temperature  of all to explain the elongation data."])
examples.append(["%(prog)s static DBDIR/153 OUTDIR/153 -v", "Process all sensors, for each of them use the temperature data of all and the elongation data the others to explain the elongation data (deconvolution with multiple inputs)."])
examples.append(["%(prog)s static DBDIR/153 OUTDIR/153 --time0 2016-03-01 --Ntrn 1000 -v", "Change the length of the training period to 1000 hours starting from the begining of the truncated data set which is 2016-03-01."])
examples.append(["%(prog)s static DBDIR/153 OUTDIR/153 --component=seasonal -v", "Process the seasonal component of data."])
examples.append(["%(prog)s dynamic DBDIR/153 OUTDIR/153 -v", "Use the dynamic model (Kalman filter) to process all sensors."])

__example__ = "Some examples of use (change the path seperator '/' to '\\' on Windows platform):" + "".join([examplestyle(x) for x in examples])


def main():
    # usage_msg = '%(prog)s [options] <infile> [outdir]'
    # parser = argparse.ArgumentParser(description=__script__, usage=usage_msg)
    parser = argparse.ArgumentParser(description=__script__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=__warning__ + "\n\n" + __example__)

    parser.add_argument('projdir', help='directory of a project in the database including the preprocessed static data.')
    parser.add_argument('outdir', type=str, default=None, help='directory where results (figures and data files) are saved.')

    sensor_opts = parser.add_argument_group('Sensor options')
    sensor_opts.add_argument("--alocs", dest="alocs", type=lambda s: [int(x) for x in s.split(',')], default=None, help="Location key IDs of sensors to be analyzed (default=all sensors).", metavar="integers separated by \',\'")
    sensor_opts.add_argument("--component", dest="component", type=str, default="all", help="Type of component of data for analysis: all (default), trend, seasonal.", metavar="string")

    wdata_opts = parser.add_argument_group('Data truncation options')
    wdata_opts.add_argument('--time0', dest='time0', type=str, default=None, help='Starting timestamp (default=the begining of data set).', metavar='YYYY-MM-DD')
    wdata_opts.add_argument('--time1', dest='time1', type=str, default=None, help='Ending timestamp (default=the ending of data set).', metavar='YYYY-MM-DD')

    ddata_opts = parser.add_argument_group('Component decomposition options')
    ddata_opts.add_argument('--mwmethod', dest='mwmethod', type=str, default='mean', help='Type of moving window mean estimator for decomposition of component: mean (default), median.', metavar='string')
    ddata_opts.add_argument('--mwsize', dest='mwsize', type=int, default=24, help='Length of the moving window (default=24).', metavar='integer')
    ddata_opts.add_argument('--kzord', dest='kzord', type=int, default=1, help='Order of Kolmogorov-Zurbenko filter (default=1).', metavar='integer')

    model_opts = parser.add_argument_group('Model options')
    model_opts.add_argument('--wsize', dest='wsize', type=int, default=24*10, help='Length of the moving window for analysis of thermal coefficients (default=24*10).', metavar='integer')
    model_opts.add_argument('--luwsize', dest='luwsize', type=int, default=24*5, help='Length of the smoothing window for estimation of thermal delay (default=24*5).', metavar='integer')
    # model_opts.add_argument('--dlrange', dest='dlrange', type=tuple, default=(-6,6), help='Range of search for the optimal delay, default=(-6,6)', metavar='[integer, integer]')
    model_opts.add_argument('--dlrange', dest='dlrange', type=lambda s: [int(x) for x in s.split(',')], default=[-6,6], help='Range of search for the optimal delay, default=[-6,6]', metavar='\"integer,  integer\"')
    model_opts.add_argument('--const', dest='const', action='store_true', default=False, help='Add constant trend in the convolution model (default: no constant trend).')

    parser.add_argument('-v', '--verbose', dest='verbose', action='count', default=0, help='Print message.')

    options = parser.parse_args()

    # check the input data file
    options.infile = os.path.join(options.projdir, 'Preprocessed_static.pkl')
    if not os.path.isfile(options.infile):
        print('Preprocessed static data not found! Have you already run osmos_preprocess?')
        raise FileNotFoundError(options.infile)

    if not os.path.isdir(options.outdir):
        raise FileNotFoundError(options.outdir)

    # create a subfolder for output
    func_name = __name__[__name__.rfind('.')+1:]
    outdir = os.path.join(options.outdir, func_name, 'component[{}]_wsize[{}]_const[{}]_dlrange[{}]_luwsize[{}]_alocs[{}]'.format(options.component.upper(), options.wsize, options.const, options.dlrange, options.luwsize, options.alocs))
    try:
        os.makedirs(outdir)
    except OSError:
        pass

    outfile = os.path.join(outdir, 'Results')

    _ = Thermal_static_data(options.infile, outfile, options)

    # # final output file name
    # if options.alocs is None:
    #     outfile0 = 'Results_All'
    # else:
    #     outfile0 = 'Results_{}'.format(options.alocs).replace(' ', '')
    # outfile = os.path.join(outdir, outfile0)





    # if not os.path.isfile(options.infile):
    #     raise FileNotFoundError(options.infile)
    # # output directory
    # if options.outdir is None:
    #     idx2 = options.infile.rfind(os.path.sep, 0)
    #     idx1 = options.infile.rfind(os.path.sep, 0, idx2)
    #     idx0 = options.infile.rfind(os.path.sep, 0, idx1)
    #     outdir0 = os.path.join(options.infile[:idx0], 'Outputs', options.infile[idx1+1:idx2])
    # else:
    #     outdir0 = options.outdir

    # func_name = __name__[__name__.rfind('.')+1:]
    # outdir = os.path.join(outdir0,'{}_[{}_wsize={}_dlrange={}]'.format(func_name, options.component.upper(), options.wsize, tuple(options.dlrange)))
    # try:
    #     os.makedirs(outdir)
    # except OSError:
    #     pass


if __name__ == "__main__":
    main()
