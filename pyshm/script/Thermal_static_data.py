#!/usr/bin/env python

"""Analysis of static data using the simple linear model with estimation of thermal delay.
"""

import sys, os, argparse
from pyshm.script import static_data_analysis_template, examplestyle, warningstyle


class Options:
    verbose=False  # print message
    info=False  # only print information about the project
    mwmethod="mean"  # method for computation of trend component
    mwsize=24  # size of moving window for computation of trend component
    kzord=1  # order of KZ filter
    component="Trend"  # name of the component for analysis, ["All", "Seasonal", "Trend"]
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

    # Dictionaries estimation
    D = {}  # thermal delay
    A = {}  # slope or thermal expansion coefficient
    B = {}  # intercept

    Yprd0 = {}  # final prediction from inputs
    # Yerr0 = {}  # error of prediction

    Tidx = Xcpn.index  # time index
    Locations = list(Xcpn.keys())

    if options.alocs is None:  # sensors to be analyzed
        options.alocs = Locations.copy()  # is not given use all sensors

    if options.verbose:
        print("Analysis of thermal properties of the \'{}\' component...".format(options.component.upper()))

    for aloc in options.alocs:
        if options.verbose:
            print("\tProcessing the location {}...".format(aloc))

        # training period and data
        xobs, yobs = Xcpn[aloc], Ycpn[aloc]
        (tidx0, tidx1), Ntrn = Stat.training_period(len(Tidx), tidx0=options.sidx, Ntrn=options.Ntrn)
        xtrn, ytrn = xobs.iloc[tidx0:tidx1], yobs.iloc[tidx0:tidx1]
        # update options
        options.trnperiod = (tidx0, tidx1)
        options.Ntrn = Ntrn
        
        # Estimation of delay: use the differentials, not the original values
        delay = Stat.global_thermal_delay(xtrn.diff(options.dstep), ytrn.diff(options.dstep), dlrng=options.dlrng)
        # Estimation of thermal coefficients given the thermal delay
        a, b, _ = Stat.estimate_thermal_coefficient(xtrn, ytrn, delay=delay, dstep=options.dstep, robust=options.robust, shrink=options.shrink)

        if options.verbose>1:
            print("\t\tthermal delay = {}, thermal coefficient = {:.3f}".format(delay, a))

        # Prediction
        Yprd0[aloc] = a * Tools.roll_fill(xobs, delay)
        D[aloc] = delay
        A[aloc] = a
        B[aloc] = b

    return {'Delay':D, 'Slope':A, 'Intercept':B, 'Yprd':pd.DataFrame(Yprd0, index=Tidx)}

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
    mainparser = argparse.ArgumentParser(description=__script__,
                                             formatter_class=argparse.RawDescriptionHelpFormatter,
                                             epilog=__warning__ + "\n\n" + __example__)

    mainparser.add_argument("projdir", help="directory of a project in the database including the preprocessed static data.")
    mainparser.add_argument("outdir", type=str, default=None, help="directory where results are saved.")

    sensor_opts = mainparser.add_argument_group("Sensor options")
    sensor_opts.add_argument("--alocs", dest="alocs", type=lambda s: [int(x) for x in s.split(',')], default=None, help="Location key IDs of sensors to be analyzed (default=all sensors).", metavar="integers separated by \',\'")
    sensor_opts.add_argument("--component", dest="component", type=str, default="trend", help="Type of component of data for analysis: all (default), trend, seasonal.", metavar="string")

    wdata_opts = mainparser.add_argument_group("Data truncation options")
    wdata_opts.add_argument("--time0", dest="time0", type=str, default=None, help="Starting timestamp (default=the begining of data set).", metavar="YYYY-MM-DD")
    wdata_opts.add_argument("--time1", dest="time1", type=str, default=None, help="Ending timestamp (default=the ending of data set).", metavar="YYYY-MM-DD")

    ddata_opts = mainparser.add_argument_group("Component decomposition options")
    ddata_opts.add_argument("--mwmethod", dest="mwmethod", type=str, default="mean", help="Type of moving window mean estimator for decomposition of component: mean (default), median.", metavar="string")
    ddata_opts.add_argument("--mwsize", dest="mwsize", type=int, default=24, help="Length of the moving window (default=24).", metavar="integer")
    ddata_opts.add_argument("--kzord", dest="kzord", type=int, default=1, help="Order of Kolmogorov-Zurbenko filter (default=1).", metavar="integer")

    tdata_opts = mainparser.add_argument_group("Training data options")
    tdata_opts.add_argument("--sidx", dest="sidx", type=int, default=0, help="starting time index (an integer) of the training data relative to time0 (default=0).", metavar="integer")
    tdata_opts.add_argument("--Ntrn", dest="Ntrn", type=int, default=3*30*24, help="Length of the training data (default=24*30*3).", metavar="integer")

    model_opts = mainparser.add_argument_group("Model options")
    model_opts.add_argument('--dlrng', dest='dlrng', nargs=2, type=int, default=(-6,6), help='Range of search for the optimal delay, default=[-6,6].', metavar='integer')
    # model_opts.add_argument("--dlrng", dest="dlrng", type=lambda s: [int(x) for x in s.split(',')], default=[-6,6], help="Range of search for the optimal delay, default=[-6,6]", metavar="integer, integer")
    model_opts.add_argument("--dstep", dest="dstep", type=int, default=2, help="Step for calculation of differential (default=2).", metavar="integer")
    model_opts.add_argument("--shrink", dest="shrink", type=float, default=3*10**-3, help="Threshold value of thermal coefficient (default=3*10**-3).", metavar="float")
    model_opts.add_argument("--robust", dest="robust", action="store_true", default=False, help="Use robust linear regression.")
    # model_opts.add_argument('--luwsize', dest='luwsize', type=int, default=24*5, help='Length of the smoothing window for estimation of thermal delay (default=24*5).', metavar='integer')
    # model_opts.add_argument('--globaldelay', dest='const', action='store_true', default=False, help='Add constant trend in the convolution model (default: no constant trend).')
    # model_opts.add_argument('--global', dest='const', action='store_true', default=False, help='Add constant trend in the convolution model (default: no constant trend).')
    # model_opts.add_argument('--dlrange', dest='dlrange', type=tuple, default=(-6,6), help='Range of search for the optimal delay, default=(-6,6)', metavar='[integer, integer]')
    # model_opts.add_argument('--dlrange', dest='dlrange', type=lambda s: [int(x) for x in s.split(',')], default=[-6,6], help='Range of search for the optimal delay, default=[-6,6]', metavar='\"integer,  integer\"')

    mainparser.add_argument("-v", "--verbose", dest="verbose", action="count", default=0, help="Print message.")

    options = mainparser.parse_args()

    # check the input data file
    options.infile = os.path.join(options.projdir, "Preprocessed_static.pkl")
    if not os.path.isfile(options.infile):
        print("Preprocessed static data not found.")
        raise FileNotFoundError(options.infile)

    if not os.path.isdir(options.outdir):
        raise FileNotFoundError(options.outdir)

    # create a subfolder for output
    options.func_name = __name__[__name__.rfind('.')+1:]
    outdir = os.path.join(options.outdir, options.func_name, "component[{}]_alocs[{}]_robust[{}]".format(options.component.upper(), options.alocs, options.robust))
    try:
        os.makedirs(outdir)
    except OSError:
        pass

    outfile = os.path.join(outdir, "Results")

    _ = Thermal_static_data(options.infile, outfile, options)


if __name__ == "__main__":
    main()
