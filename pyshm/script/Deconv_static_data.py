#!/usr/bin/env python

"""Analysis of static data using the vectorial deconvolution model.
"""

import sys, os, argparse
from pyshm.script import static_data_analysis_template, examplestyle, warningstyle


class Options:
    verbose=False  # print message
    info=False  # only print information about the project
    alocs=None  # locations of active sensors
    xlocs=None  # locations for x inputs
    ylocs=None  # locations for y inputs
    mwmethod='mean'  # method for computation of trend component
    mwsize=24  # size of moving window for computation of trend component
    kzord=1  # order of KZ filter
    component='Trend'  # name of the component for analysis, ['All', 'Seasonal', 'Trend']
    time0=None  # beginning of the data set, a string
    time1=None  # ending of the data set, a string
    lagx=12  # kernel length of x inputs
    lagy=6  # kernel length of y inputs
    dtx=0  # artificial delay in x inputs
    dty=0  # artificial delay in y inputs
    subcommand='static'  # type of deconvolution model

    # Static method only
    Ntrn=3*30*24  # length of training data
    sidx=0  # beginning index (relative to tidx0) of training data
    # Dynamic method only
    sigmaq2=1e-6
    sigmar2=1e-4
    kalman='smoother'

# def get_locs(locs):
#     return [int(s) for s in locs.split(',')]


@static_data_analysis_template
def Deconv_static_data(Xcpn, Ycpn, options):
    """Wrapper function of _Deconv_static_data for decorator functional.

    Args:
        infile (str): name of pickle file containing the preprocessed static data
        outdir0 (str): name of the directory for the output
        options (Options): instance containing the fields of the class Options
    Return:
        same as _Deconv_static_data.
    """
    return _Deconv_static_data(Xcpn, Ycpn, options)


def _Deconv_static_data(Xcpn, Ycpn, options):
    """Deconvolution of static data.

    Args:
        Xcpn (pandas DataFrame): X input
        Ycpn (pandas DataFrame): Y input
        options (Options): instance containing the fields of the class Options
    Return:
        a dictionary containing the following fields:
            Yprd: final prediction from inputs
            Aprd: contribution of the first group of inputs
            Bprd: contribution of the second group of inputs, if exists
            Yerr: error of prediction
            Knel: kernel matrices
            Mxd: objects of deconvolution model
    """

    from pyshm import Models
    import numpy as np
    import pandas as pd

    Yprd0 = {}  # final prediction from inputs
    Aprd0 = {}  # contribution of the first group of inputs
    Bprd0 = {}  # contribution of the second group of inputs, if exists
    Yerr0 = {}  # error of prediction
    Knel = {}  # kernel matrices
    Mxd = {}  # objects of deconvolution model

    Tidx = Xcpn.index  # time index
    Locations = list(Xcpn.keys())  # all sensors

    if options.alocs is None:  # sensors to be analyzed
        options.alocs = Locations.copy()  # is not given use all sensors

    # modify the default value of the active sensor set
    if options.xlocs is None or len(options.xlocs)==0:
        # temperature sensors not given: use alocs
        options.xlocs = options.alocs.copy()
    else:  # keep valid sensors
        options.xlocs = list(filter(lambda x:x in Locations, options.xlocs))
    if options.ylocs is None or len(options.ylocs)==0:
        # elongation sensors not given: use xlocs
        options.ylocs = options.xlocs.copy()
    else:  # keep valid sensors
        options.ylocs = list(filter(lambda x:x in Locations, options.ylocs))
    if options.verbose > 1:
        print('Active sensors for temperature: {}\nActive sensors for elongation: {}\n'.format(options.xlocs, options.ylocs))

    # compute valid values for training period
    if options.subcommand.upper() == 'STATIC':
        options.trnperiod, options.Ntrn = Models.MxDeconv._training_period(len(Xcpn), tidx0=options.sidx, Ntrn=options.Ntrn)

    if options.verbose:
        print('Deconvolution of the \'{}\' component...'.format(options.component.upper()))
        if options.verbose > 1 and options.subcommand.upper() == 'STATIC':
            print('Training period: from {} to {}, around {} days.'.format(Tidx[options.trnperiod[0]], Tidx[options.trnperiod[1]-1], int((options.trnperiod[1]-options.trnperiod[0])/24)))

    for aloc in options.alocs:
        if options.verbose:
            print('   Processing the location {}...'.format(aloc))

        # Sensors of temperature contribution
        xlocs = options.xlocs.copy()
        # Sensors of elongation contribution
        ylocs = options.ylocs.copy()
        if aloc in ylocs:
            ylocs.remove(aloc)

        # Data of observations
        # Xobs = np.asarray(Xcpn[[aloc]]).T
        Yobs = np.asarray(Ycpn[[aloc]]).T

        # Data for training and prediction
        Xvar = np.asarray(Xcpn[xlocs]).T
        Yvar = np.asarray(Ycpn[ylocs]).T if len(ylocs)>0 else None
        # print(Xvar.shape, len(ylocs))

        # Apply delay to avoid over-fitting
        Xvar = np.roll(Xvar, options.dtx, axis=1); Xvar[:,:options.dtx] = np.nan
        if Yvar is not None:
            Yvar = np.roll(Yvar, options.dty, axis=1); Yvar[:,:options.dty] = np.nan

        # Construction of the deconvolution model
        if options.lagx>0 and len(xlocs)>0 and options.lagy>0 and len(ylocs)>0:  # Full model
            if options.subcommand.upper() == 'STATIC':  # static model
                mxd = Models.DiffDeconv(Yobs, Xvar, options.lagx, Yvar, options.lagy)
            else:  # dynamic model
                mxd = Models.DiffDeconvBM(Yobs, Xvar, options.lagx, Yvar, options.lagy)
        elif options.lagx>0 and len(xlocs)>0:  # Half model, X input only
            if options.subcommand.upper() == 'STATIC':  # static model
                mxd = Models.DiffDeconv(Yobs, Xvar, options.lagx)
            else:  # dynamic model
                mxd = Models.DiffDeconvBM(Yobs, Xvar, options.lagx)
        elif options.lagy>0 and len(ylocs)>0:  # Half model, X input only
            if options.subcommand.upper() == 'STATIC':  # static model
                mxd = Models.DiffDeconv(Yobs, Yvar, options.lagy)
            else:  # dynamic model
                mxd = Models.DiffDeconvBM(Yobs, Yvar, options.lagy)
        else:
            raise ValueError('Incompatible set of parameters.')

        # Model fitting and prediction
        if options.subcommand.upper() == 'STATIC':
            res_fit = mxd.fit(constflag=False, tidx0=options.sidx, Ntrn=options.Ntrn)
            res_predict = mxd.predict(Xvar, Yvar)
            kmat = mxd._As  # kernel matrices
        else:
            res_fit = mxd.fit(sigmaq2=options.sigmaq2, sigmar2=options.sigmar2, x0=0, p0=1)
            res_predict, kmat, _ = mxd.predict(smooth=options.kalman.upper()=='SMOOTHER')

        # The last [0] is for taking the first row (which is also the only row)
        Yprd0[aloc] = res_predict[0][0]
        Aprd0[aloc] = res_predict[1][0][0]
        if len(res_predict[1]) > 1:
            Bprd0[aloc] = res_predict[1][1][0]
        Yerr0[aloc] = Yobs[0] - Yprd0[aloc]
        Knel[aloc] = kmat
        Mxd[aloc] = mxd

    Yprd = pd.DataFrame(Yprd0, columns=options.alocs, index=Tidx)
    Yerr = pd.DataFrame(Yerr0, columns=options.alocs, index=Tidx)
    Aprd = pd.DataFrame(Aprd0, columns=options.alocs, index=Tidx)
    Bprd = pd.DataFrame(Bprd0, columns=options.alocs, index=Tidx) if len(Bprd0)>0 else None

    return {'Yprd':Yprd, 'Yerr':Yerr, 'Aprd':Aprd, 'Bprd':Bprd, 'Knel':Knel, 'Mxd':Mxd}


# __all__ = ['Deconv_static_data', 'Options']

__script__ = __doc__

__warning__ = "Warning:" + warningstyle("\n  This script can be applied only on data preprocessed by the script osmos_preprocessing (which is typically named Preprocessed_static.pkl). Two distinct models (static and dynamic) are implemented and accessible via the corresponding subcommand.")

examples = []
examples.append(["%(prog)s -h", "Print this help messages (about common parameters)"])
examples.append(["%(prog)s static -h", "Print help messages about the static model"])
examples.append(["%(prog)s dynamic -h", "Print help messages about the dynamic model"])
examples.append(["%(prog)s static DBDIR/153/Preprocessed_static.pkl OUTDIR/153 --alocs=754 --time0 2016-03-01 --time1 2016-08-01 -vv", "Apply the static model on the preprocessed data of the project of PID 153 for the period from 2016-03-01 to 2016-08-01, and save the results in the directory named OUTDIR/153. Process only the sensor of location 754 (--alocs=754), use the temperature of the same sensor to explain the elongation data (scalar model). Print supplementary messages."])
examples.append(["%(prog)s static DBDIR/153/Preprocessed_static.pkl OUTDIR/153 --alocs=754,755 --ylocs=0 -v", "Process the sensors of location 754 and 755, for each of them use the temperature of both to explain the elongation data (vectorial model)."])
examples.append(["%(prog)s static DBDIR/153/Preprocessed_static.pkl OUTDIR/153 --ylocs=0 -v", "Process all sensors, for each of them use the temperature  of all to explain the elongation data."])
examples.append(["%(prog)s static DBDIR/153/Preprocessed_static.pkl OUTDIR/153 -v", "Process all sensors, for each of them use the temperature data of all and the elongation data the others to explain the elongation data (deconvolution with multiple inputs)."])
examples.append(["%(prog)s static DBDIR/153/Preprocessed_static.pkl OUTDIR/153 --time0 2016-03-01 --Ntrn 1000 -v", "Change the length of the training period to 1000 hours starting from the begining of the truncated data set which is 2016-03-01."])
examples.append(["%(prog)s static DBDIR/153/Preprocessed_static.pkl OUTDIR/153 --component=seasonal -v", "Process the seasonal component of data."])
examples.append(["%(prog)s dynamic DBDIR/153/Preprocessed_static.pkl OUTDIR/153 -v", "Use the dynamic model (Kalman filter) to process all sensors."])

__example__ = "Some examples of use (change the path seperator '/' to '\\' on Windows platform):" + "".join([examplestyle(x) for x in examples])

def main():
    usage_msg = '%(prog)s <subcommand> <infile> <outdir> [options]'
    # parser = argparse.ArgumentParser(description=__script__, usage=usage_msg)
    parser = argparse.ArgumentParser(description=__script__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=__warning__ + "\n\n" + __example__)

    subparsers = parser.add_subparsers(title='subcommands', description='method of deconvolution')
    parser_static = subparsers.add_parser('static', help='Deconvolution using static model')
    parser_dynamic = subparsers.add_parser('dynamic', help='Deconvolution using dynamic model')

    parser.add_argument('infile', type=str, help='preprocessed data file containing all sensors of one project.')
    # parser.add_argument('outdir', nargs='?', type=str, default=None, help='directory where results (figures and data files) will be saved.')
    parser.add_argument('outdir', type=str, help='directory where results (figures and data files) will be saved.')

    sensor_opts = parser.add_argument_group('Sensor options')
    sensor_opts.add_argument('--alocs', dest='alocs', type=lambda s: [int(x) for x in s.split(',')], default=None, help='Location key IDs of sensors to be analyzed (default=all sensors).', metavar='integers separated by \',\'')
    sensor_opts.add_argument('--xlocs', dest='xlocs', type=lambda s: [int(x) for x in s.split(',')], default=None, help='Location key IDs of active temperature sensors (default=all sensors).', metavar='integers separated by \',\'')
    sensor_opts.add_argument('--ylocs', dest='ylocs', type=lambda s: [int(x) for x in s.split(',')], default=None, help='Location key IDs of active elongation sensors (default=same as xlocs).', metavar='integers separated by \',\'')
    sensor_opts.add_argument('--component', dest='component', type=str, default='Trend', help='Type of component of data for analysis: Trend (default), Seasonal, All.', metavar='string')

    wdata_opts = parser.add_argument_group('Data truncation options')
    wdata_opts.add_argument('--time0', dest='time0', type=str, default=None, help='Starting timestamp (default=the begining of data set).', metavar='YYYY-MM-DD')
    wdata_opts.add_argument('--time1', dest='time1', type=str, default=None, help='Ending timestamp (default=the ending of data set).', metavar='YYYY-MM-DD')

    ddata_opts = parser.add_argument_group('Component decomposition options')
    ddata_opts.add_argument('--mwmethod', dest='mwmethod', type=str, default='mean', help='Type of moving window mean estimator for decomposition of component: mean (default), median.', metavar='string')
    ddata_opts.add_argument('--mwsize', dest='mwsize', type=int, default=24, help='Length of the moving window (default=24).', metavar='integer')
    ddata_opts.add_argument('--kzord', dest='kzord', type=int, default=1, help='Order of Kolmogorov-Zurbenko filter (default=1).', metavar='integer')

    model_opts = parser.add_argument_group('Model options')
    # model_opts.add_argument('--const', dest='const', action='store_true', default=False, help='Add constant trend in the convolution model (default: no constant trend).')
    model_opts.add_argument('--lagx', dest='lagx', type=int, default=12, help='Length of the convolution kernel of temperature (default=12). It will be desactivated if set to 0.', metavar='integer')
    model_opts.add_argument('--lagy', dest='lagy', type=int, default=6, help='Length of the convolution kernel of elongation (default=6). It will be desactivated if set to 0.', metavar='integer')
    model_opts.add_argument('--dtx', dest='dtx', type=int, default=0, help='Artificial delay (in hours) applied on the temperature data to avoid over-fitting (default=0).', metavar='integer')
    model_opts.add_argument('--dty', dest='dty', type=int, default=0, help='Artificial delay (in hours) applied on the elongation data to avoid over-fitting (default=0).', metavar='integer')

    parser.add_argument('-v', '--verbose', dest='verbose', action='count', default=0, help='Print messages.')
    # parser.add_argument('--info', dest='info', action='store_true', default=False, help='Print only information about the project.')

    # tdata_opts = parser_static.add_argument_group('Training data options for static method')
    parser_static.add_argument('--sidx', dest='sidx', type=int, default=0, help='starting time index (an integer) of the training data relative to time0 (default=0).', metavar='integer')
    parser_static.add_argument('--Ntrn', dest='Ntrn', type=int, default=3*30*24, help='Length of the training data (default=24*30*3).', metavar='integer')

    # kalman_opts = parser_dynamic.add_argument_group('Kalman filter options for dynamic method')
    parser_dynamic.add_argument('--sigmaq2', dest='sigmaq2', type=float, default=1e-6, help='Variance of transition noise (default=1e-6).', metavar='float')
    parser_dynamic.add_argument('--sigmar2', dest='sigmar2', type=float, default=1e-4, help='Variance of observation noise (default=1e-4).', metavar='float')
    parser_dynamic.add_argument('--kalman', dest='kalman', type=str, default='smoother', help='Method of estimation of Kalman filter: filter, smoother (default).', metavar='string')

    options = parser.parse_args()
    options.subcommand = sys.argv[1]  # name of the subcommand: method of deconvolution  <--- TODO: get the subcommand name

    # check the input data file
    if not os.path.isfile(options.infile):
        raise FileNotFoundError(options.infile)

    if not os.path.isdir(options.outdir):
        raise FileNotFoundError(options.outdir)
    # if options.outdir is None:
    #     idx2 = options.infile.rfind(os.path.sep, 0)
    #     idx1 = options.infile.rfind(os.path.sep, 0, idx2)
    #     idx0 = options.infile.rfind(os.path.sep, 0, idx1)
    #     outdir0 = os.path.join(options.infile[:idx0], 'Outputs', options.infile[idx1+1:idx2])

    # create a subfolder for output
    func_name = __name__[__name__.rfind('.')+1:]
    outdir = os.path.join(options.outdir, '{}_[{}_{}]'.format(func_name, options.subcommand.upper(), options.component.upper()))
    # outdir = os.path.join(options.outdir, '{}_[{}_{}_lagx={}_lagy={}_timerange=({},{})]'.format(func_name, options.subcommand.upper(), options.component.upper(), options.lagx, options.lagy, options.timerange[0], options.timerange[1]))
    try:
        os.makedirs(outdir)
    except OSError:
        pass

    # final output file name
    if options.alocs is None:
        outfile0 = 'Results_All'
    else:
        outfile0 = 'Results_{}'.format(options.alocs).replace(' ', '')
    outfile = os.path.join(outdir, outfile0)

    # No handle of exceptions so that any exception will result a system exit in the terminal.
    _ = Deconv_static_data(options.infile, outfile, options)

if __name__ == "__main__":
    main()
