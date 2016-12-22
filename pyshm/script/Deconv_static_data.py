#!/usr/bin/env python

"""Analysis of static data using the vectorial deconvolution model.
"""

from pyshm.script import static_data_analysis_template

class Options:
    verbose=False  # print message
    info=False  # only print information about the project
    xlocs=[]  # locations for x inputs
    ylocs=[]  # locations for y inputs
    mwmethod='mean'  # method for computation of trend component
    mwsize=24  # size of moving window for computation of trend component
    kzord=1  # order of KZ filter
    component='Trend'  # name of the component for analysis, ['All', 'Seasonal', 'Trend']
    timerange=[None,None]  # beginning of the data set, a string
    lagx=12  # kernel length of x inputs
    lagy=6  # kernel length of y inputs
    dtx=0  # artificial delay in x inputs
    dty=0  # artificial delay in y inputs
    Ntrn=3*30*24  # length of training data
    sidx=0  # beginning index (relative to tidx0) of training data
    # constflag=False


# def get_locs(locs):
#     return [int(s) for s in locs.split(',')]


@static_data_analysis_template
def Deconv_static_data(options, Xcpn, Ycpn):
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

    from pyshm import Models
    import numpy as np
    import pandas as pd

    Yprd0 = {}  # final prediction from inputs
    Aprd0 = {}  # contribution of the first group of inputs
    Bprd0 = {}  # contribution of the second group of inputs, if exists
    Yerr0 = {}  # error of prediction
    Mxd = {}  # objects of deconvolution model

    Tidx = Xcpn.index  # time index
    Locations = list(Xcpn.keys())  # location id of sensors

    # modify the default value of active sensor set
    if options.xlocs is None or len(options.xlocs)==0:
        options.xlocs = Locations.copy()
    if options.ylocs is None or len(options.ylocs)==0:
        options.ylocs = options.xlocs.copy()
    if options.verbose>1:
        print('Active sensors for temperature: {}\nActive sensors for elongation: {}\n'.format(options.xlocs, options.ylocs))

    # compute valid values for training period
    options.trn_idx, options.Ntrn = Models.MxDeconv._training_period(len(Xcpn), tidx0=options.sidx, Ntrn=options.Ntrn)

    if options.verbose:
        print('Deconvolution of the \'{}\' component...'.format(options.component.upper()))
        if options.verbose>1:
            print('Training period: from {} to {}, around {} days.'.format(Tidx[options.trn_idx[0]], Tidx[options.trn_idx[1]-1], int((options.trn_idx[1]-options.trn_idx[0])/24)))

    for aloc in Locations:
        if options.verbose:
            print('   Processing the location {}...'.format(aloc))

        # Sensors of temperature contribution
        xlocs = options.xlocs.copy()  # use all temperatures
        # Sensors of elongation contribution
        ylocs = options.ylocs.copy(); ylocs.remove(aloc)

        # Data of observations
        # Xobs = np.asarray(Xcpn[[aloc]]).T
        Yobs = np.asarray(Ycpn[[aloc]]).T

        # Data for training and prediction
        Xvar = np.asarray(Xcpn[xlocs]).T
        Yvar = np.asarray(Ycpn[ylocs]).T

        # Apply delay to avoid over-fitting
        Xvar = np.roll(Xvar, options.dtx, axis=1); Xvar[:,:options.dtx] = np.nan
        Yvar = np.roll(Yvar, options.dty, axis=1); Yvar[:,:options.dty] = np.nan

        # Deconvolution model
        if options.lagx>0 and options.lagy>0:
            mxd = Models.DiffDeconv(Yobs, Xvar, options.lagx, Yvar, options.lagy)
        elif options.lagx>0:
            mxd = Models.DiffDeconv(Yobs, Xvar, options.lagx)
        elif options.lagy>0:
            mxd = Models.DiffDeconv(Yobs, Yvar, options.lagy)
        else:
            raise ValueError('lagx and lagy can not be both 0.')

        # Model fitting and prediction
        res_fit = mxd.fit(constflag=False, tidx0=options.sidx, Ntrn=options.Ntrn)
        res_predict = mxd.predict(Xvar, Yvar)

        # The last [0] is for taking the first row (which is also the only row).
        Yprd0[aloc] = res_predict[0][0]
        Aprd0[aloc] = res_predict[1][0][0]
        if len(res_predict[1])>1:
            Bprd0[aloc] = res_predict[1][1][0]
        Yerr0[aloc] = Yobs[0] - Yprd0[aloc]
        Mxd[aloc] = mxd

    Yprd = pd.DataFrame(Yprd0, columns=Ycpn.columns, index=Tidx)
    Yerr = pd.DataFrame(Yerr0, columns=Ycpn.columns, index=Tidx)
    Aprd = pd.DataFrame(Aprd0, columns=Ycpn.columns, index=Tidx)
    Bprd = pd.DataFrame(Bprd0, columns=Ycpn.columns, index=Tidx) if len(Bprd0)>0 else None

    return {'Yprd':Yprd, 'Yerr':Yerr, 'Aprd':Aprd, 'Bprd':Bprd, 'Mxd':Mxd}


# __all__ = ['Deconv_static_data', 'Options']

__script__ = __doc__


import sys, os, argparse
# from optparse import OptionParser, parser.add_argument_group       # command line arguments parser

def main():
    usage_msg = '%(prog)s [options] <infile> [outdir]'

    parser = argparse.ArgumentParser(description=__script__, usage=usage_msg)

    parser.add_argument('infile', type=str, help='preprocessed data file containing all sensors of one project (see the script Preprocessing_of_data.py)')
    parser.add_argument('outdir', nargs='?', type=str, default=None, help='directory where results (figures and data files) are saved.')

    sensor_opts = parser.add_argument_group('Sensor options')
    sensor_opts.add_argument('--xlocs', dest='xlocs', type=lambda s: [int(x) for x in s.split(',')], default=None, help='Location key IDs of active temperature sensors (default=all sensors).', metavar='integers separated by \',\'')
    sensor_opts.add_argument('--ylocs', dest='ylocs', type=lambda s: [int(x) for x in s.split(',')], default=None, help='Location key IDs of active elongation sensors (default=same as xlocs).', metavar='integers separated by \',\'')
    sensor_opts.add_argument('--component', dest='component', type=str, default='Trend', help='Type of component of data for analysis: Trend (default), Seasonal, All.', metavar='string')

    wdata_opts = parser.add_argument_group('Data truncation options')
    wdata_opts.add_argument('--timerange', dest='timerange', nargs=2, type=str, default=[None,None], help='Starting and ending timestamp index (default=the whole data set).', metavar='YYYY-MM-DD')

    ddata_opts = parser.add_argument_group('Component decomposition options')
    ddata_opts.add_argument('--mwmethod', dest='mwmethod', type=str, default='mean', help='Type of moving window mean estimator for decomposition of component: mean (default), median.', metavar='string')
    ddata_opts.add_argument('--mwsize', dest='mwsize', type=int, default=24, help='Length of the moving window (default=24).', metavar='integer')
    ddata_opts.add_argument('--kzord', dest='kzord', type=int, default=1, help='Order of Kolmogorov-Zurbenko filter (default=1).', metavar='integer')

    tdata_opts = parser.add_argument_group('Training data options')
    tdata_opts.add_argument('--sidx', dest='sidx', type=int, default=0, help='starting time index (an integer) of the training data relative to tidx0 (default=0).', metavar='integer')
    tdata_opts.add_argument('--Ntrn', dest='Ntrn', type=int, default=3*30*24, help='Length of the training data (default=24*30*3).', metavar='integer')

    model_opts = parser.add_argument_group('Model options')
    # model_opts.add_argument('--const', dest='const', action='store_true', default=False, help='Add constant trend in the convolution model (default: no constant trend).')
    model_opts.add_argument('--lagx', dest='lagx', type=int, default=12, help='Length of the convolution kernel of temperature (default=12). It will be desactivated if set to 0.', metavar='integer')
    model_opts.add_argument('--lagy', dest='lagy', type=int, default=6, help='Length of the convolution kernel of elongation (default=6). It will be desactivated if set to 0.', metavar='integer')
    model_opts.add_argument('--dtx', dest='dtx', type=int, default=0, help='Artificial delay (in hours) applied on the temperature data to avoid over-fitting (default=0).', metavar='integer')
    model_opts.add_argument('--dty', dest='dty', type=int, default=0, help='Artificial delay (in hours) applied on the elongation data to avoid over-fitting (default=0).', metavar='integer')

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
    outdir = os.path.join(outdir0, '{}_[{}_lagx={}_lagy={}]'.format(func_name, options.component.upper(), options.lagx, options.lagy))
    try:
        os.makedirs(outdir)
    except OSError:
        pass
    outfile = os.path.join(outdir, 'Results.pkl')

    _ = Deconv_static_data(options.infile, outfile, options)


if __name__ == "__main__":
    main()
