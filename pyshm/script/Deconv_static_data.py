#!/usr/bin/env python

"""Analysis of static data using the vectorial deconvolution model.
"""

# class Options:
#     tidx0=None
#     tidx1=None
#     component='Trend'
#     lagx=10
#     lagy=24
#     Ntrn=3*30*24
#     sidx=0
#     const=False
#     xlocs=[]
#     ylocs=[]
#     verbose=False


def get_locs(locs):
    return [int(s) for s in locs.split(',')]


def Deconv_static_data(infile, outdir0, options):
    import os
    import pickle
    import numpy as np
    import pandas as pd

    from pyshm import OSMOS, Models

    # Load preprocessed static data
    Data0, Tall0, Eall0, Locations = OSMOS.load_static_data(infile)
    Midx0 = OSMOS.concat_mts(Data0, 'Missing')  # indicator of missing data

    if options.info:
        ss = str(Locations)[1:-1].replace(',', '')
        print('Location key IDs:', ss)

        for loc in Locations:
            print('Location {}, from {} to {}'.format(loc, Data0[loc].index[0], Data0[loc].index[-1]))
        t0, t1 = OSMOS.common_time_range(Data0)
        print('Common period: from {} to {}'.format(t0, t1))
        sys.exit()

    if options.xlocs is None or len(options.xlocs)==0:
        options.xlocs = Locations.copy()
    else:
        options.xlocs = get_locs(options.xlocs)
    if options.ylocs is None or len(options.ylocs)==0:
        options.ylocs = options.xlocs.copy()
    else:
        options.ylocs = get_locs(options.ylocs)

    if options.verbose:
        print('Sensors for temperature: {}\nSensors for elongation: {}\n'.format(options.xlocs, options.ylocs))

    # Selection of component
    if options.component.upper() in ['SEASONAL', 'TREND']:
        if options.verbose:
            # print('Decomposition of signals...')
            print('Moving window estimator: {}\nMoving window size: {}\nOrder of the KZ filter: {}\n'.format(options.mwmethod, options.mwsize, options.kzord))

        # Decomposition of signals
        Ttrd0, Tsnl0 = OSMOS.trend_seasonal_decomp(Tall0, mwsize=options.mwsize, kzord=options.kzord, method=options.mwmethod)
        Etrd0, Esnl0 = OSMOS.trend_seasonal_decomp(Eall0, mwsize=options.mwsize, kzord=options.kzord, method=options.mwmethod)

        if options.component.upper() == 'SEASONAL':
            Xcpn0, Ycpn0 = Tsnl0, Esnl0
        else:
            Xcpn0, Ycpn0 = Ttrd0, Etrd0
    elif options.component.upper() == 'ALL':
            Xcpn0, Ycpn0 = Tall0, Eall0
    else:
        raise Exception('Unknown type of component:', options.component)

    # Data truncation
    # t0, t1 = map(str, OSMOS.common_time_range(Data0))
    # if options.tidx0 is None:
    #     options.tidx0 = t0

    # options.tidx0 and options.tidx1 are respectively the beginning and ending timestamps
    Xcpn = Xcpn0[options.tidx0:options.tidx1]
    Ycpn = Ycpn0[options.tidx0:options.tidx1]
    Midx = Midx0[options.tidx0:options.tidx1]  # indicator of missing value
    Tidx = Xcpn.index  # time index

    # Preparation of training data
    options.Ntrn = min(options.Ntrn, len(Xcpn))
    Yprd0 = {}  # final prediction from inputs
    Aprd0 = {}  # contribution of the first group of inputs
    Bprd0 = {}  # contribution of the second group of inputs, if exists
    Yerr0 = {}  # error of prediction
    Mxd = {}  # objects of deconvolution model

    if options.verbose:
        print('Deconvolution of the \'{}\' component...'.format(options.component.upper()))

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

    outdir = os.path.join(outdir0,'Deconv_[{}_lagx={}_lagy={}]'.format(options.component.upper(), options.lagx, options.lagy))
    try:
        os.makedirs(outdir)
    except OSError:
        pass

    fname = os.path.join(outdir, 'Results.pkl')
    try:
        with open(fname, 'wb') as fp:
            pickle.dump({'Xcpn':Xcpn, 'Ycpn':Ycpn, 'Yprd':Yprd, 'Yerr':Yerr, 'Aprd':Aprd, 'Bprd':Bprd, 'options':options, 'Mxd':Mxd, 'Midx':Midx}, fp)
        if options.verbose:
            print('\nResults saved in {}'.format(fname))
    except Exception as msg:
        print(msg)
        # print(Fore.RED + 'Warning: ', msg)
        # print(Style.RESET_ALL)


__all__ = ['Deconv_static_data']

__script__ = __doc__


import sys, os
from optparse import OptionParser, OptionGroup       # command line arguments parser

def main():
    # Load data
    usage_msg = '{} [options] <input_data_file> [output_directory]'.format(sys.argv[0])
    parm_msg = '  input_data_file : data file (containing all sensors of one project) returned by the script Preprocessing_of_data.py\n  output_directory : directory where results (figures and data files) are saved.'
    # , a sub-directory of the same name as the input data file will be created.'

    parser = OptionParser(usage=usage_msg+'\n'+parm_msg)

    # parser.add_option('--pfname', dest='pfname', type='string', default=None, help='Load pre-computed ARX kernels from a pickle file (default: estimate ARX kernels from data).')
    # parser.add_option('--loc', dest='loc', type='int', default=None, help='Location key ID of the sensor. If not given all sensors will be processed.')
    # parser.add_option('--dropbg', dest='dropbg', action='store_true', default=False, help='Drop the first 15 days of data.')

    sensor_opts = OptionGroup(parser, 'Sensor options')
    sensor_opts.add_option('--xlocs', dest='xlocs', type='string', default=None, help='Location key IDs of input sensors (temperature) separated by \',\' (default=all sensors).', metavar='string')
    sensor_opts.add_option('--ylocs', dest='ylocs', type='string', default=None, help='Location key IDs of output sensors (elongation) separated by \',\' (default=same as xlocs).', metavar='string')
    sensor_opts.add_option('--component', dest='component', type='string', default='Trend', help='Type of component of data for analysis: Trend (default), Seasonal, All.', metavar='string')
    parser.add_option_group(sensor_opts)

    wdata_opts = OptionGroup(parser, 'Data truncation options')
    wdata_opts.add_option('--tidx0', dest='tidx0', type='string', default=None, help='Starting timestamp index (default=begining of whole data set).', metavar='a string of format \'YYYY-MM-DD\'')
    wdata_opts.add_option('--tidx1', dest='tidx1', type='string', default=None, help='Ending timestamp index (default=end of whole data set).', metavar='a string of format \'YYYY-MM-DD\'')
    # wdata_opts.add_option('--ndays', dest='ndays', type='int', default=0, help='Ignore the first n days of the raw data (default=0).', metavar='integer')
    # parser.add_option('--cmptrn', dest='cmptrn', type='string', default=None, help='Type of component of data for training (default: same as --component).')
    parser.add_option_group(wdata_opts)

    ddata_opts = OptionGroup(parser, 'Component decomposition options (only for --component=Seasonal* or Trend*.)')
    ddata_opts.add_option('--mwmethod', dest='mwmethod', type='string', default='mean', help='Type of moving window mean estimator for decomposition of component: mean (default), median.', metavar='string')
    ddata_opts.add_option('--mwsize', dest='mwsize', type='int', default=24, help='Length of the moving window (default=24).', metavar='integer')
    ddata_opts.add_option('--kzord', dest='kzord', type='int', default=1, help='Order of Kolmogorov-Zurbenko filter (default=1).', metavar='integer')
    parser.add_option_group(ddata_opts)

    tdata_opts = OptionGroup(parser, 'Training data options')
    tdata_opts.add_option('--sidx', dest='sidx', type='int', default=0, help='starting time index (an integer) of the training data relative to tidx0 (default=0).', metavar='integer')
    tdata_opts.add_option('--Ntrn', dest='Ntrn', type='int', default=3*30*24, help='Length of the training data (default=24*30*3).', metavar='integer')
    parser.add_option_group(tdata_opts)

    model_opts = OptionGroup(parser, 'Model options')
    # model_opts.add_option('--const', dest='const', action='store_true', default=False, help='Add constant trend in the convolution model (default: no constant trend).')
    model_opts.add_option('--lagx', dest='lagx', type='int', default=12, help='Length of the convolution kernel of temperature (default=12). It will be desactivated if set to 0.', metavar='integer')
    model_opts.add_option('--lagy', dest='lagy', type='int', default=6, help='Length of the convolution kernel of elongation (default=6). It will be desactivated if set to 0.', metavar='integer')
    model_opts.add_option('--dtx', dest='dtx', type='int', default=0, help='Artificial delay (in hours) applied on the temperature data to avoid over-fitting (default=0).', metavar='integer')
    model_opts.add_option('--dty', dest='dty', type='int', default=0, help='Artificial delay (in hours) applied on the elongation data to avoid over-fitting (default=0).', metavar='integer')
    parser.add_option_group(model_opts)

    parser.add_option('-v', '--verbose', dest='verbose', action='store_true', default=False, help='Print message.')
    parser.add_option('--info', dest='info', action='store_true', default=False, help='Print only information about the project.')

    (options, args) = parser.parse_args()
    # ARX = {}  # dictionary for pre-computed ARX kernels or for output

    if len(args) < 1:
        print('Usage: ' + usage_msg)
        print(parm_msg)

        sys.exit(0)
    else:  # check datadir
        infile = args[0]
        if not os.path.isfile(infile):
            raise FileNotFoundError(infile)
        # if options.pfname is not None:
        #     if not os.path.isfile(options.pfname):
        #         raise FileNotFoundError(options.pfname)
        #     else:
        #         with open(options.pfname, 'rb') as fp:
        #             ARX = pickle.load(fp)

        # output directory
        if len(args)==2:
            outdir0 = args[1]
        else:
            idx2 = infile.rfind(os.path.sep, 0)
            idx1 = infile.rfind(os.path.sep, 0, idx2)
            idx0 = infile.rfind(os.path.sep, 0, idx1)
            outdir0 = os.path.join(infile[:idx0], 'Outputs', infile[idx1+1:idx2])
            # print(outdir0)

    Deconv_static_data(infile, outdir0, options)


if __name__ == "__main__":
    print(__script__)
    # print()
    main()
