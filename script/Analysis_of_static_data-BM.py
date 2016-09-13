#!/usr/bin/env python

import sys
import os
# import glob
from optparse import OptionParser       # command line arguments parser
import pickle
# import datetime
# import dateutil
# from collections import namedtuple
# import warnings
# import itertools
# import copy

from Pyshm import OSMOS, Tools, Stat, Kalman

import pandas as pd
import statsmodels.api as sm
import numpy as np
from numpy import newaxis, mean, sqrt, zeros, zeros_like, squeeze, asarray, abs, linspace, fft, arange, nan
from numpy.linalg import norm, svd, inv
import scipy
from scipy import stats

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Hide annoying trace back message
# sys.excepthook = lambda exctype,exc,traceback : print("{}: {}".format(exctype.__name__,exc))

__script__ = 'Analysis of static data using a BM model with Kalman filter'


def main():
    # Load data
    usage_msg = '{} [options] input_data_file output_directory'.format(sys.argv[0])
    parser = OptionParser(usage_msg)

    parser.add_option('--loc', dest='loc', type='int', default=None, help='Location key ID of the sensor. If not given all sensors will be processed.')
    parser.add_option('--tidx0', dest='tidx0', type='string', default=None, help='Data truncation: starting timestamp index (default=begining of whole data set).')
    parser.add_option('--tidx1', dest='tidx1', type='string', default=None, help='Data truncation :ending timestamp index (default=end of whole data set).')
    parser.add_option('--component', dest='component', type='string', default='AllDiff-AllDiff', help='Type of component of data for analysis, must be like X-Y with X and Y in : All, AllDiff (default), Seasonal, SeasonalDiff, Trend, TrendDiff.')
    # parser.add_option('--cmptrn', dest='cmptrn', type='string', default=None, help='Type of component of data for training (default: same as --component).')
    parser.add_option('--Nh', dest='Nh_usr', type='int', default='10', help='Length of the auto-regression kernel (default=10, if 0 the kernel is not used, if <0 use BIC to determine the optimal length).')
    parser.add_option('--Ng', dest='Ng_usr', type='int', default='24', help='Length of the convolution kernel (default=24, if 0 the kernel is not used, if <0 use BIC to determine the optimal length).')
    parser.add_option('--sigmar2', dest='sigmar2_usr', type='float', default=None, help='Variance of observation noise sigmar^2 (default=None, determined by parameter estimation).')
    parser.add_option('--sigmaq2', dest='sigmaq2_usr', type='float', default=None, help='Variance of innovation noise sigmaq^2 (default=None, determined by parameter estimation).')
    parser.add_option('--Ntrn', dest='Ntrn', type='int', default=3*30*24, help='Length of the training data (default=24*30*3), useful only for parameter estimation.')
    parser.add_option('--sidx', dest='sidx', type='int', default=0, help='starting index (integer) of the training data relative to tidx0 (default=0), useful only for parameter estimation.')
    parser.add_option('--niter', dest='niter', type='int', default=5, help='number of fixed point iteration for parameter estimation (default=5).')
    parser.add_option('--noconst', dest='const', action='store_false', default=True, help='Do not add constant trend in the convolution model (default: add constant trend).')
    parser.add_option('-v', '--verbose', dest='verbose', action='store_true', default=False, help='Print message.')

    (options, args) = parser.parse_args()

    if len(args) < 2:
        print('Usage: ' + usage_msg)

        sys.exit(0)
    else:  # check datadir
        infile = args[0]
        if not os.path.isfile(infile):
            raise FileNotFoundError(infile)

        # output directory
        idx = infile.rfind('/')
        outdir0 = args[1] + '/' + infile[idx+1:-4]

    # Load raw data
    with open(infile, 'rb') as fp:
        All = pickle.load(fp)['Static_Data']

    Locations = list(All.keys()); Locations.sort() # Locations of the current project

    # if options.cmptrn is None:
    #     options.cmptrn = options.component

    if options.loc is not None and options.loc in Locations:
        ListLoc = [options.loc]
    else:
        ListLoc = Locations

    for loc in ListLoc:
        if options.verbose:
            print('\nLocation: {}'.format(loc))

        # Data truncation
        if options.tidx0 is not None and options.tidx1 is not None:
            Data0 = All[loc][options.tidx0:options.tidx1]
        elif options.tidx0 is not None:
            Data0 = All[loc][options.tidx0:]
        elif options.tidx1 is not None:
            Data0 = All[loc][:options.tidx1]
        else:
            Data0 = All[loc]

        Xall = Data0['Temperature'].copy()
        Yall = Data0['Elongation'].copy()
        Tidx = Xall.index  # time index
        Nidx = np.isnan(np.asarray(Xall)) # nan index

        # Components for prediction
        Xraw, Yraw = OSMOS.choose_component(Data0, options.component)
        Xdata, Ydata = np.asarray(Xraw), np.asarray(Yraw)

        # Training data
        Xtrn = Xdata[options.sidx:options.sidx+options.Ntrn].copy()
        Ytrn = Ydata[options.sidx:options.sidx+options.Ntrn].copy()
        # Xtrn[np.isnan(Xtrn)] = 0
        # Ytrn[np.isnan(Ytrn)] = 0

        # Optimal length of kernels
        if options.Nh_usr < 0:
            AIC, BIC = Tools.optimal_kernel_length_AR(Xtrn, Ytrn)
            options.Nh = np.argmin(BIC)+1
            if options.verbose:
                print('Optimal length of AR kernel: {}'.format(options.Nh))
        else:
            options.Nh = max(options.Nh_usr, 0)

        if options.Ng_usr < 0:
            AIC, BIC = Tools.optimal_kernel_length_conv(Xtrn, Ytrn)
            options.Ng = np.argmin(BIC)+1
            if options.verbose:
                print('Optimal length of convolution kernel: {}'.format(options.Ng))
        else:
            options.Ng = max(options.Ng_usr, 0)

        if (options.Nh==0 and options.Ng==0):
            raise ValueError('Nh and Ng must not be both == 0')

        Nq = options.Nh+options.Ng+int(options.const)  # total dimension of the state vector

        # Estimation of parameters
        if options.sigmaq2_usr is None or options.sigmar2_usr is None:
            ssfunc = lambda sigmaq2, sigmar2, X0, P0: \
            Kalman.Kalman_ARX_wrapper(Ytrn, options.Nh, Xtrn, options.Ng, sigmaq2, sigmar2, X0, P0, cflag=options.const)

            if options.verbose:
                print('----Estimation of parameters----')
            options.sigmaq2, options.sigmar2, options.X0, options.P0 = \
            Kalman.FP_Parms_Estimation(ssfunc, sigmaq2=np.ones(Nq), sigmar2=1, niter=options.niter, verbose=options.verbose)

            if options.verbose:
                print('Estimate of sigmaq2:', options.sigmaq2)
                print('Estimate of sigmar2:', options.sigmar2)
                print('Initial value X0:', options.X0)
        else:
            options.sigmaq2, options.sigmar2 = options.sigmaq2_usr, options.sigmar2_usr
            options.X0, options.P0 = None, None

        # Apply KF
        res_tn, res_tt, res_tm = \
        Kalman.Kalman_ARX(Ydata, options.Nh, Xdata, options.Ng, options.sigmaq2, options.sigmar2, X0=options.X0, P0=options.P0, cflag=options.const)

        # Contents of the results
        # Xtn, Ptn, Ytn, Etn = res_tn
        # Xtt, Ptt, Ytt, Ett = res_tt
        # Xtm, Ptm, Ytm, Etm = res_tm

        # Choose a type of result
        Res = {} # dictionary for keeping results
        Res['Xall'] = Xall  # Temperature, raw
        Res['Yall'] = Yall  # Elongation, raw
        Res['component'] = options.component  # type of component for whole data set
        Res['sigmaq2'] = options.sigmaq2
        Res['sigmar2'] = options.sigmar2
        Res['Xall'] = Xall  # Temperature, raw
        Res['Yall'] = Yall  # Elongation, raw
        Res['Xdata'] = Xdata  # prediction data set: Temperature
        Res['Ydata'] = Ydata  # prediction data set: Elongation
        Res['X0'] = options.X0
        Res['P0'] = options.P0
        Res['const'] = options.const
        Res['Nh'] = options.Nh
        Res['Ng'] = options.Ng
        Res['sidx'] = options.sidx  # starting index of the training data
        Res['Ntrn'] = options.Ntrn  # length of the training data

        toto = {}
        for tname, res in zip(['smoother', 'filter', 'predictor'], [res_tn, res_tt, res_tm]):
            toto[tname] = res
        Res['Result'] = toto

        outdir = outdir0+'/BM_[{}]_[Nh={}_Ng={}_const={}]/'.format(options.component, options.Nh, options.Ng, options.const)
        try:
            os.makedirs(outdir)
        except OSError:
            pass

        fname = outdir+'/{}'.format(loc)
        try:
            with open(fname+'.pkl', 'wb') as fp:
                pickle.dump(Res, fp)
            if options.verbose:
                print('Results saved in {}.pkl'.format(fname))
        except Exception as msg:
            print(msg)
            # print(Fore.RED + 'Warning: ', msg)
            # print(Style.RESET_ALL)


if __name__ == "__main__":
    print(__script__)
    print()
    main()
