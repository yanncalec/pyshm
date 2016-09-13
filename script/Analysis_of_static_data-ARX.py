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

from Pyshm import Tools, Stat, OSMOS

import pandas as pd
# import statsmodels.api as sm
import numpy as np
from numpy.linalg import norm

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Hide annoying trace back message
# sys.excepthook = lambda exctype,exc,traceback : print("{}: {}".format(exctype.__name__,exc))

__script__ = 'Analysis of static data using the ARX model.'


def main():
    # Load data
    usage_msg = '{} [options] input_data_file output_directory'.format(sys.argv[0])
    parm_msg = 'input_data_file : the file (containing all sensors of one project) returned by the script Preprocessing_of_data.py or Decomposition_of_static_data.py\noutput_directory : the directory where results (figures and data files) are saved, a sub-directory of the same name as the input data file will be created.'

    parser = OptionParser(usage_msg)

    # parser.add_option('--pfname', dest='pfname', type='string', default=None, help='Load pre-computed ARX kernels from a pickle file (default: estimate ARX kernels from data).')
    parser.add_option('--loc', dest='loc', type='int', default=None, help='Location key ID of the sensor. If not given all sensors will be processed.')
    parser.add_option('--tidx0', dest='tidx0', type='string', default=None, help='Data truncation: starting timestamp index (default=begining of whole data set).')
    parser.add_option('--tidx1', dest='tidx1', type='string', default=None, help='Data truncation: ending timestamp index (default=end of whole data set).')
    parser.add_option('--component', dest='component', type='string', default='AllDiff-AllDiff', help='Type of component of data for analysis, must be like X-Y with X and Y in : All, AllDiff (default), Seasonal, SeasonalDiff, Trend, TrendDiff.')
    # parser.add_option('--cmptrn', dest='cmptrn', type='string', default=None, help='Type of component of data for training (default: same as --component).')
    parser.add_option('--penalh', dest='penalh', type='float', default=5e-1, help='Use penalization for the AR kernel (default=5e-1).')
    parser.add_option('--penalg', dest='penalg', type='float', default=5e-1, help='Use penalization for the convolution kernel  (default=5e-1).')
    parser.add_option('--Nh', dest='Nh_usr', type='int', default='10', help='Length of the auto-regression kernel (default=10, if 0 the kernel is not used, if <0 use BIC to determine the optimal length).')
    parser.add_option('--Ng', dest='Ng_usr', type='int', default='24', help='Length of the convolution kernel (default=24, if 0 the kernel is not used, if <0 use BIC to determine the optimal length).')
    parser.add_option('--Ntrn', dest='Ntrn', type='int', default=3*30*24, help='Length of the training data (default=24*30*3).')
    parser.add_option('--sidx', dest='sidx', type='int', default=0, help='starting index (integer) of the training data relative to tidx0 (default=0).')
    parser.add_option('--noconst', dest='const', action='store_false', default=True, help='Do not add constant trend in the convolution model (default: add constant trend).')
    parser.add_option('-v', '--verbose', dest='verbose', action='store_true', default=False, help='Print message.')

    (options, args) = parser.parse_args()
    # ARX = {}  # dictionary for pre-computed ARX kernels or for output

    if len(args) < 2:
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
        idx = infile.rfind('/')
        outdir0 = args[1] + '/' + infile[idx+1:-4]

    # Load raw data
    with open(infile, 'rb') as fp:
        All = pickle.load(fp)['Static_Data']

    Locations = list(All.keys()); Locations.sort() # Locations of the current project

    if options.loc is not None and options.loc in Locations:
        ListLoc = [options.loc]
    else:
        ListLoc = Locations

    for loc in ListLoc:
        if options.verbose:
            print('\nLocation: {}'.format(loc))

        # Res = ARX.copy() if len(ARX)>0 else {}  # dictionary for keeping results
        Res = {}

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

        # Components for training
        Xraw, Yraw = OSMOS.choose_component(Data0, options.component)
        Xdata, Ydata = np.asarray(Xraw), np.asarray(Yraw)

        # Training data
        Xtrn = Xdata[options.sidx:options.sidx+options.Ntrn].copy()
        Ytrn = Ydata[options.sidx:options.sidx+options.Ntrn].copy()
        Xtrn[np.isnan(Xtrn)] = 0
        Ytrn[np.isnan(Ytrn)] = 0

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

        # penalization on the decay of kernel
        wh = Tools.exponential_weight(options.Nh, w0=options.penalh) if options.Nh>0 else np.zeros(options.Nh)
        wg = Tools.exponential_weight(options.Ng, w0=options.penalg) if options.Ng>0 else np.zeros(options.Ng)
        pcoef = np.hstack([wh, wg, 0]) if options.const else np.hstack([wh, wg])

        h0, g0, c0, err0, A0 = Tools.ARX_fit(Ytrn, options.Nh, Xtrn, options.Ng, bflag=True, pcoef=None, cflag=options.const)  # without penalization
        h1, g1, c1, err1, _ = Tools.ARX_fit(Ytrn, options.Nh, Xtrn, options.Ng, bflag=True, pcoef=pcoef, cflag=options.const)  # with penalization

        # save result in a dictionary
        Res['AR.Kernel_np'] = h0  # AR kernel, without penalization
        Res['Conv.Kernel_np'] = g0 # Convolution kernel, without penalization
        Res['Constant_np'] = c0 # constant trend, without penalization
        Res['AR.Kernel'] = h1  # AR kernel
        Res['AR.Penal'] = options.penalh  # penalization for AR kernel
        Res['Conv.Kernel'] = g1  # Conv. kernel
        Res['Conv.Penal'] = options.penalg  # penalization for conv.kernel
        Res['Constant'] = c1  # constant trend
        Res['Cond.Number'] = np.linalg.cond(A0)  # condition number of the linear regression matrix
        Res['Xall'] = Xall  # Temperature, raw
        Res['Yall'] = Yall  # Elongation, raw
        Res['sidx'] = options.sidx  # starting index of the training data
        Res['Ntrn'] = options.Ntrn  # length of the training data
        Res['component'] = options.component  # type of component for whole data set
        # Res['cmptrn'] = options.cmptrn  # type of component for training data set
        Res['Nh'] = options.Nh
        Res['Ng'] = options.Ng

        # Components for prediction
        Xraw, Yraw = OSMOS.choose_component(Data0, options.component)
        Xdata, Ydata = np.asarray(Xraw), np.asarray(Yraw)

        # Prediction with the estimated or precomputed kernel
        Yprd = Tools.ARX_prediction(Ydata, h1, X=Xdata, g=g1, c=c1)
        Yprd[Nidx] = np.nan
        Res['Xdata'] = Xdata  # prediction data set: Temperature
        Res['Ydata'] = Ydata  # prediction data set: Elongation
        Res['Yprd'] = Yprd  # results of prediction
        Res['Rel.Error'] = Tools.safe_norm(Yprd-Ydata)/Tools.safe_norm(Ydata)

        if options.verbose:
            print('AR.Kernel: ', Res['AR.Kernel'])
            print('Conv.Kernel: ', Res['Conv.Kernel'])
            print('Constant: ', Res['Constant'])
            print('Condition number: {}'.format(Res['Cond.Number']))
            print('Rel.Error: ', Res['Rel.Error'])

        outdir = outdir0+'/ARX_[{}]_[Nh={}_Ng={}_const={}]/'.format(options.component, options.Nh, options.Ng, options.const)
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
