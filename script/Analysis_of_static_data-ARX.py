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

from Seim import Tools, Stat
from OSMOS import OSMOS

import pandas as pd
# import statsmodels.api as sm
import numpy as np
from numpy.linalg import norm

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Hide annoying trace back message
# sys.excepthook = lambda exctype,exc,traceback : print("{}: {}".format(exctype.__name__,exc))

import matplotlib.pyplot as plt
# from matplotlib.pyplot import figure

import matplotlib.colors as colors
# color_list = list(colors.cnames.keys())
color_list = ['green', 'pink', 'lightgrey', 'magenta', 'cyan', 'red', 'yelow',
              'purple', 'blue', 'mediumorchid', 'chocolate', 'blue',
              'blueviolet', 'brown']

import mpld3
plt.style.use('ggplot')

__script__ = 'Analysis of static data using the deconvolution model.'


def main():
    # Load data
    usage_msg = '{} [options] input_data_file output_directory'.format(sys.argv[0])
    parm_msg = 'input_data_file : the file (containing all sensors of one project) returned by the script Preprocessing_of_data.py or Decomposition_of_static_data.py\noutput_directory : the directory where results (figures and data files) are saved, a sub-directory of the same name as the input data file will be created.'

    parser = OptionParser(usage_msg)

    # parser.add_option('--pfname', dest='pfname', type='string', default=None, help='Load pre-computed ARX kernels from a pickle file (default: estimate ARX kernels from data).')
    parser.add_option('--loc', dest='loc', type='int', default=None, help='Location key ID of the sensor. If not given all sensors will be processed.')
    parser.add_option('--tidx0', dest='tidx0', type='string', default=None, help='Data truncation: starting timestamp index (default=begining of whole data set).')
    parser.add_option('--tidx1', dest='tidx1', type='string', default=None, help='Data truncation :ending timestamp index (default=end of whole data set).')
    parser.add_option('--cmpdta', dest='cmpdta', type='string', default='AllDiff-AllDiff', help='Type of component of data for analysis: All, AllDiff (default), Seaonal, SeaonalDiff, Trend, TrendDiff.')
    parser.add_option('--cmptrn', dest='cmptrn', type='string', default=None, help='Type of component of data for training (default: same as --cmpdta).')
    parser.add_option('--penalh', dest='penalh', type='float', default=5e-1, help='Use penalization for the ar kernel.')
    parser.add_option('--Np', dest='Np', type='int', default='10', help='Length of the auto-regression kernel (default=10).')
    parser.add_option('--Nq', dest='Nq', type='int', default='24', help='Length of the convolution kernel (default=24).')
    # parser.add_option('--BIC', dest='BIC', action='store_true', default=False, help='Use Bayesian Information Criterion to determine the optimal length.')
    parser.add_option('--Ntrn', dest='Ntrn', type='int', default=3*30*24, help='Length of the training data (default=24*30*3).')
    parser.add_option('--sidx', dest='sidx', type='int', default=0, help='starting index (integer) of the training data relative to tidx0 (default=0).')
    parser.add_option('--penalg', dest='penalg', type='float', default=5e-1, help='Use penalization for the convolution kernel.')
    parser.add_option('--noconst', dest='const', action='store_false', default=True, help='Do not add constant trend in the convolution model (default: add constant trend).')
    parser.add_option('-v', '--verbose', dest='verbose', action='store_true', default=False, help='Print message.')

    (options, args) = parser.parse_args()
    ARX = {}  # dictionary for pre-computed ARX kernels or for output

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

    # Least-square estimation
    options.Np = max(0, options.Np)
    options.Nq = max(0, options.Nq)
    if (options.Np==0 or options.Nq==0):
        raise ValueError('Np and Nq must not be both == 0')

    if options.cmptrn is None:
        options.cmptrn = options.cmpdta

    if options.cmptrn != options.cmpdta:
        outdir = outdir0+'/ARX_[{}_{}]_[Np={}_Nq={}_const={}]/'.format(options.cmptrn, options.cmpdta, options.Np, options.Nq, options.const)
    else:
        outdir = outdir0+'/ARX_[{}]_[Np={}_Nq={}_const={}]/'.format(options.cmpdta, options.Np, options.Nq, options.const)
    try:
        os.makedirs(outdir)
    except OSError:
        pass

    if options.loc is not None and options.loc in Locations:
        ListLoc = [options.loc]
    else:
        ListLoc = Locations

    for loc in ListLoc:
        Res = ARX.copy() if len(ARX)>0 else {}  # dictionary for keeping results

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
        Xraw, Yraw = OSMOS.choose_component(Data0, options.cmptrn)
        Xdata, Ydata = np.asarray(Xraw), np.asarray(Yraw)

        # Training data
        Xtrn = Xdata[options.sidx:options.sidx+options.Ntrn].copy()
        Ytrn = Ydata[options.sidx:options.sidx+options.Ntrn].copy()
        Xtrn[np.isnan(Xtrn)] = 0
        Ytrn[np.isnan(Ytrn)] = 0

        # penalization on the decay of kernel
        wh = Tools.exponential_weight(options.Np, w0=options.penalh) if options.Np>0 else np.zeros(options.Np)
        wg = Tools.exponential_weight(options.Nq, w0=options.penalg) if options.Nq>0 else np.zeros(options.Nq)
        pcoef = np.hstack([wh, wg, 0]) if options.const else np.hstack([wh, wg])

        h0, g0, c0, err0, _ = Tools.ARX_fit(Ytrn, options.Np, Xtrn, options.Nq, bflag=True, pcoef=None, cflag=options.const)
        h1, g1, c1, err1, _ = Tools.ARX_fit(Ytrn, options.Np, Xtrn, options.Nq, bflag=True, pcoef=pcoef, cflag=options.const)

        # save result in a dictionary
        Res['AR.Kernel_np'] = h0  # AR kernel, without penalization
        Res['Conv.Kernel_np'] = g0 # Convolution kernel, without penalization
        Res['Constant_np'] = c0 # constant trend, without penalization
        Res['AR.Kernel'] = h1  # AR kernel
        Res['AR.Penal'] = options.penalh  # penalization for AR kernel
        Res['Conv.Kernel'] = g1  # Conv. kernel
        Res['Conv.Penal'] = options.penalg  # penalization for conv.kernel
        Res['Constant'] = c1  # constant trend
        Res['Xall'] = Xall  # Temperature, raw
        Res['Yall'] = Yall  # Elongation, raw
        Res['sidx'] = options.sidx  # starting index of the training data
        Res['Ntrn'] = options.Ntrn  # length of the training data
        Res['cmpdta'] = options.cmpdta  # type of component for whole data set
        Res['cmptrn'] = options.cmptrn  # type of component for training data set

        # Components for prediction
        Xraw, Yraw = OSMOS.choose_component(Data0, options.cmpdta)
        Xdata, Ydata = np.asarray(Xraw), np.asarray(Yraw)

        # Prediction with the estimated or precomputed kernel
        Yprd = Tools.ARX_prediction(Ydata, h1, X=Xdata, g=g1, c=c1)
        Yprd[Nidx] = np.nan
        Res['Xdata'] = Xdata  # prediction data set: Temperature
        Res['Ydata'] = Ydata  # prediction data set: Elongation
        Res['Yprd'] = Yprd  # results of prediction
        Res['Rel.Error'] = Tools.safe_norm(Yprd-Ydata)/Tools.safe_norm(Ydata)

        if options.verbose:
            print('\nLocation: {}'.format(loc))
            print('AR.Kernel: ', Res['AR.Kernel'])
            print('Conv.Kernel: ', Res['Conv.Kernel'])
            print('Constant: ', Res['Constant'])
            print('Rel.Error: ', Res['Rel.Error'])

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
