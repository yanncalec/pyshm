#!/usr/bin/env python

import sys, os
# import glob
from optparse import OptionParser, OptionGroup       # command line arguments parser
# import pickle
# # import datetime
# # import dateutil
# # from collections import namedtuple
# # import warnings
# # import itertools
# # import copy
#
# from Pyshm import Tools, Stat, OSMOS
#
# import pandas as pd
# # import statsmodels.api as sm
# import numpy as np
# from numpy.linalg import norm
#
# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)
#
# Hide annoying trace back message
# sys.excepthook = lambda exctype,exc,traceback : print("{}: {}".format(exctype.__name__,exc))

__script__ = 'Analysis of static data using the ARX model.'


def main():
    # Load data
    usage_msg = '{} [options] <input_data_file> <location_key_id> <output_directory>'.format(sys.argv[0])
    parm_msg = '  input_data_file : data file (containing all sensors of one project) returned by the script Preprocessing_of_data.py or Decomposition_of_static_data.py\n  location_key_id : ID of the sensor to be analyzed\n  output_directory : directory where results (figures and data files) are saved.'
    # , a sub-directory of the same name as the input data file will be created.'

    parser = OptionParser(usage=usage_msg+'\n'+parm_msg)

    # parser.add_option('--pfname', dest='pfname', type='string', default=None, help='Load pre-computed ARX kernels from a pickle file (default: estimate ARX kernels from data).')
    # parser.add_option('--loc', dest='loc', type='int', default=None, help='Location key ID of the sensor. If not given all sensors will be processed.')
    # parser.add_option('--dropbg', dest='dropbg', action='store_true', default=False, help='Drop the first 15 days of data.')
    model_opts = OptionGroup(parser, 'Model options')
    model_opts.add_option('--component', dest='component', type='string', default='AllDiff-AllDiff', help='Type of component of data for analysis, with X and Y in : All, AllDiff (default), Seasonal, SeasonalDiff, Trend, TrendDiff.', metavar='a string of format \'X-Y\'')
    model_opts.add_option('--const', dest='const', action='store_true', default=False, help='Add constant trend in the convolution model (default: no constant trend).')
    model_opts.add_option('--Nh', dest='Nh', type='int', default='10', help='Length of the auto-regression kernel (default=10, if 0 the kernel is not used, if <0 use BIC to determine the optimal length).', metavar='integer')
    model_opts.add_option('--Ng', dest='Ng', type='int', default='24', help='Length of the convolution kernel (default=24, if 0 the kernel is not used, if <0 use BIC to determine the optimal length).', metavar='integer')
    model_opts.add_option('--penalh', dest='penalh', type='float', default=5e-1, help='Use penalization for the AR kernel (default=5e-1).', metavar='positive number')
    model_opts.add_option('--penalg', dest='penalg', type='float', default=5e-1, help='Use penalization for the convolution kernel  (default=5e-1).', metavar='positive number')
    parser.add_option_group(model_opts)

    wdata_opts = OptionGroup(parser, 'Data truncation options')
    wdata_opts.add_option('--tidx0', dest='tidx0', type='string', default=None, help='Starting timestamp index (default=begining of whole data set).', metavar='a string of format \'YYYY-MM-DD\'')
    wdata_opts.add_option('--tidx1', dest='tidx1', type='string', default=None, help='Ending timestamp index (default=end of whole data set).', metavar='a string of format \'YYYY-MM-DD\'')
    # parser.add_option('--cmptrn', dest='cmptrn', type='string', default=None, help='Type of component of data for training (default: same as --component).')
    parser.add_option_group(wdata_opts)

    tdata_opts = OptionGroup(parser, 'Training data options')
    tdata_opts.add_option('--sidx', dest='sidx', type='int', default=0, help='starting time index (an integer) of the training data relative to tidx0 (default=0).', metavar='integer')
    tdata_opts.add_option('--Ntrn', dest='Ntrn', type='int', default=3*30*24, help='Length of the training data (default=24*30*3).', metavar='integer')
    parser.add_option_group(tdata_opts)

    parser.add_option('-v', '--verbose', dest='verbose', action='store_true', default=False, help='Print message.')

    (options, args) = parser.parse_args()
    # ARX = {}  # dictionary for pre-computed ARX kernels or for output

    if len(args) < 3:
        print('\nUsage: ' + usage_msg)
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

        loc = int(args[1])

        # output directory
        outdir0 = args[2]
        # idx = infile.rfind(os.path.sep)
        # loc = int(args[1])
        # outdir0 = os.path.join(args[2], infile[idx+1:-4])

    from pyshm.OSMOS_pkg.Analysis_of_static_data_ARX import Analysis_of_static_data_ARX

    Analysis_of_static_data_ARX(infile, loc, outdir0, options=options)


if __name__ == "__main__":
    print(__script__)
    # print()
    main()
