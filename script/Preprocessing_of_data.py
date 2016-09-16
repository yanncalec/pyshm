#!/usr/bin/env python

import sys, os
from optparse import OptionParser       # command line arguments parser

# import sys
# import os
# import glob
# from optparse import OptionParser       # command line arguments parser
# import pickle
# # import datetime
# # import dateutil
# # from collections import namedtuple
# # import warnings
# # import itertools
# import copy
# import pandas as pd
#

# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)

# Hide annoying trace back message
# sys.excepthook = lambda exctype,exc,traceback : print("{}: {}".format(exctype.__name__,exc))

# color_list = list(colors.cnames.keys())
# color_list = ['green', 'pink', 'lightgrey', 'magenta', 'cyan', 'red', 'yelow', 'purple', 'blue', 'mediumorchid', 'chocolate', 'blue', 'blueviolet', 'brown']

__script__ = 'Apply preprocessing on the data of a project and assemble the results into a single pandas data sheet.'


def main():
    usage_msg = '{} [options] directory_of_PID'.format(sys.argv[0])
    parser = OptionParser(usage_msg)

    # parser.add_option('-p', '--PID', dest='PID', type='int', default=None, help='Project Key ID. If not given all projects presented in the destination data directory will be processeded.')
    parser.add_option('-d', '--dflag', dest='dflag', action='store_true', default=False, help='Remove possible dynamic data.')
    parser.add_option('-s', '--sflag', dest='sflag', action='store_true', default=False, help='Remove synchronization error.')
    parser.add_option('-o', '--oflag', dest='oflag', action='store_true', default=False, help='Remove outliers.')
    parser.add_option('-f', '--fflag', dest='fflag', action='store_true', default=False, help='Fill missing data of < 12h.')
    # parser.add_option('-r', '--rflag', dest='rflag', action='store_true', default=False, help='Resampling with the step=1h.')  # this option is applied by default
    parser.add_option('-t', '--tflag', dest='tflag', action='store_true', default=False, help='Filter also the temperature.')
    parser.add_option('-j', '--jflag', dest='jflag', action='store_true', default=False, help='Detect jumps.')
    parser.add_option('--force', dest='force', action='store_true', default=False, help='Force re-computation.')
    parser.add_option('--plotstatic', dest='plotstatic', action='store_true', default=False, help='Plot static data of all sensor and save the figure.')
    parser.add_option('-v', '--verbose', dest='verbose', action='store_true', default=False, help='Print message.')

    (options, args) = parser.parse_args()

    if len(args) < 1:
        print('\nUsage: '+usage_msg)
        sys.exit(0)
    else:  # check datadir
        datadir = args[0]
        if not os.path.isdir(datadir):
            raise FileNotFoundError(datadir)

    from pyshm.OSMOS_pkg.Preprocessing_of_data import Preprocessing_of_data
    Preprocessing_of_data(datadir, options)

if __name__ == '__main__':
    print(__script__)
    # print()
    main()
