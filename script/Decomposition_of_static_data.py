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
import pandas as pd
import statsmodels.api as sm
import numpy as np

from Pyshm import Tools, Stat, OSMOS

# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)

# Hide annoying trace back message
# sys.excepthook = lambda exctype,exc,traceback : print("{}: {}".format(exctype.__name__,exc))

import matplotlib.pyplot as plt
import matplotlib.colors as colors
# color_list = list(colors.cnames.keys())
color_list = ['green', 'pink', 'lightgrey', 'magenta', 'cyan', 'red', 'yelow', 'purple', 'blue', 'mediumorchid', 'chocolate', 'blue', 'blueviolet', 'brown']

import mpld3
plt.style.use('ggplot')

__script__ = 'Decomposition of static data into seasonal and trend components.'


def main():
    usage_msg = '{} [options] directory_of_PID'.format(sys.argv[0])
    parser = OptionParser(usage_msg)

    parser.add_option('--mwsize', dest='mwsize', type='int', default=24, help='Size of the moving window (default=24).')
    parser.add_option('--kzord', dest='kzord', type='int', default=1, help='Order of the recursive KZ filter (default=1).')
    parser.add_option('--lambda', dest='lamb', type='int', default=129600*30, help='Penality constant for the HP filter (default=129600*30).')
    parser.add_option('--causal', dest='causal', action='store_true', default=False, help='Use causal moving average.')
    parser.add_option('--method', dest='method', type='string', default='moving_average', help='Method of decomposition: moving_average (default), moving_median, hp_filter.')
    parser.add_option('-v', '--verbose', dest='verbose', action='store_true', default=False, help='Print message.')

    (options, args) = parser.parse_args()

    if len(args) < 1:
        print(usage_msg)
        sys.exit(0)
    else:  # check datadir
        datadir = args[0]
        if not os.path.isdir(datadir):
            raise FileNotFoundError(datadir)

    mwsize = options.mwsize  # window size for moving average
    kzord = options.kzord  # order of the recursive KZ filter
    lamb = options.lamb  # Penality constant for the HP filter
    causal = options.causal
    method = options.method

    # if options.verbose:
    #     print('Method:', options.method)

    fname = datadir + '/Processed.pkl'
    with open(fname, 'rb') as fp:
        toto = pickle.load(fp)

    Sdata0 = toto['Static_Data']  # Static data
    # Ddata0 = toto['Dynamic_Data'] # Dynamic data

    # AEm, ATm = {}, {}  # Trend component: elongation, temperature
    # AEd, ATd = {}, {}  # Seasonal component
    # AE, AT = {}, {}  # Raw signals
    All = {}

    for loc, data0 in Sdata0.items():
        Xt, Yt = data0['Temperature'], data0['Elongation']

        if method == 'moving_average':
            # Xm = Tools.KZ_filter(Xt, mwsize, k=kzord, method='mean', causal=causal)
            # Ym = Tools.KZ_filter(Yt, mwsize, k=kzord, method='mean', causal=causal)
            Xm = Tools.KZ_filter_pandas(Xt, mwsize, k=kzord, method='mean', causal=causal)
            Ym = Tools.KZ_filter_pandas(Yt, mwsize, k=kzord, method='mean', causal=causal)
            parm = 'mwsize={}_kzord={}_causal={}'.format(mwsize, kzord, causal)
        elif method == 'moving_median':
            Xm = Tools.KZ_filter_pandas(Xt, mwsize, k=kzord, method='median', causal=causal)
            Ym = Tools.KZ_filter_pandas(Yt, mwsize, k=kzord, method='median', causal=causal)
            parm = 'mwsize={}_kzord={}_causal={}'.format(mwsize, kzord, causal)
        elif method == 'hp_filter':
            _, Xm = sm.tsa.filters.hpfilter(Xt, lamb=lamb)
            _, Ym = sm.tsa.filters.hpfilter(Yt, lamb=lamb)
            parm = 'lambda={}'.format(lamb)

        # Nidx = np.isnan(Xt)
        # Xm[Nidx] = np.nan
        # Ym[Nidx] = np.nan

        Xd, Yd = Xt - Xm, Yt - Ym
        #     # Denoising by a LU filter
        #     Xd = pd.Series(Tools.LU_mean(Xd, 3), index=Xt.index)
        #     Yd = pd.Series(Tools.LU_mean(Yd, 3), index=Yt.index)

        # ATm[loc], AEm[loc] = Xm.copy(), Ym.copy()
        # ATd[loc], AEd[loc] = Xd.copy(), Yd.copy()
        # AT[loc], AE[loc] = Xt.copy(), Yt.copy()

        All[loc] = pd.DataFrame(data={'Temperature_trend':Xm, 'Temperature_seasonal':Xd, 'Temperature':Xt, 'Elongation_trend':Ym, 'Elongation_seasonal':Yd, 'Elongation':Yt}, index=Xt.index)

    # Creat directory for output
    datadir1 = datadir + '/Decomposition/'
    try:
        os.makedirs(datadir1)
    except OSError:
        pass

    fname = datadir1+'/{}_[{}].pkl'.format(method, parm)

    try:
        with open(fname, 'wb') as fp:
            # toto = {}
            # toto['Static_Data'] = {'Temperature_trend':ATm, 'Temperature_seasonal':ATd, 'Temperature':AT,
            #                        'Elongation_trend':AEm, 'Elongation_seasonal':AEd, 'Elongation':AE}
            # pickle.dump(toto, fp)
            pickle.dump({'Static_Data': All}, fp)
        if options.verbose:
            print('Results saved in {}'.format(fname))
    except Exception as msg:
        print(msg)


if __name__ == '__main__':
    print(__script__)
    print()
    main()
