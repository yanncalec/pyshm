'''Decomposition of static data into seasonal and trend components.'''

import sys
import os
import pickle
import pandas as pd
import numpy as np

from .. import Tools
# from . import OSMOS

# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)

# Hide annoying trace back message
# sys.excepthook = lambda exctype,exc,traceback : print("{}: {}".format(exctype.__name__,exc))

# import matplotlib.pyplot as plt
# import matplotlib.colors as colors

# import mpld3
# plt.style.use('ggplot')

class Options:
    mwsize=24 # window size for moving average
    kzord=1 # order of the recursive KZ filter
    lamb=129600*30 # Penality constant for the HP filter
    causal=False
    method='moving_average'
    verbose=False

def Decomposition_of_static_data(datadir, options=None):

    if options is None:
        options = Options()

    fname = os.path.join(datadir, 'Processed.pkl')
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

        if options.method == 'moving_average':
            # Xm = Tools.KZ_filter(Xt, mwsize, k=kzord, method='mean', causal=causal)
            # Ym = Tools.KZ_filter(Yt, mwsize, k=kzord, method='mean', causal=causal)
            Xm = Tools.KZ_filter_pandas(Xt, options.mwsize, k=options.kzord, method='mean', causal=options.causal)
            Ym = Tools.KZ_filter_pandas(Yt, options.mwsize, k=options.kzord, method='mean', causal=options.causal)
            parm = 'mwsize={}_kzord={}_causal={}'.format(options.mwsize, options.kzord, options.causal)
        elif options.method == 'moving_median':
            Xm = Tools.KZ_filter_pandas(Xt, options.mwsize, k=options.kzord, method='median', causal=options.causal)
            Ym = Tools.KZ_filter_pandas(Yt, options.mwsize, k=options.kzord, method='median', causal=options.causal)
            parm = 'mwsize={}_kzord={}_causal={}'.format(options.mwsize, options.kzord, options.causal)
        elif options.method == 'hp_filter':
            import statsmodels.api as sm
            _, Xm = sm.tsa.filters.hpfilter(Xt, lamb=options.lamb)
            _, Ym = sm.tsa.filters.hpfilter(Yt, lamb=options.lamb)
            parm = 'lambda={}'.format(options.lamb)
        else:
            raise Exception('{}: Unknown method.'.format(options.method))

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

    # # Creat directory for output
    # datadir1 = os.path.join(datadir, 'Decomposition')
    # try:
    #     os.makedirs(datadir1)
    # except OSError:
    #     pass
    #
    # fname = os.path.join(datadir1, '{}_[{}].pkl'.format(options.method, parm))

    # output file name
    fname = os.path.join(datadir, 'Decomposed.pkl')

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
