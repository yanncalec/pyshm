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
from libs import OSMOS, Tools, Stat, Kalman

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

import matplotlib.pyplot as plt
# from matplotlib.pyplot import figure
import matplotlib.colors as colors
# color_list = list(colors.cnames.keys())
color_list = ['green', 'pink', 'lightgrey', 'magenta', 'cyan', 'red', 'yelow',
              'purple', 'blue', 'mediumorchid', 'chocolate', 'blue',
              'blueviolet', 'brown']

import mpld3
plt.style.use('ggplot')

__script__ = 'Analysis of static data using a BM model'


def main():
    # Load data
    usage_msg = '{} [options] input_data_file output_directory'.format(sys.argv[0])
    parser = OptionParser(usage_msg)

    parser.add_option('--loc', dest='loc', type='int', default=None, help='Location key ID of the sensor. If not given all sensors will be processed.')
    parser.add_option('--Nq', dest='Nq', type='int', default=10, help='Length of the convolution kernel (default=10).')
    parser.add_option('--Ntrn', dest='Ntrn', type='int', default=8*30*24, help='Length of the training data (default=24*30*8).')
    parser.add_option('--vthresh', dest='vthresh', type='float', default=4., help='Threshold value for event detection.')
    parser.add_option('--mwsize', dest='mwsize', type='int', default=24*10, help='Size of the moving window for local statistics (default=24*10).')
    parser.add_option('--component', dest='component', type='string', default='AllDiff', help='Type of components to be analysed: All, AllDiff (default), Seaonal, SeaonalDiff, Trend, TrendDiff.')
    parser.add_option('--sigmaq2', dest='sigmaq2', type='float', default=1e-6, help='sigmar^2.')
    parser.add_option('--sigmar2', dest='sigmar2', type='float', default=1e-5, help='sigmaq^2.')
    parser.add_option('--parmestim', dest='parmestim', type='string', default=None, help='method for parameter estimation: LS, MLE (default=No estimation of parameters).')
    parser.add_option('--ftype', dest='ftype', type='string', default='smoother', help='Method of Kalman filter: filter, smoother (default), predictor.')
    parser.add_option('-p', '--penal', dest='penal', action='store_true', default=False, help='Use penalization in the least-sqaure estimation.')
    parser.add_option('-n', '--nrml', dest='nrml', action='store_true', default=False, help='Apply normalization of data (default: no normalization).')
    parser.add_option('-v', '--verbose', dest='verbose', action='store_true', default=False, help='Print message.')

    (options, args) = parser.parse_args()

    if len(args) == 0:
        print(usage_msg)
        sys.exit(0)
    else:  # check datadir
        infile = args[0]
        if not os.path.isfile(infile):
            raise FileNotFoundError(infile)

        idx = infile.rfind('/')
        outdir = args[1] + '/' + infile[idx+1:-4]
        try:
            os.makedirs(outdir)
        except OSError:
            pass

    mwsize = options.mwsize  # window size for moving average

    # Load raw data
    with open(infile, 'rb') as fp:
        Data0 = pickle.load(fp)['Static_Data']

    Locations = list(Data0.keys()); Locations.sort() # Locations of the current project

    Nq = options.Nq  # length of the convolution kernel
    # Ntrn = options.Ntrn  # length of training data
    # pcoef = np.sqrt(Ntrn) * Tools.exponential_weight(Nq, w0=5e-1)
    # vthresh = options.vthresh  # threshold for the detection

    figdir = outdir+'/Nq={}'.format(Nq)
    try:
        os.makedirs(figdir)
    except OSError:
        pass

    if options.loc is not None and options.loc in Locations:
        loc_list = [options.loc]
    else:
        loc_list = Locations

    Res = {} # dictionary for keeping results

    for loc in loc_list:
        Res[loc] = {}

        # choose component
        if options.component == 'All':
            Xraw, Yraw = Data0[loc]['Temperature'].copy(), Data0[loc]['Elongation'].copy()
        elif options.component == 'AllDiff':
            Xraw, Yraw = Data0[loc]['Temperature'].diff(), Data0[loc]['Elongation'].diff()
        elif options.component == 'Seasonal':
            Xraw, Yraw = Data0[loc]['Temperature_seasonal'].copy(), Data0[loc]['Elongation_seasonal'].copy()
        elif options.component == 'SeasonalDiff':
            Xraw, Yraw = Data0[loc]['Temperature_seasonal'].diff(), Data0[loc]['Elongation_seasonal'].diff()
        elif options.component == 'Trend':
            Xraw, Yraw = Data0[loc]['Temperature_trend'].copy(), Data0[loc]['Elongation_trend'].copy()
        elif options.component == 'TrendDiff':
            Xraw, Yraw = Data0[loc]['Temperature_trend'].diff(), Data0[loc]['Elongation_trend'].diff()
        else:
            raise NotImplementedError('Unknown type of component: {}'.format(options.component))

        Xall = Data0[loc]['Temperature'].copy()
        Yall = Data0[loc]['Elongation'].copy()
        Tidx = Xall.index  # time index
        Nidx = np.isnan(np.asarray(Xall)) # nan index

        if options.nrml:
            nX, nY = Tools.safe_norm(np.asarray(Xraw)), Tools.safe_norm(np.asarray(Yraw))
        else:
            nX, nY = 1, 1
        Xdata, Ydata = np.asarray(Xraw)/nX, np.asarray(Yraw)/nY

        if options.verbose:
            print('\nLocation: {}'.format(loc))

        # Estimation of mean kernel by least-square
        if options.sigmaq2 is None or options.sigmar2 is None or options.parmestim is not None:
            if options.verbose:
                print('Estimation of parameters by {} method...'.format(options.parmestim))

            Xtrn = Xdata[:options.Ntrn].copy()
            Ytrn = Ydata[:options.Ntrn].copy()

            if options.parmestim == 'LS':
                sigmaq2_vec, sigmar2 = Tools.LS_estimation_4SS(Xtrn, Ytrn, Nq, mwsize=mwsize, penal=options.penal, dT=1, causal=False) # set penal=True to avoid numerical instability
                # sigmaq2 = sigmaq2_vec[0] * 1
                sigmaq2 = sigmaq2_vec * 1
            # elif options.parmestim == 'MLE':
            #     parms = Kalman.Kalman_MLE()
            else:
                raise NotImplementedError('Unknown method of estimation: '+options.parmestim)
        else:
            sigmaq2, sigmar2 = options.sigmaq2, options.sigmar2

        if options.verbose:
            print('Apply Kalman Filter with the parameters: sigmaq2={}, sigmar2={}'.format(sigmaq2, sigmar2))

        # Deconvolution by KF
        LXtn, LPtn, res, B0 = Kalman.Kalman_deconvolution_BM(Xdata, Ydata, Nq=Nq, X0=None, P0=None, sigmar2=sigmar2, sigmaq2=sigmaq2)

        # Assemble results
        Xtt = asarray(res[0])
        Ptt = asarray(res[1])
        Xtm = asarray(res[2])
        Ptm = asarray(res[3])
        Etm = squeeze(asarray(res[4], dtype=np.float64))
        St = squeeze(asarray(res[5]))
        Kt = asarray(res[6])
        LLHt = asarray(res[8])
        # dLLHt = asarray(res[9])
        Xtn = asarray(LXtn)
        Ptn = asarray(LPtn)

        Ytt = np.sum(B0 * Xtt, axis=1) # Filteration
        Ytn = np.sum(B0 * Xtn, axis=1) # Smoothing
        Ytm = np.sum(B0 * Xtm, axis=1) # Prediction

        # Errors
        Ett = Ydata - Ytt
        Etn = Ydata - Ytn
        # Etm = Ydata - Ytm

        # Choose a set of result: smoothing/filteration/prediction
        if options.ftype == 'filter': # Use KF result
            Xto, Yto, Pto = Xtt, Ytt, Ptt
        elif options.ftype == 'smoother':  # Use Kalman smoother result
            Xto, Yto, Pto = Xtn, Ytn, Ptn
        else:  # Use Kalman predictor result
            Xto, Yto, Pto = Xtm, Ytm, Ptm #

        Xto, Yto, Pto = Xtn*(nY/nX), Ytn*nY, Ptn*(nY/nX)**2  # de-normalization to restore true values

        # Residual
        Eto0 = pd.Series(Yto-Yraw)
        Eto = asarray(Eto0)
        Sto = asarray(Eto0.rolling(mwsize, min_periods=1, center=True).std())  # standard deviation
        Mto = asarray(Eto0.rolling(mwsize, min_periods=1, center=True).mean())  # mean

        # Post-processing:
        Xto[Nidx,] = nan
        Yto[Nidx,] = nan
        Pto[Nidx,] = nan
        Eto[Nidx,] = nan
        Sto[Nidx,] = nan

        # Save results
        Res[loc]['Xto'] = Xto.copy()
        Res[loc]['Yto'] = Yto.copy()
        Res[loc]['Pto'] = Pto.copy()

        if options.verbose:
            print('Relative error: {}'.format(Tools.safe_norm(Eto)/Tools.safe_norm(Yraw)))

        # Plot
        fig, axes = plt.subplots(4, 1, figsize=(20,20), sharex=True)

        axa = axes[0]
        axa.plot(Tidx, Ydata, color='b', alpha=0.5, linewidth=2, label='Elongation')
        axa.plot(Tidx, Yto, color='c', alpha=0.7, label='Deconvolution')
        axb = axa.twinx()
        axb.patch.set_alpha(0.0)
        axb.plot(Tidx, Xdata, color='r', alpha=0.5, label='Temperature')
        axa.legend(loc='upper left')
        axb.legend(loc='upper right')
        axa.set_title('{} components of the location {}'.format(options.component, loc))

        axa = axes[1]
        axa.plot(Yall, color='b', alpha=0.5, label='Elongation')
        axb = axa.twinx()
        axb.patch.set_alpha(0.0)
        axb.plot(Xall, color='r', alpha=0.5, label='Temperature')
        axa.legend(loc='upper left')
        axb.legend(loc='upper right')
        axa.set_title('Signals of the location {}'.format(loc))

        axa = axes[2]
        axa.plot(Tidx, abs(Eto/Sto))
        axa.set_title('Normalized residual of the deconvolution $v_t/\sigma_t$')
        axa.set_ylim((0,6))
        axa.fill_between(Tidx, 0, options.vthresh, color='g', alpha=0.1)

        # axa = axes[k]
        # axa.plot(Ydata0.index, Nto)
        # # axes.set_ylim((-6,6))
        # # vthresh = 4
        # # axes.fill_between(Ydata0.index, -vthresh, vthresh, color='g', alpha=0.1)
        # axa.hlines(0, Ydata0.index[0], Ydata0.index[-1], color='y', linewidth=3, alpha=0.5)
        # axa.set_title('Normalized mean residual of the deconvolution $\mu_t/\sigma_t$')
        # k+=1

        axa = axes[3]
        # Pto[:Nq,:,:] = nan # remove the begining
        for i, c in zip(range(3), ['r', 'g', 'b', 'm', 'c']):
            axa.plot(Tidx, Xto[:,i], color=c, label='$a_{{{}}}$'.format(i))
            Kr = -stats.norm.ppf(0.05/2)*np.sqrt(Pto[:,i,i])
            axa.fill_between(Tidx, Xto[:,i]-Kr, Xto[:,i]+Kr, color=c, alpha=0.1)

        x0, x1 = axa.get_xlim()
        axa.hlines(0, x0, x1, color='y', linewidth=3, alpha=0.7)
        axa.legend()
        axa.set_title('Evolution of the convolution kernel')

        fname = figdir + '/{}_Kalman_{}_msigmaq={:.2e}_sigmar={:.2e}'.format(loc, options.component, mean(sigmaq2), sigmar2) + ('_nrml' if options.nrml else '')
        mpld3.save_html(fig, fname+'.html')
        fig.savefig(fname+'.pdf', bbox_inches='tight')
        plt.close(fig)

        # Plot the kernel
        fig = plt.figure(figsize=(10,8))
        Xto1 = Xto.copy(); Xto1[np.isnan(Xto)]=0
        plt.plot(np.mean(Xto1,axis=0), 'b')
        plt.xlim(-1, Nq)
        plt.title('Mean value of the convolution kernel for {} component'.format(options.component))

        fig.savefig(fname+'_Kernel.pdf', bbox_inches='tight')
        plt.close(fig)

        try:
            with open(fname+'.pkl', 'wb') as fp:
                pickle.dump(Res, fp)
            if options.verbose:
                print('Results saved in {}.pkl'.format(fname))
        except Exception as msg:
            print(Fore.RED + 'Warning: ', msg)
            print(Style.RESET_ALL)



if __name__ == "__main__":
    print(__script__)
    print()
    main()
