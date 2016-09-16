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
# import statsmodels.api as sm
import numpy as np
from numpy.linalg import norm
import scipy
from scipy import interpolate, stats, signal, sparse

from Pyshm import OSMOS, Tools, Stat

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Hide annoying trace back message
# sys.excepthook = lambda exctype,exc,traceback : print("{}: {}".format(exctype.__name__,exc))

import matplotlib.pyplot as plt
# from matplotlib.pyplot import figure

import matplotlib.colors as colors
# color_list = list(colors.cnames.keys())
color_list = ['red', 'green', 'blue', 'magenta', 'cyan', 'pink', 'lightgrey', 'yelow',
              'purple', 'mediumorchid', 'chocolate', 'blue', 'blueviolet', 'brown']

import mpld3
plt.style.use('ggplot')

__script__ = 'Plot results of the analysis of static data returned by BM.'


def main():
    # Load data
    usage_msg = '{} [options] input_data_file [output_directory]'.format(sys.argv[0])
    parm_msg = 'input_data_file : the file returned by the script Analysis_of_static_data-BM.py\noutput_directory : the directory where results are saved (default: in the same directory as input_data_file).'

    parser = OptionParser(usage_msg)

    parser.add_option('--ftype', dest='ftype', type='string', default='smoother', help='Method of Kalman filter: filter, smoother (default), predictor.')
    parser.add_option('--vthresh', dest='vthresh', type='float', default=4., help='Threshold value for event detection.')
    parser.add_option('--mwsize0', dest='mwsize0', type='int', default=6, help='Size of the moving window for local statistics (default=6).')
    parser.add_option('--mwsize1', dest='mwsize1', type='int', default=24*10, help='Size of the moving window for global statistics (default=24*10).')
    parser.add_option('--mad', dest='mad', action='store_true', default=False, help='Use median based estimator')
    parser.add_option('-v', '--verbose', dest='verbose', action='store_true', default=False, help='Print message.')

    (options, args) = parser.parse_args()

    if len(args) < 1:
        print('Usage: ' + usage_msg)
        print(parm_msg)

        sys.exit(0)
    else:  # check datadir
        infile = args[0]
        if not os.path.isfile(infile):
            raise FileNotFoundError(infile)
        idx0 = infile.rfind('/')+1
        idx1 = infile.rfind('.')
        loc = int(infile[idx0:idx1])

        if len(args) == 2:
            figdir = args[1]
        else:
            idx = infile.rfind('/')
            figdir = infile[:idx]

        try:
            os.makedirs(figdir)
        except OSError:
            pass
        # if not os.path.isdir(outdir):
        #     raise FileNotFoundError(outdir)

    # Load raw data
    try:
        with open(infile, 'rb') as fp:
            Res = pickle.load(fp)
    except Exception as msg:
        print(msg)

    Xall = Res['Xall']
    Yall = Res['Yall']
    Xdata = Res['Xdata']
    Ydata = Res['Ydata']
    sidx = Res['sidx']
    Ntrn = Res['Ntrn']
    component = Res['component']
    Nh = Res['Nh']
    Ng = Res['Ng']
    const = Res['const']
    Tidx = Xall.index  # Time index
    Nidx = np.isnan(np.asarray(Xall)) # nan index

    Xto, Pto, Yto, Eto = Res['Result'][options.ftype]

    # Average value of kernels
    tail = int(Xto.shape[0]/4) # use only a last part of the total length
    toto = np.mean(Xto[max(0, Xto.shape[0]-tail):,], axis=0)

    h1 = toto[:Nh]
    g1 = toto[Nh:-1] if const else toto[Nh:]
    c1 = toto[-1] if const else 0

    if options.verbose:
        print('\nLocation: {}'.format(loc))
        print('AR.Kernel: ', h1)
        print('Conv.Kernel: ', g1)
        print('Constant: ', c1)

    # Post-processing
    Xto[Nidx,] = np.nan
    Yto[Nidx,] = np.nan
    Pto[Nidx,] = np.nan
    Eto[Nidx,] = np.nan

    Hto = Xto[:, :Nh]  # AR kernel
    PHto = Pto[:, :Nh, :Nh]
    Gto = Xto[:, Nh:-1] if const else Xto[:, Nh:]  # Conv. kernel
    PGto = Pto[:, Nh:-1, Nh:-1] if const else Pto[:, Nh:, Nh:]
    Cto = Xto[:, -1] if const else np.zeros(Xto.shape[0])  # constant trend
    PCto = Pto[:, -1, -1] if const else np.zeros(Xto.shape[0])  # constant trend

    # Residual of prediction
    Err = pd.DataFrame(Ydata-Yto, index=Tidx)

    if options.mad:
        mErr0 = Err.rolling(window=options.mwsize0, min_periods=1).median() #.bfill()
        sErr0 = 1.4826 * (Err-mErr0).abs().rolling(window=options.mwsize0, min_periods=1).median() #.bfill()
        mErr1 = Err.rolling(window=options.mwsize1, min_periods=1).median() #.bfill()
        sErr1 = 1.4826 * (Err-mErr1).abs().rolling(window=options.mwsize1, min_periods=1).median() #.bfill()
    else:
        mErr0 = Err.rolling(window=options.mwsize0, min_periods=1).mean() #.bfill()
        sErr0 = Err.rolling(window=options.mwsize0, min_periods=1).std() #.bfill()
        mErr1 = Err.rolling(window=options.mwsize1, min_periods=1).mean() #.bfill()
        sErr1 = Err.rolling(window=options.mwsize1, min_periods=1).std() #.bfill()

    # drop the begining
    mErr0.iloc[:int(options.mwsize0*1.1)]=np.nan
    sErr0.iloc[:int(options.mwsize0*1.1)]=np.nan
    mErr1.iloc[:int(options.mwsize1*1.1)]=np.nan
    sErr1.iloc[:int(options.mwsize1*1.1)]=np.nan

    mErr0.iloc[Nidx] = np.nan
    sErr0.iloc[Nidx] = np.nan
    mErr1.iloc[Nidx] = np.nan
    sErr1.iloc[Nidx] = np.nan

    # Plot the kernel
    fig, axes = plt.subplots(1,2,figsize=(20,5))
    # plot(AData[loc].index[tidx0+max(ng,nh):twsize-max(ng,nh)+tidx0], err)
    axa = axes[0]
    axa.plot(h1, 'b')
    axa.set_title('Kernel of auto-regression')
    _ = axa.set_xlim(-1,)

    axa = axes[1]
    axa.plot(g1)
    axa.set_title('Kernel of convolution')
    _ = axa.set_xlim(-1,)

    fig.savefig(figdir+'/{}_Kernel.pdf'.format(loc), bbox_inches='tight')
    plt.close(fig)

    # Plot the residual
    nfig, k = 4, 0
    fig, axes = plt.subplots(nfig,1, figsize=(20, nfig*5), sharex=True)

    axa = axes[k]
    axa.plot(Yall, color='b', alpha=0.5, label='Elongation')
    axb = axa.twinx()
    axb.patch.set_alpha(0.0)
    axb.plot(Xall, color='r', alpha=0.5, label='Temperature')
    axa.legend(loc='upper left')
    axb.legend(loc='upper right')
    axa.set_title('Signals of the location {}'.format(loc))
    k+=1

    axa = axes[k]
    axa.plot(Tidx, Ydata, color='b', alpha=0.5, linewidth=2, label='Elongation')
    axa.plot(Tidx, Yto, color='c', alpha=0.7, label='Prediction')
    axb = axa.twinx()
    axb.patch.set_alpha(0.0)
    sgn = np.sign(g1[0]) if len(g1)>0 else 1  # adjust the sign of Xdata
    axb.plot(Tidx, sgn*Xdata, color='r', alpha=0.5, label='Temperature')
    axa.legend(loc='upper left')
    axb.legend(loc='upper right')
    t0, t1 = Tidx[sidx], Tidx[sidx+Ntrn]
    # axa.fill_betweenx(np.arange(-100,100), t0, t1, color='c', alpha=0.2)
    axa.axvspan(t0, t1, color='c', alpha=0.2)
    axa.set_title('{} components of the location {} (sign adjusted for the temperature)'.format(component, loc))
    k+=1

    axa = axes[k]
    # axa.plot(abs(Err-mErr0)/sErr1)
    axa.plot(abs(Err-mErr1)/sErr1)
    # axa.plot(Err/sErr1)
    # axa.set_ylim((0,6))
    axa.fill_between(Tidx, 0, options.vthresh, color='c', alpha=0.2)
    axa.set_title('Normalized residual')
    k+=1

    axa = axes[k]
    axa.plot(mErr0, color='b', alpha=0.5, label='Local mean window={}'.format(options.mwsize0))
    axa.plot(mErr1, color='c', label='Local mean window={}'.format(options.mwsize1))
    axa.legend(loc='upper left')
    axb = axa.twinx()
    axb.patch.set_alpha(0.0)
    axb.plot(sErr0, color='r', alpha=0.5, label='Local standard deviation window={}'.format(options.mwsize0))
    axb.plot(sErr1, color='m', label='Local standard deviation window={}'.format(options.mwsize1))
    axb.legend(loc='upper right')
    axa.set_title('Local mean and standard deviation of the residual')
    k+=1

    # axa = axes[k]
    # axa.plot(mErr1/sErr1)
    # # axa.plot(mErr0/sErr0)
    # # axes.set_ylim((-6,6))
    # # vthresh = 4
    # # axes.fill_between(Ydata0.index, -vthresh, vthresh, color='g', alpha=0.1)
    # # axa.hlines(0, Tidx[0], Tidx[-1], color='b', linewidth=3, alpha=0.2)
    # axa.set_title('Normalized mean of the residual')
    # k+=1
    #
    # axa = axes[k]
    # axa.plot(Err)
    # axa.set_title('Residual')
    # k+=1

    fname = figdir+'/{}'.format(loc)
    fig.savefig(fname+'.pdf', bbox_inches='tight')
    mpld3.save_html(fig, fname+'.html')
    plt.close(fig)

    # Plot the evolution of kernels
    fig, axes = plt.subplots(3,1,figsize=(20,15))

    Nc = 3 # number of the leading coefficients
    alpha = 0.1

    axa=axes[0]
    for i,c in zip(range(min(Nc, Nh)), color_list):
        axa.plot(Tidx, Hto[:,i], color=c, label='$h_{}$'.format(i))
        Kr = -stats.norm.ppf(alpha/2)*np.sqrt(PHto[:,i,i])
        axa.fill_between(Tidx, Hto[:,i]-Kr, Hto[:,i]+Kr, color=c, alpha=0.2)

    x0, x1 = axa.get_xlim()
    axa.hlines(0, x0, x1, color='y', linewidth=3, alpha=0.7)
    axa.legend()
    axa.set_title('Evolution of the AR kernel')

    axa=axes[1]
    for i,c in zip(range(min(Nc, Ng)), color_list):
        axa.plot(Tidx, Gto[:,i], color=c, label='$g_{}$'.format(i))
        Kr = -stats.norm.ppf(alpha/2)*np.sqrt(PGto[:,i,i])
        axa.fill_between(Tidx, Gto[:,i]-Kr, Gto[:,i]+Kr, color=c, alpha=0.2)

    x0, x1 = axa.get_xlim()
    axa.hlines(0, x0, x1, color='y', linewidth=3, alpha=0.7)
    axa.legend()
    axa.set_title('Evolution of the convolution kernel')

    axa=axes[2]
    axa.plot(Tidx, Cto, color='r')
    Kr = -stats.norm.ppf(alpha/2)*np.sqrt(PCto)
    axa.fill_between(Tidx, Cto-Kr, Cto+Kr, color='r', alpha=0.2)
    axa.set_title('Evolution of the constant trend')

    fname = figdir+'/{}_evolution'.format(loc)
    fig.savefig(fname+'.pdf', bbox_inches='tight')
    mpld3.save_html(fig, fname+'.html')
    plt.close(fig)

    if options.verbose:
        print('Figures saved in {}'.format(figdir))


if __name__ == "__main__":
    print(__script__)
    print()
    main()
