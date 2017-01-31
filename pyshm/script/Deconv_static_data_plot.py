#!/usr/bin/env python

"""Plot results of deconvolution of static data.
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from pyshm import Stat, Tools
from pyshm.script import load_data#, dictkey2int

class Options:
    pass


xobs_style = {'color': 'r', 'linewidth': 1, 'alpha':0.5, 'label':'Temperature: observation'}
yobs_style = {'color': 'b', 'linewidth': 1, 'alpha':0.5, 'label':'Elongation: observation'}
yprd_style = {'color': 'g', 'linewidth': 2, 'alpha':0.8, 'label':'Elongation: prediction'}
aprd_style = {'color': 'r', 'linewidth': 2, 'alpha':0.8, 'label':'Elongation: thermal contribution'}
bprd_style = {'color': 'b', 'linewidth': 2, 'alpha':0.8, 'label':'Elongation: non-thermal contribution'}
yerr_style = {'color': 'c', 'linewidth': 2, 'alpha':0.8, 'label':'Elongation: residual'}


def plot_results(xobs0, yobs0, yprd0, aprd0, bprd0, component,
                 trnperiod=None, Midx=None, mad=False,
                 mwsize=240, vthresh=3., minperiod=24, ithresh=0.7):
    """Plot the results of one sensor with statistical analysis.

    Args:
        xobs0, yobs0: X and Y components of one sensor
        yprd0, aprd0, bprd0: Y prediction, thermal and non-thermal contributions
        component: type of component ('seasonal' or 'trend' or 'all')
        trnperiod: training period starting and ending index
        Midx: missing index
        other parameters: see main()
    """
    assert(len(xobs0)==len(yobs0)==len(yprd0)==len(aprd0)==len(bprd0))
    xobs = xobs0.copy(); xobs[Midx] = np.nan
    yobs = yobs0.copy(); yobs[Midx] = np.nan
    yprd = yprd0.copy(); yprd[Midx] = np.nan
    aprd = aprd0.copy(); aprd[Midx] = np.nan
    bprd = bprd0.copy(); bprd[Midx] = np.nan
    yerr = yobs - yprd  # residual

    # statistics
    # mXXX for mean, sXXX for standard deviation, nXXX for normalization
    merr, serr = Stat.local_mean_std(yerr, mwsize, mad=mad, causal=False, drop=True)

    # drop the missing values
    merr[Midx] = np.nan
    serr[Midx] = np.nan

    nfig, k = 4, 0
    fig, axes = plt.subplots(nfig, 1, figsize=(20, nfig*5), sharex=True)

    # Observation and prediction
    axa = axes[k]
    axa.plot(yobs, **yobs_style)
    axa.plot(yprd, **yprd_style)
    axa.plot(yerr, **yerr_style)
    axa.legend(loc='upper left', fancybox=True, framealpha=0.5)
    axb = axa.twinx()
    axb.patch.set_alpha(0.0)
    # axb.plot(xobs, **xobs_style)
    axb.plot(xobs, color='r', alpha=0.5, linewidth=1, label='Temperature: observation')
    axb.legend(loc='upper right', fancybox=True, framealpha=0.5)
    # mark the training period
    if trnperiod is not None:
        t0, t1 = xobs.index[trnperiod[0]], xobs.index[trnperiod[1]-1]
        ylim = axb.get_ylim()
        axb.fill_betweenx(ylim, t0, t1, color='c', alpha=0.2)
        # axa.axvspan(t0, t1, color='c', alpha=0.2)
    axa.set_title('Observations and prediction of elongation')
    k+=1

    # Prediction, Thermal and non-thermal contribution
    axa = axes[k]
    axa.plot(yprd, **yprd_style)
    axa.plot(aprd, **aprd_style)
    if bprd is None:
        axa.set_title('Prediction and thermal contribution')
    else:
        axa.plot(bprd, **bprd_style)
        # axa.plot(bprd, color='b', alpha=0.8, linewidth=2, label='Non-thermal contribution')
        axa.set_title('Prediction, thermal and non-thermal contributions')
    axa.legend(loc='upper left', fancybox=True, framealpha=0.5)
    k+=1

    # Local statistics of residual
    axa = axes[k]
    axa.plot(yerr, **yerr_style)
    axa.plot(merr, color='b', alpha=0.5, linewidth=2, label='Local mean with window={}'.format(mwsize))
    axa.legend(loc='upper left', fancybox=True, framealpha=0.5)
    axb = axa.twinx()
    axb.patch.set_alpha(0.0)
    axb.plot(serr, color='r', alpha=0.5, linewidth=2, label='Local standard deviation with window={}'.format(mwsize))
    axb.legend(loc='upper right', fancybox=True, framealpha=0.5)
    axa.set_title('Local statistics and instability periods')
    k+=1

    # Component-dependent analysis
    axa = axes[k]
    if component.upper() == 'TREND' or component.upper() == 'ALL':
        # Trend component: relative instability period detection based on Hurst exponent
        herr, berr, verr = Stat.Hurst(yerr, 24*5, sclrng=(0,7))  # Hurst exponent                              
        herr[np.isnan(herr)] = np.inf  # nan -> inf to avoid warning of comparison 
        hidc = Tools.L_filter(herr>ithresh, wsize=minperiod)  # filtered indicator of instability
        tidx = Tools.find_block_true(hidc)  # starting-ending indexes of blocks of instability

        axa.plot(yerr, **yerr_style)
        axa.legend(loc='upper left', fancybox=True, framealpha=0.5)

        axb = axa.twinx()
        axb.patch.set_alpha(0.0)

        herr = pd.Series(herr, index=xobs.index); herr[Midx] = np.nan  # convert to pandas
        axb.plot(herr, color='tomato', alpha=0.8, linewidth=2, label='Hurst exponent')
        axb.hlines(ithresh, xobs.index[0], xobs.index[-1], linewidth=2, color='tomato')
        ylim = axb.get_ylim()
        # ylim = (-100, 100)
        for (t0, t1) in tidx:
            axb.fill_betweenx(ylim, xobs.index[t0], xobs.index[t1-1], color='r', alpha=0.2)
                    
        axb.legend(loc='upper right', fancybox=True, framealpha=0.5)
        axb.set_title('Hurst exponent and relative instability periods')
    elif component.upper() == 'SEASONAL':
        # Seasonal component: event detection based on normalized residual
        nerr = abs(yerr-merr)/serr  # normalized error
        axa.plot(nerr, color='r', alpha=0.8, linewidth=1, label='Normalized residual')
        axa.set_title('Normalized residual: (error-mean)/std')
        # axa.set_ylim((0,6))
        axa.fill_between(xobs.index, 0, vthresh, color='c', alpha=0.2)
    else:
        raise TypeError('Unknown type of component:', component)
    
    k+=1

    return fig, axes


def plot_static_kernel(Knel):
    """Plot mean value of the static kernel.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    # from pyshm import Stat

    fig, axes = plt.subplots(1,len(Knel),figsize=(6*len(Knel), 4))
    for n,K in enumerate(Knel):
        toto = np.asarray(K)
        # print(toto.shape)
        A = np.mean(toto, axis=-1).T
        # print(A.shape)
        axa = axes[n]
        axa.plot(A[0])
        axa.set_title('Kernel of the group {}'.format(n))
    return fig, axes


def mean_dynamic_kernel(Knel):
    """Calculate mean value (of different groups) of the dynamic kernel.
    """

    Ng = len(Knel[0]) # number of groups
    Nt = len(Knel)  # duration
    A = []

    for g in range(Ng):
        toto = np.asarray([K[g] for K in Knel])
        A.append(np.mean(toto, axis=-1).transpose(2,1,0)[0])
    return A


def plot_dynamic_kernel(Knel, Tidx, ncoef=3):
    """Plot mean value of the dynamic kernel.
    Args:
        Knel (list):  Knel[t][g][n] is the n-th kernel matrix (of shape 1-by-?) of the group g at the time index t.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    As = mean_dynamic_kernel(Knel)
    fig, axes = plt.subplots(len(As),1,figsize=(20, 5*len(As)))

    for g,A in enumerate(As):
        axa = axes[g]
        for c in range(ncoef):
            axa.plot(Tidx[:A.shape[1]], A[c,:], label='coefficient {}'.format(c))
        axa.legend()
    return fig, axes


import sys, os, argparse

__script__ = __doc__

def main():
    # usage_msg = '%(prog)s [options] <infile> [outdir]'
    # parser = argparse.ArgumentParser(description=__script__, usage=usage_msg)
    parser = argparse.ArgumentParser(description=__script__)

    parser.add_argument('infile', type=str, help='file returned by the script of data analysis')
    parser.add_argument('outdir', nargs='?', type=str, default=None, help='directory where figures are saved (default: in the same folder as infile).')

    lstat_opts = parser.add_argument_group('Local statistic options')  # local statistics
    lstat_opts.add_argument('--mad', dest='mad', action='store_true', default=False, help='Use median based estimator (default: use empirical estimator).')
    lstat_opts.add_argument('--mwsize', dest='mwsize', type=int, default=240, help='Size of the moving window (default=240).', metavar='integer')
    lstat_opts.add_argument('--vthresh', dest='vthresh', type=float, default=3., help='Threshold value for event detection in seasonal components (default=4).', metavar='float')
    #     lstat_opts.add_argument('--causal', dest='causal', action='store_true', default=False, help='Use causal window (default: non causal).')

    hurst_opts = parser.add_argument_group('Hurst exponent options for trend component')  # Hurst exponent
    # hurst_opts.add_argument('--mwsizehurst', dest='mwsizehurst', type=int, default=24*5, help='Size of the moving window (default=120).', metavar='integer')
    hurst_opts.add_argument('--ithresh', dest='ithresh', type=float, default=0.7, help='Threshold value for instability detection (default=0.7).', metavar='float')
    hurst_opts.add_argument('--minperiod', dest='minperiod', type=int, default=24, help='Minimal length of instability period (default=24).', metavar='integer')

    parser.add_argument('-v', '--verbose', dest='verbose', action='count', default=False, help='Print message.')

    options = parser.parse_args()

    # Lazy import
    import pickle
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import mpld3
    plt.style.use('ggplot')

    if not os.path.isfile(options.infile):
        raise FileNotFoundError(options.infile)

    # Load raw data
    Res = load_data(options.infile)
    # print(Res['options'], Res['trnperiod'], Res['Ntrn'])
    # output directory for figures
    if options.outdir is not None:
        figdir0 = options.outdir
    else:
        idx = options.infile.rfind('.')
        figdir0 = options.infile[:idx]

    Xcpn = Res['Xcpn']  # Component of temperature
    Ycpn = Res['Ycpn']  # Component of elongation
    Yprd = Res['Yprd']  # Prediction of elongation
    # Yerr = Res['Yerr']  # Error of prediction
    algo_options = dict(Res['algo_options'])  # options of parameters
    Aprd = Res['Aprd']  # Contribution of first group of inputs
    Bprd = Res['Bprd'] if algo_options['lagy']>0 else None  # Contribution of second group of inputs
    Knel = Res['Knel'] if 'Knel' in Res else None  # Kernel matrices
    # Mxd = Res['Mxd']  # Objects of deconvolution model
    Midx = Res['Midx']  # Indicator of missing values
    # Tidx = Xcpn.index  # Time index

    component = algo_options['component']
    staticflag = algo_options['subcommand'].upper() == 'STATIC'
    if staticflag:
        trnperiod = algo_options['trnperiod']
    else:
        trnperiod = None
        sigmar2 = algo_options['sigmar2']
        sigmaq2 = algo_options['sigmaq2']
        kalman = algo_options['kalman']

    # Plot all results
    Locations = list(Yprd.keys())
    for loc in Locations:
        if options.verbose:
            print('Plotting the result of location {}...'.format(loc))

        figdir = os.path.join(figdir0, str(loc))
        try:
            os.makedirs(figdir)
        except OSError:
            pass

        # Plot the prediction results
        fig, axes = plot_results(Xcpn[loc], Ycpn[loc], Yprd[loc], Aprd[loc],
                                 Bprd[loc] if Bprd is not None else None, component,
                                 trnperiod=trnperiod, Midx=Midx[loc], mad=options.mad,
                                 mwsize=options.mwsize, vthresh=options.vthresh,
                                 minperiod=options.minperiod, ithresh=options.ithresh)
        # plt.suptitle('Location {}, {} component'.format(loc, component), position=(0.5,1.1),fontsize=20)
        # plt.tight_layout()
        # fname = os.path.join(figdir, 'Results')
        # fig.savefig(fname+'.pdf', bbox_inches='tight')
        # mpld3.save_html(fig, fname+'.html')
        # plt.close(fig)

        # # Plot and analyse the residuals
        # fig, axes = plot_residuals(Xcpn[loc], Ycpn[loc], Yprd[loc], component, Midx=Midx[loc], mad=options.mad,
        #                            mwsize=options.mwsize, vthresh=options.vthresh,
        #                            minperiod=options.minperiod, ithresh=options.ithresh)
        # # plt.suptitle('Location {}, {} component'.format(loc, component), position=(0.5,1.1),fontsize=20)
        # plt.tight_layout()
        if component.upper() in ['TREND', 'ALL']:
            fname = os.path.join(figdir, 'Results_[window={}_thresh={}_mad={}]'.format(options.mwsize, options.ithresh, options.mad))
        else:
            fname = os.path.join(figdir, 'Results_[window={}_thresh={}_mad={}]'.format(options.mwsize, options.vthresh, options.mad))
            # fname = os.path.join(figdir, 'Residual_[window={}_mad={}_causal={}]'.format(options.mwsize, options.mad, options.causal))
        fig.savefig(fname+'.pdf', bbox_inches='tight')
        mpld3.save_html(fig, fname+'.html')
        plt.close(fig)

        if Knel:  # if kernel is stored in the data file
            if staticflag:
                fig, axes = plot_static_kernel(Knel[str(loc)])  # str(loc) since in Json file keys of a dictionary must be string
                plt.suptitle('Location {}, {} component'.format(loc, component), position=(0.5,1.1),fontsize=20)
                plt.tight_layout()
                fname = os.path.join(figdir, 'Kernel_static')
                fig.savefig(fname+'.pdf', bbox_inches='tight')
            else:
                fig, axes = plot_dynamic_kernel(Knel[str(loc)], Xcpn.index)
                plt.suptitle('Location {}, {} component'.format(loc, component), position=(0.5,1.1),fontsize=20)
                plt.tight_layout()
                fname = os.path.join(figdir, 'Kernel_dynamic')
                fig.savefig(fname+'.pdf', bbox_inches='tight')


    if options.verbose:
        print('Figures saved in {}'.format(figdir0))


if __name__ == "__main__":
    print(__script__)
    # print()
    main()


# def plot_residuals(Xcpn, Ycpn, Yprd, mwsize, vthresh, Midx=None, mad=False, causal=False, component='TREND'):
#     """Plot the residuals of one sensor with statistical analysis.

#     Args:
#         Xcpn,Ycpn: X and Y components of one sensor
#         Yprd,Aprd,Bprd: Y prediction, thermal and non-thermal contributions
#         trn_idx: training period starting and ending index
#     """
#     Err = Ycpn - Yprd  # residual
#     Tidx = Err.index  # time index

#     # statistics
#     # mXXX for mean, sXXX for standard deviation, nXXX for normalization
#     mwsize0 = mwsize; mwsize1 = max(2, 2*mwsize0//3)
#     mErr0, sErr0 = Stat.local_mean_std(Err, mwsize0, mad=mad, causal=causal, drop=True)
#     mErr1, sErr1 = Stat.local_mean_std(Err, mwsize1, mad=mad, causal=causal, drop=True)

#     # drop the missing values
#     mErr0[Midx==True] = np.nan
#     sErr0[Midx==True] = np.nan
#     mErr1[Midx==True] = np.nan
#     sErr1[Midx==True] = np.nan
#     # if Midx is not None:
#     #     for loc,val in mErr.items():
#     #         val[Midx==True] = np.nan
#     #     # sErr.iloc[:int(mwsize*1.1)]=np.nan
#     #     for loc,val in sErr.items():
#     #         val[Midx==True] = np.nan

#     nErr0 = abs(Err-mErr0)/sErr0  # normalized error
#     nErr1 = abs(Err-mErr1)/sErr1  # normalized error

#     nfig, k = 3, 0
#     fig, axes = plt.subplots(nfig,1, figsize=(20, nfig*5), sharex=True)

#     # Observation and prediction
#     axa = axes[k]
#     axa.plot(Ycpn, color='b', alpha=0.5, linewidth=1, label='Elongation observation')
#     axa.plot(Yprd, color='g', alpha=0.8, linewidth=2, label='Elongation prediction')
#     axa.legend(loc='upper left', fancybox=True, framealpha=0.5)
#     axb = axa.twinx()
#     axb.patch.set_alpha(0.0)
#     axb.plot(Xcpn, color='r', alpha=0.5, linewidth=1, label='Temperature observation')
#     axb.legend(loc='upper right', fancybox=True, framealpha=0.5)
#     axa.set_title('Observations and prediction of elongation')
#     k+=1

#     # Residual
#     axa = axes[k]
#     axa.plot(Err, color='b', alpha=0.5, linewidth=1, label='Residual')
#     axa.plot(mErr0, color='c', alpha=0.8, linewidth=2, label='Local mean with window={}'.format(mwsize0))
#     # axa.plot(mErr1, color='navy', alpha=0.8, linewidth=2, label='Local mean with window={}'.format(mwsize1))
#     axa.legend(loc='upper left', fancybox=True, framealpha=0.5)
#     axb = axa.twinx()
#     axb.patch.set_alpha(0.0)
#     axb.plot(sErr0, color='r', alpha=0.8, linewidth=2, label='Local standard deviation with window={}'.format(mwsize0))
#     axb.plot(sErr1, color='tomato', alpha=0.8, linewidth=2, label='Local standard deviation with window={}'.format(mwsize1))
#     axb.legend(loc='upper right', fancybox=True, framealpha=0.5)
#     axa.set_title('Residual and local statistics with moving window')
#     k+=1

#     # Alarm
#     # Normalized residual for seasonal component
#     axa = axes[k]
#     if component.upper() == 'TREND' or component.upper() == 'ALL':
#         axa.plot(sErr1/sErr0, color='r', alpha=0.8, linewidth=1, label='Normalized variation')
#         # axa.plot(sErr0, color='r', alpha=0.8, linewidth=1, label='Normalized variation')
#         axa.set_title('Normalized variation: std/sqrt(window size)')
#         # axa.fill_between(Tidx, 0, 3*vthresh, color='c', alpha=0.2)
#     else:
#         axa.plot(nErr0, color='r', alpha=0.8, linewidth=1, label='Normalized residual')
#         axa.set_title('Normalized residual: (error-mean)/std')
#         # axa.set_ylim((0,6))
#         axa.fill_between(Tidx, 0, vthresh, color='c', alpha=0.2)
#     axa.legend(loc='upper right', fancybox=True, framealpha=0.5)
#     k+=1

#     return fig, axes





# def plot_results(xobs0, yobs0, yprd0, aprd0, bprd0, trnperiod=None, Midx=None):
#     """Plot the prediction and residuals of one sensor without statistical analysis.

#     Args:
#         xobs,yobs: X and Y components of one sensor
#         yprd,aprd,bprd: Y prediction, thermal and non-thermal contributions
#         trnperiod (tuple): starting and ending index of the training period
#     """
#     assert(len(xobs0)==len(yobs0)==len(yprd0)==len(aprd0)==len(bprd0))
#     xobs = xobs0.copy(); xobs[Midx] = np.nan
#     yobs = yobs0.copy(); yobs[Midx] = np.nan
#     yprd = yprd0.copy(); yprd[Midx] = np.nan
#     aprd = aprd0.copy(); aprd[Midx] = np.nan
#     bprd = bprd0.copy(); bprd[Midx] = np.nan
#     yerr = yobs - yprd  # residual

#     nfig, k = 2, 0
#     fig, axes = plt.subplots(nfig, 1, figsize=(20, nfig*5), sharex=True)

#     # Observation and prediction
#     axa = axes[k]
#     axa.plot(yobs, **yobs_style)
#     axa.plot(yprd, **yprd_style)
#     axa.plot(yerr, **yerr_style)
#     # axa.plot(yobs, color='b', alpha=0.5, linewidth=1, label='Elongation: observation')
#     # axa.plot(yprd, color='g', alpha=0.8, linewidth=2, label='Elongation: prediction')
#     # axa.plot(yerr, color='c', alpha=0.8, linewidth=2, label='Elongation: residual')
#     axa.legend(loc='upper left', fancybox=True, framealpha=0.5)
#     axb = axa.twinx()
#     axb.patch.set_alpha(0.0)
#     # axb.plot(xobs, **xobs_style)
#     axb.plot(xobs, color='r', alpha=0.5, linewidth=1, label='Temperature: observation')
#     axb.legend(loc='upper right', fancybox=True, framealpha=0.5)
#     # mark the training period
#     if trnperiod is not None:
#         t0, t1 = xobs.index[trnperiod[0]], xobs.index[trnperiod[1]-1]
#         ylim = axb.get_ylim()
#         axb.fill_betweenx(ylim, t0, t1, color='c', alpha=0.2)
#         # axa.axvspan(t0, t1, color='c', alpha=0.2)
#     axa.set_title('Observations and prediction of elongation')
#     k+=1

#     # Prediction, Thermal and non-thermal contribution
#     axa = axes[k]
#     axa.plot(yprd, **yprd_style)
#     axa.plot(aprd, **aprd_style)
#     # axa.plot(yprd, color='g', alpha=0.8, linewidth=2, label='Elongation: prediction')
#     # axa.plot(aprd, color='r', alpha=0.8, linewidth=2, label='Thermal contribution')
#     if bprd is None:
#         axa.set_title('Prediction and thermal contribution')
#     else:
#         axa.plot(bprd, **bprd_style)
#         # axa.plot(bprd, color='b', alpha=0.8, linewidth=2, label='Non-thermal contribution')
#         axa.set_title('Prediction, thermal and non-thermal contributions')
#     axa.legend(loc='upper left', fancybox=True, framealpha=0.5)
#     k+=1

#     # # Thermal and non-thermal residuals
#     # axa = axes[k]
#     # axa.plot(yerr, **yerr_style)
#     # # axa.plot(yerr, color='b', alpha=0.8, linewidth=2, label='Elongation residual')
#     # axa.plot(yobs-aprd, color='r', alpha=0.8, linewidth=2, label='Residual: Thermal contribution')
#     # # axa.plot((yobs-bprd), color='c', alpha=0.8, linewidth=2, label='Non-thermal residual')
#     # axa.legend(loc='upper left', fancybox=True, framealpha=0.5)
#     # axa.set_title('Thermal and non-thermal residuals')
#     # k+=1
#     return fig, axes
