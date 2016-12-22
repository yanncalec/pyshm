#!/usr/bin/env python

"""Plot results of deconvolution of static data.
"""


import sys, os, argparse

__script__ = __doc__


def plot_results(Xcpn, Ycpn, Yprd, Aprd, Bprd, trn_idx=None):
    """Plot the prediction and residuals of one sensor without statistical analysis.

    Args:
        Xcpn,Ycpn: X and Y components of one sensor
        Yprd,Aprd,Bprd: Y prediction, thermal and non-thermal contributions
        trn_idx: training period starting and ending index
    """
    import matplotlib.pyplot as plt

    Err = Ycpn - Yprd
    Tidx = Xcpn.index

    nfig, k = 3, 0
    fig, axes = plt.subplots(nfig,1, figsize=(20, nfig*5), sharex=True)

    # Observation and prediction
    axa = axes[k]
    axa.plot(Ycpn, color='b', alpha=0.5, linewidth=1, label='Elongation observation')
    axa.plot(Yprd, color='g', alpha=0.8, linewidth=2, label='Elongation prediction')
    axa.plot(Err, color='c', alpha=0.8, linewidth=2, label='Residual')
    axa.legend(loc='upper left', fancybox=True, framealpha=0.5)
    axb = axa.twinx()
    axb.patch.set_alpha(0.0)
    axb.plot(Xcpn, color='r', alpha=0.5, linewidth=1, label='Temperature observation')
    axb.legend(loc='upper right', fancybox=True, framealpha=0.5)
    # mark the training period
    if trn_idx is not None:
        t0, t1 = Tidx[trn_idx[0]], Tidx[trn_idx[1]-1]
        ylim = axb.get_ylim()
        # axa.fill_betweenx(np.arange(-100,100), t0, t1, color='c', alpha=0.2)
        axb.fill_betweenx(ylim, t0, t1, color='c', alpha=0.2)
        # axa.axvspan(t0, t1, color='c', alpha=0.2)
    axa.set_title('Observations and prediction of elongation')
    k+=1

    # Prediction, Thermal and non-thermal contribution
    axa = axes[k]
    axa.plot(Yprd, color='g', alpha=0.8, linewidth=2, label='Elongation prediction')
    axa.plot(Aprd, color='r', alpha=0.8, linewidth=2, label='Prediction: Thermal contribution')
    if Bprd is None:
        axa.set_title('Prediction and thermal contribution')
    else:
        axa.plot(Bprd, color='b', alpha=0.8, linewidth=2, label='Prediction: Non-thermal contribution')
        axa.set_title('Prediction, thermal and non-thermal contributions')
    axa.legend(loc='upper left', fancybox=True, framealpha=0.5)
    k+=1

    # Thermal and non-thermal residuals
    axa = axes[k]
    axa.plot(Err, color='royalblue', alpha=0.8, linewidth=2, label='Elongation residual')
    axa.plot(Ycpn-Aprd, color='r', alpha=0.8, linewidth=2, label='Residual: Thermal contribution')
    # axa.plot((Ycpn-Bprd), color='c', alpha=0.8, linewidth=2, label='Non-thermal residual')
    axa.legend(loc='upper left', fancybox=True, framealpha=0.5)
    axa.set_title('Thermal and non-thermal residuals')
    k+=1
    return fig, axes


def plot_residuals(Xcpn, Ycpn, Yprd, mwsize, vthresh, Midx=None, mad=False, causal=False):
    import matplotlib.pyplot as plt
    import numpy as np

    Err = Ycpn - Yprd
    Tidx = Err.index

    # local and global statistics
    # mXXX for mean, sXXX for standard deviation, nXXX for normalization
    if mad:  # use median-based estimator
        mErr = Err.rolling(window=mwsize, min_periods=1, center=not causal).median() #.bfill()
        sErr = 1.4826 * (Err-mErr).abs().rolling(window=mwsize, min_periods=1, center=not causal).median() #.bfill()
    else:
        mErr = Err.rolling(window=mwsize, min_periods=1, center=not causal).mean() #.bfill()
        sErr = Err.rolling(window=mwsize, min_periods=1, center=not causal).std() #.bfill()

    # drop the begining
    mErr.iloc[:int(mwsize*1.1)]=np.nan
    sErr.iloc[:int(mwsize*1.1)]=np.nan
    # drop the missing values
    mErr[Midx==True] = np.nan
    sErr[Midx==True] = np.nan
    # if Midx is not None:
    #     for loc,val in mErr.items():
    #         val[Midx==True] = np.nan
    #     # sErr.iloc[:int(mwsize*1.1)]=np.nan
    #     for loc,val in sErr.items():
    #         val[Midx==True] = np.nan

    nErr = abs(Err-mErr)/sErr  # normalized error

    nfig, k = 3, 0
    fig, axes = plt.subplots(nfig,1, figsize=(20, nfig*5), sharex=True)

    # Observation and prediction
    axa = axes[k]
    axa.plot(Ycpn, color='b', alpha=0.5, linewidth=1, label='Elongation observation')
    axa.plot(Yprd, color='g', alpha=0.8, linewidth=2, label='Elongation prediction')
    axa.legend(loc='upper left', fancybox=True, framealpha=0.5)
    axb = axa.twinx()
    axb.patch.set_alpha(0.0)
    axb.plot(Xcpn, color='r', alpha=0.5, linewidth=1, label='Temperature observation')
    axb.legend(loc='upper right', fancybox=True, framealpha=0.5)
    axa.set_title('Observations and prediction of elongation')
    k+=1

    # Residual
    axa = axes[k]
    axa.plot(Err, color='b', alpha=0.5, linewidth=1, label='Residual')
    axa.plot(mErr, color='c', alpha=0.8, linewidth=2, label='Local mean')
    axa.legend(loc='upper left', fancybox=True, framealpha=0.5)
    axb = axa.twinx()
    axb.patch.set_alpha(0.0)
    axb.plot(sErr, color='r', alpha=0.8, linewidth=2, label='Local standard deviation')
    axb.legend(loc='upper right', fancybox=True, framealpha=0.5)
    axa.set_title('Residual and local statistics with moving window of size {}'.format(mwsize))
    k+=1

    # Normalized residual
    axa = axes[k]
    axa.plot(nErr, color='r', alpha=0.8, linewidth=1, label='Normalized residual')
    # axa.set_ylim((0,6))
    axa.fill_between(Tidx, 0, vthresh, color='c', alpha=0.2)
    axa.set_title('Normalized residual: (error-mean)/std')
    k+=1

    return fig, axes


def main():
    # Load data
    usage_msg = '%(prog)s [options] <infile> [outdir]'

    parser = argparse.ArgumentParser(description=__script__, usage=usage_msg)

    parser.add_argument('infile', type=str, help='file returned by the script of data analysis')
    parser.add_argument('outdir', nargs='?', type=str, default=None, help='directory where figures are saved (default: in the same folder as infile).')

    parser.add_argument('--vthresh', dest='vthresh', type=float, default=3., help='Threshold value for event detection (default=4).')
    parser.add_argument('--mwsize', dest='mwsize', type=int, default=24, help='Size of the moving window for local statistics (default=24).')
    parser.add_argument('--mad', dest='mad', action='store_true', default=False, help='Use median based estimator (default: use empirical estimator).')
    parser.add_argument('--causal', dest='causal', action='store_true', default=False, help='Use causal window (default: non causal).')
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

    # output directory for figures
    if options.outdir is not None:
        figdir0 = options.outdir
    else:
        idx = options.infile.rfind(os.path.sep)
        figdir0 = options.infile[:idx]

    # Load raw data
    with open(options.infile, 'rb') as fp:
        Res = pickle.load(fp)

    Xcpn = Res['Xcpn']  # Component of temperature
    Ycpn = Res['Ycpn']  # Component of elongation
    Yprd = Res['Yprd']  # Prediction of elongation
    Yerr = Res['Yerr']  # Error of prediction
    Aprd = Res['Aprd']  # Contribution of first group of inputs
    Bprd = Res['Bprd']  # Contribution of second group of inputs
    saved_options=Res['options']  # options of parameters
    Mxd = Res['Mxd']  # Objects of deconvolution model
    Midx = Res['Midx']  # Indicator of missing values
    # print(len(Midx), len(Xcpn))
    Tidx = Xcpn.index  # Time index

    trn_idx = saved_options.trn_idx
    Ntrn = saved_options.Ntrn
    component = saved_options.component

    # Plot all results
    Locations = list(Xcpn.keys())
    for loc in Locations:
        if options.verbose:
            print('Plotting the result of location {}...'.format(loc))

        figdir = os.path.join(figdir0, str(loc))
        try:
            os.makedirs(figdir)
        except OSError:
            pass

        if saved_options.lagy > 0: # if the non-thermal contribution exists
            fig, axes = plot_results(Xcpn[loc], Ycpn[loc], Yprd[loc], Aprd[loc], Bprd[loc], trn_idx)
        else:
            fig, axes = plot_results(Xcpn[loc], Ycpn[loc], Yprd[loc], Aprd[loc], None, trn_idx)
        plt.suptitle('Location {}, {} component'.format(loc, component), position=(0.5,1.1),fontsize=20)
        plt.tight_layout()
        fname = os.path.join(figdir, 'Results')
        fig.savefig(fname+'.pdf', bbox_inches='tight')
        mpld3.save_html(fig, fname+'.html')
        plt.close(fig)

        fig, axes = plot_residuals(Xcpn[loc], Ycpn[loc], Yprd[loc], options.mwsize, Midx=Midx[loc], vthresh=options.vthresh, mad=options.mad, causal=options.causal)
        plt.suptitle('Location {}, {} component'.format(loc, component), position=(0.5,1.1),fontsize=20)
        plt.tight_layout()
        fname = os.path.join(figdir, 'Residual_[window={}_mad={}_causal={}]'.format(options.mwsize, options.mad, options.causal))
        fig.savefig(fname+'.pdf', bbox_inches='tight')
        mpld3.save_html(fig, fname+'.html')
        plt.close(fig)

    if options.verbose:
        print('Figures saved in {}'.format(figdir0))


if __name__ == "__main__":
    print(__script__)
    # print()
    main()
