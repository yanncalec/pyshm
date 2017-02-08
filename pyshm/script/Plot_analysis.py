#!/usr/bin/env python

"""Plot and interpret the data or results of analysis.
"""

import sys, os, argparse
import numpy as np
import scipy as sp
import pandas as pd
import json
from pyshm import OSMOS, Tools, Stat
from pyshm.script import static_data_analysis_template, examplestyle, warningstyle, load_result_of_analysis
from pyshm.script import MyEncoder, to_json


import matplotlib
# matplotlib.use("qt5agg")
import matplotlib.pyplot as plt
# import matplotlib.colors as colors
import mpld3
plt.style.use('ggplot')

# class Options:
#     verbose=False  # print message
#     info=False  # only print information about the project
#     subcommand='static'  # type of deconvolution model


def detect_instability_period(yerr, mwsize, sclrng, thresh, gap):
    """Instability period detection based on Hurst exponent.
    """
    hexp, *_ = Stat.Hurst(yerr, mwsize, sclrng=sclrng, wvlname="haar")  # Hurst exponent
    hexp1 = hexp.copy(); hexp1[np.isnan(hexp)] = -np.inf  # nan -> -inf to avoid warning of comparison
    # indicator of instability
    # hidc = sp.signal.convolve(hexp1>ithresh, np.ones(gap, dtype=bool), mode="same")>0
    hidc = Tools.L_filter(hexp1 > thresh, wsize=gap)  # non linear filter, slower
    hblk = Tools.find_block_true(hidc)  # starting-ending indexes of blocks of instability
    
    return hexp, hblk
    

def compute_local_statistics(Yerr, mad, mwsize):
    """Compute the local statistics: mean and standard deviation and normalized observation.
    """
    Merr0 = {}; Serr0 = {}; Nerr0 = {};
    
    for loc, yerr in Yerr.items():
        # yerr = Ycpn[loc] - Yprd[loc]
        # moving average and standard deviation
        merr, serr = Stat.local_statistics(yerr, mwsize, mad=mad, causal=False, drop=True)
        nerr = abs(yerr-merr)/serr  # normalized error
        Merr0[loc], Serr0[loc], Nerr0[loc] = merr, serr, nerr

    Merr = pd.DataFrame(Merr0, columns=list(Yerr.keys()), index=Yerr.index)
    Serr = pd.DataFrame(Serr0, columns=list(Yerr.keys()), index=Yerr.index)
    Nerr = pd.DataFrame(Nerr0, columns=list(Yerr.keys()), index=Yerr.index)

    return Merr, Serr, Nerr


def raw_plot(figdir, Rdata, Sdata, Ddata, html=False):
    """Plot raw data.

    Args:
        figdir (str): name of directory for output
        Rdata: raw data returned bo OSMOS.load_raw_data()
        Sdata: raw static data
        Ddata: raw dynamic data
        html(bool): if true generate mpld3 html plots in addition of pdf files.
    """

    # plot static data of all sensors in a single file
    if html:
        figdir_html = os.path.join(figdir, 'html')
        try:
            os.makedirs(figdir_html)
        except:
            pass

    figdir_pdf = os.path.join(figdir, 'pdf')
    try:
        os.makedirs(figdir_pdf)
    except:
        pass

    fig, axes = plt.subplots(2,1,figsize=(20,10), sharex=True)

    for loc, val in Sdata.items():
        Xt, Yt = val['Temperature'], val['Elongation']
        axes[0].plot(Xt, label='{}'.format(loc))
        axes[1].plot(Yt, label='{}'.format(loc))

    # axes[0].legend()
    # axes[1].legend()
    axes[0].legend(fancybox=True, framealpha=0.5)
    axes[1].legend(fancybox=True, framealpha=0.5)
    axes[0].set_ylabel('Temperature')
    axes[1].set_ylabel('Elongation')
    plt.tight_layout()

    if html:
        mpld3.save_html(fig, os.path.join(figdir_html, 'All_static.html'))
    fig.savefig(os.path.join(figdir_pdf, 'All_static.pdf'))
    plt.close(fig)

    # plot all data of each sensor in separated files

    for loc, val in Rdata.items():
        fig, axes = plt.subplots(2,1,figsize=(20,10), sharex=True)
        Xt, Yt = val['Temperature'], val['Elongation']
        axes[0].plot(Xt,'r')#, label='{}'.format(loc))
        axes[1].plot(Yt,'b')
        axes[0].set_ylabel('Temperature')
        axes[1].set_ylabel('Elongation')

        # highlight dynamic events
        for v in Ddata[loc]:
            # axes[1].axvspan(v.index[0], v.index[-1], color='r', alpha=0.3)
            axes[1].plot(v, 'r', alpha=0.5)  # down-sampling dynamic events

        plt.tight_layout()

        if html:
            mpld3.save_html(fig, os.path.join(figdir_html, '{}.html'.format(loc)))
        fig.savefig(os.path.join(figdir_pdf, '{}.pdf'.format(loc)))
        plt.close(fig)


def preprocess_plot(figdir, Sdata, marknan=True, markjump=True, html=False):
    """Plot preprocessed static data.
    """

    # plot static data of all sensors in a single file
    if html:
        figdir_html = os.path.join(figdir, 'html')
        try:
            os.makedirs(figdir_html)
        except:
            pass

    figdir_pdf = os.path.join(figdir, 'pdf')
    try:
        os.makedirs(figdir_pdf)
    except:
        pass

    fig, axes = plt.subplots(2,1,figsize=(20,10), sharex=True)

    for loc, val in Sdata.items():
        Xt, Yt = val['Temperature'], val['Elongation']
        axes[0].plot(Xt, label='{}'.format(loc))
        axes[1].plot(Yt, label='{}'.format(loc))

    axes[0].legend(fancybox=True, framealpha=0.5)
    axes[1].legend(fancybox=True, framealpha=0.5)
    axes[0].set_ylabel('Temperature')
    axes[1].set_ylabel('Elongation')
    plt.tight_layout()

    if html:
        mpld3.save_html(fig, os.path.join(figdir_html, 'All_static.html'))
    fig.savefig(os.path.join(figdir_pdf, 'All_static.pdf'))
    plt.close(fig)

    # plot all data of each sensor in separated files
    for loc, val in Sdata.items():
        fig, axes = plt.subplots(2,1,figsize=(20,10), sharex=True)
        Xt, Yt = val['Temperature'], val['Elongation']
        axes[0].plot(Xt,'r')#, label='{}'.format(loc))
        axes[1].plot(Yt,'b')

        ylim0 = axes[0].get_ylim()
        ylim1 = axes[1].get_ylim()

        # mark missing data
        blk = Tools.find_block_true(val['Missing'])
        for (t0,t1) in blk:
            axes[0].fill_betweenx(ylim0, Xt.index[t0], Xt.index[t1-1], color='r', alpha=0.1)
            axes[1].fill_betweenx(ylim1, Xt.index[t0], Xt.index[t1-1], color='b', alpha=0.1)
        # mark jump
        for jidx, jval in enumerate(list(val['Jump'])):
            if jval:
                axes[1].vlines(Xt.index[jidx], ylim1[0], ylim1[1], color='c', linewidth=2, alpha=0.2)
        # blk = Tools.find_block_true(val['Jump'])
        # for (t0,t1) in blk:
        #     axes[1].fill_betweenx(ylim1, Xt.index[t0], Xt.index[t1-1], color='c', linewidth=2, alpha=0.1)

        # reset the ylim to prevent strange bugs
        axes[0].set_ylim(ylim0)
        axes[1].set_ylim(ylim1)

        axes[0].set_ylabel('Temperature')
        axes[1].set_ylabel('Elongation')
        plt.tight_layout()

        if html:
            mpld3.save_html(fig, os.path.join(figdir_html, '{}.html'.format(loc)))

        fig.savefig(os.path.join(figdir_pdf, '{}.pdf'.format(loc)))
        plt.close(fig)


# Shorthand for styles
xobs_style = {'color': 'r', 'linewidth': 1, 'alpha':0.5, 'label':'Temperature: observation'}
yobs_style = {'color': 'b', 'linewidth': 1, 'alpha':0.5, 'label':'Elongation: observation'}
yprd_style = {'color': 'g', 'linewidth': 2, 'alpha':0.8, 'label':'Elongation: prediction'}
aprd_style = {'color': 'r', 'linewidth': 2, 'alpha':0.8, 'label':'Elongation: thermal contribution'}
bprd_style = {'color': 'b', 'linewidth': 2, 'alpha':0.8, 'label':'Elongation: non-thermal contribution'}
yerr_style = {'color': 'c', 'linewidth': 2, 'alpha':0.8, 'label':'Elongation: residual'}
merr_style = {'color': 'b', 'linewidth': 1, 'alpha':0.5, 'label':'Local mean'}
serr_style = {'color': 'r', 'linewidth': 2, 'alpha':0.5, 'label':'Local standard deviation'}
nerr_style = {'color': 'r', 'linewidth': 1, 'alpha':0.8, 'label':'Normalized residual'}
hexp_style = {'color': 'r', 'linewidth': 2, 'alpha':0.8, 'label':'Hurst exponent'}
tdly_style = {'color': 'r', 'linewidth': 2, 'alpha':0.5, 'label':'Thermal delay'}
tcof_style = {'color': 'b', 'linewidth': 2, 'alpha':0.8, 'label':'Thermal coefficient'}


def thermal_plot_one_sensor(component, xobs0, yobs0, yprd0,
                            tdly0, tcof0, tcor0,
                            merr0, serr0, nerr0,
                            hexp0, hblk, hthresh, vthresh, mask):
    """Plot the results of one sensor with statistical analysis.

    Args:
        component: type of component ('seasonal' or 'trend' or 'all')
        xobs0, yobs0: X and Y components of one sensor
        yprd0, aprd0, bprd0: Y prediction, thermal and non-thermal contributions
        trnperiod: training period starting and ending index
        midx: missing index
    """
    # check length
    assert(len(xobs0)==len(yobs0)==len(yprd0))

    Tidx = xobs0.index
    # mask the missing values
    xobs = np.ma.array(xobs0, mask=mask)
    yobs = np.ma.array(yobs0, mask=mask)
    yprd = np.ma.array(yprd0, mask=mask)
    tdly = np.ma.array(tdly0, mask=mask)
    tcof = np.ma.array(tcof0, mask=mask)
    tcor = np.ma.array(tcor0, mask=mask)
    merr = np.ma.array(merr0, mask=mask)    
    serr = np.ma.array(serr0, mask=mask)    
    nerr = np.ma.array(nerr0, mask=mask)
    hexp = np.ma.array(hexp0, mask=mask) if hexp0 is not None else None    
    yerr = yobs - yprd  # residual

    nfig = 2 # if component.upper() in ['TREND', 'SEASONAL'] else 5
    k = 0
    fig, axes = plt.subplots(nfig, 1, figsize=(20, nfig*5), sharex=True)

    # Observation and prediction
    axa = axes[k]
    axa.plot(Tidx, yobs, **yobs_style)
    axa.plot(Tidx, yprd, **yprd_style)
    axa.plot(Tidx, yerr, **yerr_style)
    axa.legend(loc='upper left', fancybox=True, framealpha=0.5)
    axb = axa.twinx()
    axb.patch.set_alpha(0.0)
    axb.plot(Tidx, xobs, **xobs_style)
    axb.legend(loc='upper right', fancybox=True, framealpha=0.5)
    axa.set_title('Observations and prediction of elongation')
    k+=1

    # Thermal delay and thermal coefficient
    axa = axes[k]
    axa.plot(Tidx, tdly, **tdly_style)
    axa.legend(loc='upper left', fancybox=True, framealpha=0.5)
    axb = axa.twinx()
    axb.patch.set_alpha(0.0)
    axb.plot(Tidx, tcof, **tcof_style)
    axa.set_title('Thermal delay and thermal coefficient')
    axb.legend(loc='upper right', fancybox=True, framealpha=0.5)
    k+=1

    return fig, axes


def deconv_plot_one_sensor(component, xobs0, yobs0, yprd0, aprd0, bprd0,
                           merr0, serr0, nerr0,
                           hexp0, hblk, hthresh, vthresh,
                           trnperiod, mask):
    """Plot the results of one sensor with statistical analysis.

    Args:
        component: type of component ('seasonal' or 'trend' or 'all')
        xobs0, yobs0: X and Y components of one sensor
        yprd0, aprd0, bprd0: Y prediction, thermal and non-thermal contributions
        trnperiod: training period starting and ending index
        midx: missing index
    """
    # check length
    assert(len(xobs0)==len(yobs0)==len(yprd0)==len(aprd0))
    if bprd0 is not None:
        assert(len(xobs0)==len(bprd0))

    Tidx = xobs0.index
    # mask the missing values
    xobs = np.ma.array(xobs0, mask=mask)
    yobs = np.ma.array(yobs0, mask=mask)
    yprd = np.ma.array(yprd0, mask=mask)
    aprd = np.ma.array(aprd0, mask=mask)    
    bprd = np.ma.array(bprd0, mask=mask) if bprd0 is not None else None
    merr = np.ma.array(merr0, mask=mask)    
    serr = np.ma.array(serr0, mask=mask)    
    nerr = np.ma.array(nerr0, mask=mask)
    hexp = np.ma.array(hexp0, mask=mask) if hexp0 is not None else None
    
    # xobs = xobs0.copy(); xobs[midx] = np.nan
    # yobs = yobs0.copy(); yobs[midx] = np.nan
    # yprd = yprd0.copy(); yprd[midx] = np.nan
    # aprd = aprd0.copy(); aprd[midx] = np.nan
    # if bprd0 is not None:
    #     bprd = bprd0.copy(); bprd[midx] = np.nan
    # else:
    #     bprd = bprd0

    yerr = yobs - yprd  # residual

    # # drop the missing values
    # merr[midx] = np.nan
    # serr[midx] = np.nan

    nfig = 4 if component.upper() in ['TREND', 'SEASONAL'] else 5
    k = 0
    fig, axes = plt.subplots(nfig, 1, figsize=(20, nfig*5), sharex=True)

    # Observation and prediction
    axa = axes[k]
    axa.plot(Tidx, yobs, **yobs_style)
    axa.plot(Tidx, yprd, **yprd_style)
    axa.plot(Tidx, yerr, **yerr_style)
    axa.legend(loc='upper left', fancybox=True, framealpha=0.5)
    axb = axa.twinx()
    axb.patch.set_alpha(0.0)
    axb.plot(Tidx, xobs, **xobs_style)
    axb.legend(loc='upper right', fancybox=True, framealpha=0.5)
    # mark the training period
    if trnperiod is not None:
        t0, t1 = Tidx[trnperiod[0]], Tidx[trnperiod[1]-1]
        ylim = axb.get_ylim()
        axb.fill_betweenx(ylim, t0, t1, color='c', alpha=0.2)
        # axa.axvspan(t0, t1, color='c', alpha=0.2)
    axa.set_title('Observations and prediction of elongation')
    k+=1

    # Prediction, Thermal and non-thermal contribution
    axa = axes[k]
    axa.plot(Tidx, yprd, **yprd_style)
    axa.plot(Tidx, aprd, **aprd_style)
    if bprd is None:
        axa.set_title('Prediction and thermal contribution')
    else:
        axa.plot(Tidx, bprd, **bprd_style)
        axa.set_title('Prediction, thermal and non-thermal contributions')
    axa.legend(loc='upper left', fancybox=True, framealpha=0.5)
    k+=1

    # Local statistics of residual
    axa = axes[k]
    axa.plot(Tidx, yerr, **yerr_style)
    axa.plot(Tidx, merr, **merr_style)
    axa.legend(loc='upper left', fancybox=True, framealpha=0.5)
    axb = axa.twinx()
    axb.patch.set_alpha(0.0)
    axb.plot(Tidx, serr, **serr_style)
    axb.legend(loc='upper right', fancybox=True, framealpha=0.5)
    axa.set_title('Local statistics')
    k+=1

    # Component-dependent analysis
    if component.upper() in ['TREND', 'ALL']:
        axa = axes[k]
        # Trend component: relative instability period detection based on Hurst exponent
        # tidx = Tools.find_block_true(hidc)  # starting-ending indexes of blocks of instability

        axa.plot(Tidx, yerr, **yerr_style)
        axa.plot(Tidx, merr, **merr_style)
        axa.legend(loc='upper left', fancybox=True, framealpha=0.5)

        axb = axa.twinx()
        axb.patch.set_alpha(0.0)

        axb.plot(Tidx, hexp, **hexp_style)
        axb.hlines(hthresh, Tidx[0], Tidx[-1], linewidth=2, color='tomato')
        ylim = axb.get_ylim()
        for (t0, t1) in hblk:
            axb.fill_betweenx(ylim, Tidx[t0], Tidx[t1-1], color='r', alpha=0.2)
        axb.set_ylim(ylim)
        axb.legend(loc='upper right', fancybox=True, framealpha=0.5)
        axb.set_title('Hurst exponent and relative instability periods')
        k+=1
    if component.upper() in ['SEASONAL', 'ALL'] :
        axa = axes[k]
        # Seasonal component: event detection based on normalized residual
        axa.plot(Tidx, nerr, **nerr_style)
        axa.hlines(vthresh, Tidx[0], Tidx[-1], linewidth=2, color='tomato')
        # ylim = axa.get_ylim()
        # for idx in eidx:
        #     axa.vlines(Tidx[idx], ylim[0], ylim[1], color='c', linewidth=2, alpha=0.2)
        # axa.set_ylim(ylim)
        # axa.fill_between(Tidx, 0, options.vthresh, color='c', alpha=0.2)
        axa.set_title('Normalized residual: (error-mean)/std')
        k+=1

    return fig, axes


def deconv_plot_static_kernel(Krnl):
    """Plot mean value of the static kernel of deconvolution anaylsis.

    """
    import matplotlib.pyplot as plt
    import numpy as np
    # from pyshm import Stat

    fig, axes = plt.subplots(1,len(Krnl),figsize=(6*len(Krnl), 4))
    for n,K in enumerate(Krnl):
        toto = np.asarray(K)
        # print(toto.shape)
        A = np.mean(toto, axis=-1).T
        # print(A.shape, n)
        axa = axes if len(Krnl)==1 else axes[n]
        axa.plot(A[0])
        axa.set_title('Kernel of the group {}'.format(n))
    return fig, axes


def mean_dynamic_kernel(Krnl):
    """Calculate mean value (of different groups) of the dynamic kernel of
    deconvolution analysis.

    """

    Ng = len(Krnl[0]) # number of groups
    Nt = len(Krnl)  # duration
    A = []

    for g in range(Ng):
        toto = np.asarray([K[g] for K in Krnl])
        A.append(np.mean(toto, axis=-1).transpose(2,1,0)[0])
    return A


def deconv_plot_dynamic_kernel(Krnl, Tidx, ncoef=3):
    """Plot mean value of the dynamic kernel.

    Args:
        Krnl (list): Krnl[t][g][n] is the n-th kernel matrix (of shape 1-by-?)
        of the group g at the time index t.

    """
    import matplotlib.pyplot as plt
    # import numpy as np

    As = mean_dynamic_kernel(Krnl)
    fig, axes = plt.subplots(len(As),1,figsize=(20, 5*len(As)))

    for g, A in enumerate(As):
        axa = axes[g] if len(As)>1 else axes
        for c in range(ncoef):
            axa.plot(Tidx[:A.shape[1]], A[c,:], label='coefficient {}'.format(c))
        axa.legend()
    return fig, axes

__script__ = __doc__

__warning__ = "Warning:" + warningstyle("\n ")

examples = []
examples.append(["%(prog)s -h", "Print this help messages (about common parameters)"])
examples.append(["%(prog)s raw DBDIR/153/Raw.pkl", "Apply preprocessing with default parameters on the project of PID 153 (the project lied in the database directory DBDIR), plot the static data in a subfolder named figures/Static and print messages."])
examples.append(["%(prog)s --sflag -v DBDIR/036", "Apply preprocessing by removing syncrhonisation error on the project of PID 36."])
__example__ = "Some examples of use (change the path seperator '/' to '\\' on Windows platform):" + "".join([examplestyle(x) for x in examples])


def main():
    # usage_msg = '%(prog)s <subcommand> <infile> <outdir> [options]'
    # parser = argparse.ArgumentParser(description=__script__, usage=usage_msg)
    mainparser = argparse.ArgumentParser(description=__script__,
                                         formatter_class=argparse.RawDescriptionHelpFormatter,
                                         epilog=__warning__ + "\n\n" + __example__)#,add_help=False)

    subparsers = mainparser.add_subparsers(title='subcommands', description='type of plot', dest='subcommand')
    parser_raw = subparsers.add_parser('raw', help='Plot raw data')
    # parser_raw = subparsers.add_parser('raw', parents=[mainparser], help='Plot raw data')
    parser_preproc = subparsers.add_parser('preprocess', help='Plot preprocessed static data')
    parser_deconv = subparsers.add_parser('deconv', help='Plot and analyse the results of deconvolution (cf. osmos_deconv)')
    parser_thermal = subparsers.add_parser('thermal', help='Plot the results of thermal analysis (cf. osmos_thermal)')

    for parser in [parser_raw, parser_preproc, parser_deconv, parser_thermal]:
        parser.add_argument('infile', type=str, help='input data file.')
        parser.add_argument('outdir', nargs='?', type=str, default=None, help='directory where results (figures and data files) will be saved.')
        # parser.add_argument('outdir', type=str, default=None, help='directory where results (figures and data files) will be saved.')
        parser.add_argument('-v', '--verbose', dest='verbose', action='count', default=0, help='Print messages.')
        # parser.add_argument('--info', dest='info', action='store_true', default=False, help='Print only information about the project.')
        parser.add_argument('--html', dest='html', action='store_true', default=False, help='Generate plots in html format (in addition of pdf format).')

    for parser in [parser_deconv, parser_thermal]:
        lstat_opts = parser.add_argument_group('Options for local statistics')  # local statistics
        lstat_opts.add_argument('--mad', dest='mad', action='store_true', default=False, help='Use median based estimator (default: use empirical estimator).')
        lstat_opts.add_argument('--mwsize', dest='mwsize', type=int, default=24*5, help='Size of the moving window (default=120).', metavar='integer')
        lstat_opts.add_argument('--vthresh', dest='vthresh', type=float, default=3., help='Threshold value for event detection in seasonal components (default=3.).', metavar='float')
        #     lstat_opts.add_argument('--causal', dest='causal', action='store_true', default=False, help='Use causal window (default: non causal).')

        hurst_opts = parser.add_argument_group('Options for the Hurst exponent (trend component only)')  # Hurst exponent
        hurst_opts.add_argument('--hthresh', dest='hthresh', type=float, default=0.6, help='Threshold value for instability detection (default=0.6).', metavar='float')
        hurst_opts.add_argument('--hwsize', dest='hwsize', type=int, default=24*15, help='Size of the moving window for computation of Hurst exponent (default=360).', metavar='integer')
        hurst_opts.add_argument('--hgap', dest='hgap', type=int, default=24, help='Minimal length of instability period (default=24).', metavar='integer')
        hurst_opts.add_argument('--hrng', dest='hrng', nargs=2, type=int, default=(0,8), help='Wavelet scale range index for computation of Hurst exponent (default=(0,8)).', metavar='integer')

    options = mainparser.parse_args()

    # check the input data file
    if not os.path.isfile(options.infile):
        raise FileNotFoundError(options.infile)

    # if not os.path.isdir(options.outdir):
    #     raise FileNotFoundError(options.outdir)

    if options.outdir is None:
        idx = options.infile.rfind(os.path.sep, 0)
        options.outdir = options.infile[:idx]
        # print(options.outdir)
        # raise SystemExit
        # outdir0 = os.path.join(options.infile[:idx0], 'Outputs', options.infile[idx1+1:idx2])

    # subfolder for output
    # outdir = os.path.join(options.outdir, 'data') #{}/_[{}_{}]'.format(options.subcommand.upper(), options.component.upper()))

    # do plot
    if options.subcommand.upper() == 'RAW':
        Rdata, Sdata, Ddata, Locations = OSMOS.load_raw_data(options.infile)

        idx0 = options.infile.rfind(os.path.sep, 0)
        idx1 = options.infile.find('.', idx0)
        figdir = os.path.join(options.outdir, options.infile[idx0+1:idx1])
        try:
            os.makedirs(figdir)
        except:
            pass
        if options.verbose:
            print('Generating plots of raw data in the directory {} ...'.format(figdir))

        raw_plot(figdir, Rdata, Sdata, Ddata, html=options.html)
    elif options.subcommand.upper() == 'PREPROCESS':
        Data, *_ = OSMOS.load_static_data(options.infile)

        idx0 = options.infile.rfind(os.path.sep, 0)
        idx1 = options.infile.find('.', idx0)
        figdir = os.path.join(options.outdir, options.infile[idx0+1:idx1])
        # figdir = os.path.join(options.outdir, 'preprocess')
        try:
            os.makedirs(figdir)
        except:
            pass
        if options.verbose:
            print('Generating plots of preprocessed static data in the directory {} ...'.format(figdir))

        preprocess_plot(figdir, Data, marknan=True, markjump=True, html=options.html)
    elif options.subcommand.upper() in ['DECONV', 'THERMAL']:
        Res = load_result_of_analysis(options.infile)

        # idx0 = options.infile.rfind(os.path.sep, 0)
        # idx1 = options.infile.find('.', idx0)
        # figdir = os.path.join(options.outdir, options.infile[idx0+1:idx1])
        figdir = options.outdir
        try:
            os.makedirs(figdir)
        except:
            pass

        Locations = list(Res['Yprd'].keys())
        Ycpn = Res['Ycpn'][Locations]  # Res['Ycpn'] contains elongation data of all sensors
        Yprd = Res['Yprd']
        Yerr = Ycpn - Yprd  # residual
        algo_options = dict(Res['algo_options'])  # options of parameters
        component = algo_options['component']  # name of the component being analyzed
    
        # Local statistics
        if options.verbose:
            print('Computing local statistics...')
        Merr, Serr, Nerr = compute_local_statistics(Yerr, options.mad, options.mwsize)
        # Indexes of transient event: defined only for seasonal and all component
        if component.upper() in ['SEASONAL', 'ALL']:
            Eidx = {loc: np.where(Nerr[loc] > options.vthresh)[0] for loc in Nerr}
        else:
            Eidx = None
        # Hurst exponents and index ranges of instability period: defined only for trend or all component
        if component.upper() in ['TREND', 'ALL']:
            if options.verbose:
                print('Computing the Hurst exponent...')
            Hexp0 = {}; Hblk = {}
            mYerr = Yerr.rolling(window=24, min_periods=1, center=True).median() #.bfill()
            for loc, yerr0 in mYerr.items():
                yerr = yerr0.copy(); yerr.loc[Res['Midx'][loc]] = np.nan
                Hexp0[loc], Hblk[loc] = detect_instability_period(yerr, options.hwsize, options.hrng, options.hthresh, options.hgap)
            Hexp = pd.DataFrame(Hexp0, index=Yerr.index)
        else:
            Hexp, Hblk = None, None

        # Export the result of analysis
        resdic = {'Merr':Merr, 'Serr':Serr, 'Nerr':Nerr, 'Eidx':Eidx, 'Hexp':Hexp, 'Hblk':Hblk, 'Midx':Res['Midx']}
        resjson = to_json(resdic, verbose=options.verbose)
        # print(resjson.keys())
        outfile0 = os.path.join(figdir, 'Analysis_residual')
        with open(outfile0+'.json', 'w') as fp:
            json.dump(resjson, fp, cls=MyEncoder)

        # Make plots
        if options.verbose:
            print('Generating plots of results of deconvolution in the directory {} ...'.format(figdir))
        if options.subcommand.upper() == "THERMAL":
            for loc in Locations:
                if options.verbose:
                    print('\tPlotting the result of location {}...'.format(loc))

                # Plot the prediction results
                mask = np.logical_or(Res['Midx'][loc], np.isnan(Res['Yprd'][loc]))
                fig, axes = thermal_plot_one_sensor(component, Res['Xcpn'][loc], Res['Ycpn'][loc], Res['Yprd'][loc],
                                                    Res['Delay'][loc], Res['Slope'][loc], Res['Correlation'][loc],
                                                    Merr[loc], Serr[loc], Nerr[loc],
                                                    Hexp[loc] if Hexp is not None else None,
                                                    Hblk[loc] if Hblk is not None else None,
                                                    options.hthresh, options.vthresh, mask)
                # plt.suptitle('Location {}, {} component'.format(loc, component), position=(0.5,1.1),fontsize=20)
                # plt.tight_layout()

                figdir1 = os.path.join(figdir,''.format(), str(loc))
                try:
                    os.makedirs(figdir1)
                except OSError:
                    pass

                if component.upper() == 'TREND':
                    fname = os.path.join(figdir1, 'mad[{}]_mwsize[{}]_hwsize[{}]_hthresh[{}]_hrng[{}]'.format(options.mad, options.mwsize, options.hwsize, options.hthresh, options.hrng))
                elif component.upper() == 'ALL':
                    fname = os.path.join(figdir1, 'mad[{}]_mwsize[{}]_vthresh[{}]_hwsize[{}]_hthresh[{}]_hrng[{}]'.format(options.mad, options.mwsize, options.vthresh, options.hwsize, options.hthresh, options.hrng))
                else:
                    fname = os.path.join(figdir1, 'mad[{}]_mwsize[{}]_vthresh[{}]'.format(options.mad, options.mwsize, options.vthresh))
                fig.savefig(fname+'.pdf', bbox_inches='tight')
                if options.html:
                    mpld3.save_html(fig, fname+'.html')
                plt.close(fig)
        elif options.subcommand.upper() == "DECONV":
            staticflag = algo_options['subcommand'].upper() == 'STATIC'  # method of analysis
            trnperiod = algo_options['trnperiod'] if staticflag else None   # training period, for static model only
            
            # if staticflag:
            #     trnperiod = algo_options['trnperiod']  # training period, for static model only
            # else:
            #     trnperiod = None  # No training period for dynamic model
                # sigmar2 = algo_options['sigmar2']
                # sigmaq2 = algo_options['sigmaq2']
                # kalman = algo_options['kalman']
            
            for loc in Locations:
                if options.verbose:
                    print('\tPlotting the result of location {}...'.format(loc))

                # Plot the prediction results
                mask = np.logical_or(Res['Midx'][loc], np.isnan(Res['Yprd'][loc]))
                fig, axes = deconv_plot_one_sensor(component, Res['Xcpn'][loc], Res['Ycpn'][loc], Res['Yprd'][loc], Res['Aprd'][loc],
                                                   Res['Bprd'][loc] if Res['Bprd'] is not None else None,
                                                   Merr[loc], Serr[loc], Nerr[loc],
                                                   Hexp[loc] if Hexp is not None else None,
                                                   Hblk[loc] if Hblk is not None else None,
                                                   options.hthresh, options.vthresh,
                                                   trnperiod, mask)
                # plt.suptitle('Location {}, {} component'.format(loc, component), position=(0.5,1.1),fontsize=20)
                # plt.tight_layout()

                figdir1 = os.path.join(figdir,''.format(), str(loc))
                try:
                    os.makedirs(figdir1)
                except OSError:
                    pass

                if component.upper() == 'TREND':
                    fname = os.path.join(figdir1, 'mad[{}]_mwsize[{}]_hwsize[{}]_hthresh[{}]_hrng[{}]'.format(options.mad, options.mwsize, options.hwsize, options.hthresh, options.hrng))
                elif component.upper() == 'ALL':
                    fname = os.path.join(figdir1, 'mad[{}]_mwsize[{}]_vthresh[{}]_hwsize[{}]_hthresh[{}]_hrng[{}]'.format(options.mad, options.mwsize, options.vthresh, options.hwsize, options.hthresh, options.hrng))
                else:
                    fname = os.path.join(figdir1, 'mad[{}]_mwsize[{}]_vthresh[{}]'.format(options.mad, options.mwsize, options.vthresh))
                fig.savefig(fname+'.pdf', bbox_inches='tight')
                if options.html:
                    mpld3.save_html(fig, fname+'.html')
                plt.close(fig)

                # Plot kernels
                if staticflag:
                    # print(str(loc))
                    # print(Krnl[str(loc)])
                    fig, axes = deconv_plot_static_kernel(Res['Krnl'][str(loc)])  # str(loc) since in Json file keys of a dictionary must be string
                    plt.suptitle('Location {}, {} component'.format(loc, component), position=(0.5,1.1),fontsize=20)
                    plt.tight_layout()
                    fname = os.path.join(figdir1, 'Kernel_static')
                    fig.savefig(fname+'.pdf', bbox_inches='tight')
                else:
                    fig, axes = deconv_plot_dynamic_kernel(Res['Krnl'][str(loc)], Res['Xcpn'].index)
                    plt.suptitle('Location {}, {} component'.format(loc, component), position=(0.5,1.1),fontsize=20)
                    plt.tight_layout()
                    fname = os.path.join(figdir1, 'Kernel_dynamic')
                    fig.savefig(fname+'.pdf', bbox_inches='tight')            
        else:
            raise TypeError('Unknown command')
        
        if options.verbose:
            print('Results saved in {}'.format(figdir))
        # deconv_plot(figdir, component, Res['Xcpn'], Res['Ycpn'], Res['Yprd'], Res['Aprd'], Res['Bprd'], Res['Krnl'], Merr, Serr, Hexp, Hblk, Res['Midx'], options)
        


if __name__ == "__main__":
    main()
