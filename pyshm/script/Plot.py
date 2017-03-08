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
import warnings

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


def detect_periods_of_instability(hexp, hthresh, hgap=0, mask=None):
    # hexp = np.asarray(hexp0).copy(); hexp[np.isnan(hexp0)] = -np.inf
    if hgap > 0:
        # with post-processing
        hidc = Tools.L_filter(np.int32(hexp>hthresh), wsize=hgap)>0  # non linear filter, slower
        # hidc = sp.signal.convolve(hexp>ithresh, np.ones(options.hgap, dtype=bool), mode="same")>0
    else:
        # no post-processing
        hidc = hexp>hthresh
    # apply the mask
    if mask is not None:
        hidc[np.where(mask)[0]] = False
    return Tools.find_block_true(hidc)  # starting-ending indexes of blocks of instability


def add_subplot_row(fig, **kwargs):
    """Update a figure of subplots (of shape N rows by 1 column) by adding one more row at the end.
    """
    for n in range(len(fig.axes)):
        fig.axes[n].change_geometry(len(fig.axes)+1,1,n+1)
    fig.add_subplot(len(fig.axes)+1,1,len(fig.axes)+1, **kwargs)
    return fig


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
        Xt, Yt = val['Temperature'], val['Elongation']
        fig, axes = plt.subplots(1,1,figsize=(20,5))
        axa = axes
        axa.plot(Yt,'b', alpha=0.8)
        axa.legend(loc='upper left', fancybox=True, framealpha=0.5)
        axb = axa.twinx()
        axb.patch.set_alpha(0.0)
        axb.plot(Xt,'r', label="Temperature", alpha=0.8)#, label='{}'.format(loc))
        axb.legend(loc='upper right', fancybox=True, framealpha=0.5)

        ylim0 = axa.get_ylim()
        # ylim1 = axes[1].get_ylim()

        # mark missing data
        blk = Tools.find_block_true(val['Missing'])
        for (t0,t1) in blk:
            axa.fill_betweenx(ylim0, Xt.index[t0], Xt.index[t1-1], color='r', alpha=0.1)
            # axes[1].fill_betweenx(ylim1, Xt.index[t0], Xt.index[t1-1], color='b', alpha=0.1)
        # mark jump
        for jidx, jval in enumerate(list(val['Jump'])):
            if jval:
                axa.vlines(Xt.index[jidx], ylim1[0], ylim1[1], color='c', linewidth=2, alpha=0.2)
        # blk = Tools.find_block_true(val['Jump'])
        # for (t0,t1) in blk:
        #     axes[1].fill_betweenx(ylim1, Xt.index[t0], Xt.index[t1-1], color='c', linewidth=2, alpha=0.1)

        # reset the ylim to prevent strange bugs
        axa.set_ylim(ylim0)
        # axes[1].set_ylim(ylim1)

        axb.set_ylabel('Temperature')
        axa.set_ylabel('Elongation')
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


def plot_obs_prd(axa, xobs0, yobs0, yprd0, mask, trnperiod=None):
    # Observation and prediction
    xobs = np.ma.array(xobs0, mask=mask)
    yobs = np.ma.array(yobs0, mask=mask)
    yprd = np.ma.array(yprd0, mask=mask)
    yerr = np.ma.array(yobs-yprd, mask=mask)  # residual
    Tidx = xobs0.index

    axa.plot(Tidx, yobs, **yobs_style)
    axa.plot(Tidx, yprd, **yprd_style)
    axa.plot(Tidx, yerr, **yerr_style)
    axa.legend(loc='upper left', fancybox=True, framealpha=0.5)

    # mark the training period
    if trnperiod is not None:
        t0, t1 = Tidx[trnperiod[0]], Tidx[trnperiod[1]-1]
        ylim = axa.get_ylim()
        axa.fill_betweenx(ylim, t0, t1, color='c', alpha=0.1)
        axa.set_ylim(ylim)
        # axa.axvspan(t0, t1, color='c', alpha=0.2)

    axa.set_title('Observations and prediction of elongation')

    axb = axa.twinx()
    axb.patch.set_alpha(0.0)
    axb.plot(Tidx, xobs, **xobs_style)
    axb.legend(loc='upper right', fancybox=True, framealpha=0.5)

    return axa


def plot_local_statistics(axa, yerr0, merr0, serr0, mask):
    # Observation and prediction
    yerr = np.ma.array(yerr0, mask=mask)  # residual
    merr = np.ma.array(merr0, mask=mask)  # local mean
    serr = np.ma.array(serr0, mask=mask)  # local std
    Tidx = yerr0.index

    axa.plot(Tidx, yerr, **yerr_style)
    axa.plot(Tidx, merr, **merr_style)
    axa.legend(loc='upper left', fancybox=True, framealpha=0.5)
    axb = axa.twinx()
    axb.patch.set_alpha(0.0)
    # rerr = np.roll(serr, -1) / serr
    # derr = np.zeros(len(serr))
    # for t in range(len(serr)):
    #     if t>0:
    #         toto = serr[:t].copy(); toto = toto[~np.isnan(toto)]
    #         # print(len(toto), np.median(toto))
    #         derr[t] = np.median(toto) if len(toto)>0 else np.nan
    #     else:
    #         derr[0] = serr[0]
    # mserr = np.cumsum(serr)/np.arange(1,len(serr)+1)
    # rerr = serr / derr
    axb.plot(Tidx, serr, **serr_style)
    axb.legend(loc='upper right', fancybox=True, framealpha=0.5)
    axa.set_title('Local mean and local standard deviation of the residual')

    return axa


def plot_alarm_transient(axa, nerr0, vthresh, mask):
    """Plot the alarm of transient events for the seasonal component.
    """
    nerr = np.ma.array(nerr0, mask=mask)
    Tidx = nerr0.index

    # axa = fig.axes[-1]
    axa.plot(Tidx, nerr, **nerr_style)
    axa.hlines(vthresh, Tidx[0], Tidx[-1], linewidth=2, color='tomato')
    # ylim = axa.get_ylim()
    # for idx in eidx:
    #     axa.vlines(Tidx[idx], ylim[0], ylim[1], color='c', linewidth=2, alpha=0.2)
    # axa.set_ylim(ylim)
    # axa.fill_between(Tidx, 0, options.vthresh, color='c', alpha=0.2)
    axa.set_title('Normalized residual: (error-mean)/std')

    return axa


def plot_alarm_period(axa, yerr0, merr0, hexp0, hthresh, hblk, mask):
    """Plot the alarm of instability periods for the trend component.
    """
    yerr = np.ma.array(yerr0, mask=mask)
    # plot merr only for ALL component
    merr = np.ma.array(merr0, mask=mask) if merr0 is not None else None
    hexp = np.ma.array(hexp0, mask=mask)
    Tidx = yerr0.index

    axa.plot(Tidx, yerr, **yerr_style)
    if merr is not None:
        axa.plot(Tidx, merr, **merr_style)
    axa.legend(loc='upper left', fancybox=True, framealpha=0.5)

    axb = axa.twinx()
    axb.patch.set_alpha(0.0)
    axb.plot(Tidx, hexp, **hexp_style)
    axb.hlines(hthresh, Tidx[0], Tidx[-1], linewidth=2, color='tomato')

    ylim = axb.get_ylim()
    # print(hblk)
    for (t0, t1) in hblk:
        axb.fill_betweenx(ylim, Tidx[t0], Tidx[t1-1], color='r', alpha=0.2)
    axb.set_ylim(ylim)

    axb.legend(loc='upper right', fancybox=True, framealpha=0.5)
    axb.set_title('Hurst exponent and relative instability periods')

    return axa


def deconv_plot_contrib(axa, yprd0, aprd0, bprd0, mask):
    # xobs = np.ma.array(xobs0, mask=mask)
    # yobs = np.ma.array(yobs0, mask=mask)
    yprd = np.ma.array(yprd0, mask=mask)
    aprd = np.ma.array(aprd0, mask=mask)
    bprd = np.ma.array(bprd0, mask=mask)
    Tidx = yprd0.index

    axa.plot(Tidx, yprd, **yprd_style)
    axa.plot(Tidx, aprd, **aprd_style)
    axa.plot(Tidx, bprd, **bprd_style)
    axa.set_title('Prediction, thermal and non-thermal contributions')
    axa.legend(loc='upper left', fancybox=True, framealpha=0.5)

    return axa


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
    # parser_deconv = subparsers.add_parser('deconv', help='Plot and analyse the results of deconvolution (cf. osmos_deconv)')
    # parser_thermal = subparsers.add_parser('thermal', help='Plot the results of thermal analysis (cf. osmos_thermal)')
    parser_analyse = subparsers.add_parser('analyse', help='Plot the results of analysis (cf. osmos_deconv and osmos_thermal)')

    for parser in [parser_raw, parser_preproc, parser_analyse]:
        parser.add_argument('infile', type=str, help='input data file.')
        parser.add_argument('outdir', nargs='?', type=str, default=None, help='directory where results (figures and data files) will be saved.')
        # parser.add_argument('outdir', type=str, default=None, help='directory where results (figures and data files) will be saved.')
        parser.add_argument('-v', '--verbose', dest='verbose', action='count', default=0, help='Print messages.')
        # parser.add_argument('--info', dest='info', action='store_true', default=False, help='Print only information about the project.')
        parser.add_argument('--html', dest='html', action='store_true', default=False, help='Generate plots in html format (in addition of pdf format).')

    for parser in [parser_analyse]:
        lstat_opts = parser.add_argument_group('Options for local statistics')  # local statistics
        lstat_opts.add_argument('--vthresh', dest='vthresh', type=float, default=3., help='Threshold value for event detection in seasonal components (default=3.).', metavar='float')

        # Hurst exponent
        hurst_opts = parser.add_argument_group('Options for the Hurst exponent (trend component only)')
        hurst_opts.add_argument('--hthresh', dest='hthresh', type=float, default=0.6, help='Threshold value between 0. and 1. for instability detection (default=0.6).', metavar='float')
        hurst_opts.add_argument('--hgap', dest='hgap', type=int, default=0, help='Minimal length of instability period (default=0).', metavar='integer')

    options = mainparser.parse_args()
    warnings.simplefilter('error', UserWarning)

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
            print('Generating plots...')

        raw_plot(figdir, Rdata, Sdata, Ddata, html=options.html)
    elif options.subcommand.upper() == 'PREPROCESS':
        Data, *_ = OSMOS.load_static_data(options.infile)

        idx0 = options.infile.rfind(os.path.sep, 0)
        idx1 = options.infile.rfind('.', idx0)
        figdir = os.path.join(options.outdir, options.infile[idx0+1:idx1])
        # figdir = os.path.join(options.outdir, 'preprocess')
        try:
            os.makedirs(figdir)
        except:
            pass
        if options.verbose:
            print('Generating plots...')

        preprocess_plot(figdir, Data, marknan=True, markjump=True, html=options.html)
    elif options.subcommand.upper() == "ANALYSE":
        # idx0 = options.infile.rfind(os.path.sep, 0)
        # idx1 = options.infile.find('.', idx0)
        # figdir = os.path.join(options.outdir, options.infile[idx0+1:idx1])
        figdir = options.outdir
        try:
            os.makedirs(figdir)
        except:
            pass

        # load results of analysis
        Res = load_result_of_analysis(options.infile)
        Locations = list(Res['Yprd'].keys())  # list of sensors
        idx = Res["func_name"].find('_')
        analyse_type = Res["func_name"][:idx]  # type of analysis of the result
        # Yprd = Res['Yprd']  # predictions
        # Xcpn = Res['Xcpn'][Locations]  # Res['Xcpn'] contains temperature data of all sensors
        # Ycpn = Res['Ycpn'][Locations]  # Res['Ycpn'] contains elongation data of all sensors, extract only those having some prediction results
        # Yerr = Ycpn - Yprd  # residual
        # Midx = Res['Midx']  # indexes of missing data
        algo_options = dict(Res['algo_options'])  # options of parameters
        component = algo_options['component']  # name of the component being analyzed

        # if analyse_type.upper() == "DECONV":
        #     staticflag = algo_options['subcommand'].upper() == 'STATIC'  # method of analysis
        #     trnperiod = algo_options['trnperiod'] if staticflag else None   # training period, for static model only
        # else:
        #     staticflag = None
        #     trnperiod = None

        if analyse_type.upper() == "DECONV":
            staticflag = algo_options['subcommand'].upper() == 'STATIC'  # method of analysis
        else:
            staticflag = None

        trnperiod = algo_options['trnperiod'] if 'trnperiod' in algo_options else None   # training period, for static model only

        if options.verbose:
            print('Generating alarms...')

        # Indexes of transient event: defined only for seasonal and all component
        if component.upper() in ['SEASONAL', 'ALL']:
            Eidx = Res['Nerr'] > options.vthresh
        else:
            Eidx = None

        # Hurst exponents and index ranges of instability period: defined only for trend or all component
        if component.upper() in ['TREND', 'ALL']:
            Hblk = {}
            # indicator of instability
            for loc, hexp in Res["Hexp"].items():
                Hblk[loc] = detect_periods_of_instability(hexp, options.hthresh, hgap=options.hgap, mask=Res["Midx"][loc])
        else:
            Hblk = None

        # Export alarms
        outfile = os.path.join(figdir, 'Alarms') + '.json'
        alarms = {"Periods of instability":Hblk, "Transient events":Eidx}
        resjson = to_json(alarms, verbose=options.verbose)
        with open(outfile, 'w') as fp:
            json.dump(resjson, fp, cls=MyEncoder)

        # Make plots
        if options.verbose:
            print('Generating plots...')
        # per sensor plot
        for loc in Locations:
            if options.verbose:
                print('\tPlotting the result of location {}...'.format(loc))

            figdir1 = os.path.join(figdir, '{}'.format(loc))
            try:
                os.makedirs(figdir1)
            except:
                pass

            mask = np.logical_or(Res['Midx'][loc], np.isnan(Res['Yprd'][loc]))

            nfig = 3  # number of base figures
            if analyse_type.upper() == "DECONV":
                nfig += 1
            if component.upper() == "ALL":
                nfig += 1

            fig, axes = plt.subplots(nfig, 1, figsize=(20, 5*nfig), sharex=True)
            k = 0

            plot_obs_prd(axes[k], Res["Xcpn"][loc], Res["Ycpn"][loc], Res["Yprd"][loc], mask, trnperiod=trnperiod)
            k += 1

            plot_local_statistics(axes[k], Res["Ycpn"][loc]-Res["Yprd"][loc], Res["Merr"][loc], Res["Serr"][loc], mask)
            k += 1

            if analyse_type.upper() == "DECONV":
                deconv_plot_contrib(axes[k], Res["Yprd"][loc], Res["Aprd"][loc], Res["Bprd"][loc], mask)
                k += 1

            if component.upper() in ["SEASONAL", "ALL"]:
                plot_alarm_transient(axes[k], Res["Nerr"][loc], options.vthresh, mask)
                k += 1

            if component.upper() == "ALL":
                plot_alarm_period(axes[k], Res["Ycpn"][loc]-Res["Yprd"][loc], Res["Merr"][loc], Res["Hexp"][loc], options.hthresh, Hblk[loc], mask)
                k += 1
            if component.upper() =="TREND":
                plot_alarm_period(axes[k], Res["Ycpn"][loc]-Res["Yprd"][loc], None, Res["Hexp"][loc], options.hthresh, Hblk[loc], mask)
                k += 1

            fig.tight_layout()

            if component.upper() == 'TREND':
                fname = os.path.join(figdir1, 'hthresh[{}]'.format(options.hthresh))
            elif component.upper() == 'ALL':
                fname = os.path.join(figdir1, 'hthresh[{}]_vthresh[{}]'.format(options.hthresh, options.vthresh))
            else:
                fname = os.path.join(figdir1, 'vthresh[{}]'.format(options.vthresh))

            fig.savefig(fname+'.pdf', bbox_inches='tight')
            if options.html:
                mpld3.save_html(fig, fname+'.html')
            plt.close(fig)

            if analyse_type.upper() == "DECONV":
                # Plot kernels
                if staticflag:
                    fig, axes = deconv_plot_static_kernel(Res['Krnl'][str(loc)])  # str(loc) since in Json file keys of a dictionary must be string
                    # plt.suptitle('Location {}, {} component'.format(loc, component), position=(0.5,1.1),fontsize=20)
                    plt.tight_layout()
                    fname = os.path.join(figdir, 'Kernel_static')
                    fig.savefig(fname+'.pdf', bbox_inches='tight')
                else:
                    fig, axes = deconv_plot_dynamic_kernel(Res['Krnl'][str(loc)], Res['Xcpn'].index)
                    plt.suptitle('Location {}, {} component'.format(loc, component), position=(0.5,1.1),fontsize=20)
                    plt.tight_layout()
                    fname = os.path.join(figdir, 'Kernel_dynamic')
                    fig.savefig(fname+'.pdf', bbox_inches='tight')
    else:
        raise TypeError('Unknown command')

    if options.verbose:
        print('Results saved in {}'.format(figdir))


if __name__ == "__main__":
    main()
