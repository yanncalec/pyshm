#!/usr/bin/env python

"""Plot and interpret the data or results of analysis.
"""

import sys, os, argparse
import numpy as np
import scipy as sp
import pandas as pd
from pyshm import OSMOS, Tools, Stat
from pyshm.script import static_data_analysis_template, examplestyle, warningstyle, load_result_of_analysis

# from pyshm.script import Deconv_static_data_plot as DP

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


# Shorthand for styles used in deconv_static_plot
xobs_style = {'color': 'r', 'linewidth': 1, 'alpha':0.5, 'label':'Temperature: observation'}
yobs_style = {'color': 'b', 'linewidth': 1, 'alpha':0.5, 'label':'Elongation: observation'}
yprd_style = {'color': 'g', 'linewidth': 2, 'alpha':0.8, 'label':'Elongation: prediction'}
aprd_style = {'color': 'r', 'linewidth': 2, 'alpha':0.8, 'label':'Elongation: thermal contribution'}
bprd_style = {'color': 'b', 'linewidth': 2, 'alpha':0.8, 'label':'Elongation: non-thermal contribution'}
yerr_style = {'color': 'c', 'linewidth': 2, 'alpha':0.8, 'label':'Elongation: residual'}

def deconv_static_plot_one_sensor(xobs0, yobs0, yprd0, aprd0, bprd0, component,
                                  trnperiod=None, Midx=None, mad=False,
                                  mwsize=240, vthresh=3., ithresh=0.6):
    """Plot the results of one sensor with statistical analysis.

    Args:
        xobs0, yobs0: X and Y components of one sensor
        yprd0, aprd0, bprd0: Y prediction, thermal and non-thermal contributions
        component: type of component ('seasonal' or 'trend' or 'all')
        trnperiod: training period starting and ending index
        Midx: missing index
        other parameters: see main()
    """
    # check length
    assert(len(xobs0)==len(yobs0)==len(yprd0)==len(aprd0))
    if bprd0 is not None:
        assert(len(xobs0)==len(bprd0))

    xobs = xobs0.copy(); xobs[Midx] = np.nan
    yobs = yobs0.copy(); yobs[Midx] = np.nan
    yprd = yprd0.copy(); yprd[Midx] = np.nan
    aprd = aprd0.copy(); aprd[Midx] = np.nan
    if bprd0 is not None:
        bprd = bprd0.copy(); bprd[Midx] = np.nan
    else:
        bprd = bprd0

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
    axa.set_title('Local statistics')
    k+=1

    # Component-dependent analysis
    axa = axes[k]
    if component.upper() == 'TREND' or component.upper() == 'ALL':
        # Trend component: relative instability period detection based on Hurst exponent

        # gap: fill the gap less than this value in the detected period
        # hwsize: moving window size for smoothing of Hurst exponent
        # hrng: range of scale for Hurst exponent
        # two groups of values:
        gap = 24*5; hwsize = 24*10; hrng = (0,8)
        # gap = 24*2; hwsize = 24*5; hrng = (0,7)

        herr0, berr, verr = Stat.Hurst(yerr, hwsize, sclrng=hrng)  # Hurst exponent
        herr1 = herr0.copy(); herr1[np.isnan(herr0)] = np.inf  # nan -> inf to avoid warning of comparison
        # indicator of instability
        # hidc = sp.signal.convolve(herr1>ithresh, np.ones(gap, dtype=bool), mode="same")>0
        hidc = Tools.L_filter(herr1>ithresh, wsize=gap)  # non linear filter, slower
        tidx = Tools.find_block_true(hidc)  # starting-ending indexes of blocks of instability

        axa.plot(yerr, **yerr_style)
        axa.legend(loc='upper left', fancybox=True, framealpha=0.5)

        axb = axa.twinx()
        axb.patch.set_alpha(0.0)

        herr = pd.Series(herr0, index=xobs.index); herr[Midx] = np.nan  # convert to pandas
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


def deconv_static_plot_kernel(Knel):
    """Plot mean value of the static kernel of deconvolution anaylsis.

    """
    import matplotlib.pyplot as plt
    import numpy as np
    # from pyshm import Stat

    fig, axes = plt.subplots(1,len(Knel),figsize=(6*len(Knel), 4))
    for n,K in enumerate(Knel):
        toto = np.asarray(K)
        # print(toto.shape)
        A = np.mean(toto, axis=-1).T
        # print(A.shape, n)
        axa = axes if len(Knel)==1 else axes[n]
        axa.plot(A[0])
        axa.set_title('Kernel of the group {}'.format(n))
    return fig, axes


def deconv_dynamic_mean_kernel(Knel):
    """Calculate mean value (of different groups) of the dynamic kernel of
    deconvolution analysis.

    """

    Ng = len(Knel[0]) # number of groups
    Nt = len(Knel)  # duration
    A = []

    for g in range(Ng):
        toto = np.asarray([K[g] for K in Knel])
        A.append(np.mean(toto, axis=-1).transpose(2,1,0)[0])
    return A


def deconv_dynamic_plot_kernel(Knel, Tidx, ncoef=3):
    """Plot mean value of the dynamic kernel.

    Args:
        Knel (list): Knel[t][g][n] is the n-th kernel matrix (of shape 1-by-?)
        of the group g at the time index t.

    """
    import matplotlib.pyplot as plt
    import numpy as np

    As = deconv_dynamic_mean_kernel(Knel)
    fig, axes = plt.subplots(len(As),1,figsize=(20, 5*len(As)))

    for g,A in enumerate(As):
        axa = axes[g]
        for c in range(ncoef):
            axa.plot(Tidx[:A.shape[1]], A[c,:], label='coefficient {}'.format(c))
        axa.legend()
    return fig, axes


def deconv_static_plot(figdir0, Res, options):
    # extract the variables from the dictionary
    Xcpn = Res['Xcpn']  # Component of temperature
    Ycpn = Res['Ycpn']  # Component of elongation
    Yprd = Res['Yprd']  # Prediction of elongation
    # Yerr = Res['Yerr']  # Error of prediction
    algo_options = dict(Res['algo_options'])  # options of parameters
    Aprd = Res['Aprd']  # Contribution of first group of inputs
    # Bprd = Res['Bprd'] if algo_options['lagy']>0 else None  # Contribution of second group of inputs
    Bprd = Res['Bprd']  # Contribution of second group of inputs
    Knel = Res['Knel'] if 'Knel' in Res else None  # Kernel matrices
    # Mxd = Res['Mxd']  # Objects of deconvolution model: exists only in the pickle file but not the json file
    Midx = Res['Midx']  # Indicator of missing values
    # Tidx = Xcpn.index  # Time index

    component = algo_options['component']  # name of the component being analyzed
    staticflag = algo_options['subcommand'].upper() == 'STATIC'  # method of analysis
    if staticflag:
        trnperiod = algo_options['trnperiod']  # training period, for static model only
    else:
        trnperiod = None
        sigmar2 = algo_options['sigmar2']
        sigmaq2 = algo_options['sigmaq2']
        kalman = algo_options['kalman']

    # Plot all results
    Locations = list(Yprd.keys())
    for loc in Locations:
        if options.verbose:
            print('\tPlotting the result of location {}...'.format(loc))

        # Plot the prediction results
        fig, axes = deconv_static_plot_one_sensor(Xcpn[loc], Ycpn[loc], Yprd[loc], Aprd[loc],
                                                  Bprd[loc] if Bprd is not None else None, component,
                                                  trnperiod=trnperiod, Midx=Midx[loc], mad=options.mad,
                                                  mwsize=options.mwsize,
                                                  vthresh=options.vthresh, ithresh=options.ithresh)
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

        figdir = os.path.join(figdir0,''.format(), str(loc))
        try:
            os.makedirs(figdir)
        except OSError:
            pass

        if component.upper() in ['TREND', 'ALL']:
            fname = os.path.join(figdir, 'window[{}]_thresh[{}]_mad[{}]'.format(options.mwsize, options.ithresh, options.mad))
        else:
            fname = os.path.join(figdir, 'window[{}]_thresh[{}]_mad[{}]'.format(options.mwsize, options.vthresh, options.mad))
            # fname = os.path.join(figdir, 'Residual_[window={}_mad={}_causal={}]'.format(options.mwsize, options.mad, options.causal))
        fig.savefig(fname+'.pdf', bbox_inches='tight')
        if options.html:
            mpld3.save_html(fig, fname+'.html')
        plt.close(fig)

        if Knel:  # if kernel is stored in the data file
            if staticflag:
                # print(str(loc))
                # print(Knel[str(loc)])
                fig, axes = deconv_static_plot_kernel(Knel[str(loc)])  # str(loc) since in Json file keys of a dictionary must be string
                plt.suptitle('Location {}, {} component'.format(loc, component), position=(0.5,1.1),fontsize=20)
                plt.tight_layout()
                fname = os.path.join(figdir, 'Kernel_static')
                fig.savefig(fname+'.pdf', bbox_inches='tight')
            else:
                fig, axes = deconv_dynamic_plot_kernel(Knel[str(loc)], Xcpn.index)
                plt.suptitle('Location {}, {} component'.format(loc, component), position=(0.5,1.1),fontsize=20)
                plt.tight_layout()
                fname = os.path.join(figdir, 'Kernel_dynamic')
                fig.savefig(fname+'.pdf', bbox_inches='tight')


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
        parser.add_argument('--html', dest='html', action='store_true', default=False, help='Generate plots in html format in addition of pdf files.')

    lstat_opts = parser_deconv.add_argument_group('Options for local statistics')  # local statistics
    lstat_opts.add_argument('--mad', dest='mad', action='store_true', default=False, help='Use median based estimator (default: use empirical estimator).')
    lstat_opts.add_argument('--mwsize', dest='mwsize', type=int, default=240, help='Size of the moving window (default=240).', metavar='integer')
    lstat_opts.add_argument('--vthresh', dest='vthresh', type=float, default=3., help='Threshold value for event detection in seasonal components (default=3.).', metavar='float')
    #     lstat_opts.add_argument('--causal', dest='causal', action='store_true', default=False, help='Use causal window (default: non causal).')

    hurst_opts = parser_deconv.add_argument_group('Options for the Hurst exponent (trend component only)')  # Hurst exponent
    # lstat_opts.add_argument('--hwsize', dest='hwsize', type=int, default=240, help='Size of the moving window (default=240) for Hurst exponent.', metavar='integer')
    hurst_opts.add_argument('--ithresh', dest='ithresh', type=float, default=0.6, help='Threshold value for instability detection (default=0.6).', metavar='float')
    # hurst_opts.add_argument('--minperiod', dest='minperiod', type=int, default=24, help='Minimal length of instability period (default=24).', metavar='integer')

    options = mainparser.parse_args()
    # options.subcommand = sys.argv[1]  # name of the subcommand: method of deconvolution  <--- TODO: get the subcommand name
    # print(options.subcommand)

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
    elif options.subcommand.upper() == 'DECONV':
        Res = load_result_of_analysis(options.infile)

        idx0 = options.infile.rfind(os.path.sep, 0)
        idx1 = options.infile.find('.', idx0)
        figdir = os.path.join(options.outdir, options.infile[idx0+1:idx1])
        try:
            os.makedirs(figdir)
        except:
            pass
        if options.verbose:
            print('Generating plots of results of deconvolution in the directory {} ...'.format(figdir))

        # toto = deconv_static_result_analysis(Res, options)
        deconv_static_plot(figdir, Res, options)


if __name__ == "__main__":
    main()
