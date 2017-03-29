#!/usr/bin/env python

"""Plot and interpret the data or results of analysis.
"""

import sys, os, argparse
import numpy as np
import scipy
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
    # if hgap > 0:
    #     # with post-processing
    #     hidc = Tools.L_filter(np.int32(hexp>hthresh), wsize=hgap)>0  # non linear filter, slower
    #     # hidc = scipy.signal.convolve(hexp>ithresh, np.ones(options.hgap, dtype=bool), mode="same")>0
    # else:
    #     # no post-processing
    hidc = hexp>hthresh
    # apply the mask
    if mask is not None:
        hidc[np.where(mask)[0]] = False
    blk = Tools.find_block_true(hidc)  # starting-ending indexes of blocks of instability
    return [b for b in blk if b[1]-b[0] > hgap]

# def blk2ts(blk, tidx):
#     """
#     """
#     ts = []
#     for b in blk:
#         ts.append([str(tidx[b[0]]), str(tidx[b[1]])])
#     return ts

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
                axa.vlines(Xt.index[jidx], ylim0[0], ylim0[1], color='c', linewidth=2, alpha=0.2)
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
xobs_style = {'color': 'red', 'linewidth': 1, 'alpha':0.75, 'label':'Temperature: observation'}
yobs_style = {'color': 'blue', 'linewidth': 1, 'alpha':0.75, 'label':'Elongation: observation'}
yprd_style = {'color': 'orange', 'linewidth': 1, 'alpha':0.75, 'label':'Elongation: thermal prediction'}
# aprd_style = {'color': 'r', 'linewidth': 2, 'alpha':0.8, 'label':'Elongation: thermal contribution'}
# bprd_style = {'color': 'b', 'linewidth': 2, 'alpha':0.8, 'label':'Elongation: non-thermal contribution'}
yerr_style = {'color': 'darkcyan', 'linewidth': 2, 'alpha':1., 'label':'Elongation: thermal residual'}
yssp_style = {'color': 'green', 'linewidth': 1, 'alpha':0.75, 'label':'Elongation: non-thermal subspace projection'}
yerp_style = {'color': 'darkblue', 'linewidth': 2, 'alpha':1., 'label':'Elongation: final residual'}

ys = {'color': 'black', 'linewidth': 2, 'alpha':1., 'label':'Elongation: residual'}
ms = {'color': 'b', 'linewidth': 1, 'alpha':0.5, 'label':'Local mean'}
ss = {'color': 'r', 'linewidth': 2, 'alpha':0.5, 'label':'Local standard deviation'}

merr_style = {'color': 'b', 'linewidth': 1, 'alpha':0.5, 'label':'Local mean'}
merp_style = {'color': 'b', 'linewidth': 1, 'alpha':0.5, 'label':'Local mean'}
serr_style = {'color': 'r', 'linewidth': 2, 'alpha':0.5, 'label':'Local standard deviation'}
serp_style = {'color': 'r', 'linewidth': 2, 'alpha':0.5, 'label':'Local standard deviation'}
nerr_style = {'color': 'r', 'linewidth': 1, 'alpha':0.8, 'label':'Normalized residual'}
nerp_style = {'color': 'r', 'linewidth': 1, 'alpha':0.8, 'label':'Normalized residual'}
hexp_style = {'color': 'r', 'linewidth': 2, 'alpha':0.8, 'label':'Hurst exponent'}
tdly_style = {'color': 'r', 'linewidth': 2, 'alpha':0.5, 'label':'Thermal delay'}
tcof_style = {'color': 'b', 'linewidth': 2, 'alpha':0.8, 'label':'Thermal coefficient'}


def summary_plot(Tcpn, Ecpn, Eprd, Eerr, Essp, cdim, Eerp, figdir, html=False):
    nfig=6
    fig, _ = plt.subplots(nfig,1, figsize=(20,nfig*5), sharex=True)
    axes = fig.axes
    Tcpn.plot(ax=axes[0])
    axes[0].set_title("Temperature: observation")
    axes[0].legend(loc="lower left", fancybox=True, framealpha=0.5)
    Ecpn.plot(ax=axes[1])
    axes[1].set_title("Elongation: observation")
    axes[1].legend(loc="lower left", fancybox=True, framealpha=0.5)
    Eprd.plot(ax=axes[2])
    axes[2].set_title("Elongation: thermal prediction")
    axes[2].legend(loc="lower left", fancybox=True, framealpha=0.5)
    Eerr.plot(ax=axes[3])
    axes[3].set_title("Elongation: thermal residual")
    axes[3].legend(loc="lower left", fancybox=True, framealpha=0.5)
    axa = axes[4]
    Essp.plot(ax=axa)
    Essp.mean(axis=1).plot(ax=axa, linewidth=3, alpha=0.3, color="b", linestyle="-", label="Mean behavior")
    axa.legend(loc="lower left", fancybox=True, framealpha=0.5)
    axa.set_title("Elongation: non-thermal subspace projection of dimension {}".format(cdim))
    axa = axes[5]
    Eerp.plot(ax=axa)
    axa.legend(loc="lower left", fancybox=True, framealpha=0.5)
    axa.set_title("Elongation: final residual")
    fig.tight_layout()
    # save figure
    fname = os.path.join(figdir, "Summary")
    fig.savefig(fname+".pdf")
    if html:
        mpld3.save_html(fig, fname+".html")


def pca_plot(Yerr, figdir):
    Locations = list(Yerr.columns)
    if len(Locations) >= 2:
        _, (U, S), _ = Stat.ssproj(np.asarray(Yerr).T, cdim=1, corrflag=False)

        # Regrouping of sensors
        Scof = (U @ np.diag(np.sqrt(S/S[0])))  # Scof[:,:3] are the first 3 PCA coefficients

        fig, axa = plt.subplots(1,1, figsize=(5,5))
        for n in range(Scof.shape[0]):
            xx,yy = Scof[n,0],Scof[n,1]
            axa.scatter(xx,yy)
            axa.text(xx+.001, yy+.001, str(Locations[n]))
        axa.set_xlabel('1st pv')
        axa.set_ylabel('2nd pv')
        # axa.set_zlabel('3rd pv')
        axa.set_title("PCA coefficients of sensors")
        fig.tight_layout()
        # save figure
        fname = os.path.join(figdir, "PCA_2d")
        fig.savefig(fname+".pdf")

    if len(Locations) >= 3:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        axa = fig.add_subplot(111, projection='3d')

        for n in range(Scof.shape[0]):
            xx,yy,zz = Scof[n,0],Scof[n,1],Scof[n,2]
            axa.scatter(xx,yy,zz)
            axa.text(xx+.001, yy+.001, zz+.001, str(Locations[n]))
        axa.set_xlabel('1st pv')
        axa.set_ylabel('2nd pv')
        axa.set_zlabel('3rd pv')
        axa.set_title("PCA coefficients of sensors")
        fig.tight_layout()
        # save figure
        fname = os.path.join(figdir, "PCA_3d")
        fig.savefig(fname+".pdf")


def locations_plot(Tcpn, Ecpn, Eprd, figdir, html=False):
    nfig = len(Tcpn.columns)
    fig,_ = plt.subplots(nfig,1,figsize=(20,nfig*5), sharex=True)
    axes = fig.axes
    for axa, loc in zip(axes, alocs):
        axa.plot(Ecpn[loc], 'b', label="Elongation data", alpha=0.5)
        axa.plot(Eprd[loc], 'g', label="Elongation prediction")
        axa.plot(Eerr[loc], 'c', label="Elongation residual")
        axa.legend(loc="upper left", fancybox=True, framealpha=0.5)
        axa.set_title("Location {}".format(loc))
        axb = axa.twinx()
        axb.patch.set_alpha(0.0)
        axb.plot(Tcpn[loc], 'r', label="Temperature data", alpha=0.5)
        axb.legend(loc="upper right", fancybox=True, framealpha=0.5)
    fig.tight_layout()
    # save figure
    fname = os.path.join(figdir, "Locations")
    fig.savefig(fname+".pdf")
    if html:
        mpld3.save_html(fig, fname+".html")


def plot_obs_prd(axa, xobs0, yobs0, yprd0, yerr0, yssp0, yerp0, mask, trnperiod=None):
    # Observation and prediction
    Tidx = xobs0.index
    xobs = np.ma.array(xobs0, mask=mask)
    yobs = np.ma.array(yobs0, mask=mask)
    yprd = np.ma.array(yprd0, mask=mask)
    yerr = np.ma.array(yerr0, mask=mask)

    axa.plot(Tidx, yobs, **yobs_style)
    axa.plot(Tidx, yprd, **yprd_style)
    axa.plot(Tidx, yerr, **yerr_style)

    if yssp0 is not None:
        yssp = np.ma.array(yssp0, mask=mask)
        axa.plot(Tidx, yssp, **yssp_style)
    if yerp0 is not None:
        yerp = np.ma.array(yerp0, mask=mask)
        axa.plot(Tidx, yerp, **yerp_style)

    axa.legend(loc='lower left', fancybox=True, framealpha=0.5)

    # mark the training period
    if trnperiod is not None:
        t0, t1 = Tidx[trnperiod[0]], Tidx[trnperiod[1]-1]
        ylim = axa.get_ylim()
        axa.fill_betweenx(ylim, t0, t1, color='c', alpha=0.1)
        axa.set_ylim(ylim)
        # axa.axvspan(t0, t1, color='c', alpha=0.2)

    axa.set_title('Observations and predictions')

    axb = axa.twinx()
    axb.patch.set_alpha(0.0)
    axb.plot(Tidx, xobs, **xobs_style)
    axb.legend(loc='lower right', fancybox=True, framealpha=0.5)

    return axa


def plot_local_statistics(axa, yerr0, merr0, serr0, sthresh, sblk, mask):
    # Observation and prediction
    yerr = np.ma.array(yerr0, mask=mask)  # residual
    merr = np.ma.array(merr0, mask=mask)  # local mean
    serr = np.ma.array(serr0, mask=mask)  # local std
    Tidx = yerr0.index

    axa.plot(Tidx, yerr, **ys)
    axa.plot(Tidx, merr, **ms)
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
    axb.plot(Tidx, serr, **ss)
    axb.hlines(sthresh, Tidx[0], Tidx[-1], linewidth=2, color='tomato')

    ylim = axb.get_ylim()
    for (t0, t1) in sblk:
        axb.fill_betweenx(ylim, Tidx[t0], Tidx[t1-1], color='r', alpha=0.2)
    axb.set_ylim(ylim)

    axb.legend(loc='upper right', fancybox=True, framealpha=0.5)
    axa.set_title('Local mean and local standard deviation of the residual')

    return axa


def plot_alarm_transient(axa, nerr0, nthresh, mask):
    """Plot the alarm of transient events for the seasonal component.
    """
    nerr = np.ma.array(nerr0, mask=mask)
    Tidx = nerr0.index

    # axa = fig.axes[-1]
    axa.plot(Tidx, nerr, **nerr_style)
    axa.hlines(nthresh, Tidx[0], Tidx[-1], linewidth=2, color='tomato')
    # ylim = axa.get_ylim()
    # for idx in eidx:
    #     axa.vlines(Tidx[idx], ylim[0], ylim[1], color='c', linewidth=2, alpha=0.2)
    # axa.set_ylim(ylim)
    # axa.fill_between(Tidx, 0, options.nthresh, color='c', alpha=0.2)
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

    axa.plot(Tidx, yerr, **ys)
    axa.plot(Tidx, merr, **ms)
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


# def mean_dynamic_kernel(Krnl):
#     """Calculate mean value (of different groups) of the dynamic kernel of
#     deconvolution analysis.
#     """
#     Ng = len(Krnl[0]) # number of groups
#     Nt = len(Krnl)  # duration
#     A = []
#     for g in range(Ng):
#         toto = np.asarray([K[g] for K in Krnl])
#         A.append(np.mean(toto, axis=-1).transpose(2,1,0)[0])
#     return A


def deconv_plot_dynamic_kernel(Krnl, Tidx, ncoef=10):
    """Plot mean value of the dynamic kernel.

    Args:
        Krnl (list): Krnl[t][g][n] is the n-th kernel matrix (of shape 1-by-?)
        of the group g at the time index t.

    """
    import matplotlib.pyplot as plt
    # import numpy as np

    Krnl_mean = np.mean(Krnl, axis=0)

    fig, axa = plt.subplots(1,1,figsize=(20, 5))
    for c in range(min(Krnl.shape[0], ncoef)):
        axa.plot(Tidx, Krnl[c,:], label='coefficient {}'.format(c))
        axa.legend()
    axa.plot(Tidx, Krnl_mean, linewidth=3, label='mean coefficient')
    axa.legend(loc='upper right', fancybox=True, framealpha=0.5)
    return fig, axa


def deconv_plot_mean_dynamic_kernel(Krnl, Kcov, Tidx, pval=0.9):
    """Plot mean value of the dynamic kernel.

    Args:
        Krnl (list): Krnl[t][g][n] is the n-th kernel matrix (of shape 1-by-?)
        of the group g at the time index t.

    """
    import matplotlib.pyplot as plt
    # import numpy as np

    Krnl_mean = np.mean(Krnl, axis=0)
    fig, axa = plt.subplots(1,1,figsize=(20, 5))
    axa.plot(Tidx, Krnl_mean, label='mean coefficient')
    if pval > 0:
        a = scipy.special.erfinv(pval) * np.sqrt(2)  #
        Kcov_mean = np.sum(Kcov)/(Kcov.shape[0])**2
        v = a * np.sqrt(Kcov_mean)
        y1 = Krnl_mean - v
        y2 = Krnl_mean + v
        axa.fill_between(Tidx,y1,y2=y2)

    axa.legend(loc='upper right', fancybox=True, framealpha=0.5)
    return fig, axa

    # As = mean_dynamic_kernel(Krnl)
    # fig, axes = plt.subplots(len(As),1,figsize=(20, 5*len(As)))

    # for g, A in enumerate(As):
    #     axa = axes[g] if len(As)>1 else axes
    #     for c in range(ncoef):
    #         axa.plot(Tidx[:A.shape[1]], A[c,:], label='coefficient {}'.format(c))
    #     axa.legend()
    # return fig, axes


__script__ = __doc__

# __warning__ = "Warning:" + warningstyle("\n ")

examples = []
examples.append(["%(prog)s -h", "Print this help messages (about common parameters)"])
examples.append(["%(prog)s raw DBDIR/153/Raw.pkl OUTDIR/153", "Plot raw data (both static and dynamic) of the project 153 in the folder OUTDIR/153/Raw."])
examples.append(["%(prog)s preprocess DBDIR/153/Preprocessed_static.pkl DBDIR/153 --html", "Plot preprocessed static data of the project 153 in the folder DBDIR/153/Preprocessed_static additionally with the output in html format."])
__example__ = "Some examples of use (change the path seperator '/' to '\\' on Windows platform):" + "".join([examplestyle(x) for x in examples])


def main():
    # usage_msg = '%(prog)s <subcommand> <infile> <outdir> [options]'
    # parser = argparse.ArgumentParser(description=__script__, usage=usage_msg)
    mainparser = argparse.ArgumentParser(description=__script__,
                                         formatter_class=argparse.RawDescriptionHelpFormatter,
                                        #  epilog=__warning__ + "\n\n" +
                                         epilog = __example__)#,add_help=False)

    subparsers = mainparser.add_subparsers(title='subcommands', description='type of plot', dest='subcommand')
    parser_data = subparsers.add_parser('data', help='plot raw data or preprocessed static data')
    # parser_raw = subparsers.add_parser('raw', parents=[mainparser], help='Plot raw data')
    # parser_preproc = subparsers.add_parser('preprocess', help='plot preprocessed static data')
    # parser_deconv = subparsers.add_parser('deconv', help='Plot and analyse the results of deconvolution (cf. osmos_deconv)')
    # parser_thermal = subparsers.add_parser('thermal', help='Plot the results of thermal analysis (cf. osmos_thermal)')
    parser_analyse = subparsers.add_parser('analyse', help='plot the results of analysis (cf. osmos_deconv and osmos_thermal)')

    for parser in [parser_data, parser_analyse]:
        parser.add_argument('infile', type=str, help='input data file.')
        parser.add_argument('outdir', nargs='?', type=str, default=None, help='directory where results (figures and data files) will be saved.')
        # parser.add_argument('outdir', type=str, default=None, help='directory where results (figures and data files) will be saved.')
        parser.add_argument('-v', '--verbose', dest='verbose', action='count', default=0, help='print messages.')
        # parser.add_argument('--info', dest='info', action='store_true', default=False, help='Print only information about the project.')
        parser.add_argument('--html', dest='html', action='store_true', default=False, help='Generate plots in html format (in addition of pdf format).')

    for parser in [parser_analyse]:
        lstat_opts = parser.add_argument_group('Options for event and instability detection')  # local statistics
        lstat_opts.add_argument('--nthresh', dest='nthresh', type=float, default=4., help='threshold value of normalized residual for transient event detection in seasonal components (default=4.).', metavar='float')
        lstat_opts.add_argument('--sthresh', dest='sthresh', type=float, default=0.015, help='threshold value of variance for instability detection in trend components (default=0.015).', metavar='float')
        lstat_opts.add_argument('--hthresh', dest='hthresh', type=float, default=0.8, help='threshold value of Hurst exponent (between 0 and 1) for instability detection in trend components (default=0.8).', metavar='float')
        lstat_opts.add_argument('--gap', dest='gap', type=int, default=24*5, help='minimal length of instability period (default=24*5).', metavar='integer')
        lstat_opts.add_argument('--drophead', dest='drophead', type=int, default=0, help='drop head of the results of BM model (default=24*30).', metavar='integer')
        lstat_opts.add_argument('--pval', dest='pval', type=float, default=0.9, help='level of the interval of confidence (between 0. and 1.) around the coefficients of the Brownian motion model (default=0.9).', metavar='float')

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
    if options.subcommand.upper() == "DATA":
        if options.infile[-7:].upper() == "RAW.PKL":
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
        elif options.infile[-23:].upper() == "PREPROCESSED_STATIC.PKL":
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
        else:
            raise NameError("Name of the input data file must be 'Raw' or 'Preprocessed_static'.")
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
        Locations = list(Res['Yprd'].keys())  # list of active sensors
        idx = Res["func_name"].find('_')
        analyse_type = Res["func_name"][:idx]  # type of analysis of the result
        Yprd = Res['Yprd']  # predictions
        Xcpn = Res['Xcpn'][Locations]  # Res['Xcpn'] contains temperature data of all sensors
        Ycpn = Res['Ycpn'][Locations]  # Res['Ycpn'] contains elongation data of all sensors, extract only those having some prediction results
        Yerr = Res['Yerr']  # thermal residual
        Yssp = Res['Yssp']  # subspace projection
        cdim = Res['cdim']  # dimension of subspace
        Yerp = Res['Yerp']  # final residual
        Midx = Res['Midx']  # indexes of missing data
        Merr = Res['Merr']
        Serr = Res['Serr']
        Nerr = Res['Nerr']
        Merp = Res['Merp']
        Serp = Res['Serp']
        Nerp = Res['Nerp']
        Hexp = Res['Hexp']
        Tidx = Xcpn.index

        algo_options = dict(Res['algo_options'])  # options of parameters
        component = algo_options['component']  # name of the component being analyzed

        # if analyse_type.upper() == "DECONV":
        #     staticflag = algo_options['subcommand'].upper() == 'STATIC'  # method of analysis
        #     trnperiod = algo_options['trnperiod'] if staticflag else None   # training period, for static model only
        # else:
        #     staticflag = None
        #     trnperiod = None

        if analyse_type.upper() == "DECONV":
            staticflag = algo_options['subcommand'].upper() == 'LS'  # method of analysis
        else:
            staticflag = None
        trnperiod = algo_options['trnperiod']

        # Generate alarms
        if options.verbose:
            print('Generating alarms...')
        # Indexes of transient event: defined only for seasonal and all component
        if component.upper() in ['SEASONAL', 'ALL']:
            Eidx={}
            for loc, nerr in Nerr.items():
                toto = nerr > options.nthresh
                Eidx[str(loc)] = [str(n) for n,v in toto.items() if v]
                # Eidx[loc] = [str(Tidx[n]) for n,v in enumerate(toto) if v]
            # Eidx = [str(Tidx[n]) for n,v in enumerate(Nerr > options.nthresh) if v]
        else:
            Eidx = None
        # Hurst exponents and index ranges of instability period: defined only for trend or all component
        if component.upper() in ['TREND', 'ALL']:
            Sblk, Sblk_ts = {}, {}
            # indicator of instability based on local variance
            for loc, sexp in Serp.items():
                Sblk[loc] = detect_periods_of_instability(sexp, options.sthresh, hgap=options.gap, mask=Midx[loc])
                # Sblk_ts[str(loc)] = blk2ts(Sblk[loc], Tidx)
                Sblk_ts[str(loc)] = [[str(Tidx[b[0]]), str(Tidx[b[1]-1])] for b in Sblk[loc]]
            Hblk, Hblk_ts = {}, {}
            # indicator of instability based on Hurst exponent
            for loc, hexp in Hexp.items():
                Hblk[loc] = detect_periods_of_instability(hexp, options.hthresh, hgap=options.gap, mask=Midx[loc])
                # Hblk_ts[str(loc)] = blk2ts(Hblk[loc], Tidx)
                Hblk_ts[str(loc)] = [[str(Tidx[b[0]]), str(Tidx[b[1]-1])] for b in Hblk[loc]]
        else:
            Hblk, Hblk_ts = None, None
            Sblk, Sblk_ts = None, None

        # Export alarms
        outfile = os.path.join(figdir, 'Alarms') + '.json'
        alarms = {"Periods of instability":Sblk_ts, "Periods of relative instability":Hblk_ts, "Transient events":Eidx}
        # print(Eidx)
        # resjson = to_json(alarms, verbose=options.verbose)
        # print(resjson)
        with open(outfile, 'w') as fp:
            # json.dump(resjson, fp, cls=MyEncoder)
            # json.dump(alarms, fp, cls=MyEncoder)
            json.dump(alarms, fp)
        if options.verbose:
            print('Alarms exported in\n{}'.format(outfile))

        # Make plots
        if options.verbose:
            print('Generating plots...')

        # summary plot (no summary for seasonal since too cumbersome)
        if component.upper() in ["TREND", "ALL"]:
            summary_plot(Xcpn, Ycpn, Yprd, Yerr, Yssp, cdim, Yerp, figdir, html=options.html)
            pca_plot(Yerr, figdir)

        # per sensor plot
        for n, loc in enumerate(Locations):
            figdir1 = os.path.join(figdir, '{}'.format(loc))
            try:
                os.makedirs(figdir1)
            except:
                pass

            if options.verbose:
                print('\tPlotting the result of location {}...'.format(loc))

            mask = np.logical_or(Midx[loc], np.isnan(Yprd[loc]))  # mask of nans
            if not staticflag:
                mask[:options.drophead] = True
            if component.upper() == "TREND":
                nfig = 3
                fig, axes = plt.subplots(nfig, 1, figsize=(20, 5*nfig), sharex=True)
                k = 0
                plot_obs_prd(axes[k], Xcpn[loc], Ycpn[loc], Yprd[loc], Yerr[loc], Yssp[loc], Yerp[loc], mask, trnperiod=trnperiod)
                k += 1
                plot_local_statistics(axes[k], Yerp[loc], Merp[loc], Serp[loc], options.sthresh, Sblk[loc], mask)
                k += 1
                plot_alarm_period(axes[k], Yerp[loc], Merp[loc], Hexp[loc], options.hthresh, Hblk[loc], mask)
                k += 1
            elif component.upper() == "SEASONAL":
                nfig = 2
                fig, axes = plt.subplots(nfig, 1, figsize=(20, 5*nfig), sharex=True)
                k = 0
                plot_obs_prd(axes[k], Xcpn[loc], Ycpn[loc], Yprd[loc], Yerr[loc], None, None, mask, trnperiod=trnperiod)
                k += 1
                plot_alarm_transient(axes[k], Nerr[loc], options.nthresh, mask)
                k += 1
            else:
                nfig = 4
                fig, axes = plt.subplots(nfig, 1, figsize=(20, 5*nfig), sharex=True)
                k = 0
                plot_obs_prd(axes[k], Xcpn[loc], Ycpn[loc], Yprd[loc], Yerr[loc], Yssp[loc], Yerp[loc], mask, trnperiod=trnperiod)
                k += 1
                plot_local_statistics(axes[k], Yerp[loc], Merp[loc], Serp[loc], options.sthresh, Sblk[loc], mask)
                k += 1
                plot_alarm_period(axes[k], Yerp[loc], Merp[loc], Hexp[loc], options.hthresh, Hblk[loc], mask)
                k += 1
                plot_alarm_transient(axes[k], Nerr[loc], options.nthresh, mask)
                k += 1
            fig.tight_layout()

            fname = os.path.join(figdir1, 'hthresh[{}]_sthresh[{}]_nthresh[{}]'.format(options.hthresh, options.sthresh, options.nthresh))
            fig.savefig(fname+'.pdf', bbox_inches='tight')
            if options.html:
                mpld3.save_html(fig, fname+'.html')
            plt.close(fig)

            # Plot kernels
            if staticflag:
                Amat = Res['Amat'][n,:]
                nx = len(Locations)
                # lag = len(Amat) / nx
                fig, axa = plt.subplots(1,1,figsize=(10,5))
                axa.plot(np.mean(Amat.reshape((-1,nx)), axis=-1)) # [::nx])
                axa.set_title('Location {}: convolution kernel'.format(loc))
                # fig, axes = deconv_plot_static_kernel(Amat)  # str(loc) since in Json file keys of a dictionary must be string
                # # plt.suptitle('Location {}, {} component'.format(loc, component), position=(0.5,1.1),fontsize=20)
                # plt.tight_layout()
                fname = os.path.join(figdir1, 'Kernel_static')
                fig.savefig(fname+'.pdf', bbox_inches='tight')
            else:
                Amat, Acov = Res['Amatc'][:,n,:].T, Res['Acovc'][:,n,:].T
                fig, axa = deconv_plot_dynamic_kernel(Amat, Tidx)
                axa.set_title('Location {}: evolution of the convolution kernel'.format(loc))
                plt.tight_layout()
                fname = os.path.join(figdir1, 'Kernel_dynamic')
                fig.savefig(fname+'.pdf', bbox_inches='tight')

                fig, axa = deconv_plot_mean_dynamic_kernel(Amat, Acov, Tidx, pval=options.pval)
                axa.set_title('Location {}: evolution of the convolution kernel (mean coefficient)'.format(loc))
                plt.tight_layout()
                fname = os.path.join(figdir1, 'Kernel_dynamic_mean')
                fig.savefig(fname+'.pdf', bbox_inches='tight')
    else:
        raise TypeError('Unknown command')

    if options.verbose:
        print('Results saved in\n{}'.format(figdir))


if __name__ == "__main__":
    main()
