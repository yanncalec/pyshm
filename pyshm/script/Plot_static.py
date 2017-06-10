#!/usr/bin/env python

"""Plot results of analysis.
"""

import sys, os, argparse
import numpy as np
import scipy
import scipy.special
import pandas as pd
import json
# from pyshm import OSMOS, Tools, Stat, Models
# from pyshm.script import static_data_analysis_template, examplestyle, warningstyle, load_result_of_analysis
# from pyshm.script import MyEncoder, to_json
import warnings

import matplotlib
matplotlib.use("macosx")
import matplotlib.pyplot as plt
# import matplotlib.colors as colors
import mpld3
plt.style.use('ggplot')

def read_df(Results, key):
    if not key in Results:
        return None
    else:
        X = Results[key].copy()
        if 'index' in X:
            X.index = X['index']
            del X['index']
        return X

def location_plot(Tcpn, Ecpn, Eprd, Eerr, Essp, Astd, Apers, Atran, figdir, html=False):
    SNR = 10 * np.log10(Ecpn.var()/Eerr.var())  # Eprd.var()/Eerr.var() is not correct!
    Eerp = Eerr - Essp
    nfigs = 4
    for n, loc in enumerate(Tcpn.columns):
        fig, _ = plt.subplots(nfigs, 1, figsize=(20,5*nfigs), sharex=True)
        axa = fig.axes[0]
        axa.plot(Ecpn[loc], color='b', alpha=0.4, label='Elongation: observation')
        axa.plot(Eprd[loc], color='c', alpha=0.5, label='Elongation: thermal prediction')
        axa.plot(Eerr[loc], color='g', alpha=0.8, label='Elongation: thermal residual')
        axa.plot(Essp[loc], color='orange', alpha=0.5, label='Elongation: subspace projection')
        axa.plot(Eerp[loc], color='darkblue', alpha=0.8, label='Elongation: non-thermal residual')
        axa.legend(loc='lower left', fancybox=True, framealpha=0.5)
        axb = axa.twinx(); axb.patch.set_alpha(0.0)
        axb.plot(Tcpn[loc], color='r', alpha=0.4, label='Temperaure: observation')
        axb.legend(loc='upper right', fancybox=True, framealpha=0.5)
        # axa.set_title('Location: {}'.format(loc))
        axa.set_title('Location: {}, SNR: {:0.3} db'.format(loc, SNR[loc]))

        if Astd is not None:
            axa = fig.axes[1]
            axa.plot(Eerr[loc], color='g', alpha=0.8, label='Elongation: thermal residual')
            axa.legend(loc='upper left')
            axb = axa.twinx(); axb.patch.set_alpha(0.0)
            axb.plot(Astd[loc], color='r', alpha=0.4, label='Local standard deviation')
            axb.legend(loc='upper right')

        if Apers is not None:
            axa = fig.axes[2]
            axa.plot(Eerr[loc], color='g', alpha=0.8, label='Elongation: thermal residual')
            axa.legend(loc='upper left')
            axb = axa.twinx(); axb.patch.set_alpha(0.0)
            axb.plot(Apers[loc], color='r', alpha=0.4, label='Hurst exponent')
            axb.legend(loc='upper right')

        if Atran is not None:
            axa = fig.axes[3]
            axa.plot(Atran[loc], color='r', alpha=0.4)
            axa.set_title('Transient events')

        fig.tight_layout()
        # save figure
        fname = os.path.join(figdir, str(loc))
        fig.savefig(fname+".pdf")
        if html:
            mpld3.save_html(fig, fname+".html")


def clustering_plot_2d(Scof, Locations, figdir):
    fig,_ = plt.subplots(1,3, figsize=(15,5))
    axa = fig.axes[0]
    axb = fig.axes[1]
    axc = fig.axes[2]

    for n,loc in enumerate(Locations):
        xx,yy = Scof[n,[0,1]]
        axa.scatter(xx,yy)
        axa.text(xx+.001, yy+.001, str(loc))

        xx,yy = Scof[n,[0,2]]
        axb.scatter(xx,yy)
        axb.text(xx+.001, yy+.001, str(loc))

        xx,yy = Scof[n,[1,2]]
        axc.scatter(xx,yy)
        axc.text(xx+.001, yy+.001, str(loc))

    # axa.set_zlabel('3rd pv')
    axa.set_xlabel('1st pv')
    axa.set_ylabel('2nd pv')
    axa.set_title("PCA coefficients of sensors")
    axb.set_xlabel('1st pv')
    axb.set_ylabel('3nd pv')
    axb.set_title("PCA coefficients of sensors")
    axc.set_xlabel('2st pv')
    axc.set_ylabel('3nd pv')
    axc.set_title("PCA coefficients of sensors")
    fig.tight_layout()
    # save figure
    fname = os.path.join(figdir, "PCA_2d")
    fig.savefig(fname+".pdf")


def clustering_plot_3d(Scof, Locations, figdir):
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


def summary_plot(Tcpn, Ecpn, Eprd, Eerr, Virt, Eerp, figdir, html=False):
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
    Virt.plot(ax=axes[4])
    axes[4].legend(loc="lower left", fancybox=True, framealpha=0.5)
    axes[4].set_title("Elongation: virtual sensor(s)")
#     axa = axes[4]
#     Essp.plot(ax=axa)
#     Essp.mean(axis=1).plot(ax=axa, linewidth=3, alpha=0.3, color="b", linestyle="-", label="Mean behavior")
#     axa.legend(loc="lower left", fancybox=True, framealpha=0.5)
#     axa.set_title("Elongation: non-thermal subspace projection of dimension {}".format(cdim))
    axa = axes[5]
    Eerp.plot(ax=axa)
    axa.legend(loc="lower left", fancybox=True, framealpha=0.5)
    axa.set_title("Elongation: non-thermal residual")
    fig.tight_layout()

    # save figure
    fname = os.path.join(figdir, "Summary")
    fig.savefig(fname+".pdf")
    if html:
        mpld3.save_html(fig, fname+".html")


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


# def deconv_plot_dynamic_kernel(Krnl, Tidx, ncoef=10):
#     """Plot mean value of the dynamic kernel.

#     Args:
#         Krnl (list): Krnl[t][g][n] is the n-th kernel matrix (of shape 1-by-?)
#         of the group g at the time index t.

#     """
#     import matplotlib.pyplot as plt
#     # import numpy as np

#     Krnl_mean = np.mean(Krnl, axis=0)

#     fig, axa = plt.subplots(1,1,figsize=(20, 5))
#     for c in range(min(Krnl.shape[0], ncoef)):
#         axa.plot(Tidx, Krnl[c,:], label='coefficient {}'.format(c))
#         axa.legend()
#     axa.plot(Tidx, Krnl_mean, linewidth=3, label='mean coefficient')
#     axa.legend(loc='upper right', fancybox=True, framealpha=0.5)
#     return fig, axa


def plot_mean_dynamic_kernel(Krnl_mean, Kcov_mean, pval=0.9):
    """Plot mean value of the dynamic kernel.

    Args:
        Krnl (list): Krnl[t][g][n] is the n-th kernel matrix (of shape 1-by-?)
        of the group g at the time index t.

    """
    # import matplotlib.pyplot as plt
    # import numpy as np
    Tidx = Krnl_mean.index
    # Krnl_mean = np.mean(Krnl, axis=0)
    fig, axa = plt.subplots(1,1,figsize=(20, 5))
    axa.plot(Krnl_mean, label='mean coefficient')
    if pval > 0:
        a = scipy.special.erfinv(pval) * np.sqrt(2)  #
        # Kcov_mean = np.sum(Kcov)/(Kcov.shape[0])**2
        v = a * np.sqrt(Kcov_mean)
        y1 = Krnl_mean - v
        y2 = Krnl_mean + v
        axa.fill_between(Tidx, y1, y2=y2)
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
__example__ = []

def main():
    # usage_msg = '%(prog)s <infile> [options]'
    # parser = argparse.ArgumentParser(description=__script__, usage=usage_msg)
    parser = argparse.ArgumentParser(description=__script__,
                                    formatter_class=argparse.RawDescriptionHelpFormatter,
                                    #  epilog=__warning__ + "\n\n" +
                                    epilog = __example__)#,add_help=False)

    parser.add_argument('infile', help='directory of the local OSMOS database')

    # parser.add_argument('-p', '--PID', dest='PID', type=str, default=None, help='project Key ID to be processed (by default all projects presented on the remote server will be processed)', metavar='integer')
    # parser.add_argument('--info', dest='info', action='store_true', default=False, help='save the list of available projects')
    # parser.add_argument('-s', '--sflag', dest='sflag', action='store_true', default=False, help="remove synchronization error")
    # parser.add_argument('-o', '--oflag', dest='oflag', action='store_true', default=False, help="remove outliers")
    # parser.add_argument('-t', '--tflag', dest='tflag', action='store_true', default=False, help="apply the preprocessing on the temperature data")
    # parser.add_argument('-j', '--jflag', dest='jflag', action='store_true', default=False, help="detect jumps in the deformation data")
    # parser.add_argument('-n', dest='nh', action='store', type=int, default=12, help="gaps (in hour) larger than this value will be marked as nan (default 12)", metavar="int")
    # parser.add_argument('--port', dest='port', action='store', type=int, default=27017, help="port of local MongoDB (default=27017)", metavar="int")
    # parser.add_argument("--json", dest="json", action="store_true", default=False, help="save results in json format")
    parser.add_argument('--html', dest='html', action='store_true', default=False, help='Generate plots in html format (in addition of pdf format).')
    # parser.add_argument("--plot", dest="plot", action="store_true", default=False, help="plot data")
    parser.add_argument('-v', '--verbose', dest='verbose', action='count', default=0, help='print messages')

    options = parser.parse_args()

    if not os.path.isfile(options.infile):
        raise FileNotFoundError(options.infile)
    options.figdir = options.infile[:options.infile.rfind(os.path.sep)]  # output directory for figures

    if options.verbose:
        print("Generating figures...")

    Results = pd.read_excel(options.infile, sheetname=None)
    # print(Results.keys())

    # class Options:
    #     def __init__(self, **entries):
    #         self.__dict__.update(entries)

    # with open(fname, 'r') as fp:
    #     toto = json.load(fp)
    # options = Options(**toto)

    Tcpn = read_df(Results, 'Temperature')
    Ecpn = read_df(Results, 'Elongation')
    Eprd = read_df(Results, 'Prediction')
    Eerr = Ecpn - Eprd
    Virt = read_df(Results, 'Virtual sensors')
    Essp = read_df(Results, 'Subspace projection')
    Eerp = Eerr - Essp
    Scof = np.asarray(Results['PCA coefficients']).T
    Atran = read_df(Results, 'Transient')
    Astd = read_df(Results, 'Std')
    Apers = read_df(Results, 'Persistence')

    Locations = list(Tcpn.keys())

    Virt_sm = Virt.rolling(24, center=True).mean()
    cdim = Virt.shape[1]

    summary_plot(Tcpn, Ecpn, Eprd, Eerr, Virt, Eerp, options.figdir, html=options.html)

    location_plot(Tcpn, Ecpn, Eprd, Eerr, Essp, Astd, Apers, Atran, options.figdir, html=options.html)

    # Amat_ls = read_df(Results, "Flattened kernel")
    # Amatc_ls = read_df(Results, "Flattened reduced kernel")

    # Amat_bm = read_df(Results, "Mean kernel")
    # Acov_bm = read_df(Results, "Mean var. of kernel")
    # Amatc_bm = read_df(Results, "Mean reduced kernel")
    # Acovc_bm = read_df(Results, "Mean var. of reduced kernel")

    # if Amat_bm is not None and Amatc_bm is not None:
    #     # BM model
    #     # print(Amat_bm.head())
    #     # print(Acov_bm.head())
    #     # print(Amatc_bm.head())
    #     # print(Acovc_bm.head())
    #     for loc in Amatc_bm.columns:
    #         fig, axa = plot_mean_dynamic_kernel(Amat_bm[loc], Acov_bm[loc], pval=0.)
    #         axa.set_title('Location {}: evolution of the convolution kernel (mean coefficient)'.format(loc))
    #         # plt.tight_layout()
    #         fname = os.path.join(options.figdir, '{}_mean_kernel.pdf'.format(loc))
    #         fig.savefig(fname, bbox_inches='tight')
    #         plt.close(fig)


    if len(Locations) > 2:
        clustering_plot_2d(Scof, Locations, options.figdir)
        clustering_plot_3d(Scof, Locations, options.figdir)

    if options.verbose:
        print("Figures saved in {}".format(options.figdir))


if __name__ == '__main__':
    main()
