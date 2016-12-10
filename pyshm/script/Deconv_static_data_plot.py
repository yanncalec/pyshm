#!/usr/bin/env python

"""Plot results of the analysis of static data returned by deconvolution.
"""


import sys, os
from optparse import OptionParser       # command line arguments parser

__script__ = __doc__


def main():
    # Load data
    usage_msg = '{} [options] <input_data_file> [output_directory]'.format(sys.argv[0])
    parm_msg = '\tinput_data_file : file returned by the script of data analysis\n\toutput_directory :  directory where results are saved (default: in the same folder as input_data_file).'

    parser = OptionParser(usage=usage_msg+'\n'+parm_msg)

    parser.add_option('--vthresh', dest='vthresh', type='float', default=4., help='Threshold value for event detection (default=4).')
    parser.add_option('--mwsize0', dest='mwsize0', type='int', default=6, help='Size of the moving window for local statistics (default=6).')
    parser.add_option('--mwsize1', dest='mwsize1', type='int', default=24*10, help='Size of the moving window for global statistics (default=24*10).')
    parser.add_option('--mad', dest='mad', action='store_true', default=False, help='Use median based estimator (default: use empirical estimator).')
    parser.add_option('-v', '--verbose', dest='verbose', action='store_true', default=False, help='Print message.')

    (options, args) = parser.parse_args()

    if len(args) < 1:
        print('Usage: ' + usage_msg)
        print(parm_msg)

        sys.exit(0)
    else:
        # Lazy import
        import pickle
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import mpld3
        plt.style.use('ggplot')

        infile = args[0]
        if not os.path.isfile(infile):
            raise FileNotFoundError(infile)

        # loc = int(args[1])

        if len(args)==2:
            figdir0 = args[1]
        else:
            idx = infile.rfind(os.path.sep)
            figdir0 = infile[:idx]

    # Load raw data
    with open(infile, 'rb') as fp:
        Res = pickle.load(fp)

    Xcpn = Res['Xcpn']  # Component of temperature
    Ycpn = Res['Ycpn']  # Component of elongation
    Yprd = Res['Yprd']  # Prediction of elongation
    Yerr = Res['Yerr']  # Error of prediction
    Aprd = Res['Tprd']  # Contribution of first group of inputs
    Bprd = Res['Eprd']  # Contribution of second group of inputs
    saved_options=Res['options']  # options of parameters
    Mxd = Res['Mxd']  # Objects of deconvolution model
    Midx = Res['Midx']  # Indicator of missing values

    Tidx = Xcpn.index

    sidx = saved_options.sidx
    Ntrn = saved_options.Ntrn
    component = saved_options.component

    # local and global statistics
    if options.mad:  # use median-based estimator
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

    nErr0 = abs(Err-mErr0)/sErr0
    nErr1 = abs(Err-mErr1)/sErr1

    for ridx, loc in enumerate(saved_options.ylocs):
        if options.verbose:
            print('Plotting the result of location {}...'.format(loc))

        g0 = Gs[:,ridx,:].mean(axis=1)
        h0 = Hs[:,ridx,:].mean(axis=1)

        figdir = os.path.join(figdir0, str(loc))
        try:
            os.makedirs(figdir)
        except OSError:
            pass
        # if not os.path.isdir(outdir):
        #     raise FileNotFoundError(outdir)

        # Plot the kernel
        fig, axes = plt.subplots(1,2,figsize=(20,5))
        # plot(AData[loc].index[tidx0+max(ng,nh):twsize-max(ng,nh)+tidx0], err)
        axa = axes[0]
        axa.plot(h0, 'b', label='Least square')
        # axa.plot(h1, 'r', label='Penalization')
        axa.legend(loc='upper right')
        # axa.set_title('Kernel of auto-regression, penalization={}'.format(penalh))
        axa.set_title('Kernel of auto-regression')
        _ = axa.set_xlim(-1,)

        axa = axes[1]
        axa.plot(g0, 'b', label='Least square')
        # axa.plot(g1, 'r', label='Penalization')
        axa.legend(loc='upper right')
        axa.set_title('Kernel of convolution')
        _ = axa.set_xlim(-1,)

        fig.savefig(figdir+'/Kernels.pdf', bbox_inches='tight')
        plt.close(fig)

        # Plot the residual
        nfig, k = 5, 0
        fig, axes = plt.subplots(nfig,1, figsize=(20, nfig*5), sharex=True)

        # Raw data
        axa = axes[k]
        axa.plot(Yall[loc], color='b', alpha=0.5, label='Elongation')
        axb = axa.twinx()
        axb.patch.set_alpha(0.0)
        axb.plot(Xall[loc], color='r', alpha=0.5, label='Temperature')
        axa.legend(loc='upper left')
        axb.legend(loc='upper right')
        axa.set_title('Signals of the location {}'.format(loc))
        k+=1

        # User-specified component and ARX-prediction
        axa = axes[k]
        axa.plot(Yvar[loc], color='b', alpha=0.5, linewidth=2, label='Elongation')
        axa.plot(Yprd[loc], color='c', alpha=0.7, label='Prediction')
        axb = axa.twinx()
        axb.patch.set_alpha(0.0)
        sgn = np.sign(g0[0]) if len(g0)>0 else 1  # adjust the sign of Xvar
        axb.plot(sgn*Xvar[loc], color='r', alpha=0.5, label='Temperature')
        axa.legend(loc='upper left')
        axb.legend(loc='upper right')
        t0, t1 = Tidx[sidx], Tidx[sidx+Ntrn-1]
        # print(t0,t1)
        # axa.fill_betweenx(np.arange(-100,100), t0, t1, color='c', alpha=0.2)
        # axa.axvspan(t0, t1, color='c', alpha=0.2)
        axa.set_title('{} components of the location {} (sign adjusted for the temperature)'.format(component, loc))
        k+=1

        # Normalized residual
        axa = axes[k]
        axa.plot(nErr1[loc])
        # axa.set_ylim((0,6))
        axa.fill_between(Tidx, 0, options.vthresh, color='c', alpha=0.2)
        axa.set_title('Normalized residual: (error-mean)/std')
        k+=1

        # Local mean and standard deviation of the residual
        axa = axes[k]
        axa.plot(mErr0[loc], color='b', alpha=0.5, label='Local mean window={}'.format(options.mwsize0))
        axa.plot(mErr1[loc], color='c', label='Local mean window={}'.format(options.mwsize1))
        axa.legend(loc='upper left')
        axa.set_title('Local mean of the residual')
        k+=1

        axa = axes[k]
        # axb = axa.twinx()
        # axb.patch.set_alpha(0.0)
        axa.plot(sErr0[loc], color='r', alpha=0.5, label='Local standard deviation window={}'.format(options.mwsize0))
        axa.plot(sErr1[loc], color='m', label='Local standard deviation window={}'.format(options.mwsize1))
        axa.legend(loc='upper left')
        axa.set_title('Local standard deviation of the residual')
        k+=1

        # # Normalized mean
        # axa = axes[k]
        # axa.plot(mErr0/sErr0, label='window={}'.format(options.mwsize0))
        # axa.plot(mErr1/sErr1, label='window={}'.format(options.mwsize1))
        # axa.legend()
        # axa.set_title('Normalized mean of the: residual mean/std')
        # k+=1
        #
        # # Residual
        # axa = axes[k]
        # axa.plot(Err)
        # axa.set_title('Residual')
        # k+=1

        fname = os.path.join(figdir, 'Plots')
        fig.savefig(fname+'.pdf', bbox_inches='tight')
        mpld3.save_html(fig, fname+'.html')
        plt.close(fig)

    if options.verbose:
        print('Figures saved in {}'.format(figdir0))


if __name__ == "__main__":
    print(__script__)
    # print()
    main()
