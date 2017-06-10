#!/usr/bin/env python

"""Analysis of static data.
"""

import os, sys, argparse
import json
import numpy as np
import scipy
import pandas as pd
from pymongo import MongoClient
import warnings
from sklearn.cluster import KMeans #, AffinityPropagation
import pywt

import matplotlib
# matplotlib.use("macosx")
import matplotlib.pyplot as plt
# import matplotlib.colors as colors
import mpld3
plt.style.use('ggplot')

import pyshm
# from pyshm import OSMOS, Tools, Stat, Models
# from pyshm.script import static_data_analysis_template, examplestyle, warningstyle, load_result_of_analysis
# from pyshm.script import MyEncoder, to_json
# from pyshm.script import examplestyle, warningstyle

# __seperator__ = '-'*50

# key2str = lambda D: {str(k):v for k,v in D.items()}

# transform raw elongation to mm

def compute_local_statistics(Yerr, mad, mwsize, win_type='boxcar'):
    """Compute the local statistics: mean and standard deviation and normalized
    observation.

    """
    Merr0 = {}; Serr0 = {}; Nerr0 = {}

    for loc, yerr in Yerr.items():
        # yerr = Ycpn[loc] - Yprd[loc]
        # moving average and standard deviation
        merr, serr = pyshm.Stat.local_statistics(yerr, mwsize, mad=mad, causal=False, drop=False, win_type=win_type)
        nerr = abs(yerr-merr)/serr  # normalized error
        Merr0[loc], Serr0[loc], Nerr0[loc] = merr, serr, nerr

    Merr = pd.DataFrame(Merr0, columns=list(Yerr.keys()), index=Yerr.index)
    Serr = pd.DataFrame(Serr0, columns=list(Yerr.keys()), index=Yerr.index)
    Nerr = pd.DataFrame(Nerr0, columns=list(Yerr.keys()), index=Yerr.index)

    return Merr, Serr, Nerr

# union2list = lambda L1, L2: L1+[x for x in L2 if x not in L1]

def Hurstfunc(loc, X, mwsize, hrng):
    Y, *_ = pyshm.Stat.Hurst(np.asarray(X), mwsize, sclrng=hrng, wvlname="haar")  # Hurst exponent
    return {loc: Y}


def plot_static(Data, fname):
    # import matplotlib
    # # matplotlib.use("qt5agg")
    # import matplotlib.pyplot as plt
    # # import matplotlib.colors as colors
    # import mpld3
    # plt.style.use('ggplot')

    fig, _ = plt.subplots(3,1,figsize=(20,15), sharex=True)
    axa = fig.axes[0]
    axa.plot(Data['ElongationTfm'], 'b', alpha=0.5, label='Elongation')
    axa.legend(loc='upper left')
    axb = axa.twinx()
    axb.plot(Data['TemperatureTfm'], 'r', alpha=0.5, label='Temperature')
    axb.legend(loc='upper right')
    # axa.set_title('UID: {}'.format(uid))

    axa = fig.axes[1]
    axa.plot(Data['ElongationRaw'], 'b', alpha=0.5, label='Elongation Raw')
    axa.legend(loc='upper left')
    axb = axa.twinx()
    axb.plot(Data['TemperatureRaw'], 'r', alpha=0.5, label='Temperature Raw')
    axb.legend(loc='upper right')

    axa = fig.axes[2]
    # axa.plot(Data['parama'], 'b', alpha=0.5, label='a')
    # axa.plot(Data['paramb'], 'r', alpha=0.5, label='b')
    # axa.plot(Data['paramc'], 'g', alpha=0.5, label='c')
    axa.plot(Data['Reference'], 'c', alpha=0.7, label='Reference')
    axa.legend(loc='upper left')
    # axb = axa.twinx()
    # axb.legend(loc='upper right')

    plt.tight_layout()
    fig.savefig(fname+".pdf", bbox_inches='tight')
    plt.close(fig)

    return fig


__script__ = __doc__

# __warning__ = "Warning:" + warningstyle("\n This script can be applied only on data preprocessed by the script osmos_preprocessing (the data file is typically named Preprocessed_static.pkl). Two distinct models (static and dynamic) are implemented and are accessible via the corresponding subcommand.")

# examples = []
# examples.append(["%(prog)s -h", "Print this help messages (about common parameters)"])
# examples.append(["%(prog)s ls -h", "Print help messages about the static model"])
# examples.append(["%(prog)s bm -h", "Print help messages about the dynamic model"])
# # __example__ = "Some examples of use (change the path seperator '/' to '\\' on Windows platform):" + "".join([examplestyle(x) for x in examples])
# __example__ = "Some examples of use (change the path seperator '/' to '\\' on Windows platform):" + "".join(examples)

# ls_examples = []
# ls_examples.append(["%(prog)s DBDIR/153 OUTDIR/153 --alocs 754 --component trend --time0 2016-03-01 --time1 2016-08-01 -vv", "On the location 754 of the project of PID 153 (preprocessed data), apply the least-square model on the trend component for the period from 2016-03-01 to 2016-08-01 and save the results in the directory named OUTDIR/153. Print supplementary messages."])
# ls_examples.append(["%(prog)s DBDIR/153 OUTDIR/153 --alocs 754,755 -v", "Process the locations 754 and 755, for each of them use the temperature of both to explain the elongation data (vectorial model)."])
# ls_examples.append(["%(prog)s DBDIR/153 OUTDIR/153 -v", "Process all sensors, for each of them use the temperature of all to explain the elongation data."])
# ls_examples.append(["%(prog)s DBDIR/153 OUTDIR/153 --time0 2016-03-01 --Ntrn 1000 -v", "Change the length of the training period to 1000 hours starting from the begining of the truncated data set which is 2016-03-01."])
# ls_examples.append(["%(prog)s DBDIR/153 OUTDIR/153 --component=seasonal -v", "Process the seasonal component of data."])
# __ls_example__ = "Some examples of use (change the path seperator '/' to '\\' on Windows platform):" + "".join([examplestyle(x) for x in ls_examples])

# bm_examples = []
# bm_examples.append(["%(prog)s DBDIR/153 OUTDIR/153 -v", "Use the BM model to process all sensors."])
# __bm_example__ = "Some examples of use (change the path seperator '/' to '\\' on Windows platform):" + "".join([examplestyle(x) for x in bm_examples])

def main():
    usage_msg = "%(prog)s <subcommand> <pid> <dbdir> [options]"
    # parser = argparse.ArgumentParser(description=__script__, usage=usage_msg)
    mainparser = argparse.ArgumentParser(description=__script__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter,)
                                    #  epilog="\n\n" + __example__)
                                    #  epilog=__warning__ + "\n\n" + __example__)

    subparsers = mainparser.add_subparsers(title="subcommands", description="Perform deconvolution or statistical analysis", dest="subcommand")
    parser_ls = subparsers.add_parser("ls", help="Deconvolution using the least-square model",
                                        formatter_class=argparse.RawDescriptionHelpFormatter,)
                                        # epilog=__ls_example__)
    parser_bm = subparsers.add_parser("bm", help="Deconvolution using the Brownian motion model",
                                        formatter_class=argparse.RawDescriptionHelpFormatter,)
                                        # epilog=__bm_example__)
    # parser_analysis = subparsers.add_parser("analysis", help="Statistical analysis of results of deconvolution",
    #                                        formatter_class=argparse.RawDescriptionHelpFormatter,
    #                                        epilog=__analysis_example__)

    for parser in [parser_ls, parser_bm]:
        # parser.add_argument("projdir", help="directory of a project in the database including the preprocessed static data.")
        # parser.add_argument("infile", type=str, help="json data file containing all sensors of one project.")
        # parser.add_argument("dbdir", nargs='?', type=str, default=None, help="directory where results (figures and data files) will be saved.")
        parser.add_argument("pid", type=int, help="project key ID.")
        parser.add_argument("dbdir", type=str, help="directory of local database and outputs.")
        parser.add_argument("-v", "--verbose", dest="verbose", action="count", default=0, help="print messages.")

        # parser.add_argument("--raw", dest="raw", action="store_true", default=False, help="use non-transformed raw data (default: use transformed data).")

        mongo_opts = parser.add_argument_group("MongoDB options")
        mongo_opts.add_argument("--hostname", dest="hostname", type=str, default="localhost", help="name of the MongoDB server (default=localhost).", metavar="string")
        mongo_opts.add_argument("--port", dest="port", type=int, default=27017, help="port of the MongoDB server(default=27017).", metavar="integer")
        mongo_opts.add_argument("--update", dest="update", action="store_true", default=False, help="force updating local database from the MongoDB server.")
        mongo_opts.add_argument("--plot", dest="plot", action="store_true", default=False, help="plot data.")

        # sensor_opts = parser.add_argument_group("Sensor options")

        wdata_opts = parser.add_argument_group("Data options")
        wdata_opts.add_argument("--alocs", dest="alocs", type=lambda s: [int(x) for x in s.split(',')], default=None, help="location ID of sensors to be analyzed (default=all sensors).", metavar="integers separated by \',\'")
        wdata_opts.add_argument("--time0", dest="time0", type=str, default=None, help="starting timestamp (default=the begining of data set).", metavar="YYYY-MM-DD")
        wdata_opts.add_argument("--time1", dest="time1", type=str, default=None, help="ending timestamp (default=the ending of data set).", metavar="YYYY-MM-DD")
        wdata_opts.add_argument("--component", dest="component", type=str, default="all", help="type of component of data to be analyzed: 'all', 'trend', 'seasonal', 'raw' (default='all').", metavar="string")
        wdata_opts.add_argument("--mwmethod", dest="mwmethod", type=str, default="mean", help="type of moving window estimator for decomposition of component: mean (default), median.", metavar="string")
        wdata_opts.add_argument("--mwsize", dest="mwsize", type=int, default=24, help="length of the moving window (default=24).", metavar="integer")
        wdata_opts.add_argument("--kzord", dest="kzord", type=int, default=2, help="order of moving window filter (default=2).", metavar="integer")

        tdata_opts = parser.add_argument_group("Training data options")
        tdata_opts.add_argument("--sidx", dest="sidx", type=int, default=24*10, help="starting time index (an integer) of the training data relative to time0 (default=24*10).", metavar="integer")
        tdata_opts.add_argument("--Ntrn", dest="Ntrn", type=int, default=6*30*24, help="length of the training data (default=24*30*6).", metavar="integer")

        model_opts = parser.add_argument_group("Model options")
        model_opts.add_argument("--lag", dest="lag", type=int, default=6, help="length of the convolution kernel (default=6)", metavar="integer")
        model_opts.add_argument("--pord", dest="pord", type=int, default=None, help="order of non-thermal polynomial process (default=1 for trend or all component, 0 for raw or seasonal component).", metavar="integer")
        model_opts.add_argument("--dord", dest="dord", type=int, default=None, help="order of derivative (default=1 for trend or all component, 0 for raw or seasonal component).", metavar="integer")
        model_opts.add_argument("--pflag", dest="pflag", action="store_true", default=False, help="add polynomial trends in the prediction (default=no polynomial trend).")
        model_opts.add_argument("--vthresh", dest="vthresh", type=float, default=0.01, help="percentage of tolerable information loss for dimension reduction. The principle dimensions corresponding to the percentage of (1-vthresh) will be kept, i.e. 1 percent of information is discarded if vthresh=0.01. No dimension reduction if set to 0 (default=0.01). This parameter has some effect of regularization.", metavar="float")
        # dimr_opts.add_argument("--corrflag", dest="corrflag", action="store_true", default=False, help="use correlation matrix in dimension reduction.")

        cluster_opts = parser.add_argument_group("Clustering and low rank approximation options")
        cluster_opts.add_argument("--cthresh", dest="cthresh", type=float, default=0.1, help="percentage of tolerable information loss in clustering (default=0.1).", metavar="float")
        # cluster_opts.add_argument("--update", dest="update", action="store_true", default=False, help="force updating local database from the MongoDB server.")
        cluster_opts.add_argument("--cdim", dest="cdim", type=int, default=None, help="reduced dimension in low rank approximation. If not set it will be determined automatically from the value of cthresh, otherwise cthresh will be ignored (default=None).", metavar="integer")

        # ddata_opts = parser.add_argument_group("Smoothing options")

        alarm_opts = parser.add_argument_group("Alarm options")
        # alarm_opts.add_argument("--mwmethod", dest="mwmethod", type=str, default="mean", help="type of moving window estimator for decomposition of component: mean (default), median.", metavar="string")
        # alarm_opts.add_argument("--mwsize", dest="mwsize", type=int, default=24, help="length of the moving window (default=24).", metavar="integer")
        alarm_opts.add_argument("--tran", dest="thresh_tran_alarm", type=float, default=0.6, help="threshold of transient events (default=0.6).", metavar="float")
        alarm_opts.add_argument("--pers", dest="thresh_pers_alarm", type=float, default=0.6, help="threshold of persistence or Hurst exponent (default=0.6).", metavar="float")
        alarm_opts.add_argument('--std', dest='thresh_std_alarm', type=float, default=0.015, help='threshold of standard deviation (default=0.015).', metavar='float')
        # alarm_opts.add_argument('--gap', dest='gap', type=int, default=24*5, help='minimal length of instability period (default=24*5).', metavar='integer')

    # for parser in [parser_ls, parser_bm, parser_analysis]:
    #     parser.add_argument("-v", "--verbose", dest="verbose", action="count", default=0, help="Print messages.")

    regressor_opts = parser_ls.add_argument_group("Linear regression options (ls model only)")
    regressor_opts.add_argument("--Nexp", dest="Nexp", type=int, default=0, help="number of experiments in RANSAC (default=0, no RANSAC).", metavar="integer")
    regressor_opts.add_argument("--snr2", dest="snr2", type=float, default=10**4, help="squared signal-to-noise ratio of the Gaussian polynomial process (default=1e4), no effect if clen2 is not set.", metavar="float")
    regressor_opts.add_argument("--clen2", dest="clen2", type=float, default=None, help="squared correlation length of the Gaussian polynomial process (default=None, use deterministic polynomial process).", metavar="float")
    regressor_opts.add_argument("--dspl", dest="dspl", type=int, default=1, help="down-sampling rate of training data for acceleration on large training dataset (default=1, no down-sampling).", metavar="integer")
    regressor_opts.add_argument("--vreg", dest="vreg", type=float, default=0, help="factor of regularization (default=0).", metavar="float")

    kalman_opts = parser_bm.add_argument_group("Kalman filter options (bm model only)")
    kalman_opts.add_argument("--sigmaq2", dest="sigmaq2", type=float, default=10**-6, help="variance of transition noise (default=1e-6).", metavar="float")
    kalman_opts.add_argument("--sigmar2", dest="sigmar2", type=float, default=10**2, help="variance of observation noise (default=1e+2).", metavar="float")
    kalman_opts.add_argument("--kalman", dest="kalman", type=str, default="smoother", help="method of estimation of Kalman filter: filter, smoother (default).", metavar="string")
    kalman_opts.add_argument("--rescale", dest="rescale", action="store_true", default=False, help="rescale the input and output variables to normalize the amplitude.")

    options = mainparser.parse_args()

    # set pord and dord automatically according to the component's type
    if options.component.upper() == 'RAW':
        if options.verbose:
            print("Recommended value for the component {}:".format(options.component.upper()))
            print("\tpord=0, dord=0, pflag=True")
        if options.pord is None:
            options.pord = 0
        if options.dord is None:
            options.dord = 0
    elif options.component.upper() in ['ALL', 'TREND']:
        if options.verbose:
            print("Recommended value for the component {}:".format(options.component.upper()))
            print("\tpord=1, dord=0, pflag=False")
        if options.pord is None:
            options.pord = 1
        if options.dord is None:
            options.dord = 0
    else:
        if options.verbose:
            print("Recommended value for the component {}:".format(options.component.upper()))
            print("\tpord=0, dord=0, pflag=False")
        if options.pord is None:
            options.pord = 0
        if options.dord is None:
            options.dord = 0

    # print(options.pord, options.dord)
    # the following parameters are hard-coded:
    options.snl_threshold = 0.1  # threshold for the energy of the seasonal component
    options.thresh_tran_lowlevel = 2.  # threshold for transient alarms
    options.mwsize_tran_rng = (24, 240)  # range of size of moving window for alarms of transient events
    options.max_thermal_delay = 6  # maximum value of thermal delay, >= 1
    options.mwsize_std_rng = (24*5, 24*30)   # range of size of moving window for alarms of std
    options.hurst_sclrng = (0, 8)  # scale range of wavelet coefficient for alarms of persistence or Hurst exponent
    options.hurst_mwsize = 24*10   # size of moving window for alarms of persistence or Hurst exponent
    options.gap = 24*5  # minimal length of alarms for std of persistence
    # vol_dp_rng = (1, 24)


    ##### Step 1. Load data from local database or from MongoDB #####
    # check local database directory
    if not os.path.isdir(options.dbdir):
        raise FileNotFoundError("Local database directory not found: {}".format(options.dbdir))

    options.projdir = os.path.join(options.dbdir, str(options.pid))  # project directory
    options.infile_data = os.path.join(options.projdir, "preprocessed_static.xlsx")  # input data file of the project
    options.infile_info = os.path.join(options.dbdir, "LIRIS_info.xlsx")  # input LIRIS information file of the project

    options.func_name = __file__[__file__.rfind(os.path.sep)+1 : __file__.rfind('.')]
    if options.subcommand.upper() == 'LS':
        options.outdir = os.path.join(options.projdir, options.func_name, "model[{}]_component[{}]_alocs[{}]_[from_{}_to_{}]_Ntrn[{}]_lag[{}]_pord[{}]_dord[{}]_pflag[{}]_vthresh[{}]_clen2[{}]_snr2[{:.1e}]_dspl[{}]_Nexp[{}]_vreg[{}]".format(options.subcommand.upper(), options.component.upper(), options.alocs, options.time0, options.time1, options.Ntrn, options.lag, options.pord, options.dord, options.pflag, options.vthresh, options.clen2, options.snr2, options.dspl, options.Nexp, options.vreg))  # output directory
        # options.outdir = os.path.join(options.projdir, options.func_name, "model[{}]_component[{}]_alocs[{}]_[from_{}_to_{}]_Ntrn[{}]_lag[{}]_pord[{}]_dord[{}]_pflag[{}]_vthresh[{}]_clen2[{}]_snr2[{:.1e}]_dspl[{}]_Nexp[{}]_vreg[{:.1e}]".format(options.subcommand.upper(), options.component.upper(), options.alocs, options.time0, options.time1, options.Ntrn, options.lag, options.pord, options.dord, options.pflag, options.vthresh, options.clen2, options.snr2, options.dspl, options.Nexp, options.vreg))  # output directory
    else:
        options.outdir = os.path.join(options.projdir, options.func_name, "model[{}]_component[{}]_alocs[{}]_[from_{}_to_{}]_Ntrn[{}]_lag[{}]_pord[{}]_dord[{}]_pflag[{}]_vthresh[{}]_sigmaq2[{:.1e}]_sigmar2[{:.1e}]_rescale[{}]".format(options.subcommand.upper(), options.component.upper(), options.alocs, options.time0, options.time1, options.Ntrn, options.lag, options.pord, options.dord, options.pflag, options.vthresh, options.sigmaq2, options.sigmar2, options.rescale))

    if os.path.isfile(options.infile_data) and os.path.isfile(options.infile_info) and not options.update:  # if both files exist, use local database
        if options.verbose:
            # print(__seperator__)
            print("Loading data from local database...")

        LIRIS_info = pd.read_excel(options.infile_info)  # info of all projects
        LIRIS = LIRIS_info[LIRIS_info['pid']==options.pid].reset_index(drop=True)  # info of this pid
        Sdata0 = pd.read_excel(options.infile_data, sheetname=None)  # <=== this is time consuming
        Sdata = {}
        for loc, val in Sdata0.items():
            Sdata[int(loc)] = val
    else:  # otherwise update local data base
        # 1. Get LIRIS sensor information
        if not os.path.isfile(options.infile_info):
            if options.verbose:
                # print(__seperator__)
                print("Retrieving information of LIRIS sensors of the project {}...".format(options.pid))

            # LIRIS_info = pyshm.OSMOS.retrieve_LIRIS_info([options.pid])
            LIRIS_info_full = pyshm.OSMOS.retrieve_LIRIS_info(list(range(1,500)))

            if LIRIS_info_full is None:
                raise ValueError("Failed to retrieve information.")

            # save essential information in an Excel file
            LIRIS_info = LIRIS_info_full[['pid', 'uid', 'locationkeyid', 'parama', 'paramb', 'paramc']]
            writer = pd.ExcelWriter(options.infile_info)
            LIRIS_info.to_excel(writer)
            writer.save()
            if options.verbose:
                print("Information saved in {}".format(options.infile_info))
        else:
            LIRIS_info = pd.read_excel(os.path.join(options.dbdir, 'LIRIS_info.xlsx'))

        # 2. Get LIRIS data
        if not os.path.isfile(options.infile_data):
            if options.verbose:
                print("Retrieving data of the project {} from MongoDB...".format(options.pid))

            LIRIS = LIRIS_info[LIRIS_info['pid']==options.pid].reset_index(drop=True)  # info of this pid
            Sdata, _ = pyshm.OSMOS.retrieve_data(options.hostname, options.port, options.pid, LIRIS)
            if len(Sdata) > 0:
                try:
                    os.makedirs(options.projdir)
                    if options.verbose > 0:
                        print("Create the project folder {}".format(options.projdir))
                except Exception:
                    pass

                # save data in an Excel file
                writer = pd.ExcelWriter(options.infile_data)
                for loc, data in Sdata.items():
                    # print(loc)
                    # print(data.head())
                    data.to_excel(writer, str(loc))
                writer.save()
                if options.verbose:
                    print("Data saved in {}".format(options.infile_data))
            else:
                raise Exception("No data found in MongoDB.")

    Locations = list(Sdata.keys())  # All locations with data
    Locations.sort()

    if options.verbose:
        print("ID of available locations: {}".format(Locations))

    if options.plot:
        if options.verbose:
            print('Generating plots...')
        for loc, X in Sdata.items():
            # print(fname1, loc)
            plot_static(X, os.path.join(options.projdir, str(loc)))
        if options.verbose:
            print("Plots saved in {}".format(options.projdir))
        sys.exit(0)

    ##### Step 2. Preparation of data for analysis #####
    if options.alocs is None:  # sensors to be analyzed
        options.alocs = Locations.copy()  # if not given use all sensors
    else:
        options.alocs.sort()  # sort in increasing order
    if options.verbose:
        print('Active locations: {}'.format(options.alocs))

    # Concatenate data of different sensors
    # non-transformed (raw) data of observation
    Tobs_raw0 = pyshm.OSMOS.concat_mts(Sdata, 'TemperatureRaw')
    Eobs_raw0 = pyshm.OSMOS.concat_mts(Sdata, 'ElongationRaw')
    Ref0 = pyshm.OSMOS.concat_mts(Sdata, 'Reference')
    Midx0 = pyshm.OSMOS.concat_mts(Sdata, 'Missing')
    # transformed data of observation
    # toto = pyshm.OSMOS.raw2celsuis(np.asarray(Tobs_raw0).T)
    # Tobs_tfm0 = pd.DataFrame(toto.T, columns=Tobs_raw0.columns, index=Tobs_raw0.index)
    Tobs_tfm0 = pyshm.OSMOS.raw2celsuis(Tobs_raw0)
    Parms = {int(val['locationkeyid']):tuple(val[['parama', 'paramb', 'paramc']]) for n, val in LIRIS.iterrows() if val['locationkeyid'] in Locations}
    # print(Parms)
    Eobs_tfm0 = pyshm.OSMOS.raw2millimeters(Eobs_raw0/Ref0, Parms)
    # toto={loc:pyshm.OSMOS.raw2mm(Parms[loc], np.asarray(Eobs_raw0[loc]/Ref0[loc])) for loc in options.alocs}
    # Eobs_tfm0 = pd.DataFrame(toto, index=Eobs_raw0.index)

    # Truncation of data
    #
    Tobs_raw = Tobs_raw0[options.alocs][options.time0:options.time1]
    Eobs_raw = Eobs_raw0[options.alocs][options.time0:options.time1]
    Tobs_tfm = Tobs_tfm0[options.alocs][options.time0:options.time1]
    Eobs_tfm = Eobs_tfm0[options.alocs][options.time0:options.time1]
    Ref = Ref0[options.alocs][options.time0:options.time1]
    Midx = Midx0[options.alocs][options.time0:options.time1]
    Time_idx = Tobs_raw.index  # time indexes of truncated data

    if options.component.upper() == 'RAW':  # work on raw data
        Tcpn_tfm = Tobs_tfm
        Ecpn_tfm = Eobs_tfm
        Xvar = np.asarray(Tobs_raw).T  # input variable
        Yvar = np.asarray(Eobs_raw).T  # output variable
    else:  # work on transformed data
        # Decomposition of data
        if options.component.upper() in ['SEASONAL', 'TREND']:
            Ttrd_tfm0, Tsnl_tfm0 = pyshm.OSMOS.trend_seasonal_decomp(Tobs_tfm0, mwsize=options.mwsize, method=options.mwmethod, kzord=options.kzord, causal=False, luwsize=0)
            Etrd_tfm0, Esnl_tfm0 = pyshm.OSMOS.trend_seasonal_decomp(Eobs_tfm0, mwsize=options.mwsize, method=options.mwmethod, kzord=options.kzord, causal=False, luwsize=0)
            Tsnl_tfm = Tsnl_tfm0[options.alocs][options.time0:options.time1]
            Esnl_tfm = Esnl_tfm0[options.alocs][options.time0:options.time1]
            Ttrd_tfm = Ttrd_tfm0[options.alocs][options.time0:options.time1]
            Etrd_tfm = Etrd_tfm0[options.alocs][options.time0:options.time1]

            if options.component.upper() == 'SEASONAL':
                # It seems useful to affirm whether a significant seaonal component is present (by testing the variance)
                ratio_snl = np.sqrt(np.diag(Esnl_tfm.cov()) / np.diag(Eobs_tfm.cov()))
                idx_snl = ratio_snl < options.snl_threshold
                if idx_snl.all():
                    raise Exception('No significant seasonal component detected!')
                Tcpn_tfm = Tsnl_tfm
                Ecpn_tfm = Esnl_tfm
            else:
                Tcpn_tfm = Ttrd_tfm
                Ecpn_tfm = Etrd_tfm
        elif options.component.upper() == 'ALL':
            Tcpn_tfm = Tobs_tfm
            Ecpn_tfm = Eobs_tfm
        else:
            raise TypeError('Unknown type of component:', options.component)
        Xvar = np.asarray(Tcpn_tfm).T
        Yvar = np.asarray(Ecpn_tfm).T

    # valid training period
    options.trnperiod, options.Ntrn = pyshm.Stat.training_period(len(Time_idx), tidx0=options.sidx, Ntrn=options.Ntrn)

    ##### Step 3. Deconvolution model #####
    if options.verbose:
        print("Applying the deconvolution model...")
        if options.verbose > 1:
            print("\tActive sensors: {}".format(options.alocs))
            print("\tLength of FIR kernel: {}".format(options.lag))
            print("\tDimension reduction threshold: {}".format(options.vthresh))
            print("\tOrder of polynomial process: {}".format(options.pord))
            print("\tOrder of derivative: {}".format(options.dord))
            print("\tTraining period: from {} to {}, about {} days.".format(Time_idx[options.trnperiod[0]], Time_idx[options.trnperiod[1]-1], int((options.trnperiod[1]-options.trnperiod[0])/24)))

    if options.subcommand.upper() == "LS":
        Yprd, Amat, Amatc = pyshm.Models.deconv(Yvar, Xvar, options.lag, dord=options.dord,
        pord=options.pord, snr2=options.snr2, clen2=options.clen2, dspl=options.dspl, sidx=options.sidx, Ntrn=options.Ntrn, vthresh=options.vthresh, Nexp=options.Nexp, pflag=options.pflag)
    elif options.subcommand.upper() == "BM":
        smoothflag = options.kalman.upper() == "SMOOTHER"
        # # full-vectorial version: time and space consuming
        # (Yprd0,Ycov), ((Amat,Acov), *_) = Models.deconv_bm(Yvar, Xvar, options.lag, dord=options.dord, pord=options.pord, sigmaq2=options.sigmaq2, sigmar2=options.sigmar2, x0=0., p0=1., smooth=smoothflag, sidx=options.sidx, Ntrn=options.Ntrn, vthresh=options.vthresh, corrflag=options.corrflag)
        # Yerr0 = Yvar - Yprd0  # error of prediction
        # Yprd = pd.DataFrame(Yprd0.T, columns=options.alocs, index=Tidx)
        # Yerr = pd.DataFrame(Yerr0.T, columns=options.alocs, index=Tidx)

        # semi-vectorial version: we proceed sensor-by-sensor
        Yprd0 = []  # final prediction from external inputs
        Amat0 = []  # convolution matrices, or the thermal law
        Acov0 = []  # covariance matrix of Amat
        Amatc0 = []  # reduced convolution matrices, or the thermal law
        Acovc0 = []  # covariance matrix of Amatc
        for n, aloc in enumerate(options.alocs):
            if options.verbose:
                print("\tProcessing the location {}...".format(aloc))
            yprd, (amat,acov), (amatc,acovc) = pyshm.Models.deconv_bm(Yvar[[n],:], Xvar, options.lag, dord=options.dord, pord=options.pord, sigmaq2=options.sigmaq2, sigmar2=options.sigmar2, x0=0., p0=1., smooth=smoothflag, sidx=options.sidx, Ntrn=options.Ntrn, vthresh=options.vthresh, rescale=options.rescale)
            # print(yprd.shape, ycov.shape, amat.shape, acov.shape)
            Yprd0.append(yprd)
            Amat0.append(np.squeeze(amat))
            Acov0.append(np.asarray([np.diag(a) for a in acov]))
            Amatc0.append(np.squeeze(amatc))
            Acovc0.append(np.asarray([np.diag(a) for a in acovc]))
        Yprd = np.squeeze(np.asarray(Yprd0))  # shape of Yprd: len(alocs)*Nt
        Amat = np.asarray(Amat0).transpose((1,0,2))  # shape of Amat: Nt*len(alocs)*(len(alocs)*lag)
        Acov = np.asarray(Acov0).transpose((1,0,2))  # shape of Acov: Nt*len(alocs)*(len(alocs)*lag)
        Amatc = np.asarray(Amatc0).transpose((1,0,2))  # shape of Amatc: Nt*len(alocs)*(len(alocs)*lag)
        Acovc = np.asarray(Acovc0).transpose((1,0,2))  # shape of Acovc: Nt*len(alocs)*(len(alocs)*lag)
        # print(Yprd.shape, Amat.shape, Acov.shape, Amatc.shape, Acovc.shape)
    else:
        raise NotImplementedError()

    # Post-processing on the residual
    if options.component.upper() == 'RAW':
        # raw value
        Eprd_raw = pd.DataFrame(Yprd.T, columns=options.alocs, index=Time_idx)
        # transformed value
        Eprd_tfm = pyshm.OSMOS.raw2millimeters(Eprd_raw/Ref, Parms)
    else:
        Eprd_tfm = pd.DataFrame(Yprd.T, columns=options.alocs, index=Time_idx)
    Eerr_tfm = Ecpn_tfm - Eprd_tfm  # residual


    ##### Step 4. Clustering of sensors and low rank approximation #####
    #
    if not options.component.upper() == 'SEASONAL':
        # projection of the residual onto a low dimension subspace
        X0 = np.asarray(Eerr_tfm).T
        toto = pyshm.Models.ssproj(X0, cdim=options.cdim, vthresh=options.cthresh, corrflag=False, dflag=False)
        U, S = toto[1][0], toto[1][1]  # basis and sv
        cdim = toto[2]  # dimension of the subspace
        Scof = (U @ np.diag(np.sqrt(S/S[0])))  # Scof[:,:3] are the first 3 PCA coefficients, division by S[0]: normalization
        Scof_pd = pd.DataFrame(Scof.T, columns=options.alocs)

        # Individual behavior: projection onto virtual sensors
        Eerr_prj = pd.DataFrame(toto[0].T, columns=options.alocs, index=Time_idx)
        # Individual behavior: residual of projection
        Eerr_res = Eerr_tfm - Eerr_prj
        # Behavior of virtual sensors
        Px = np.asarray([np.mean(U[:,[n]] @ U[:,[n]].T @ X0, axis=0) for n in range(cdim)])
        Virt = pd.DataFrame(Px.T, index=Time_idx)  # virtual sensors
        # Virt_sm = Virt.rolling(24, center=True).mean()  # smoothed virtual sensors

        # # Note: the projection method is just the svd of the covariance method:
        # Vmat = Eerr_tfm.cov()
        # U, S, _ = svd(Vmat)
        # Scof = (U @ np.sqrt(np.diag(S)))[:,:2]

        # the number of cluster is set to the dimension of the subspace preserving (1-cthresh) of information
        # nbClusters = max(np.sum(np.cumsum(S)/np.sum(S) <= 0.999), 1)
        nbClusters = cdim
        y_pred = KMeans(n_clusters=nbClusters).fit(Scof[:,:])  # use the full matrix of coefficient for clustering
        cluster_locs = []
        for n in range(nbClusters):
            G = np.where(y_pred.labels_==n)[0]
            cluster_locs.append([options.alocs[g] for g in G])

        if options.verbose:
            # print(__separator__)
            print('Dimension of low rank approximation (number of virtual sensors):', cdim)
            print("Clustering of sensors: ")
            for n, g in enumerate(cluster_locs):
                print("\tGroup {}: {}".format(n+1, g))
    else:
        Eerr_prj = None
        Eerr_res = None
        Virt = None
        cluster_locs = None
        Scof_pd = None


    ##### Step 5. Alarms #####
    if options.verbose:
        print("Generating alarms...")

    ### 5.1 Alarms of transient events (not for trend component) ###
    # A transient event happens at the moment when locally the residual
    # does not behave as a stationary gaussian process. This can be detected
    # by normalization + thresholding.
    if not options.component.upper() == 'TREND':
        # In order to make the procedure robust, we apply the multi-windows smoothing technique
        Nerr = pd.DataFrame(0, columns=options.alocs, index=Time_idx)  # normalized prediction error
        for wsize in range(*options.mwsize_tran_rng):
            _, _, nerr = compute_local_statistics(Eerr_tfm, False, wsize, win_type='triang')
            Nerr += (nerr >= options.thresh_tran_lowlevel)
            # for loc, nerr in Nerr.items():
            #     Adic[loc] += np.asarray(nerr >= options.thresh_tran, dtype=int)
        Nerr /= (options.mwsize_tran_rng[1]-options.mwsize_tran_rng[0])  # rate of hit

        # smoothing one more time to regrouping events separated by a gap of at most max_thermal_delay hours
        # note that we again applying the smoothing window of different size here to make the result robust
        # The result should be better that a simple smoothing like:
        # Alarm_tran = Alarm_tran0.rolling(window=12, min_periods=1, center=False, win_type=win_type).mean()
        #
        Atran = Nerr.copy()  # amplitude of alarms, special case of moving window size = 1
        for wsize in range(2, options.max_thermal_delay):
            Atran += Nerr.rolling(window=wsize, center=True, win_type='triang').mean()
        Atran /= options.max_thermal_delay

        Aidx = Atran > options.thresh_tran_alarm # this gives the alarms on the transient events
        Alarm_tran = {str(loc): [str(s) for s in aidx.index[aidx].tolist()] for loc, aidx in Aidx.items()}

        if options.verbose > 1:
            # print(__seperator__)
            print("Alarms of transient events :")
            for loc, val in Alarm_tran.items():
                print("\t{}: {}".format(loc, val))
                # print(Alarm_tran[loc].index[Alarm_tran[loc]])
    else:
        Atran = None
        Alarm_tran = None

    if options.component.upper() in ['RAW', 'ALL', 'TREND']:
        ### 5.2 Alarms on standard deviation (not for seasonal component) ###
        # toto = pd.DataFrame(0, columns=options.alocs, index=Time_idx)
        # for p in range(*vol_dp_rng):
        #     Atmp += np.log10(Eerr_tfm).diff(p)/p
        # toto /= (vol_dp_rng[1] - vol_dp_rng[0])

        # apply the same multi-windows smoothing technique
        Astd = pd.DataFrame(0, columns=options.alocs, index=Time_idx)
        for wsize in range(options.mwsize_std_rng[0], options.mwsize_std_rng[1]):
            _, serr, _ = compute_local_statistics(Eerr_tfm, False, wsize)
            Astd += serr
        Astd /= (options.mwsize_std_rng[1] - options.mwsize_std_rng[0])

        # detection of periods
        Alarm_std = {}
        for loc, val in Astd.items():
            toto = pyshm.Stat.detect_periods_of_instability(val, options.thresh_std_alarm, hgap=options.gap, mask=Midx[loc])
            Alarm_std[str(loc)] = [[str(Time_idx[b[0]]), str(Time_idx[b[1]-1])] for b in toto]

        if options.verbose > 1:
            # print(__seperator__)
            print("Alarms of std :")
            for loc, val in Alarm_std.items():
                print("\t{}: {}".format(loc, val))

        ### 5.3 Alarms on persistence ###
        Hexp0 = {}  # Hurst exponent
        # Bexp0 = {}  # Bias
        # Vexp0 = {}  # Variance of estimation

        # # parallel version:
        # from joblib import Parallel, delayed
        # htoto = Parallel(n_jobs=4)(delayed(Hurstfunc)(loc, mYerp[loc].diff(24*2), options.hwsize, options.hrng) for loc in alocs)
        # htoto = Parallel(n_jobs=4)(delayed(Hurstfunc)(loc, Eerr_tfm[loc], options.hwsize, options.hrng) for loc in options.alocs)
        # for h in htoto:
        #     Hexp0.update(h)
        #
        # sequential version:
        for loc in options.alocs:
            Hexp0[loc], _, _ = pyshm.Stat.Hurst(np.asarray(Eerr_tfm[loc]), options.hurst_mwsize, sclrng=options.hurst_sclrng, wvlname="haar")
            # Hexp0[loc], Bexp0[loc], Vexp0[loc] = pyshm.Stat.Hurst(np.asarray(Eerr_tfm[loc]), options.hurst_mwsize, sclrng=options.hurst_sclrng, wvlname="haar")
        Apers = pd.DataFrame(Hexp0, index=Time_idx)
        # Bexp = pd.DataFrame(Bexp0, index=Time_idx)
        # Vexp = pd.DataFrame(Vexp0, index=Time_idx)

        # detection of periods
        Alarm_pers = {}
        for loc, val in Apers.items():
            toto = pyshm.Stat.detect_periods_of_instability(val, options.thresh_pers_alarm, hgap=options.gap, mask=Midx[loc])
            Alarm_pers[str(loc)] = [[str(Time_idx[b[0]]), str(Time_idx[b[1]-1])] for b in toto]

        if options.verbose > 1:
            # print(__seperator__)
            print("Alarms of persistence :")
            for loc, val in Alarm_pers.items():
                print("\t{}: {}".format(loc, val))
    else:
        Astd = None
        Alarm_std = None
        Apers = None
        Alarm_pers = None

    ##### Step 6: save results #####
    ### 6.1 The following variables will be saved in an Excel file ###
    # Tcpn_tfm : component of transformed temperature
    # Ecpn_tfm : component of transformed elongation
    # # Midx : indicator of missing values
    # Eprd_tfm : prediction of temperature
    # # Eerr_tfm : error of prediction
    # Virt : virtual sensors
    # Eerr_prj : projection of Eerr_tfm onto Virt
    # Scof_pd : coefficients of projection
    # Atran : amplitude of rate of transient events
    # Astd : amplitude of std
    # Apers : amplitude of persistence

    try:
        os.makedirs(options.outdir)
        if options.verbose > 0:
            print("Create the output folder:\n\t{}".format(options.outdir))
    except Exception:
        pass

    options.outfile_results = os.path.join(options.outdir, "results.xlsx")
    writer = pd.ExcelWriter(options.outfile_results)

    Tcpn_tfm.to_excel(writer, sheet_name="Temperature")
    Ecpn_tfm.to_excel(writer, sheet_name="Elongation")
    Eprd_tfm.to_excel(writer, sheet_name="Prediction")
    if options.subcommand.upper() == 'LS':
        Amat_pd = pd.DataFrame(Amat.T, columns=options.alocs)
        Amat_pd.to_excel(writer, sheet_name="Flattened kernel")
        Amatc_pd = pd.DataFrame(Amatc.T, columns=options.alocs)
        Amatc_pd.to_excel(writer, sheet_name="Flattened reduced kernel")
    elif options.subcommand.upper() == 'BM':
        Amat_mean = np.mean(Amat, axis=-1)
        Amat_pd = pd.DataFrame(Amat_mean, columns=options.alocs, index=Time_idx)
        Amat_pd.to_excel(writer, sheet_name="Mean kernel")
        Acov_mean = np.mean(Acov, axis=-1)
        Acov_pd = pd.DataFrame(Acov_mean, columns=options.alocs, index=Time_idx)
        Acov_pd.to_excel(writer, sheet_name="Mean var. of kernel")
        Amatc_mean = np.mean(Amatc, axis=-1)
        Amatc_pd = pd.DataFrame(Amatc_mean, columns=options.alocs, index=Time_idx)
        Amatc_pd.to_excel(writer, sheet_name="Mean reduced kernel")
        Acovc_mean = np.mean(Acovc, axis=-1)
        Acovc_pd = pd.DataFrame(Acovc_mean, columns=options.alocs, index=Time_idx)
        Acovc_pd.to_excel(writer, sheet_name="Mean var. of reduced kernel")

    if Virt is not None:
        Virt.to_excel(writer, sheet_name="Virtual sensors")
        Eerr_prj.to_excel(writer, sheet_name="Subspace projection")
    if Scof_pd is not None:
        Scof_pd.to_excel(writer, sheet_name="PCA coefficients")
    if Atran is not None:
        Atran.to_excel(writer, sheet_name="Transient")
    if Astd is not None:
        Astd.to_excel(writer, sheet_name="Std")
    if Apers is not None:
        Apers.to_excel(writer, sheet_name="Persistence")
    writer.save()

    ### 6.2 The following variables will be saved in a json file ###
    # cluster_locs : clustering of sensors
    # Alarm_tran : alarms of transient events
    # Alarm_std : alarms of std
    # Alarm_pers : alarms of persistence
    #
    # options will be saved also in a json file

    options.outfile_alarms = os.path.join(options.outdir, 'alarms.json')
    # alarms = {"Alarm transient":key2str(Alarm_tran), "Alarm std":key2str(Alarm_std), "Alarm persistence":key2str(Alarm_pers), "Cluster":cluster_locs}
    alarms_dic = {"Alarm transient":Alarm_tran, "Alarm std":Alarm_std, "Alarm persistence":Alarm_pers, "Cluster":cluster_locs}
    with open(options.outfile_alarms, 'w') as fp:
        # json.dump(resjson, fp, cls=MyEncoder)
        # json.dump(alarms, fp, cls=MyEncoder)
        json.dump(alarms_dic, fp)

    options.outfile_options = os.path.join(options.outdir, 'options.json')
    with open(options.outfile_options, 'w') as fp:
        json.dump(vars(options), fp)

    if options.verbose:
        print("Results saved in\n{}".format(options.outfile_results))
        print('Alarms saved in\n{}'.format(options.outfile_alarms))
        print('Options saved in\n{}'.format(options.outfile_options))

if __name__ == "__main__":
    main()
