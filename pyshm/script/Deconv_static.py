#!/usr/bin/env python

"""Deconvolution of static data using the vectorial linear model.
"""

import os, sys, argparse
import json
import numpy as np
import scipy
import pandas as pd
import warnings
from sklearn.cluster import KMeans #, AffinityPropagation
import pywt
from pymongo import MongoClient

import pyshm
# from pyshm import OSMOS, Tools, Stat, Models
from pyshm.script import MyEncoder #, to_json
# from pyshm.script import examplestyle, warningstyle


__script__ = __doc__

# __warning__ = "Warning:" + warningstyle("\n This script can be applied only on data preprocessed by the script osmos_preprocessing (the data file is typically named Preprocessed_static.pkl). Two distinct models (static and dynamic) are implemented and are accessible via the corresponding subcommand.")

# examples = []
# examples.append(["%(prog)s -h", "Print this help messages (about common parameters)"])
# examples.append(["%(prog)s ls -h", "Print help messages about the static model"])
# examples.append(["%(prog)s bm -h", "Print help messages about the dynamic model"])
# # __example__ = "Some examples of use (change the path seperator '/' to '\\' on Windows platform):" + "".join([examplestyle(x) for x in examples])
# __example__ = "Some examples of use (change the path seperator '/' to '\\' on Windows platform):" + "".join(examples)

# ls_examples = []
# ls_examples.append(["%(prog)s 153 DBDIR --alocs 754 --component trend --time0 2016-03-01 --time1 2016-08-01 -vv", "On the location 754 of the project of PID 153 (preprocessed data), apply the least-square model on the trend component for the period from 2016-03-01 to 2016-08-01 and save the results in the directory named OUTDIR/153. Print supplementary messages."])
# ls_examples.append(["%(prog)s 153 DBDIR --alocs 754,755 -v", "Process the locations 754 and 755, for each of them use the temperature of both to explain the elongation data (vectorial model)."])
# ls_examples.append(["%(prog)s 153 DBDIR -v", "Process all sensors, for each of them use the temperature of all to explain the elongation data."])
# ls_examples.append(["%(prog)s 153 DBDIR --time0 2016-03-01 --Ntrn 1000 -v", "Change the length of the training period to 1000 hours starting from the begining of the truncated data set which is 2016-03-01."])
# ls_examples.append(["%(prog)s 153 DBDIR --component=seasonal -v", "Process the seasonal component of data."])
# __ls_example__ = "Some examples of use (change the path seperator '/' to '\\' on Windows platform):" + "".join([examplestyle(x) for x in ls_examples])

# bm_examples = []
# bm_examples.append(["%(prog)s 153 DBDIR -v", "Use the BM model to process all sensors."])
# __bm_example__ = "Some examples of use (change the path seperator '/' to '\\' on Windows platform):" + "".join([examplestyle(x) for x in bm_examples])

def main():
    usage_msg = "%(prog)s <subcommand> <pid> <dbdir> [options]"
    # parser = argparse.ArgumentParser(description=__script__, usage=usage_msg)
    mainparser = argparse.ArgumentParser(description=__script__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter,)
                                    #  epilog="\n\n" + __example__)
                                    #  epilog=__warning__ + "\n\n" + __example__)

    subparsers = mainparser.add_subparsers(title="subcommands", # description="Perform deconvolution",
    dest="subcommand")
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
        # parser.add_argument("--excel", dest="excel", action="store_true", default=False, help="save results in excel format.")

        # parser.add_argument("--raw", dest="raw", action="store_true", default=False, help="use non-transformed raw data (default: use transformed data).")

        mongo_opts = parser.add_argument_group("MongoDB options")
        mongo_opts.add_argument("--hostname", dest="hostname", type=str, default="localhost", help="name of the MongoDB server (default=localhost or '127.0.0.1').", metavar="string")
        mongo_opts.add_argument("--port", dest="port", type=int, default=27017, help="port of the MongoDB server (default=27017).", metavar="integer")
        mongo_opts.add_argument("--update", dest="update", action="store_true", default=False, help="force updating local database.")
        mongo_opts.add_argument("--plot_only", dest="plot_only", action="store_true", default=False, help="plot original data and exit.")

        store_opts = parser.add_argument_group("Store options")
        store_opts.add_argument("--link", dest="link", type=str, default="https://client.osmos-group.com/server/application.php", help="store page link server (default='https://client.osmos-group.com/server/application.php').", metavar="string")
        store_opts.add_argument("--login", dest="login", type=str, default="be@osmos-group.com", help="login to Store (default='be@osmos-group.com').", metavar="string")
        store_opts.add_argument("--password", dest="password", type=str, default="osmos", help="password of the login (default='osmos').", metavar="string")

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
        model_opts.add_argument("--lag", dest="lag", type=int, default=24, help="length of the convolution kernel (default=24).", metavar="integer")
        model_opts.add_argument("--pord", dest="pord", type=int, default=1, help="order of non-thermal polynomial process (default=1 for trend or all component, 0 for raw or seasonal component).", metavar="integer")
        model_opts.add_argument("--dord", dest="dord", type=int, default=0, help="order of derivative (default=1 for trend or all component, 0 for raw or seasonal component).", metavar="integer")
        model_opts.add_argument("--poly", dest="poly", action="store_true", default=False, help="add polynomial trends in the prediction (default=no polynomial trend). This option should be set in case of raw component.")
        model_opts.add_argument("--vthresh", dest="vthresh", type=float, default=10**-3, help="percentage of tolerable information loss for dimension reduction. The principle dimensions corresponding to the percentage of (1-vthresh) will be kept, i.e. 1 percent of information is discarded if vthresh=0.01. No dimension reduction if set to 0 (default=0.001).", metavar="float")
        # dimr_opts.add_argument("--corrflag", dest="corrflag", action="store_true", default=False, help="use correlation matrix in dimension reduction.")

    # for parser in [parser_ls, parser_bm, parser_analysis]:
    #     parser.add_argument("-v", "--verbose", dest="verbose", action="count", default=0, help="Print messages.")

    regressor_opts = parser_ls.add_argument_group("Linear regression options (ls model only)")
    regressor_opts.add_argument("--Nexp", dest="Nexp", type=int, default=0, help="number of experiments in RANSAC (default=0, no RANSAC).", metavar="integer")
    regressor_opts.add_argument("--snr2", dest="snr2", type=float, default=10**4, help="squared signal-to-noise ratio of the Gaussian polynomial process (default=1e4), no effect if clen2 is not set.", metavar="float")
    regressor_opts.add_argument("--clen2", dest="clen2", type=float, default=None, help="squared correlation length of the Gaussian polynomial process (default=None, use deterministic polynomial process). Setting this parameter may slow down the training.", metavar="float")
    regressor_opts.add_argument("--dspl", dest="dspl", type=int, default=1, help="down-sampling rate of training data for acceleration on large training dataset (default=1, no down-sampling).", metavar="integer")
    # regressor_opts.add_argument("--vreg", dest="vreg", type=float, default=0, help="factor of regularization (default=0).", metavar="float")

    kalman_opts = parser_bm.add_argument_group("Kalman filter options (bm model only)")
    kalman_opts.add_argument("--sigmaq2", dest="sigmaq2", type=float, default=10**-6, help="variance of transition noise (default=1e-6).", metavar="float")
    kalman_opts.add_argument("--sigmar2", dest="sigmar2", type=float, default=1e+3, help="variance of observation noise (default=1e+3).", metavar="float")
    kalman_opts.add_argument("--kalman", dest="kalman", type=str, default="smoother", help="method of estimation of Kalman filter: filter, smoother (default).", metavar="string")
    kalman_opts.add_argument("--rescale", dest="rescale", action="store_true", default=False, help="rescale the input and output variables to normalize the amplitude.")

    options = mainparser.parse_args()

    # the following parameters are hard-coded:
    options.dbname = 'OSMOS'  # name of database
    options.clname = 'Liris_Measure'  # name of collection
    options.snl_threshold = 0.1  # threshold for the energy of the seasonal component

    ##### Step 1. Load data from local database or from MongoDB #####
    # check local database directory
    if not os.path.isdir(options.dbdir):
        raise FileNotFoundError("Local database directory not found: {}".format(options.dbdir))

    options.projdir = os.path.join(options.dbdir, str(options.pid))  # project directory
    options.infile_data = os.path.join(options.projdir, "preprocessed_static.xlsx")  # input data file of the project
    options.infile_info = os.path.join(options.projdir, "liris_info.xlsx")  # input LIRIS information file of the project

    options.func_name = __file__[__file__.rfind(os.path.sep)+1 : __file__.rfind('.')]
    outdir0 = os.path.join(options.projdir, options.func_name, "model[{}]_component[{}]_alocs[{}]_[from_{}_to_{}]_Ntrn[{}]_lag[{}]_pord[{}]_dord[{}]_poly[{}]_vthresh[{:.1e}]".format(options.subcommand.upper(), options.component.upper(), options.alocs, options.time0, options.time1, options.Ntrn, options.lag, options.pord, options.dord, options.poly, options.vthresh))

    if options.subcommand.upper() == 'LS':
        # options.outdir = outdir0 + "_clen2[{}]_snr2[{:.1e}]_dspl[{}]_Nexp[{}]_vreg[{}]".format(options.clen2, options.snr2, options.dspl, options.Nexp, options.vreg)  # output directory
        options.outdir = outdir0 + "_clen2[{}]_snr2[{:.1e}]_dspl[{}]_Nexp[{}]".format(options.clen2, options.snr2, options.dspl, options.Nexp)  # output directory
    else:
        options.outdir = outdir0 + "_sigmaq2[{:.1e}]_sigmar2[{:.1e}]_rescale[{}]".format(options.sigmaq2, options.sigmar2, options.rescale)

        # os.path.join(options.projdir, options.func_name, "model[{}]_component[{}]_alocs[{}]_[from_{}_to_{}]_Ntrn[{}]_lag[{}]_pord[{}]_dord[{}]_poly[{}]_vthresh[{}]_sigmaq2[{:.1e}]_sigmar2[{:.1e}]_rescale[{}]".format(options.subcommand.upper(), options.component.upper(), options.alocs, options.time0, options.time1, options.Ntrn, options.lag, options.pord, options.dord, options.poly, options.vthresh, options.sigmaq2, options.sigmar2, options.rescale))

    if os.path.isfile(options.infile_data) and os.path.isfile(options.infile_info) and not options.update:  # if both files exist, use local database
        if options.verbose:
            # print(__seperator__)
            print("Loading data from local database...")

        LIRIS_info = pd.read_excel(options.infile_info)  # info of all projects
        LIRIS = LIRIS_info[LIRIS_info['pid']==options.pid].reset_index(drop=True)  # info of this pid
        Sdata0 = pd.read_excel(options.infile_data, sheetname=None)  # <=== this may be time consuming
        Sdata = {}
        Parms = {}
        for loc, val in Sdata0.items():
            Sdata[int(loc)] = val
            Parms[int(loc)] = tuple(np.asarray(val[['parama', 'paramb', 'paramc']]).mean(axis=0))
        # Parms = {int(val['locationkeyid']):tuple(val[['parama', 'paramb', 'paramc']]) for n, val in LIRIS.iterrows() if val['locationkeyid'] in Locations}  # parameters of transformation of raw measurements

    else:  # otherwise update local data base
        # 1. Get LIRIS sensor information
        if not os.path.isfile(options.infile_info) or options.update:
            if options.verbose:
                # print(__seperator__)
                print("Retrieving information of LIRIS sensors of the project {}...".format(options.pid))

            LIRIS_info_full = pyshm.OSMOS.retrieve_LIRIS_info([options.pid], link=options.link, login=options.login, password=options.password)
            # LIRIS_info_full = pyshm.OSMOS.retrieve_LIRIS_info(list(range(1,500)))

            if LIRIS_info_full is None:
                raise ValueError("Failed to retrieve information of the project {} from server.".format(options.pid))
            else:
                try:
                    os.makedirs(options.projdir)
                    if options.verbose > 0:
                        print("Create the project folder {}".format(options.projdir))
                except Exception:
                    pass

            # save essential information in an Excel file
            LIRIS_info = LIRIS_info_full[['pid', 'uid', 'locationkeyid', 'parama', 'paramb', 'paramc']]
            # LIRIS_info = LIRIS_info_full[['pid', 'uid', 'locationkeyid']]
            writer = pd.ExcelWriter(options.infile_info)
            LIRIS_info.to_excel(writer)
            writer.save()

            if options.verbose:
                print("Information saved in {}".format(options.infile_info))
        else:
            LIRIS_info = pd.read_excel(options.infile_info)

        # 2. Get LIRIS data
        if not os.path.isfile(options.infile_data) or options.update:
            if options.verbose:
                print("Retrieving data of the project {} from MongoDB...".format(options.pid))

            LIRIS = LIRIS_info[LIRIS_info['pid']==options.pid].reset_index(drop=True)  # info of this pid
            Sdata, Parms = pyshm.OSMOS.retrieve_data(options.hostname, options.port, options.pid, [int(v) for v in LIRIS['locationkeyid']], options.dbname, options.clname)

            if len(Sdata) > 0:
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
                raise Exception("No data found in MongoDB for the project {}.".format(options.pid))

    Locations = list(Sdata.keys())  # All locations with data
    Locations.sort()
    UIDs = {int(u['locationkeyid']): u['uid'] for n,u in LIRIS.iterrows()}  # loc -> uid mapping

    if options.verbose:
        print("ID of available locations: {}".format(Locations))

    if options.plot_only:
        if options.verbose:
            print('Generating plots...')
        figdir = os.path.join(options.projdir, "original")
        try:
            os.makedirs(figdir)
        except Exception:
            pass

        for loc, X in Sdata.items():
            # print(fname1, loc)
            plot_static(X, os.path.join(figdir, str(loc)))
        if options.verbose:
            print("Plots saved in {}".format(figdir))
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
    Tobs_tfm0 = pyshm.OSMOS.concat_mts(Sdata, 'TemperatureTfm')
    Eobs_tfm0 = pyshm.OSMOS.concat_mts(Sdata, 'ElongationTfm')
    # # or compute the transformed data by the following
    # Tobs_tfm0 = pyshm.OSMOS.raw2celsuis(Tobs_raw0)
    # Eobs_tfm0 = pyshm.OSMOS.raw2millimeters(Eobs_raw0, Ref0, Parms)
    Ref0 = pyshm.OSMOS.concat_mts(Sdata, 'Reference')
    Midx0 = pyshm.OSMOS.concat_mts(Sdata, 'Missing')

    # Truncation of data
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
        # a first run to determinate the value of cdim
        yprd0, amat0, amatc0 = pyshm.Models.deconv(Yvar, Xvar, options.lag, dord=options.dord, pord=options.pord, snr2=options.snr2, clen2=options.clen2, dspl=options.dspl, sidx=options.sidx, Ntrn=options.Ntrn, vthresh=options.vthresh, Nexp=options.Nexp, poly=options.poly)

        # a second run for a range of cdim
        yprd_list = [yprd0]
        amat_list = [amat0]
        # amatc_list = [amatc0]
        for cdim, _ in zip(range(amatc0.shape[1], amat0.shape[1]), range(10)):
            # print("second run with cdim={}".format(cdim))
            yprd0, amat0, amatc0 = pyshm.Models.deconv(Yvar, Xvar, options.lag, dord=options.dord, pord=options.pord, snr2=options.snr2, clen2=options.clen2, dspl=options.dspl, sidx=options.sidx, Ntrn=options.Ntrn, vthresh=0, cdim=cdim, Nexp=options.Nexp, poly=options.poly)
            yprd_list.append(yprd0)
            amat_list.append(amat0)
        # take the mean to reduce the high freqency anomalies
        Yprd = np.asarray(yprd_list).mean(axis=0)
        Amat = np.asarray(amat_list).mean(axis=0)
        # Amatc = np.asarray(amatc_list).mean(axis=0)

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
        # Amatc0 = []  # reduced convolution matrices, or the thermal law
        # Acovc0 = []  # covariance matrix of Amatc

        for n, aloc in enumerate(options.alocs):
            if options.verbose:
                print("\tProcessing the location {}...".format(aloc))

            yprd0, (amat0,acov0), (amatc0,acovc0) = pyshm.Models.deconv_bm(Yvar[[n],:], Xvar, options.lag, dord=options.dord, pord=options.pord, sigmaq2=options.sigmaq2, sigmar2=options.sigmar2, x0=0., p0=1., smooth=smoothflag, sidx=options.sidx, Ntrn=options.Ntrn, vthresh=options.vthresh, rescale=options.rescale)

            Yprd0.append(np.asarray(yprd0))
            Amat0.append(np.asarray(amat0))
            Acov0.append(np.asarray([np.diag(a) for a in acov0]))
            # Amatc0.append(amatc)
            # Acovc0.append(np.asarray([np.diag(a) for a in acovc]))
        Yprd = np.squeeze(np.asarray(Yprd0))  # shape : len(alocs) x Nt
        Amat = np.asarray(Amat0)  # 4d, shape : len(alocs) x Nt x 1 x (lag*len(alocs)+1)
        Acov = np.asarray(Acov0)  # 3d, shape : len(alocs) x Nt x (len(alocs)*lag+1)
        # Amatc = np.asarray(Amatc0)  # 4d, shape : len(alocs) x Nt x ? x ?
        # Acovc = np.asarray(Acovc0)  # 3d, shape : len(alocs) x Nt x ?
        # print(Amat.shape, Acov.shape, Amatc.shape, Acovc.shape)
        # print(Amat.shape, Acov.shape)

    else:
        raise NotImplementedError()

    # Post-processing on the residual
    if options.component.upper() == 'RAW':
        # raw value
        Eprd_raw = pd.DataFrame(Yprd.T, columns=options.alocs, index=Time_idx)
        # transformed value
        # print(Parms)
        Eprd_tfm = pyshm.OSMOS.raw2millimeters(Eprd_raw, Ref, Parms)
    else:
        Eprd_tfm = pd.DataFrame(Yprd.T, columns=options.alocs, index=Time_idx)
    Eerr_tfm = Ecpn_tfm - Eprd_tfm  # residual


    ##### Step 4: save results in an Excel file #####
    # Tcpn_tfm : component of transformed temperature
    # Ecpn_tfm : component of transformed elongation
    # Eprd_tfm : prediction of temperature
    # # Eerr_tfm : error of prediction = Ecpn-Eprd

    if options.verbose:
        print('Exporting results...')

    try:
        os.makedirs(options.outdir)
        # if options.verbose > 0:
        #     print("Create the output folder:\n\t{}".format(options.outdir))
    except Exception:
        pass

    # options will be saved in a json file
    options.outfile_options = os.path.join(options.outdir, 'options.json')
    with open(options.outfile_options, 'w') as fp:
        json.dump(vars(options), fp, cls=MyEncoder)

    # results will be saved in an excel file
    options.outfile_results = os.path.join(options.outdir, "results_deconv.xlsx")
    writer = pd.ExcelWriter(options.outfile_results)

    Tcpn_tfm.to_excel(writer, sheet_name="Temperature")
    Ecpn_tfm.to_excel(writer, sheet_name="Elongation")
    Eprd_tfm.to_excel(writer, sheet_name="Prediction")
    # Midx.to_excel(writer, sheet_name="Missing")

    # if options.subcommand.upper() == 'LS':
    #     Amat_pd = pd.DataFrame(Amat.T, columns=options.alocs)
    #     Amat_pd.to_excel(writer, sheet_name="Flattened kernel")
    #     Amatc_pd = pd.DataFrame(Amatc.T, columns=options.alocs)
    #     Amatc_pd.to_excel(writer, sheet_name="Flattened reduced kernel")
    # elif options.subcommand.upper() == 'BM':
    #     Amat_mean = np.mean(Amat, axis=-1)
    #     Amat_pd = pd.DataFrame(Amat_mean, columns=options.alocs, index=Time_idx)
    #     Amat_pd.to_excel(writer, sheet_name="Mean kernel")
    #     Acov_mean = np.mean(Acov, axis=-1)
    #     Acov_pd = pd.DataFrame(Acov_mean, columns=options.alocs, index=Time_idx)
    #     Acov_pd.to_excel(writer, sheet_name="Mean var. of kernel")
    #     Amatc_mean = np.mean(Amatc, axis=-1)
    #     Amatc_pd = pd.DataFrame(Amatc_mean, columns=options.alocs, index=Time_idx)
    #     Amatc_pd.to_excel(writer, sheet_name="Mean reduced kernel")
    #     Acovc_mean = np.mean(Acovc, axis=-1)
    #     Acovc_pd = pd.DataFrame(Acovc_mean, columns=options.alocs, index=Time_idx)
    #     Acovc_pd.to_excel(writer, sheet_name="Mean var. of reduced kernel")

    writer.save()

    if options.verbose:
        print("Results saved in\n{}".format(options.outfile_results))

        # if options.subcommand.upper() == 'LS':
        #     toto = pd.DataFrame(Amat.T, columns=options.alocs)
        #     Resdic["Flattened kernel"] = toto
        #     toto = pd.DataFrame(Amatc.T, columns=options.alocs)
        #     Resdic["Flattened reduced kernel"] = toto
        # elif options.subcommand.upper() == 'BM':
        #     # Resdic["Mean kernel"] = pd.DataFrame(Amat.T, columns=options.alocs)
        #     # Resdic["Mean var. of kernel"] = pd.DataFrame(Amatc.T, columns=options.alocs)

        #     # Amat_mean = np.mean(Amat, axis=-1)
        #     # Amat_pd = pd.DataFrame(Amat_mean, columns=options.alocs, index=Time_idx)
        #     # Amat_pd.to_excel(writer, sheet_name="Mean kernel")
        #     # Acov_mean = np.mean(Acov, axis=-1)
        #     # Acov_pd = pd.DataFrame(Acov_mean, columns=options.alocs, index=Time_idx)
        #     # Acov_pd.to_excel(writer, sheet_name="Mean var. of kernel")
        #     # Amatc_mean = np.mean(Amatc, axis=-1)
        #     # Amatc_pd = pd.DataFrame(Amatc_mean, columns=options.alocs, index=Time_idx)
        #     # Amatc_pd.to_excel(writer, sheet_name="Mean reduced kernel")
        #     # Acovc_mean = np.mean(Acovc, axis=-1)
        #     # Acovc_pd = pd.DataFrame(Acovc_mean, columns=options.alocs, index=Time_idx)
        #     # Acovc_pd.to_excel(writer, sheet_name="Mean var. of reduced kernel")
        #     pass

        # resjson = to_json(Resdic, verbose=options.verbose)
        # options.outfile_results = os.path.join(options.outdir, "results.json")
        # with open(options.outfile_results, 'w') as fp:
        #     json.dump(resjson, fp, cls=MyEncoder)

if __name__ == "__main__":
    main()
