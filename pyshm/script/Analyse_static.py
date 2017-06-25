#!/usr/bin/env python

"""Analysis of the results of deconvolution.
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
from joblib import Parallel, delayed

# import pyshm
from .. import OSMOS, Tools, Stat, Models
# from pyshm.script import static_data_analysis_template, examplestyle, warningstyle, load_result_of_analysis
from . import MyEncoder, load_results, compute_local_statistics, split_by_day, Hurstfunc, Options
# from pyshm.script import examplestyle, warningstyle


__script__ = __doc__


def main():
    usage_msg = "%(prog)s <subcommand> <infile> [options]"
    # parser = argparse.ArgumentParser(description=__script__, usage=usage_msg)
    parser = argparse.ArgumentParser(description=__script__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter,)
                                    #  epilog="\n\n" + __example__)
                                    #  epilog=__warning__ + "\n\n" + __example__)

    parser.add_argument('infile', type=str, help='Excel file containing the results of deconvolution.')

    parser.add_argument('-v', '--verbose', dest='verbose', action='count', default=0, help='print messages.')
    # parser.add_argument('-o', '--overwrite', dest='overwrite', action='store_true', default=False, help='overwrite on the input file.')

    mongo_opts = parser.add_argument_group("MongoDB options")
    mongo_opts.add_argument("--hostname", dest="hostname", type=str, default="localhost", help="name of the MongoDB server (default=localhost or '127.0.0.1').", metavar="string")
    mongo_opts.add_argument("--port", dest="port", type=int, default=27017, help="port of the MongoDB server (default=27017).", metavar="integer")

    cluster_opts = parser.add_argument_group("Clustering and low rank approximation options")
    cluster_opts.add_argument("--cthresh", dest="cthresh", type=float, default=1e-2, help="percentage of tolerable information loss in clustering (default=1e-2).", metavar="float")
    cluster_opts.add_argument("--cdim", dest="cdim", type=int, default=None, help="reduced dimension in low rank approximation. If not set it will be determined automatically from the value of cthresh, otherwise cthresh will be ignored (default=None).", metavar="integer")

    alarm_opts = parser.add_argument_group("Alarm options")
    # alarm_opts.add_argument("--mwmethod", dest="mwmethod", type=str, default="mean", help="type of moving window estimator for decomposition of component: mean (default), median.", metavar="string")
    # alarm_opts.add_argument("--mwsize", dest="mwsize", type=int, default=24, help="length of the moving window (default=24).", metavar="integer")
    alarm_opts.add_argument("--proj", dest="proj", action="store_true", help="compute the alarms on the residual of projection in place of the residual of prediction.")
    alarm_opts.add_argument("--tran", dest="thresh_tran_alarm", type=float, default=0.6, help="threshold of transient events (default=0.6).", metavar="float")
    alarm_opts.add_argument("--pers", dest="thresh_pers_alarm", type=float, default=0.6, help="threshold of persistence or Hurst exponent (default=0.6).", metavar="float")
    alarm_opts.add_argument('--std', dest='thresh_std_alarm', type=float, default=0.015, help='threshold of standard deviation (default=0.015).', metavar='float')
    # alarm_opts.add_argument('--gap', dest='gap', type=int, default=24*5, help='minimal length of instability period (default=24*5).', metavar='integer')

    output_opts = parser.add_argument_group("Output options")
    output_opts.add_argument("--noexcel", dest="toexcel", action="store_false", default=True, help="do not export in Excel format (default=export in Excel format).")
    output_opts.add_argument("--nomongo", dest="tomongo", action="store_false", default=True, help="do not export to MongoDB (default=export to MongoDB).")

    options = parser.parse_args()

    # the following parameters are hard-coded:
    options.dbname = 'OSMOS'  # name of database
    options.clname = 'Liris_Measure'  # name of collection
    options.snl_threshold = 0.1  # threshold for the energy of the seasonal component
    options.thresh_tran_lowlevel = 2.  # threshold for transient alarms
    options.mwsize_tran_rng = (24, 240)  # range of size of moving window for alarms of transient events
    options.max_thermal_delay = 6  # maximum value of thermal delay, >= 1
    options.mwsize_std_rng = (24*5, 24*10)   # range of size of moving window for alarms of std
    options.hurst_sclrng = (0, 8)  # scale range of wavelet coefficient for alarms of persistence or Hurst exponent
    options.hurst_mwsize = 24*10   # size of moving window for alarms of persistence or Hurst exponent
    options.gap = 24*5  # minimal length of alarms for std of persistence
    # vol_dp_rng = (1, 24)

    if not (options.toexcel or options.tomongo):
        raise ValueError("At least one format of export must be used: excel or MongoDB.")

    # Load input file
    if not os.path.isfile(options.infile):
        raise FileNotFoundError(options.infile)

    # options.figdir = options.infile[:options.infile.rfind(os.path.sep)]  # output directory for figures
    if options.verbose:
        print("Loading results...")

    results = pd.read_excel(options.infile, sheetname=None)
    # print(Results.keys())

    idx = options.infile.rfind(os.path.sep)
    options.indir = options.infile[:idx]
    options.outdir = os.path.join(options.infile[:idx], "cthresh[{}]_cdim[{}]_transient[{:.1e}]_std[{:.1e}]_pers[{:.1e}]".format(options.cthresh, options.cdim, options.thresh_tran_alarm, options.thresh_std_alarm, options.thresh_pers_alarm))

    try:
        os.makedirs(options.outdir)
    except Exception:
        pass

    # restore options of the deconvolution
    options.infile_options = os.path.join(options.indir, 'options.json')
    with open(options.infile_options, 'r') as fp:
        toto = json.load(fp)
    algo_options = Options(**toto)  # options used for the algorithm of deconvolution
    options.pid = algo_options.pid
    options.component = algo_options.component
    options.subcommand = algo_options.subcommand

    idx1 = [t for t,c in enumerate(options.indir) if c==os.path.sep]
    options.infile_info = os.path.join(options.indir[:idx1[-2]], 'liris_info.xlsx')
    LIRIS = pd.read_excel(options.infile_info)
    # LIRIS = LIRIS_info[LIRIS_info['pid']==options.pid].reset_index(drop=True)  # info of this pid
    UIDs = {int(u['locationkeyid']): u['uid'] for n,u in LIRIS.iterrows()}  # loc -> uid mapping
    # print(UIDs)

    Tcpn_tfm = load_results(results, 'Temperature')
    Ecpn_tfm = load_results(results, 'Elongation')
    Eprd_tfm = load_results(results, 'Prediction')
    # Midx = load_results(results, 'Missing')
    # print(Midx.head())
    Eerr_tfm = Ecpn_tfm - Eprd_tfm

    options.alocs = [int(loc) for loc in Tcpn_tfm.keys()]  # active locations
    Time_idx = Tcpn_tfm.index  # time index

    ##### Step 1. Clustering of sensors and low rank approximation #####
    # projection of the residual onto a low dimension subspace
    X0 = np.asarray(Eerr_tfm).T
    toto = Models.ssproj(X0, cdim=options.cdim, vthresh=options.cthresh, corrflag=False, dflag=False)
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
        print("Number of virtual sensors:", cdim)
        print("Clustering of sensors:")
        for n, g in enumerate(cluster_locs):
            print("\tGroup {}: {}".format(n+1, g))


    ##### Step 2. Alarms #####
    if options.verbose:
        print("Generating alarms...")

    Eerr_cpn = Eerr_prj if options.proj else Eerr_tfm

    ### 2.1 Alarms of transient events (not for trend component) ###
    # A transient event happens at the moment when locally the residual
    # does not behave as a stationary gaussian process. This can be detected
    # by normalization + thresholding.
    # In order to make the procedure robust, we apply the multi-windows smoothing technique
    Nerr = pd.DataFrame(0, columns=options.alocs, index=Time_idx)  # normalized prediction error
    for wsize in range(*options.mwsize_tran_rng):
        _, _, nerr = compute_local_statistics(Eerr_cpn, False, wsize, win_type='triang')
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

    ### 2.2 Alarms on standard deviation ###
    # toto = pd.DataFrame(0, columns=options.alocs, index=Time_idx)
    # for p in range(*vol_dp_rng):
    #     Atmp += np.log10(Eerr_cpn).diff(p)/p
    # toto /= (vol_dp_rng[1] - vol_dp_rng[0])

    # apply the same multi-windows smoothing technique
    Astd = pd.DataFrame(0, columns=options.alocs, index=Time_idx)
    for wsize in range(options.mwsize_std_rng[0], options.mwsize_std_rng[1]):
        _, serr, _ = compute_local_statistics(Eerr_cpn, False, wsize)
        Astd += serr
    Astd /= (options.mwsize_std_rng[1] - options.mwsize_std_rng[0])

    # detection of periods
    Alarm_std = {}
    for loc, val in Astd.items():
        toto = Stat.detect_periods_of_instability(val, options.thresh_std_alarm, hgap=options.gap)#, mask=Midx[loc])
        Alarm_std[str(loc)] = [[str(Time_idx[b[0]]), str(Time_idx[b[1]-1])] for b in toto]

    if options.verbose > 1:
        # print(__seperator__)
        print("Alarms of std :")
        for loc, val in Alarm_std.items():
            print("\t{}: {}".format(loc, val))

    ### 2.3 Alarms on persistence ###
    Hexp0 = {}  # Hurst exponent
    # Bexp0 = {}  # Bias
    # Vexp0 = {}  # Variance of estimation

    # parallel version:
    from joblib import Parallel, delayed
    htoto = Parallel(n_jobs=4)(delayed(Hurstfunc)(loc, Eerr_cpn[loc], options.hurst_mwsize, options.hurst_sclrng) for loc in options.alocs)
    for h in htoto:
        Hexp0.update(h)

    # # sequential version:
    # for loc in options.alocs:
    #     Hexp0[loc], _, _ = Stat.Hurst(np.asarray(Eerr_cpn[loc]), options.hurst_mwsize, sclrng=options.hurst_sclrng, wvlname="haar")
    #     # Hexp0[loc], Bexp0[loc], Vexp0[loc] = Stat.Hurst(np.asarray(Eerr_cpn[loc]), options.hurst_mwsize, sclrng=options.hurst_sclrng, wvlname="haar")

    Apers = pd.DataFrame(Hexp0, index=Time_idx)
    # Bexp = pd.DataFrame(Bexp0, index=Time_idx)
    # Vexp = pd.DataFrame(Vexp0, index=Time_idx)

    # detection of periods
    Alarm_pers = {}
    for loc, val in Apers.items():
        toto = Stat.detect_periods_of_instability(val, options.thresh_pers_alarm, hgap=options.gap) #, mask=Midx[loc])
        Alarm_pers[str(loc)] = [[str(Time_idx[b[0]]), str(Time_idx[b[1]-1])] for b in toto]

    if options.verbose > 1:
        # print(__seperator__)
        print("Alarms of persistence :")
        for loc, val in Alarm_pers.items():
            print("\t{}: {}".format(loc, val))

    ##### Step 3: save results #####

    ### 3.1 Alarms ###
    # Alarm_tran : alarms of transient events
    # Alarm_std : alarms of std
    # Alarm_pers : alarms of persistence

    options.outfile_alarms = os.path.join(options.outdir, 'alarms.json')
    # alarms = {"Alarm transient":key2str(Alarm_tran), "Alarm std":key2str(Alarm_std), "Alarm persistence":key2str(Alarm_pers), "Cluster":cluster_locs}
    alarms_dic = {"Alarm transient":Alarm_tran, "Alarm std":Alarm_std, "Alarm persistence":Alarm_pers}
    with open(options.outfile_alarms, 'w') as fp:
        json.dump(alarms_dic, fp, cls=MyEncoder)

    if options.verbose:
        print('Alarms saved in\n{}'.format(options.outfile_alarms))

    ### 3.2 Clustering ###
    options.outfile_cluster = os.path.join(options.outdir, 'clustering.json')
    cluster_dic = {"Cluster":cluster_locs}
    with open(options.outfile_cluster, 'w') as fp:
        json.dump(cluster_dic, fp, cls=MyEncoder)

    if options.verbose:
        print('Clustering saved in\n{}'.format(options.outfile_cluster))

    ### 3.3 The following variables will be exported to MongoDB ###
    if options.tomongo:  # export to MongoDB
        if options.verbose:
            print('Exporting results in MongoDB...')

        client = MongoClient(options.hostname, options.port)
        # collection = client['OSMOS']['Liris_Measure']  # collection
        db = client[options.dbname]

        # overwrite: first remove existing results
        clname_modified = options.clname + '_Sivienn_Modified'
        clname_virtual = options.clname + '_Sivienn_Virtual'
        clname_coeffs = options.clname + '_Sivienn_Coeffs'

        for collection in [db[clname_modified], db[clname_virtual], db[clname_coeffs]]:
            collection.delete_many({'pid': str(options.pid), 'component': options.component.upper(), 'model':options.subcommand.upper()})

        # per sensor results
        collection = db[clname_modified]
        # Xdic = {}
        for loc in options.alocs:
            toto0 = {'temperature': np.asarray(Tcpn_tfm[loc]),
                'measure': np.asarray(Ecpn_tfm[loc]),
                'prediction': np.asarray(Eprd_tfm[loc]),
                'error': np.asarray(Ecpn_tfm[loc] - Eprd_tfm[loc]),
                'ssproj': np.asarray(Eerr_prj[loc]),
                'transient': None if Atran is None else np.asarray(Atran[loc]),
                'std': None if Astd is None else np.asarray(Astd[loc]),
                'persistence': None if Apers is None else np.asarray(Apers[loc]),
                # 'date': [t.to_pydatetime() for t in Time_idx],  # <- this results a strange "NaTType does not support..." error
            }
            xdic = pd.DataFrame(toto0, index=Time_idx)

            # split the results
            # xdic_splitted = split_by_day(xdic)
            for x in split_by_day(xdic): # xdic_splitted:
                datalist = [{**u, 'date':t.to_pydatetime()} for t,u in x.iterrows()]
                t0 = x.index[0]
                collection.insert_one({'pid': str(options.pid),
                                        'model': options.subcommand.upper(),
                                        'component': options.component.upper(),
                                        'location': str(loc),
                                        # 'uid': UIDs[loc],
                                        'data': datalist,
                                        'start': t0.to_pydatetime(),
                                        'year': t0.year,
                                        'month': t0.month,
                                        'day':t0.day,
                                        })

        if Virt is not None:
            collection = db[clname_virtual]
            # vdim = Virt.shape[1]
            for x in split_by_day(Virt):
                t0 = x.index[0]
                # y = np.asarray(x)  # 2d array
                collection.insert_one({'pid': str(options.pid),
                                        'model': options.subcommand.upper(),
                                        'component': options.component.upper(),
                                        # 'vid': n,
                                        'data': [{'date':t.to_pydatetime(), 'measure':list(u)} for t,u in x.iterrows()],
                                        'start': t0.to_pydatetime(),
                                        'year': t0.year, 'month': t0.month, 'day':t0.day})
                # for n in range(vdim):
                #     collection.insert_one({'pid': str(options.pid),
                #                             'component': options.component.upper(),
                #                             'vid': n,
                #                             'data': [{'date':t.to_pydatetime(), 'measure':u[n]}, for t,u in x.iterrows()],
                #                             'start': t0.to_pydatetime(),
                #                             'year': t0.year, 'month': t0.month, 'day':t0.day})

        if Scof_pd is not None:
            collection = db[clname_coeffs]
            for loc in options.alocs:
                collection.insert_one({'pid': str(options.pid),
                                        'model': options.subcommand.upper(),
                                        'component': options.component.upper(),
                                        'location': str(loc),
                                        # 'uid': UIDs[loc],
                                        'data': list(Scof_pd[loc])})

    ### 3.4 The following variables will be saved in an Excel file ###
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

    if options.toexcel:
        options.outfile = os.path.join(options.outdir, 'results_analysis.xlsx')
        writer = pd.ExcelWriter(options.outfile)

        Tcpn_tfm.to_excel(writer, sheet_name="Temperature")
        Ecpn_tfm.to_excel(writer, sheet_name="Elongation")
        Eprd_tfm.to_excel(writer, sheet_name="Prediction")

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

        if options.verbose:
            print("Results saved in\n{}".format(options.outfile))


if __name__ == "__main__":
    main()


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

        # if Virt is not None:
        #     Resdic["Virtual sensors"] = Virt
        #     Resdic["Subspace projection"] = Eerr_prj
        # if Scof_pd is not None:
        #     Resdic["PCA coefficients"] = Scof_pd
        # if Atran is not None:
        #     Resdic["Transient"] = Atran
        # if Astd is not None:
        #     Resdic["Std"] = Astd
        # if Apers is not None:
        #     Resdic["Persistence"] = Apers

        # resjson = to_json(Resdic, verbose=options.verbose)
        # options.outfile_results = os.path.join(options.outdir, "results.json")
        # with open(options.outfile_results, 'w') as fp:
        #     json.dump(resjson, fp, cls=MyEncoder)

