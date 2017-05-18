#!/usr/bin/env python

"""Analysis of static data using the vectorial deconvolution model.
"""

import sys, os, argparse
from pyshm.script import static_data_analysis_template, examplestyle, warningstyle, load_result_of_analysis

# class Options:
#     verbose=False  # print message
#     info=False  # only print information about the project
#     alocs=None  # locations of active sensors
#     xlocs=None  # locations for x inputs
#     ylocs=None  # locations for y inputs
#     mwmethod='mean'  # method for computation of trend component
#     mwsize=24  # size of moving window for computation of trend component
#     kzord=2  # order of KZ filter
#     component='trend'  # name of the component for analysis, ['all', 'seasonal', 'trend']
#     time0=None  # beginning of the data set, a string
#     time1=None  # ending of the data set, a string
#     lagx=12  # kernel length of x inputs
#     lagy=6  # kernel length of y inputs
#     # dtx=0  # artificial delay in x inputs
#     # dty=0  # artificial delay in y inputs
#     const=False  # add constant trend in the model fitting step
#     subcommand='static'  # type of deconvolution model

#     # Static method only
#     Ntrn=3*30*24  # length of training data
#     sidx=0  # beginning index (relative to tidx0) of training data
#     # Dynamic method only
#     sigmaq2=1e-6
#     sigmar2=1e-4
#     kalman='smoother'


@static_data_analysis_template
def Deconv_static_data(Xcpn, Ycpn, options):
    """Wrapper function of _Deconv_static_data for decorator functional.

    Args:
        infile (str): name of pickle file containing the preprocessed static data
        outdir0 (str): name of the directory for the output
        options (Options): instance containing the fields of the class Options
    Return:
        same as _Deconv_static_data.
    """
    return _Deconv_static_data(Xcpn, Ycpn, options)


def _Deconv_static_data(Xcpn, Ycpn, options):
    """Deconvolution of static data.

    Args:
        Xcpn, Ycpn (pandas DataFrame): external input and observation
        options: object including all options, e.g., returned by parser.parse_args()
    Return:
        a dictionary containing the following fields:
        Yprd: final prediction from inputs
        Aprd: contribution of the first group of inputs
        Bprd: contribution of the second group of inputs, if exists
        Yerr: error of prediction
        Krnl: kernel matrices
        Mxd: objects of deconvolution model
    """

    from pyshm import Stat, Models
    import numpy as np
    import pandas as pd

    # Aprd0 = {}  # contribution of external inputs
    # Yerr0 = {}  # error of prediction
    # Krnl = {}  # kernel matrices
    # Mxd = {}  # objects of deconvolution model

    staticflag = options.subcommand.upper() == "LS"

    Tidx = Xcpn.index  # time index
    Nt = len(Tidx)  # size of measurement
    Locations = list(Xcpn.keys())  # all sensors

    if options.alocs is None:  # sensors to be analyzed
        options.alocs = Locations.copy()  # if not given use all sensors
    else:
        options.alocs.sort()  # sort in increasing order
    # data for analysis
    Xvar = np.asarray(Xcpn[options.alocs]).T
    Yvar = np.asarray(Ycpn[options.alocs]).T

    # compute valid values for training period
    options.trnperiod, options.Ntrn = Stat.training_period(Nt, tidx0=options.sidx, Ntrn=options.Ntrn)

    if options.verbose:
        print("Options:")
        print("\tActive sensors: {}".format(options.alocs))
        print("\tLength of FIR kernel: {}".format(options.lag))
        print("\tDimension reduction threshold: {}".format(options.vthresh))
        print("\tOrder of polynomial process: {}".format(options.pord))
        print("\tOrder of derivative: {}".format(options.dord))
        print("\tTraining period: from {} to {}, about {} days.".format(Tidx[options.trnperiod[0]], Tidx[options.trnperiod[1]-1], int((options.trnperiod[1]-options.trnperiod[0])/24)))

    if options.verbose:
        print("Deconvolution of the \'{}\' component...".format(options.component.upper()))

    # Proceed by sensor, although it is possible to use full vectorial version
    # for n, aloc in enumerate(options.alocs):
    #     if options.verbose:
    #         print("\tProcessing the location {}...".format(aloc))
    #     if staticflag:
    #         yprd, (amat, *_) = Models.deconv(Yvar[[n],:], Xvar, options.lag, dord=options.dord, pord=options.pord, snr2=options.snr2, clen2=options.clen2, dspl=options.dspl, sidx=options.sidx, Ntrn=options.Ntrn, vthresh=options.vthresh, corrflag=options.corrflag, Nexp=options.Nexp)
    #         Amat[aloc] = amat
    #     else:
    #         smoothflag = options.kalman.upper=="SMOOTHER"
    #         (yprd,ycov), ((amat, acov), *_) = Models.deconv_bm(Yvar[[n],:], Xvar, options.lag, dord=options.dord, pord=options.pord, sigmaq2=options.sigmaq2, sigmar2=options.sigmar2, x0=0., p0=1., smooth=smoothflag, sidx=options.sidx, Ntrn=options.Ntrn, vthresh=options.vthresh, corrflag=options.corrflag)
    #         Amat[aloc] = pd.DataFrame(np.squeeze(amat), columns=options.alocs, index=Tidx)
    #         Acov[aloc] = pd.DataFrame(np.asarray([np.diag(P) for P in acov]), columns=options.alocs, index=Tidx)
    #         # print(amat.shape, cvec.shape, err.shape, sig.shape)
    #         Yprd0.append(yprd[0])
    #         Ycov0.append(ycov[0])
    if staticflag:
        Yprd0, Amat, Amatc = Models.deconv(Yvar, Xvar, options.lag, dord=options.dord, pord=options.pord, snr2=options.snr2, clen2=options.clen2, dspl=options.dspl, sidx=options.sidx, Ntrn=options.Ntrn, vthresh=options.vthresh, Nexp=options.Nexp)

        Yprd = pd.DataFrame(Yprd0.T, columns=options.alocs, index=Tidx)
        Yerr0 = Yvar - Yprd0  # error of prediction
        Yerr = pd.DataFrame(Yerr0.T, columns=options.alocs, index=Tidx)
        Resdic = {"Yprd":Yprd, "Yerr":Yerr, "Amat":Amat, "Amatc":Amatc}
    else:
        smoothflag = options.kalman.upper() == "SMOOTHER"
        # # full-vectorial version: time and space consuming
        # (Yprd0,Ycov), ((Amat,Acov), *_) = Models.deconv_bm(Yvar, Xvar, options.lag, dord=options.dord, pord=options.pord, sigmaq2=options.sigmaq2, sigmar2=options.sigmar2, x0=0., p0=1., smooth=smoothflag, sidx=options.sidx, Ntrn=options.Ntrn, vthresh=options.vthresh, corrflag=options.corrflag)
        # Yerr0 = Yvar - Yprd0  # error of prediction
        # Yprd = pd.DataFrame(Yprd0.T, columns=options.alocs, index=Tidx)
        # Yerr = pd.DataFrame(Yerr0.T, columns=options.alocs, index=Tidx)

        # semi-vectorial version: we procede sensor-by-sensor
        Yprd0 = []  # final prediction from external inputs
        # Amat0 = []  # convolution matrices, or the thermal law
        Amatc0 = []  # reduced convolution matrices, or the thermal law
        Acovc0 = []  # covariance matrix of Amatc
        for n, aloc in enumerate(options.alocs):
            if options.verbose:
                print("\tProcessing the location {}...".format(aloc))
            yprd, (amat,acov), (amatc,acovc) = Models.deconv_bm(Yvar[[n],:], Xvar, options.lag, dord=options.dord, pord=options.pord, sigmaq2=options.sigmaq2, sigmar2=options.sigmar2, x0=0., p0=1., smooth=smoothflag, sidx=options.sidx, Ntrn=options.Ntrn, vthresh=options.vthresh, cdim=options.cdim)
            # print(yprd.shape, ycov.shape, amat.shape, acov.shape)
            Yprd0.append(yprd)
            # Amat0.append(np.squeeze(amat))
            Amatc0.append(np.squeeze(amatc))
            Acovc0.append(np.asarray([np.diag(a) for a in acovc]))
        Yprd = np.squeeze(np.asarray(Yprd0))  # shape of Yprd: len(alocs)*Nt
        # Amat = np.asarray(Amat0).transpose((1,0,2))  # shape of Amat: Nt*len(alocs)*(len(alocs)*lag)
        Amatc = np.asarray(Amatc0).transpose((1,0,2))  # shape of Amat: Nt*len(alocs)*(len(alocs)*lag)
        Acovc = np.asarray(Acovc0).transpose((1,0,2))  # shape of Acov: Nt*len(alocs)*(len(alocs)*lag)
        Yerr = Yvar - Yprd  # error of prediction
        # print(Yprd.shape, Ycov.shape, Amat.shape, Acov.shape)
        Resdic = {"Yprd":pd.DataFrame(Yprd.T, columns=options.alocs, index=Tidx),
                    "Yerr": pd.DataFrame(Yerr.T, columns=options.alocs, index=Tidx),
                    "Amatc":Amatc, "Acovc":Acovc}
    return Resdic


# __all__ = ["Deconv_static_data", "Options"]

__script__ = __doc__

# __warning__ = "Warning:" + warningstyle("\n This script can be applied only on data preprocessed by the script osmos_preprocessing (the data file is typically named Preprocessed_static.pkl). Two distinct models (static and dynamic) are implemented and are accessible via the corresponding subcommand.")

examples = []
examples.append(["%(prog)s -h", "Print this help messages (about common parameters)"])
examples.append(["%(prog)s ls -h", "Print help messages about the static model"])
examples.append(["%(prog)s bm -h", "Print help messages about the dynamic model"])

ls_examples = []
ls_examples.append(["%(prog)s DBDIR/153 OUTDIR/153 --alocs 754 --component trend --time0 2016-03-01 --time1 2016-08-01 -vv", "On the location 754 of the project of PID 153 (preprocessed data), apply the least-square model on the trend component for the period from 2016-03-01 to 2016-08-01 and save the results in the directory named OUTDIR/153. Print supplementary messages."])
ls_examples.append(["%(prog)s DBDIR/153 OUTDIR/153 --alocs 754,755 -v", "Process the locations 754 and 755, for each of them use the temperature of both to explain the elongation data (vectorial model)."])
ls_examples.append(["%(prog)s DBDIR/153 OUTDIR/153 -v", "Process all sensors, for each of them use the temperature of all to explain the elongation data."])
ls_examples.append(["%(prog)s DBDIR/153 OUTDIR/153 --time0 2016-03-01 --Ntrn 1000 -v", "Change the length of the training period to 1000 hours starting from the begining of the truncated data set which is 2016-03-01."])
ls_examples.append(["%(prog)s DBDIR/153 OUTDIR/153 --component=seasonal -v", "Process the seasonal component of data."])

bm_examples = []
bm_examples.append(["%(prog)s DBDIR/153 OUTDIR/153 -v", "Use the BM model to process all sensors."])

analysis_examples = []

__example__ = "Some examples of use (change the path seperator '/' to '\\' on Windows platform):" + "".join([examplestyle(x) for x in examples])
__ls_example__ = "Some examples of use (change the path seperator '/' to '\\' on Windows platform):" + "".join([examplestyle(x) for x in ls_examples])
__bm_example__ = "Some examples of use (change the path seperator '/' to '\\' on Windows platform):" + "".join([examplestyle(x) for x in bm_examples])
__analysis_example__ = "Some examples of use (change the path seperator '/' to '\\' on Windows platform):" + "".join([examplestyle(x) for x in analysis_examples])

def main():
    usage_msg = "%(prog)s <subcommand> <infile> <outdir> [options]"
    # parser = argparse.ArgumentParser(description=__script__, usage=usage_msg)
    mainparser = argparse.ArgumentParser(description=__script__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog="\n\n" + __example__)
                                    #  epilog=__warning__ + "\n\n" + __example__)


    subparsers = mainparser.add_subparsers(title="subcommands", description="Perform deconvolution or statistical analysis", dest="subcommand")
    parser_ls = subparsers.add_parser("ls", help="Deconvolution using the least-square model",
                                        formatter_class=argparse.RawDescriptionHelpFormatter,
                                        epilog=__ls_example__)
    parser_bm = subparsers.add_parser("bm", help="Deconvolution using the Brownian motion model",
                                        formatter_class=argparse.RawDescriptionHelpFormatter,
                                        epilog=__bm_example__)
    # parser_analysis = subparsers.add_parser("analysis", help="Statistical analysis of results of deconvolution",
    #                                        formatter_class=argparse.RawDescriptionHelpFormatter,
    #                                        epilog=__analysis_example__)

    for parser in [parser_ls, parser_bm]:
        parser.add_argument("projdir", help="directory of a project in the database including the preprocessed static data.")
        # parser.add_argument("infile", type=str, help="preprocessed data file containing all sensors of one project.")
        # parser.add_argument("outdir", nargs='?', type=str, default=None, help="directory where results (figures and data files) will be saved.")
        parser.add_argument("outdir", type=str, help="directory where results (figures and data files) will be saved.")

        parser.add_argument("-v", "--verbose", dest="verbose", action="count", default=0, help="print messages.")

        sensor_opts = parser.add_argument_group("Sensor options")
        sensor_opts.add_argument("--alocs", dest="alocs", type=lambda s: [int(x) for x in s.split(',')], default=None, help="location ID of sensors to be analyzed (default=all sensors).", metavar="integers separated by \',\'")
        sensor_opts.add_argument("--component", dest="component", type=str, default="trend", help="type of component of data to be analyzed: trend (default), seasonal, all.", metavar="string")

        wdata_opts = parser.add_argument_group("Data truncation options")
        wdata_opts.add_argument("--time0", dest="time0", type=str, default=None, help="starting timestamp (default=the begining of data set).", metavar="YYYY-MM-DD")
        wdata_opts.add_argument("--time1", dest="time1", type=str, default=None, help="ending timestamp (default=the ending of data set).", metavar="YYYY-MM-DD")

        ddata_opts = parser.add_argument_group("Component decomposition options")
        ddata_opts.add_argument("--mwmethod", dest="mwmethod", type=str, default="mean", help="type of moving window estimator for decomposition of component: mean (default), median.", metavar="string")
        ddata_opts.add_argument("--mwsize", dest="mwsize", type=int, default=24, help="length of the moving window (default=24).", metavar="integer")
        ddata_opts.add_argument("--kzord", dest="kzord", type=int, default=2, help="order of moving window filter (default=2).", metavar="integer")

        model_opts = parser.add_argument_group("Model options")
        model_opts.add_argument("--lag", dest="lag", type=int, default=24, help="length of the convolution kernel (default=24)", metavar="integer")
        model_opts.add_argument("--pord", dest="pord", type=int, default=None, help="order of non-thermal polynomial process (default=1 for trend or all component, 0 for seasonal component).", metavar="integer")
        model_opts.add_argument("--dord", dest="dord", type=int, default=None, help="order of derivative (default=1 for trend or all component, 0 for seasonal component).", metavar="integer")

        tdata_opts = parser.add_argument_group("Training data options")
        tdata_opts.add_argument("--sidx", dest="sidx", type=int, default=24*10, help="starting time index (an integer) of the training data relative to time0 (default=24*10).", metavar="integer")
        tdata_opts.add_argument("--Ntrn", dest="Ntrn", type=int, default=3*30*24, help="length of the training data (default=24*30*3).", metavar="integer")

        dimr_opts = parser.add_argument_group("Dimension reduction options")
        dimr_opts.add_argument("--vthresh", dest="vthresh", type=float, default=10**-2, help="relative threshold for dimension reduction (default=1e-3), no dimension reduction if set to 0.", metavar="float")
        dimr_opts.add_argument("--cdim", dest="cdim", type=int, default=None, help="reduced dimension, vthresh will be ignored if cdim is set to some positive integer (default=None).", metavar="integer")
        # dimr_opts.add_argument("--corrflag", dest="corrflag", action="store_true", default=False, help="use correlation matrix in dimension reduction.")

    # for parser in [parser_ls, parser_bm, parser_analysis]:
    #     parser.add_argument("-v", "--verbose", dest="verbose", action="count", default=0, help="Print messages.")

    regressor_opts = parser_ls.add_argument_group("Linear regression options (ls model only)")
    regressor_opts.add_argument("--Nexp", dest="Nexp", type=int, default=0, help="number of experiments in RANSAC (default=0, no RANSAC).", metavar="integer")
    regressor_opts.add_argument("--snr2", dest="snr2", type=float, default=10**4, help="squared signal-to-noise ratio of the Gaussian polynomial process (default=1e4), no effect if clen2 is not set.", metavar="float")
    regressor_opts.add_argument("--clen2", dest="clen2", type=float, default=None, help="squared correlation length of the Gaussian polynomial process (default=None, use deterministic polynomial process).", metavar="float")
    regressor_opts.add_argument("--dspl", dest="dspl", type=int, default=1, help="down-sampling rate of training data for acceleration on large training dataset (default=1, no down-sampling).", metavar="integer")

    kalman_opts = parser_bm.add_argument_group("Kalman filter options (bm model only)")
    kalman_opts.add_argument("--sigmaq2", dest="sigmaq2", type=float, default=10**-6, help="variance of transition noise (default=1e-6).", metavar="float")
    kalman_opts.add_argument("--sigmar2", dest="sigmar2", type=float, default=10**-6, help="variance of observation noise (default=1e-6).", metavar="float")
    kalman_opts.add_argument("--kalman", dest="kalman", type=str, default="smoother", help="method of estimation of Kalman filter: filter, smoother (default).", metavar="string")

    options = mainparser.parse_args()

    # set pord and dord automatically according to the component's type
    if options.component.upper() in ["TREND", "ALL"]:
        if options.pord is None:
            options.pord = 1
        if options.dord is None:
            options.dord = 1
    else:
        if options.pord is None:
            options.pord = 0
        if options.dord is None:
            options.dord = 0
        # options.vthresh = 0  # no dimension reduction for seasonal component

    if options.subcommand.upper() in ["LS", "BM"]:
        # check the input data file
        options.infile = os.path.join(options.projdir, "Preprocessed_static.pkl")
        if not os.path.isfile(options.infile):
            print("Preprocessed static data not found.")
            raise FileNotFoundError(options.infile)

        if not os.path.isdir(options.outdir):
            raise FileNotFoundError(options.outdir)
        # if options.outdir is None:
        #     idx2 = options.infile.rfind(os.path.sep, 0)
        #     idx1 = options.infile.rfind(os.path.sep, 0, idx2)
        #     idx0 = options.infile.rfind(os.path.sep, 0, idx1)
        #     outdir0 = os.path.join(options.infile[:idx0], "Outputs", options.infile[idx1+1:idx2])

        # create a subfolder for output
        options.func_name = __name__[__name__.rfind('.')+1:]
        outdir = os.path.join(options.outdir, options.func_name, "model[{}]_component[{}]_alocs[{}]_[from_{}_to_{}]".format(options.subcommand.upper(), options.component.upper(), options.alocs, options.time0, options.time1))
        if options.subcommand.upper() == "BM":
            outdir += "_sigmaq2[{:.1e}]_sigmar2[{:.1e}]".format(options.sigmaq2, options.sigmar2)
        outfile = os.path.join(outdir, "Results")

        # No handle of exceptions so that any exception will result a system exit in the terminal.
        _ = Deconv_static_data(options.infile, outfile, options)
    else:
        raise NotImplementedError("{}: subcommand not implemented".format(options.subcommand))

if __name__ == "__main__":
    main()
