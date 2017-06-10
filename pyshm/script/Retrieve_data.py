#!/usr/bin/env python

"""Retrieve data from the local MongoDB database.

Note:
    The projects in the local database are organized by their PID (project key ID) of 3 digits. For example, the project 24 is saved in the folder '024'.
"""

import os, sys, argparse
from pyshm.script import examplestyle, warningstyle


def plot_static(Data, fname):
    import matplotlib
    matplotlib.use("qt5agg")
    import matplotlib.pyplot as plt
    # import matplotlib.colors as colors
    import mpld3
    plt.style.use('ggplot')

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
    axa.plot(Data['parama'], 'b', alpha=0.5, label='a')
    axa.plot(Data['paramb'], 'r', alpha=0.5, label='b')
    axa.plot(Data['paramc'], 'g', alpha=0.5, label='c')
    axa.legend(loc='upper left')
    axb = axa.twinx()
    axb.plot(Data['Reference'], 'c', alpha=0.7, label='Reference')
    axb.legend(loc='upper right')

    fig.savefig(fname+".pdf", bbox_inches='tight')
    plt.close(fig)

    return fig


def Retrieve_data(dbdir, options):
    """Retrieve data from the MongoDB server.

    Args:
        dbdir (string): the directory of the whole database
        options: object including all options, e.g., returned by parser.parse_args()
    """

    import requests, json, colorama, dateutil, pickle
    import pandas as pd
    from pymongo import MongoClient
    from pyshm import OSMOS

    # Get infos on the available projects from OSMOS's server
    if options.info:
        L = []
        for pid in range(500):
            P = OSMOS._retrieve_LIRIS_info(pid)
            if P is not None:
                L.append(P)
        LIRIS_info_full = pd.concat(L).reset_index(drop=True)
        del LIRIS_info_full['location']  # remove the field location which is redundant

        # # save in an excel file
        # writer = pd.ExcelWriter(os.path.join(dbdir, 'LIRIS_info.xlsx'))
        # Liris_info.to_excel(writer)
        # writer.save()

        # save essential information in another excel file
        LIRIS_info = LIRIS_info_full[['pid', 'uid', 'locationkeyid', 'parama', 'paramb', 'paramc']]
        writer = pd.ExcelWriter(os.path.join(dbdir, 'LIRIS_info.xlsx'))
        LIRIS_info.to_excel(writer)
        writer.save()
        # sys.exit(0)
    else:
        LIRIS_info = pd.read_excel(os.path.join(dbdir, 'LIRIS_info.xlsx'))

    # Retrieve data from local MongoDB
    client = MongoClient('localhost', options.port)
    collection = client['OSMOS']['Liris_Measure']  # collection
    # equivalently, we can use
    # db = client.get_database('OSMOS')
    # collection = db.get_collection('Liris_Measure')

    # # get the UID of all available sensors
    # ListUID = collection.distinct('uid')
    # if options.verbose:
    #     print('Total number of sensor UIDs:', len(ListUID))

    # Retrieve static data
    ListPID = []  # list of available PIDs
    for p in LIRIS_info['pid']:
        if p not in ListPID:
            ListPID.append(int(p))

    # if options.verbose > 1:
    #     print('Available project key IDs:', ListPID)

    if not options.pid in ListPID:
        raise ValueError('Cannot establish the list of LIRIS for the project {}.'.format(options.pid))

    if options.verbose:
        print('Retrieving and preprocessing data of PID {}...'.format(options.pid))

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

    Rdata = {}  # dictionary of raw data
    # Sdata = {}  # dictionary of preprocessed static data
    Tdata = {}  # dictionary of preprocessed transformed static data
    Ndata = {}  # dictionary of preprocessed non-transformed static data
    Parms = {}  # dictionary of parameters

        for n, u in liris.iterrows(): # iteration on all sensors of the project
            uid = u['uid']
            loc = int(u['locationkeyid'])

            Data = []
            try:
                Data = OSMOS.mongo_load_static(collection, uid, dflag=True)
            except Exception as msg:
                print(msg)
                continue

            if len(Data)>0:
                Rdata[loc] = Data
                # S, Rtsx, Ntsx = OSMOS.resampling_time_series(Data, m=options.nh)
                # Sdata[loc] = S.loc[Rtsx]

                # Transformed
                X = Data[['TemperatureTfm', 'ElongationTfm']].rename(columns={"TemperatureTfm": "Temperature", "ElongationTfm": "Elongation"})  # rename the fields
                S, Rtsx, _ = OSMOS.resampling_time_series(X)
                Tdata[loc] = S.loc[Rtsx]
                # Tdata[loc] = OSMOS.static_data_preprocessing(X,
                #                                             sflag=options.sflag,
                #                                             oflag=options.oflag,
                #                                             jflag=options.jflag,
                #                                             tflag=options.tflag,
                #                                             nh=options.nh)

                # Non-transformed
                # X = Data[['TemperatureRaw', 'ElongationRaw', 'Reference', 'parama', 'paramb', 'paramc']].rename(columns={"TemperatureRaw": "Temperature", "ElongationRaw": "Elongation"})  # rename the fields
                X = Data[['TemperatureRaw', 'ElongationRaw', 'Reference']].rename(columns={"TemperatureRaw": "Temperature", "ElongationRaw": "Elongation"})  # rename the fields
                Ndata[loc] = OSMOS.static_data_preprocessing(X,
                                                            sflag=options.sflag,
                                                            oflag=options.oflag,
                                                            jflag=options.jflag,
                                                            tflag=options.tflag,
                                                            nh=options.nh)
                Parms[loc] = (u['parama'], u['paramb'], u['paramc'])

        if len(Rdata) > 0:  #
            projdir = os.path.join(dbdir, '{:03}'.format(pid)) # output directory
            # outdir = '/Users/hanwang/Outputs/OSMOS/{:03}'.format(pid)
            try:
                os.makedirs(projdir)
            except OSError:
                pass

            from pyshm.script import to_json, MyEncoder
            # save raw data
            resjson = to_json(Rdata, verbose=options.verbose)
            fname = os.path.join(projdir,'Raw_static.json')
            with open(fname, 'w') as fp:
                json.dump(resjson, fp, cls=MyEncoder)
            if options.verbose:
                print('Results saved in {}'.format(fname))

            # # save resampled data
            # resjson = to_json(Sdata, verbose=options.verbose)
            # fname = os.path.join(projdir,'Resampled_static.json')
            # with open(fname, 'w') as fp:
            #     json.dump(resjson, fp, cls=MyEncoder)
            # if options.verbose:
            #     print('Results saved in {}'.format(fname))

            # save transformed data
            resjson = to_json(Tdata, verbose=options.verbose)
            fname = os.path.join(projdir,'Transformed_static.json')
            with open(fname, 'w') as fp:
                json.dump(resjson, fp, cls=MyEncoder)
            if options.verbose:
                print('Results saved in {}'.format(fname))

            # save non-transformed data
            # resjson = to_json({'Data':Ndata, 'Parms':Parms}, verbose=options.verbose)
            resjson = to_json(Ndata, verbose=options.verbose)
            fname = os.path.join(projdir,'Non_transformed_static.json')
            with open(fname, 'w') as fp:
                json.dump(resjson, fp, cls=MyEncoder)
            if options.verbose:
                print('Results saved in {}'.format(fname))

            # resjson = to_json(Parms, verbose=options.verbose)
            fname = os.path.join(projdir,'Parms_of_transform.json')
            with open(fname, 'w') as fp:
                json.dump(Parms, fp, cls=MyEncoder)
            if options.verbose:
                print('Results saved in {}'.format(fname))

            if options.plot:
                if options.verbose:
                    print('Generating plots...')
                for loc, X in Rdata.items():
                    # print(fname1, loc)
                    plot_static(X, os.path.join(projdir, str(loc)))


__all__ = ['Retrieve_data', 'Options']

__script__ = __doc__

__warning__ = "Warning:" + warningstyle("\n It is advised AGAINST manual modifications of the local database (e.g., insert, delete or rename files or folders in the database directory) which may lead to its dysfunction.")

examples = []
examples.append(["%(prog)s -p 24 -v DBDIR", "Download or update the project of PID 24 in the directory DBDIR (this will create a project subfolder named 024 under DBDIR and generate a file named Raw.pkl) and print messages."])
examples.append(["%(prog)s -v DBDIR", "Download or update all available projects in the directory DBDIR."])
__example__ = "Some examples of use (change the path seperator '/' to '\\' on Windows platform):" + "".join([examplestyle(x) for x in examples])


def main():
    # usage_msg = '%(prog)s [options] <dbdir>'
    # parser = argparse.ArgumentParser(description=__script__, usage=usage_msg)
    parser = argparse.ArgumentParser(description=__script__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=__warning__ + "\n\n" + __example__)

    parser.add_argument('dbdir', help='directory of the local OSMOS database')
    parser.add_argument('-p', '--PID', dest='PID', type=int, default=None, help='project Key ID to be processed', metavar='integer')
    parser.add_argument('--info', dest='info', action='store_true', default=False, help='save the list of available projects')
    parser.add_argument('-s', '--sflag', dest='sflag', action='store_true', default=False, help="remove synchronization error")
    parser.add_argument('-o', '--oflag', dest='oflag', action='store_true', default=False, help="remove outliers")
    parser.add_argument('-t', '--tflag', dest='tflag', action='store_true', default=False, help="apply the preprocessing on the temperature data")
    parser.add_argument('-j', '--jflag', dest='jflag', action='store_true', default=False, help="detect jumps in the deformation data")
    parser.add_argument('-n', dest='nh', action='store', type=int, default=12, help="gaps (in hour) larger than this value will be marked as nan (default 12)", metavar="int")
    parser.add_argument('--port', dest='port', action='store', type=int, default=27017, help="port of local MongoDB (default=27017)", metavar="int")
    parser.add_argument("--json", dest="json", action="store_true", default=False, help="save results in json format")
    parser.add_argument("--plot", dest="plot", action="store_true", default=False, help="plot data")
    parser.add_argument('-v', '--verbose', dest='verbose', action='count', default=0, help='print messages')

    options = parser.parse_args()

    if options.PID is not None:  # string -> integer conversion
        try:
            options.PID = int(options.PID)
        except:
            raise ValueError("Invalid format of PID: {}".format(options.PID))

    # if len(args) < 1:
    #     print('Usage: '+usage_msg)
    #     print(parm_msg)
    #     sys.exit(0)
    # else:  # check dbdir
    #     dbdir = args[0]

    # check the database directory
    if not os.path.isdir(options.dbdir):
        raise FileNotFoundError(options.dbdir)
        # # create the directory if not existed
        # try:
        #     os.makedirs(options.dbdir)
        # except OSError:
        #     pass

    # core function
    Retrieve_data(options.dbdir, options)


if __name__ == '__main__':
    main()
