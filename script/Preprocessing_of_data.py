#!/usr/bin/env python

import sys
import os
import glob
from optparse import OptionParser       # command line arguments parser
import pickle
# import datetime
# import dateutil
# from collections import namedtuple
# import warnings
# import itertools
import copy
import pandas as pd

from OSMOS import OSMOS

# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)

# Hide annoying trace back message
# sys.excepthook = lambda exctype,exc,traceback : print("{}: {}".format(exctype.__name__,exc))

import matplotlib.pyplot as plt
import matplotlib.colors as colors
# color_list = list(colors.cnames.keys())
color_list = ['green', 'pink', 'lightgrey', 'magenta', 'cyan', 'red', 'yelow', 'purple', 'blue', 'mediumorchid', 'chocolate', 'blue', 'blueviolet', 'brown']

import mpld3
plt.style.use('ggplot')

__script__ = 'Apply preprocessing on updated data and assemble all processed data into a single pandas data sheet.'


def main():
    usage_msg = '{} [options] directory_of_PID'.format(sys.argv[0])
    parser = OptionParser(usage_msg)

    # parser.add_option('-p', '--PID', dest='PID', type='int', default=None, help='Project Key ID. If not given all projects presented in the destination data directory will be processeded.')
    parser.add_option('-d', '--dflag', dest='dflag', action='store_true', default=False, help='Remove possible dynamic data.')
    parser.add_option('-s', '--sflag', dest='sflag', action='store_true', default=False, help='Remove synchronization error.')
    parser.add_option('-o', '--oflag', dest='oflag', action='store_true', default=False, help='Remove outliers.')
    parser.add_option('-f', '--fflag', dest='fflag', action='store_true', default=False, help='Fill missing data of < 12h.')
    # parser.add_option('-r', '--rflag', dest='rflag', action='store_true', default=False, help='Resampling with the step=1h.')
    parser.add_option('-t', '--tflag', dest='tflag', action='store_true', default=False, help='Filter also the temperature.')
    parser.add_option('-j', '--jflag', dest='jflag', action='store_true', default=False, help='Detect jumps.')
    parser.add_option('--force', dest='force', action='store_true', default=False, help='Force re-computation.')
    parser.add_option('--plotstatic', dest='plotstatic', action='store_true', default=False, help='Plot static data of all sensor and save the figure.')
    parser.add_option('-v', '--verbose', dest='verbose', action='store_true', default=False, help='Print message.')

    (options, args) = parser.parse_args()

    if len(args) < 1:
        print('Usage: '+usage_msg)
        sys.exit(0)
    else:  # check datadir
        datadir = args[0]
        if not os.path.isdir(datadir):
            raise FileNotFoundError(datadir)

    # Process only the updated data file
    # pnames = sorted(glob.glob(datadir+'/*'))
    # ListDir = {}  # dic of data directories
    # ListPID0 = []  # list of PIDs in the data directory

    # # filter the sub-directories with a valid PID
    # for p in pnames: # iteration on folders of projects
    #     if os.path.isdir(p):
    #         idx = p.rfind('/')
    #         try:
    #             PID = int(p[idx+1:])
    #         except:
    #             continue

    #         ListDir[PID] = p
    #         ListPID0.append(PID) # Project key ID

    # # List of PID to be preprocessed
    # if options.PID is not None:  # PID given
    #     if options.PID in ListPID0:
    #         ListPID = [options.PID]
    #     else:
    #         raise KeyError('PID {} not found'.format(options.PID))
    # else:  # PID not given
    #     ListPID = ListPID0.copy()
    #     ListPID.sort()

    # for PID in ListPID:
    #     if options.verbose:
    #         print('\n-------Preprocessing the project {}-------'.format(PID))

    Sdata_all, Ddata_all = {}, {}  # dictionary for assembling data

    for q in glob.glob(datadir+'/*'): # iteration on folders of locations
        if os.path.isdir(q):
            idx = q.rfind('/')
            try:
                loc = int(q[idx+1:]) # location key ID
            except:
                continue

            # find the latest updated data file
            fnames = glob.glob(q+'/Raw_*.pkl')

            for f in fnames: # iteration on data files
                idx = f.rfind('Raw')
                g = f[:idx]+'Processed'+f[idx+3:] # name of the processed file

                if not os.path.isfile(g) or options.force: # if not processed or forced re-computation
                    if options.verbose:
                        print('Location: {}, file name: {}'.format(loc, f[idx:]))

                    with open(f, 'rb') as fp:
                        toto = pickle.load(fp)

                    RawData = OSMOS.raw2pandas(toto['Data']) # Convert to Pandas datasheet

                    try:
                        Sdata, Ddata = OSMOS.Preprocessing_by_location(RawData, MinDataLength=0,
                                                                       dflag=options.dflag,
                                                                       sflag=options.sflag,
                                                                       oflag=options.oflag,
                                                                       fflag=options.fflag,
                                                                       # rflag=options.rflag,
                                                                       rflag=True,
                                                                       tflag=options.tflag,
                                                                       jflag=options.jflag)

                        with open(g, 'wb') as fp:
                            pickle.dump({'Static_Data':Sdata,
                                         'Dynamic_Data':Ddata},
                                        fp, protocol=pickle.HIGHEST_PROTOCOL)
                    except Exception as msg:
                        print(msg)
                        continue

            # Assemble all preprocessed data files into a single pandas data sheet
            fnames = glob.glob(q+'/Processed_*.pkl')
            Sdata_list, Ddata_list = [], []  # list of static and dynamic data for assembling

            for f in fnames: # iteration on data files
                idx = f.rfind('/')

                with open(f, 'rb') as fp:
                    toto = pickle.load(fp)

                if len(toto['Static_Data']) > 0:
                    Sdata_list.append(toto['Static_Data'])
                if len(toto['Dynamic_Data']) > 0:
                    Ddata_list += toto['Dynamic_Data']  # '+=' : to fusion the content of lists

            if len(Sdata_list) > 0:
                # resampling to mark the missing data as nan
                Sdata_all[loc] = pd.concat(Sdata_list).resample('1H').asfreq()
                # if options.rflag:
                #     Sdata_all[loc] = pd.concat(Sdata_list).resample('1H').asfreq()
                # else:
                #     Sdata_all[loc] = pd.concat(Sdata_list)
            if len(Ddata_list) > 0:
                Ddata_all[loc] = copy.deepcopy(Ddata_list)

    fname = datadir+'/Processed.pkl'
    with open(fname, 'wb') as fp:
        pickle.dump({'Static_Data': Sdata_all, 'Dynamic_Data': Ddata_all},
                    fp, protocol=pickle.HIGHEST_PROTOCOL)

    if options.verbose:
        print('Results saved in {}'.format(fname))

    #### plot all sensors
    if options.plotstatic:
        fig, axes = plt.subplots(2,1,figsize=(20,10), sharex=True)

        for n, (loc, val) in enumerate(Sdata_all.items()):
            # print(AT[loc])
            Xt, Yt = val['Temperature'], val['Elongation']
            # Xt = val['Temperature'].resample('1H').asfreq()
            # Yt = val['Elongation'].resample('1H').asfreq()
            axes[0].plot(Xt, label='{}'.format(loc))
            axes[1].plot(Yt, label='{}'.format(loc))

        axes[0].legend()
        axes[1].legend()
        axes[0].set_title('Temperature')
        axes[1].set_title('Elongation')

        mpld3.save_html(fig, datadir+'/All.html')
        fig.savefig(datadir+'/All.pdf')
        plt.close(fig)


if __name__ == '__main__':
    print(__script__)
    print()
    main()
