"""Apply preprocessing on updated data and assemble all processed data into a single pandas data sheet."""

import sys, os
from optparse import OptionParser       # command line arguments parser

import glob, pickle
import colorama
import matplotlib.pyplot as plt
# import matplotlib.colors as colors
import mpld3
plt.style.use('ggplot')

import pandas as pd
import copy

from . import OSMOS

class Options:
    dflag=False
    sflag=False
    oflag=False
    fflag=False
    tflag=False
    jflag=False
    force=False
    plotstatic=False
    verbose=False


def Preprocessing_of_data(datadir, options=None):
    """
    datadir : the directory of one project in the database
    """

    if options is None:  # use default value for options
        options = Options()

    Sdata_all, Ddata_all = {}, {}  # dictionary for assembling data

    for q in glob.glob(os.path.join(datadir,'*')): # iteration on folders of locations
        if os.path.isdir(q):
            idx = q.rfind(os.path.sep)
            # print(q,idx, os.path.sep)
            try:
                loc = int(q[idx+1:]) # location key ID
            except:
                continue

            # find the latest updated data file
            fnames = glob.glob(os.path.join(q,'Raw_*.pkl'))
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
                        print(colorama.Fore.RED + 'Error: ', msg)
                        print(colorama.Style.RESET_ALL)
                        continue

            # Assemble all preprocessed data files into a single pandas data sheet
            fnames = glob.glob(os.path.join(q,'Processed_*.pkl'))
            Sdata_list, Ddata_list = [], []  # list of static and dynamic data for assembling

            for f in fnames: # iteration on data files
                idx = f.rfind(os.path.sep)

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

    fname = os.path.join(datadir,'Processed.pkl')
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

        mpld3.save_html(fig, os.path.join(datadir,'All.html'))
        fig.savefig(os.path.join(datadir, 'All.pdf'))
        plt.close(fig)
