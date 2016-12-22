from functools import wraps

def static_data_analysis_template(func):
    """Template for static data analysis algorithms.

    This functional prepares data for the core algorithm and saves the result in a dictionary. The core algorithm is passed by 'func' which is callable via the interface
        resdic = func(options, Xcpn, Ycpn, *args, **kwargs)
    the returned dictionary will be auguemented with extra information.
    """

    @wraps(func)
    def newfunc(infile, outfile, options, *args, **kwargs):
        import os, sys
        import pickle
        import json
        # import numpy as np
        # import pandas as pd
        # import colorama
        from pyshm import OSMOS

        # Load preprocessed static data
        Data0, Tall0, Eall0, Locations = OSMOS.load_static_data(infile)
        # indicator of missing data, NaN: not defined, True: missing data
        Midx0 = OSMOS.concat_mts(Data0, 'Missing')

        if options.info:
            ss = str(Locations)[1:-1].replace(',', '')
            print('Location key IDs:', ss)

            for loc in Locations:
                print('Location {}, from {} to {}'.format(loc, Data0[loc].index[0], Data0[loc].index[-1]))
            t0, t1 = OSMOS.common_time_range(Data0)
            print('Common period: from {} to {}'.format(t0, t1))
            sys.exit()

        # Selection of component
        if options.component.upper() in ['SEASONAL', 'TREND']:
            if options.verbose:
                print('Decomposition of signals...')
                print('   Moving window estimator: {}\n   Moving window size: {}\n   Order of the KZ filter: {}\n'.format(options.mwmethod, options.mwsize, options.kzord))

            # Decomposition of signals
            Ttrd0, Tsnl0 = OSMOS.trend_seasonal_decomp(Tall0, mwsize=options.mwsize, kzord=options.kzord, method=options.mwmethod)
            Etrd0, Esnl0 = OSMOS.trend_seasonal_decomp(Eall0, mwsize=options.mwsize, kzord=options.kzord, method=options.mwmethod)

            if options.component.upper() == 'SEASONAL':
                Xcpn0, Ycpn0 = Tsnl0, Esnl0
            else:
                Xcpn0, Ycpn0 = Ttrd0, Etrd0
        elif options.component.upper() == 'ALL':
                Xcpn0, Ycpn0 = Tall0, Eall0
        else:
            raise Exception('Unknown type of component:', options.component)

        # Data truncation
        # if options.timerange is None
        # tidx0 and tidx1 are respectively the beginning and ending timestamps
        tidx0, tidx1 = options.timerange
        Xcpn = Xcpn0[tidx0:tidx1]
        Ycpn = Ycpn0[tidx0:tidx1]
        Midx = Midx0[tidx0:tidx1]  # indicator of missing value
        # options.timerange = (Xcpn.index[0], Xcpn.index[-1])  # true time range

        # Call the core function
        resdic = func(options, Xcpn, Ycpn, *args, **kwargs)
        # dictionary of the results
        resdic.update({'Xcpn':Xcpn, 'Ycpn':Ycpn, 'options':options, 'Midx':Midx})

        with open(outfile, 'wb') as fp:
            # json.dump(resdic, fp, sort_keys=True)
            pickle.dump(resdic, fp)
        if options.verbose:
            print('\nResults saved in {}'.format(outfile))
        return resdic
        # try:
        #     with open(fname, 'wb') as fp:
        #         pickle.dump(resdic, fp)
        #     if options.verbose:
        #         print('\nResults saved in {}'.format(fname))
        #     return resdic
        # except Exception as msg:
        #     # print(msg)
        #     print(Fore.RED + 'Warning: ', msg)
        #     print(Style.RESET_ALL)

    return newfunc


# from . import Download_data, Preprocess_static_data, Deconv_static_data, Deconv_static_data_plot
