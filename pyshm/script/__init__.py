from functools import wraps
import json, numpy, pickle, pandas

class MyEncoder(json.JSONEncoder):
    """ Convert numpy types to json.
    """
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def load_data(fname):
    """Utility function for loading pickle or json file saved by Thermal_* or Deconv_* scripts.
    """
    idx = fname.rfind('.')
    if fname[idx+1:].lower() == 'pkl':
        with open(fname, 'rb') as fp:
            Res = pickle.load(fp)
        return Res
    elif fname[idx+1:].lower() == 'json':
        with open(fname, 'r') as fp:
            toto = json.load(fp)
        Res = {}
        for k,v in toto.items():
            try:
                Res[k] = pandas.read_json(v)
            except Exception:
                # print(k)
                Res[k] = v
        return Res
    else:
        raise TypeError('Unknown file type.')


def per_sensor_result(X, Locations, aflag=True):
    """Reorganize the result by sensor.
    
    If aflag==True all non-pandas fields present in X will be preserved.
    """
    import pandas
    
    Res = {}
    for loc in Locations:
        toto = {}
        for k, v in X.items():
            if isinstance(v, pandas.DataFrame) or isinstance(v, pandas.Series):
                toto[k] = v[loc]
        Res[loc] = pandas.DataFrame(toto)
    if aflag:
        for k, v in X.items():
            if not (isinstance(v, pandas.DataFrame) or isinstance(v, pandas.Series)):
                Res.update({k:v})

    return Res


def to_json(X):
    """Convert the values of pandas instances in a dictionary to json format.
    
    Args:
        X (dict): input dictionary
    """
    Res = {}
    for k, v in X.items():
        if isinstance(v, pandas.DataFrame) or isinstance(v, pandas.Series):
            Res[k] = v.to_json(date_unit='s', date_format='iso')
        else:
            try:
                # if the value can be converted to json, keep as it is
                json.dumps(v, cls=MyEncoder)
                Res[k] = v
            except Exception as msg: #TypeError:
                pass
    return Res


def static_data_analysis_template(func):
    """Template for static data analysis algorithms.

    This functional prepares data for the core algorithm and saves the result in a dictionary. The core algorithm is passed by 'func' which is callable via the interface
        resdic = func(options, Xcpn, Ycpn, *args, **kwargs)
    the returned dictionary will be auguemented by some extra informations.
    """

    @wraps(func)
    def newfunc(infile, outfile0, options, *args, **kwargs):
        import os, sys
        import pickle
        import json
        # import numpy as np
        import pandas as pd
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
        # tidx0 and tidx1 are respectively the beginning and ending timestamps
        if options.timerange is None:
            tidx0, tidx1 = None, None
        else:
            tidx0, tidx1 = options.timerange
        Xcpn = Xcpn0[tidx0:tidx1]
        Ycpn = Ycpn0[tidx0:tidx1]
        Midx = Midx0[tidx0:tidx1]  # indicator of missing value
        # options.timerange = (Xcpn.index[0], Xcpn.index[-1])  # true time range

        # Call the core function and save results in a dictionary
        resdic = func(Xcpn, Ycpn, options, *args, **kwargs)  # side effect on options
        resdic.update({'Xcpn':Xcpn, 'Ycpn':Ycpn, 'Midx':Midx, 'algo_options':vars(options)})

        # Save the results
        with open(outfile0+'.pkl', 'wb') as fp:
            # json.dump(resdic, fp, sort_keys=True)
            pickle.dump(resdic, fp)
        if options.verbose:
            print('\nResults saved in {}'.format(outfile0+'.pkl'))

        resjson = to_json(resdic)  # options is dropped in resjson
        with open(outfile0+'.json', 'w') as fp:
            json.dump(resjson, fp, cls=MyEncoder)
        if options.verbose:
            print('\nResults saved in {}'.format(outfile0+'.json'))

        # # Reorganized per sensor
        # toto = per_sensor_result(resdic, Locations, aflag=True)
        # resjson = {}
        # for k,v in toto.items():  # convert to json
        #     resjson[k] = v.to_json(date_unit='s', date_format='iso')
        # alljson = {'results':resjson, 'options':vars(options)}
        # with open(outfile0+'.json', 'w') as fp:
        #     json.dump(alljson, fp, sort_keys=True, cls=MyEncoder)
        # if options.verbose:
        #     print('\nResults saved in {}'.format(outfile0+'.json'))

        # for loc in Locations:
        #     fname = outfile0+'_{}.json'.format(loc)
        #     res_sensor[loc].to_json(fname, date_unit='s', date_format='iso')
        #     if options.verbose:
        #         print('Results saved in {}'.format(fname))

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

        return resdic, resjson
    return newfunc



# from . import Download_data, Preprocess_static_data, Deconv_static_data, Deconv_static_data_plot
