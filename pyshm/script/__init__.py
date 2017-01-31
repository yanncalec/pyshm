from functools import wraps
import json, numpy, pickle, pandas
import colorama

warningstyle = lambda x: colorama.Style.BRIGHT + colorama.Fore.RED + x + colorama.Style.RESET_ALL
brightstyle = lambda x: colorama.Style.BRIGHT + x + colorama.Style.RESET_ALL
examplestyle = lambda x: "\n  " + brightstyle(x[0]) + "\n\t" + x[1]


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
        # elif isinstance(obj, numpy.bool_):
        #     return bool(obj)
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
                # json will not dump dictionary unless all keys are string
                u = {str(k):v for k,v in v.items()}
                json.dumps(u, cls=MyEncoder)
                Res[k] = u
            except Exception as msg: #TypeError:
                # print(k, msg)
                pass
    return Res


def prepare_data(infile, component='Trend', mwsize=24, kzord=1, mwmethod='mean', timerange=(None,None), verbose=True):
    """Load data and separate components.
    """
    from pyshm import OSMOS
    # import sys

    # options_dict = {'component':'Seasonal', 'verbose':True, 'mwsize':24, 'kzord':1, 'mwmethod':'mean',
    #             'timerange':None, 'info':False}
    # dict_filter = lambda D,F:{k:(D[k] if k in D else v) for k,v in F.items()}
    # options = type('Options', (object,), dict_filter(kwargs, options_dict))    # Load preprocessed static data

    # Load preprocessed static data
    Data0, Tall0, Eall0, Locations = OSMOS.load_static_data(infile)
    # indicator of missing data, NaN: not defined, True: missing data
    Midx0 = OSMOS.concat_mts(Data0, 'Missing')

    # Selection of component
    if component.upper() in ['SEASONAL', 'TREND']:
        if verbose:
            print('Decomposition of signals...')
            print('   Moving window estimator: {}\n   Moving window size: {}\n   Order of the KZ filter: {}\n'.format(mwmethod, mwsize, kzord))

        # Decomposition of signals
        Ttrd0, Tsnl0 = OSMOS.trend_seasonal_decomp(Tall0, mwsize=mwsize, kzord=kzord, method=mwmethod)
        Etrd0, Esnl0 = OSMOS.trend_seasonal_decomp(Eall0, mwsize=mwsize, kzord=kzord, method=mwmethod)

        if component.upper() == 'SEASONAL':
            Xcpn0, Ycpn0 = Tsnl0, Esnl0
        else:
            Xcpn0, Ycpn0 = Ttrd0, Etrd0
    elif component.upper() == 'ALL':
            Xcpn0, Ycpn0 = Tall0, Eall0
    else:
        raise TypeError('Unknown type of component:', component)

    # Data truncation
    # tidx0 and tidx1 are respectively the beginning and ending timestamps
    tidx0, tidx1 = timerange
    Xcpn = Xcpn0[tidx0:tidx1]
    Ycpn = Ycpn0[tidx0:tidx1]
    # indicator of missing value, the nan values due forced alignment of OSMOS.concat_mts are casted to True
    Midx = Midx0[tidx0:tidx1].astype(bool)

    return Xcpn, Ycpn, Midx


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

        Xcpn, Ycpn, Midx = prepare_data(infile, component=options.component,  mwsize=options.mwsize, kzord=options.kzord, mwmethod=options.mwmethod, timerange=(options.time0, options.time1), verbose=options.verbose)

        # Call the core function and save results in a dictionary
        resdic = func(Xcpn, Ycpn, options, *args, **kwargs)  # side effect on options
        resdic.update({'Xcpn':Xcpn, 'Ycpn':Ycpn, 'Midx':Midx, 'algo_options':vars(options)})

        # # Save the results in pickle format
        # with open(outfile0+'.pkl', 'wb') as fp:
        #     # json.dump(resdic, fp, sort_keys=True)
        #     pickle.dump(resdic, fp)
        # if options.verbose:
        #     print('\nResults saved in {}'.format(outfile0+'.pkl'))

        # Save the results in json format:
        # some non-standard objects might be removed, and non-float values will be casted as float
        resjson = to_json(resdic)
        # print(resjson.keys())
        with open(outfile0+'.json', 'w') as fp:
            json.dump(resjson, fp, cls=MyEncoder)
        if options.verbose:
            print('Results saved in {}'.format(outfile0+'.json'))

        return resdic, resjson
    return newfunc


from . import Download_data
from . import Preprocess_static_data
from . import Deconv_static_data

# from .Download_data import Download_data
# from .Preprocess_static_data import Preprocess_static_data
# from .Deconv_static_data import Deconv_static_data
# , Deconv_static_data, Deconv_static_data_plot

