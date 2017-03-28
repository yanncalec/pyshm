from functools import wraps
import json, numpy, pickle, pandas
import colorama
import numpy as np
import os

# Some console text formation styles
warningstyle = lambda x: colorama.Style.BRIGHT + colorama.Fore.RED + x + colorama.Style.RESET_ALL
brightstyle = lambda x: colorama.Style.BRIGHT + x + colorama.Style.RESET_ALL
examplestyle = lambda x: "\n  " + brightstyle(x[0]) + "\n\t" + x[1]


class MyEncoder(json.JSONEncoder):
    """Utility class for convert from numpy types to json.

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


def to_json(X, verbose=False):
    """Filter and convert a dictionary to json format.

    A key-value pair in the dictionary will be convert to json if the value is one of the following types:
        - pandas DataFrame or Series
        - a numpy array
        - a dictionary of numpy array
        - None type
    otherwise it will be filtered out.

    Args:
        X (dict): input dictionary
    Returns:
        a dictionary composed of the converted key-value pairs.
    """
    import numbers

    Res = {}
    for k, v in X.items():
        if isinstance(v, pandas.DataFrame) or isinstance(v, pandas.Series):
            Res[k] = v.to_json(date_unit='s', date_format='iso')
        else:
            try:
                # if the value can be converted to json or if it is None, keep
                # as it is json will not dump dictionary unless all keys are
                # string.
                if isinstance(v, dict):
                    u = {str(k):v for k,v in v.items()}
                    json.dumps(u, cls=MyEncoder)
                    Res[k] = u
                elif isinstance(v, np.ndarray) or isinstance(v, str) or isinstance(v, numbers.Number) or v is None:
                    Res[k] = v
                else:
                    # This should not happen:
                    raise TypeError('to_json: The values of the input dictionary is not valid.')
            except Exception as msg:
                if verbose:
                    print(warningstyle("Warning:\n{}".format(msg)))
                pass
    return Res


def load_result_of_analysis(fname):
    """Utility function for loading pickle or json file saved by the scripts of data analysis.

    Args:
        fname (str): input file name
    Return:
        Res: a dictionary containing the contents of the input file
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


def static_data_analysis_template(func):
    """Template for static data analysis algorithms.

    This functional prepares data for the core algorithm and saves the
    results in a dictionary. The core algorithm is passed by 'func' which
    must be callable via the interface

        resdic = func(options, Xcpn, Ycpn, *args, **kwargs)

    the returned dictionary will be auguemented by some extra informations.

    """
    from pyshm import OSMOS
    # import os
    # import pickle
    # import sys
    # import json
    # import numpy as np
    # import pandas as pd
    # import colorama
    # import inspect

    @wraps(func)
    def newfunc(infile, outfile0, options, *args, **kwargs):

        if options.component.upper() in ['SEASONAL', 'TREND']:
            if options.verbose:
                print('Decomposition of signals...')
                print('\tMoving window estimator: {}\n\tMoving window size: {}\n\tOrder of the KZ filter: {}'.format(options.mwmethod, options.mwsize, options.kzord))

            (Xall,Xsnl,Xtrd), (Yall,Ysnl,Ytrd), Midx = OSMOS.prepare_static_data(infile, mwsize=options.mwsize, kzord=options.kzord, method=options.mwmethod, timerange=(options.time0, options.time1))

            if options.component.upper() == 'SEASONAL':
                Xcpn, Ycpn = Xsnl, Ysnl
            else:
                Xcpn, Ycpn = Xtrd, Ytrd
        elif options.component.upper() == 'ALL':
            Xcpn, Ycpn, Midx = OSMOS.truncate_static_data(infile, (options.time0, options.time1))
        else:
            raise TypeError("{}: Wrong type of component.".format(options.component))

        # Call the core function and save results in a dictionary
        resdic = func(Xcpn, Ycpn, options, *args, **kwargs)  # side effect on options
        resdic.update({'func_name':options.func_name, 'Xcpn':Xcpn, 'Ycpn':Ycpn, 'Midx':Midx, 'algo_options':vars(options)})

        # Make the output directory if necessary
        idx = outfile0.rfind(os.path.sep)
        outdir = outfile0[:idx]
        try:
            os.makedirs(outdir)
        except OSError:
            pass

        # Save the results in pickle format
        outfile = outfile0+'.pkl'
        with open(outfile, 'wb') as fp:
            pickle.dump(resdic, fp)
        # # Save the results in json format:
        # # some non-standard objects might be removed, and non-float values will be casted as float
        # resjson = to_json(resdic, verbose=options.verbose)
        # outfile = outfile0+'.json'
        # with open(outfile, 'w') as fp:
        #     json.dump(resjson, fp, cls=MyEncoder)

        if options.verbose:
            print('Results saved in\n{}'.format(outfile))

        return resdic
    return newfunc



from . import Download_data
from . import Preprocess_static_data
from . import Deconv_static_data

# from .Download_data import Download_data
# from .Preprocess_static_data import Preprocess_static_data
# from .Deconv_static_data import Deconv_static_data
# , Deconv_static_data, Deconv_static_data_plot
