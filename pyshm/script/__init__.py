from functools import wraps
import json, numpy, pickle, pandas
import colorama
import numpy as np
import pandas as pd
import os
from .. import OSMOS, Tools, Stat, Models
# import pyshm

# Some console text formation styles
warningstyle = lambda x: colorama.Style.BRIGHT + colorama.Fore.RED + x + colorama.Style.RESET_ALL
brightstyle = lambda x: colorama.Style.BRIGHT + x + colorama.Style.RESET_ALL
examplestyle = lambda x: "\n  " + brightstyle(x[0]) + "\n\t" + x[1]


class Options:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def split_by_day(Xvar):
    """Split a time series by day.

    Args:
        Xvar (pandas DataFrame): input, Xvar.index is Timestamp object.
    Return:
        a list of objects of DataFrame.
    """
    Xlist = []

    Tidx = Xvar.index
    t0, t1 = Tidx[0], Tidx[-1]
    Dlist = pd.date_range(start=str(t0.floor('D')), end=str(t1.ceil('D')), freq='D')

    for day0, day1 in zip(Dlist[:-1], Dlist[1:]):
        tbix = np.logical_and(Tidx>=day0, Tidx<day1)
    #     print(Tidx[tbix])
    #     print(toto['Temperature'][tbix])
        Xlist.append(Xvar.iloc[tbix])
    return Xlist


def compute_local_statistics(Yerr, mad, mwsize, win_type='boxcar'):
    """Compute the local statistics: mean and standard deviation and normalized
    observation.

    """
    Merr0 = {}; Serr0 = {}; Nerr0 = {}

    for loc, yerr in Yerr.items():
        # yerr = Ycpn[loc] - Yprd[loc]
        # moving average and standard deviation
        merr, serr = Stat.local_statistics(yerr, mwsize, mad=mad, causal=False, drop=False, win_type=win_type)
        nerr = abs(yerr-merr)/serr  # normalized error
        Merr0[loc], Serr0[loc], Nerr0[loc] = merr, serr, nerr

    Merr = pd.DataFrame(Merr0, columns=list(Yerr.keys()), index=Yerr.index)
    Serr = pd.DataFrame(Serr0, columns=list(Yerr.keys()), index=Yerr.index)
    Nerr = pd.DataFrame(Nerr0, columns=list(Yerr.keys()), index=Yerr.index)

    return Merr, Serr, Nerr

# union2list = lambda L1, L2: L1+[x for x in L2 if x not in L1]

def Hurstfunc(loc, X, mwsize, hrng):
    """An auxiliary function for parallel computation of Hurst exponent.
    """
    Y, *_ = Stat.Hurst(np.asarray(X), mwsize, sclrng=hrng, wvlname="haar")  # Hurst exponent
    return {loc: Y}


def load_results(Results, key):
    if not key in Results:
        return None
    else:
        X = Results[key].copy()
        if 'index' in X:
            X.index = X['index']
            del X['index']
        return X


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


# def to_json(X, verbose=False):
#     """Filter and convert a dictionary to json format.

#     A key-value pair in the dictionary will be convert to json if the value is one of the following types:
#         - pandas DataFrame or Series
#         - a numpy array
#         - a dictionary of numpy array
#         - None type
#     otherwise it will be filtered out.

#     Args:
#         X (dict): input dictionary
#     Returns:
#         a dictionary composed of the converted key-value pairs.
#     """
#     import numbers

#     Res = {}
#     for k, v in X.items():
#         if isinstance(v, pandas.DataFrame) or isinstance(v, pandas.Series):
#             Res[k] = v.to_json(date_unit='s', date_format='iso')
#         else:
#             try:
#                 # if the value can be converted to json or if it is None, keep
#                 # as it is json will not dump dictionary unless all keys are
#                 # string.
#                 if isinstance(v, dict):
#                     u = {str(k):v for k,v in v.items()}
#                     json.dumps(u, cls=MyEncoder)
#                     Res[k] = u
#                 elif isinstance(v, np.ndarray) or isinstance(v, str) or isinstance(v, numbers.Number) or v is None:
#                     Res[k] = v
#                 else:
#                     # This should not happen:
#                     raise TypeError('to_json: The values of the input dictionary is not valid.')
#             except Exception as msg:
#                 if verbose:
#                     print(warningstyle("Warning:\n{}".format(msg)))
#                 pass
#     return Res


# def load_result_of_analysis(fname):
#     """Utility function for loading pickle or json file saved by the scripts of data analysis.

#     Args:
#         fname (str): input file name
#     Return:
#         Res: a dictionary containing the contents of the input file
#     """
#     idx = fname.rfind('.')
#     if fname[idx+1:].lower() == 'pkl':
#         with open(fname, 'rb') as fp:
#             Res = pickle.load(fp)
#         return Res
#     elif fname[idx+1:].lower() == 'json':
#         with open(fname, 'r') as fp:
#             toto = json.load(fp)
#         Res = {}
#         for k,v in toto.items():
#             try:
#                 Res[k] = pandas.read_json(v)
#             except Exception:
#                 Res[k] = v
#         return Res
#     else:
#         raise TypeError('Unknown file type.')


# def per_sensor_result(X, Locations, aflag=True):
#     """Reorganize the result by sensor.

#     If aflag==True all non-pandas fields present in X will be preserved.
#     """
#     import pandas

#     Res = {}
#     for loc in Locations:
#         toto = {}
#         for k, v in X.items():
#             if isinstance(v, pandas.DataFrame) or isinstance(v, pandas.Series):
#                 toto[k] = v[loc]
#         Res[loc] = pandas.DataFrame(toto)
#     if aflag:
#         for k, v in X.items():
#             if not (isinstance(v, pandas.DataFrame) or isinstance(v, pandas.Series)):
#                 Res.update({k:v})

#     return Res


def plot_static(Data, fname):
    """Plot original static data and save figures.

    Args:
        Data: pandas DataFrame
        fname (str): name of the output figure
    """
    import matplotlib
    # matplotlib.use('qt5agg')
    matplotlib.use('Agg')
    # matplotlib.use('macosx')

    import matplotlib.pyplot as plt
    # import matplotlib.colors as colors
    # import mpld3
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
    # axa.plot(Data['parama'], 'b', alpha=0.5, label='a')
    # axa.plot(Data['paramb'], 'r', alpha=0.5, label='b')
    # axa.plot(Data['paramc'], 'g', alpha=0.5, label='c')
    axa.plot(Data['Reference'], 'c', alpha=0.7, label='Reference')
    axa.legend(loc='upper left')
    # axb = axa.twinx()
    # axb.legend(loc='upper right')

    plt.tight_layout()
    fig.savefig(fname+".pdf", bbox_inches='tight')
    plt.close(fig)

    return fig


# def static_data_analysis_template(func):
#     """Template for static data analysis algorithms.

#     This functional prepares data for the core algorithm and saves the
#     results in a dictionary. The core algorithm is passed by 'func' which
#     must be callable via the interface

#         resdic = func(options, Xcpn, Ycpn, *args, **kwargs)

#     the returned dictionary will be auguemented by some extra informations.

#     """
#     from pyshm import OSMOS
#     # import os
#     # import pickle
#     # import sys
#     # import json
#     # import numpy as np
#     # import pandas as pd
#     # import colorama
#     # import inspect

#     @wraps(func)
#     def newfunc(infile, outfile0, options, *args, **kwargs):

#         if options.component.upper() in ['SEASONAL', 'TREND']:
#             if options.verbose:
#                 print('Decomposition of signals...')
#                 print('\tMoving window estimator: {}\n\tMoving window size: {}\n\tOrder of the KZ filter: {}'.format(options.mwmethod, options.mwsize, options.kzord))

#             (Xall,Xsnl,Xtrd), (Yall,Ysnl,Ytrd), Midx = OSMOS.prepare_static_data(infile, mwsize=options.mwsize, kzord=options.kzord, method=options.mwmethod, timerange=(options.time0, options.time1))

#             if options.component.upper() == 'SEASONAL':
#                 Xcpn, Ycpn = Xsnl, Ysnl
#             else:
#                 Xcpn, Ycpn = Xtrd, Ytrd
#         elif options.component.upper() == 'ALL':
#             Xcpn, Ycpn, Midx = OSMOS.truncate_static_data(infile, (options.time0, options.time1))
#         else:
#             raise TypeError("{}: Wrong type of component.".format(options.component))

#         # Call the core function and save results in a dictionary
#         resdic = func(Xcpn, Ycpn, options, *args, **kwargs)  # side effect on options
#         resdic.update({'func_name':options.func_name, 'Xcpn':Xcpn, 'Ycpn':Ycpn, 'Midx':Midx, 'algo_options':vars(options)})

#         # Make the output directory if necessary
#         idx = outfile0.rfind(os.path.sep)
#         outdir = outfile0[:idx]
#         try:
#             os.makedirs(outdir)
#         except OSError:
#             pass

#         # Save the results in pickle format
#         outfile = outfile0+'.pkl'
#         with open(outfile, 'wb') as fp:
#             pickle.dump(resdic, fp)
#         # # Save the results in json format:
#         # # some non-standard objects might be removed, and non-float values will be casted as float
#         # resjson = to_json(resdic, verbose=options.verbose)
#         # outfile = outfile0+'.json'
#         # with open(outfile, 'w') as fp:
#         #     json.dump(resjson, fp, cls=MyEncoder)

#         if options.verbose:
#             print('Results saved in\n{}'.format(outfile))

#         return resdic
#     return newfunc


# from . import Download_data
# from . import Preprocess_static
# from . import Deconv_static

# from .Download_data import Download_data
# from .Preprocess_static_data import Preprocess_static_data
# from .Deconv_static_data import Deconv_static_data
# , Deconv_static_data, Deconv_static_data_plot
