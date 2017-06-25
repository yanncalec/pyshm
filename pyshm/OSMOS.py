"""Collection of functions related to I/O and preprocessing of OSMOS data.
"""

import json, requests
import dateutil, datetime
import os, glob, pickle, sys, pprint
from pymongo import MongoClient

import numpy as np
# import numpy.linalg as la
import scipy

import pandas as pd
import colorama
# from colorama import Fore, Back, Style

import collections  # ordered dictionary

from . import Tools
# from pyshm import Tools

class LIRIS:
    """Class for describing a LIRIS object
    """
    def __init__(self, **entries):
        self.__dict__.update(entries)


def _retrieve_LIRIS_info(PID, link, login, password):
    # disable certificate check
    requests.packages.urllib3.disable_warnings()

    # open the session
    session = requests.session()

    # get the token
    payload = {"token":"null","action":"loginClient","data":{"email":login,"password":password}}
    r = session.get(link, data=json.dumps(payload), verify=False)
    response = json.loads(r.text)
    assert(response['result'] == 'SUCCESS')  # check the response from the server

    # token to use for next requests
    token = response['data']["tokenStore"]

    payload = {"token": token,
            "action": "getlirisfromproject",
            # "action": "getlistlocationsused",  # <- incomplete information
            "data": {"projectkeyid": PID}}
    r = session.get(link, data=json.dumps(payload), verify=False)
    toto = json.loads(r.text)

    if toto['result']=='SUCCESS' and len(toto['data']['records'])>0:  # <-toto['data']
#         print(toto)
        P = pd.DataFrame(toto['data']['records'])
        P['pid'] = PID
        return P
    else:
        return None


def retrieve_LIRIS_info(PIDs, link="https://client.osmos-group.com/server/application.php", login="be@osmos-group.com", password="osmos", redundant=False):
    """Retrive LIRIS info of projects from OSMOS server.

    Args:
        PIDs (list): list of PID
    Return:
        an object of pandas DataFrame containing the information of given PIDs.
    """
    L = []
    for pid in PIDs:
        P = _retrieve_LIRIS_info(pid, link=link, login=login, password=password)
        if P is not None:
            L.append(P)
    if len(L) > 0:
        LIRIS_info = pd.concat(L).reset_index(drop=True)
        # print(LIRIS_info.keys())
        if not redundant:  # remove the field location which seems to be redundant
            del LIRIS_info['location']
    else:
        LIRIS_info = None
    return LIRIS_info


def retrieve_data(hostname, port, pid, locations, dbname='OSMOS', clname='Liris_Measure', redundant=True):
    """Retrieve data of a PID from MongoDB.

    Args:
        hostname (str): name of the MongoDB server
        port (int): port of the MongoDB server
        pid (int): project key ID
        locations (list): list of locations associated to pid
    Returns:
        Sdata, Parms: a dictionary of pandas DataFrame and the parameters of transformation.
    """
    client = MongoClient(hostname, port)
    # collection = client['OSMOS']['Liris_Measure']  # collection
    collection = client[dbname][clname]  # collection

    Sdata = {}  # dictionary of preprocessed static data
    Parms = {}  # dictionary of parameters

    for loc in locations: # iteration on all sensors of the project
        # uid = u['uid']
        # loc = int(u['locationkeyid'])

        X0 = []
        try:
            X0 = mongo_load_static(collection, loc, dflag=True)
        except Exception as msg:
            print(msg)
            continue

        if len(X0)>0:
    #         Rtsx = pd.date_range(X0.index[0].ceil(rsp), X0.index[-1].floor(rsp), freq=rsp)
    #         Sdata[loc] = X0.resample('H').ffill().loc[Rtsx]
            S, Rtsx, Ntsx = resampling_time_series(X0)
            Sdata[loc] = S.loc[Rtsx]
            Sdata[loc].loc[Ntsx] = np.nan
            toto = pd.Series(False,index=Rtsx); toto[Ntsx] = True
            Sdata[loc]['Missing'] = toto  # a field indicating missing values

            # if not redundant:  # remove redundant information
            #     del Sdata[loc]['parama'], Sdata[loc]['paramb'], Sdata[loc]['paramc']

            Parms[loc] = tuple(np.asarray(Sdata[loc][['parama', 'paramb', 'paramc']]).mean(axis=0))
    return Sdata, Parms


def update_LIRIS_data_by_project(token, session, PID, projdir, endtime=None, verbose=0):
    """Update the local database by requesting the OSMOS's server.

    Args:
        token: see Download_data.py.
        session: see Download_data.py.
        PID (int): project key id.
        projdir (str): project directory.
        endtime (str): fetch data til this time, by default fetch til today.
        verbose (int): print message.
    Returns:
        A bool variable indicating the presence of new data.
    """

    payload = {"token": token,
               "action": "getlirisfromproject",
               "data": {"projectkeyid": PID}}
    r = session.post("https://client.osmos-group.com/server/application.php",
                     data=json.dumps(payload))
    response0 = json.loads(r.text)

    nflag = False  # indicator of new data
    Liris = []  # informations of Liris capteurs

    # If the response from the server is successful and the project exists
    if 'result' in response0.keys() and response0['result'] == 'SUCCESS' and response0['data']['count'] > 0:
        if verbose > 0:
            print('number of LIRIS: {}'.format(response0['data']['count']))

        # keys of the following dictionaries are the location ID
        # Measures_dict = {} # dictionary for measurements
        # LIRIS_dict = {} # dictionary for LIRIS information

        # t_end, t_start = time_range # to get a more precise starting and ending time

        for record in response0['data']['records']:  # each record corresponds to one location/LIRIS
            liris = LIRIS(**record) #convert the dictionary to a LIRIS object
            # liris = namedtuple('LIRIS', record.keys())(*record.values())
            Liris.append(liris)

            if liris.locationkeyid is not None: # if the location is valid
                liris_info = 'Location ID: {}, Location name: {}, LIRIS ID: {}, LIRIS name:{}'.format(liris.locationkeyid, liris.locationname, liris.liriskeyid, liris.name)
                if verbose > 0:
                    print(liris_info)

                loc = liris.locationkeyid  # location
                t_start0 = dateutil.parser.parse(liris.datestart)  # official starting time
                t_end0 = dateutil.parser.parse(liris.dateend)  # official ending time
                if verbose > 1:
                    print("Official operation period: from {} to {}".format(t_start0, t_end0))

                datadir = os.path.join(projdir, '{}'.format(loc)) # Directory for saving the data of this location
                Flist = glob.glob(os.path.join(datadir,'Raw_*.pkl'.format(liris.locationkeyid))) # previously downloaded data, already sorted in increasing order
                if len(Flist) > 0:
                    fname = Flist[-1]
                    idx = fname.rfind('[')
                    t_start0 = dateutil.parser.parse(fname[idx+5:idx+24])
                    t_start0 += datetime.timedelta(0, 1) # new start time is 1s after

                Measures0 = [] # for saving of the download
                t0 = t_start0

                t_end0 = min(t_end0, datetime.datetime.today() if endtime is None else dateutil.parser.parse(endtime))
                if verbose:
                    print('Fetching data from {} to {}'.format(t_start0, t_end0))

                while t0 < t_end0: # for a valide time range
                    # increment by 30 days, which is the capacity of the Osmos server (?)
                    timeinterval = [t0, t0+datetime.timedelta(30,0)]

                    if verbose > 1:
                        print("Requesting the data of the period from {} to {}".
                              format(timeinterval[0], timeinterval[1]))

                    location = [{"locationkeyid": liris.locationkeyid,
                                 "offset": [],
                                 "round": "0"}]

                    # get data of each location
                    # Warning: with hourEnd=2359 the data of the day of timeinterval[1] are doubled
                    payload = {"token":token,"action":"getalldatajson",
                               "data":{"location":location,
                                       "start":str(timeinterval[0]),
                                       "end":str(timeinterval[1]),
                                       "hmStart":0,"hmEnd":2359,
                                       "hourStart":0,"hourEnd":0,
                                       "secondStart":0,"secondEnd":0
                                      }
                              }
                    r = session.post("https://client.osmos-group.com/server/application.php", data=json.dumps(payload))
                    response=json.loads(r.text)

                    Measures0 += response['data']['datas'][0]['measures']
                    t0 += datetime.timedelta(30, 0)

                if len(Measures0) > 0:  # if the result of requests is not empty
                    # Retrieve the real period of the new data
                    t_start1 = dateutil.parser.parse(Measures0[0][0])
                    t_end1 = dateutil.parser.parse(Measures0[-1][0])

                    # Save data only when containing more than one day of data
                    if t_end1-t_start1 > datetime.timedelta(1):
                        nflag = True  # Set the indicator of new data

                        if verbose > 0:
                            print(colorama.Fore.GREEN + 'New data (>24h) updated: from {} to {}'.format(t_start1, t_end1))
                            print(colorama.Style.RESET_ALL)

                        try:
                            os.makedirs(datadir)
                        except OSError:
                            pass

                        fname0 = 'Raw_[Start_{}]_[End_{}]].pkl'.format(to_valid_time_format(t_start1), to_valid_time_format(t_end1))
                        fname = os.path.join(datadir, fname0)
                        # print(fname)
                        # with open(fname, 'wb') as fp:
                        #     json.dumps({'LIRIS':record, 'Data':Measures0},  # record is the LIRIS information
                        #             fp)
                        with open(fname, 'wb') as fp:
                            pickle.dump({'LIRIS':record, 'Data':Measures0},  # record is the LIRIS information
                                        fp, protocol=pickle.HIGHEST_PROTOCOL)
        return nflag, Liris
    else:
        # print('result' in response0.keys())
        # print(response0['result'] == 'SUCCESS')
        # print(response0['data'])
        raise Exception('Failed response from the server or empty project.')


def to_valid_time_format(f):
    """Transform a datetime object to a string so that it can be used as a filename on Windows (and other systems).

    Examples:
        '2016-12-11 17:14:30' will be transformed to '2016-12-11-17H14M30S'
    """
    g = str(f).replace(' ', '-').replace(':','H', 1).replace(':','M', 1)+'S'#.replace('.','S', 1)
    return g


def assemble_to_pandas(projdir):
    """Assemble splitted data into a single pandas datasheet.

    Args:
        projdir (str): folder where the raw splitted data are stored. This folder is organized in sub folders named by the location key id.
    Returns:
        A pickle file named ``Raw.pkl`` in the given projdir.
    """
    pnames = glob.glob(os.path.join(projdir, '*'))

    if len(pnames) > 0 :
        Data = {}  # for raw data of pandas format, a dictionary, referred by location key id
        Static_Data = {}  # for raw static data of pandas format, a dictionary, referred by location key id
        Liris = {}

        for p in pnames:
            if os.path.isdir(p):  # for all sub-folders
                Sdic = []

                idx = p.rfind(os.path.sep) # path separator: '/' on Linux, '\' on Windows
                try:
                    loc = int(p[idx+1:])  # location key ID
                except:
                    continue

                fnames = glob.glob(os.path.join(p, 'Raw_*.pkl'))

                # assamble the splitted files
                if len(fnames)>0:
                    for f in fnames:
                        with open(f,'rb') as fp:
                            toto = pickle.load(fp)
                        Sdic += toto['Data']

                    Liris[loc] = LIRIS(**toto['LIRIS']) # object from dictionary
                    Data[loc] = raw2pandas(Sdic) # Convert to Pandas datasheet
                    # Static_Data[loc] = Data[loc][Data[loc].Type==1]

        with open(os.path.join(projdir,'Raw.pkl'), 'wb') as fp:
            # record is the LIRIS information
            pickle.dump({'LIRIS':Liris, 'Data':Data},
                        fp, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        raise Exception('Empty project.')


def raw2pandas(X):
    """Convert the raw OSMOS data to pandas format.

    Args:
        X: a list of tuples *(Time, Temperature, Elongation, Type)*. These are the data obtained directly from OSMOS server. Note that the field `Time` here is a string.
    Returns:
        a pandas datasheet with the fields 'Temperature', 'Elongation', 'Type'.
    """

    toto=pd.DataFrame(data=X, columns=['Time', 'Temperature', 'Elongation', 'Type'])
    ts=pd.to_datetime(toto.Time)  # convert from string to timestamps
    ts.name=None  # remove the name field

    x0=toto[['Temperature', 'Elongation', 'Type']]
    x0.index=ts  # change the index column to timestamps
    return x0


def static_data_preprocessing(X0,
                            sflag=False,
                            oflag=False,
                            jflag=False,
                            tflag=False,
                            nh=12):
    """Apply preprocessings routines on static data of a sensor.

    This function applies the following pre-processings (by order):

    1. remove wrong values due to the synchronisation problem (optional)
    2. remove obvious outliers (optional)
    3. resampling at a regular time step
    4. detect step jumps (optional)

    Args:
        X0 (pandas DataFrame): containing the field 'Temperature', 'Elongation', with the index being the DateTimeIndex object (not the integer index).
        sflag (bool): if True remove synchronization error
        oflag (bool): if True remove obvious outliers
        jflag (bool): if True apply step jumps detection
        tflag (bool): if True the temperature data will be processed also
        nh (int): gaps larger than nh hours will be marked as nan
    Returns:
        A pandas DataFrame sheet containing the field 'Temperature', 'Elongation' and 'Type', where 'Type' is a N-bit marker of the state of the current time-stamp

        * The first bit indicates if the value is original or interpolated
        * The second bit indicates that no original data exist in 1 hour around
        * The third bit indicates a jump of the value of deformation

        Example:
            * Type = 000: means the value of the current time-stamp is original
            * Type = 011: means the value of the current time-stamp is interpolated from points outside the 1 hour interval
    """

    Temp0 = np.asarray(X0['Temperature'], dtype=np.float64).copy() # temperature
    Elon0 = np.asarray(X0['Elongation'], dtype=np.float64).copy() # elongation
    MinDataLength = 24*60*60

    if X0.index[-1]-X0.index[0] >= datetime.timedelta(0,MinDataLength): # only for the length > sampling step
        # 1. Wrong values due to synchronisation
        # The values wsize=1, thresh=5. have been tuned for OSMOS data
        # (see projets 36, 38)
        if sflag:
            Elon0 = Tools.remove_plateau_jumps(Elon0, wsize=1, thresh=5., dratio=0.8, bflag=False)  # first correction using a small window
            Elon0 = Tools.remove_plateau_jumps(Elon0, wsize=24, thresh=20., dratio=0.8, bflag=False)  # second correction using a large window to remove residual errors
            if tflag:
                Temp0 = Tools.remove_plateau_jumps(Temp0, wsize=1, thresh=5., dratio=0.8, bflag=False)

        # 2. Outliers
        # The values wsize=10, thresh=10. have been tuned for OSMOS data
        # (see projets 46/200, 44/192, 76/369)
        if oflag:
            Elon0 = Tools.remove_plateau_jumps(Elon0, wsize=24, thresh=20., dratio=0.8, bflag=False)
            if tflag:
                Temp0 = Tools.remove_plateau_jumps(Temp0, wsize=24, thresh=5., dratio=0.8, bflag=False)

        # 3. Resampling
        toto = X0.copy()
        toto['Temperature'] = Temp0
        toto['Elongation'] = Elon0
        # toto = pd.DataFrame(data = {'Temperature':Temp0, 'Elongation':Elon0}, index=X0.index)
        X1, Rtsx, Ntsx = resampling_time_series(toto, rsp='H', m=nh)

        # 4. Jumps
        if jflag:
            Jidx = Tools.detect_step_jumps(X1['Elongation'], method='diff', thresh=8, mwsize=2, median=0.8)
        else:
            Jidx = []

        # Add the Type column
        Type1 = pd.Series(data=np.zeros(len(X1), dtype=int), index=X1.index)
        if len(Rtsx)>0:
            # Type1.loc[Rtsx]+=0b001  # loc since Rtsx are timestamp indexes
            for t in Rtsx:
                Type1[t] += 0b001
        if len(Ntsx)>0:
            # Type1.loc[Ntsx]+=0b010
            for t in Ntsx:
                Type1[t] += 0b010
        if len(Jidx)>0:
            # Type1.iloc[Jidx]+=0b100  # iloc since Jidx are integers
            for t in Jidx:
                Type1.iloc[t] += 0b100

        X1['Type']=Type1
        return X1
    else:
        raise Exception('Insufficient length of data (< {} seconds): from {} to {} ({} points)'.format(MinDataLength, X0.index[0], X0.index[1], len(X0)))


def resampling_time_series(X, rsp='H', m=6):
    """Resampling of time series with a regular step.

    Args:
        X (pandas DataFrame or Series): input time series, X.index must be object of DateTimeIndex.
        rsp (str): step of resampling, by default use 'H' which stands for 'hour'.
        m (int): resampling is considered as invalid if no original data exist in an interval of m points
    Returns:
        the following variables

        - S (pandas DataFrame or Series): resampled time series
        - Rtsx (pandas DateTimeIndex): timestamps of resampled points
        - Nidx: timestamps where no original observations exist on an interval of length m
    """
    Rtsx = pd.date_range(X.index[0].ceil(rsp), X.index[-1].floor(rsp), freq=rsp)

    Y = X.resample(rsp).mean().loc[Rtsx]
    Nblx = np.isnan(np.asarray(Y, dtype=float)).sum(axis=1)>0  # nan indicator
    Nblx = Tools.UL_filter_boundary(Nblx, m) # index where no original observations exist on an
    # Nidx = np.where(Tools.UL_filter_boundary(Nblx, m))[0] # index where no original observations exist on an interval of length m
    Ntsx = Rtsx[Nblx]

    toto = np.zeros((len(Rtsx),X.shape[1])) if X.ndim==2 else np.zeros(len(Rtsx))
    toto.fill(np.nan)

    if isinstance(X, pd.DataFrame):
        R=X.combine_first(pd.DataFrame(data=toto, index=Rtsx, columns=X.columns))
    elif isinstance(X, pd.Series):
        R=X.combine_first(pd.Series(data=toto, index=Rtsx))

    S = R.interpolate(method='slinear')  # linear interpolation may introduce artificial period

    # with the returned indexes one can select the subset of index to remove values due to interpolation on long intervals
    return S, Rtsx, Ntsx


def load_raw_data(fname, datatype='static'):
    """Load raw data from a given file.

    Note:
        Raw data are stored in pandas format and contain only three fields: 'Temperature', 'Elongation', and 'Type'.

    Args:
        fname (str): name of the pickle file containing assembled Osmos data
        datatype (str): type of data to be loaded, can be 'static', 'dynamic', or 'all'
    Returns:
        the following variables

        - Rdata (dict): raw data
        - Sdata (dict): static data, empty if datatype='dynamic'
        - Ddata (dict): dynamic data (elongation only, no temperature), empty if datatype='static'
        - Locations (list): location keyid of sensors
    """
    with open(fname, 'rb') as fp:
        toto = pickle.load(fp)

    try:
        Rdata = toto['Data']
        Liris = toto['LIRIS']
    except Exception:
        raise TypeError('{}: No raw data found in the input file.'.format(fname))

    Locations = list(Rdata.keys())
    Locations.sort()

    Sdata, Ddata = {}, {}
    for loc, val in Rdata.items():
        # Extraction of static data
        if datatype.upper() in ['STATIC', 'ALL']:
            Tidx1 = val['Type']==1
            Sdata[loc] = []
            if np.any(Tidx1): # if containing static data
                Sdata[loc] = val[Tidx1].drop('Type',1)  # Drop the column 'Type'
        # Extraction of dynamic data
        if datatype.upper() in ['DYNAMIC', 'ALL']:
            Tidx2 = val['Type']==2
            Ddata[loc] = []
            if np.any(Tidx2): # if containing dynamic data
                # 40000 microseconds = 0.04 second = 2 samples
                # dynamic data are sampled at 20 milliseconds (50Hz)
                Didx = Tools.time_findgap(val[Tidx2].index, dtuple=(0,0,20000))
                for sidx in np.split(np.arange(np.sum(Tidx2)), Didx):
                    Ddata[loc].append(val[Tidx2]['Elongation'].iloc[sidx]) # add the first event
        if datatype.upper() not in ['STATIC', 'DYNAMIC', 'ALL']:
            raise TypeError('{}: unknown type of data'.format(datatype))
    return Rdata, Sdata, Ddata, Locations


def is_transformed(fname):
    """Determine if the data in the given file is transformed
    """

    Data = load_json(fname)
    for k, val in Data.items():
        if 'Reference' in val:  # transformed data do not contain the field Reference
            break
    else:
        return True
    return False


def load_static_data(fname):
    """Load preprocessed static data from a given file.

    This function loads a pickle file that contains preprocessed static data in pandas format and
    extract the resampled values.

    Args:
        fname (str): name of the pickle file
    Returns:
        the following variables

        - Data (dict): static data of all sensors
        - Tall (pandas DataFrame): concatenated temperature of all sensors
        - Eall (pandas DataFrame): concatenated elongation of all sensors
        - Locations (list): location key IDs of sensors

    Note:

        - Preprocessed static data contain only three fields: [Temperature, Elongation, Type], which are identical to the raw data. In order to distinguish a pickle file of preprocessed data from that of raw data we can use the key 'LIRIS' which is present only in the raw data file (cf. :func:`assemble_to_pandas`) as indicator.
        - The outputs Tall, Eall may be longer than Data due to forced alignment.
    """

    if fname[fname.rfind('.')+1:].upper() == 'PKL':
        with open(fname, 'rb') as fp:
            toto = pickle.load(fp)
        try:
            Data0 = toto['Data']
        except Exception:
            raise TypeError('{}: No preprocessed data found in the input file.'.format(fname))

        if len(toto.keys()) > 1:
            # Unlike the raw data file, the preprocessed data file contains only one field.
            raise TypeError('{}: Not a valid file of preprocessed data.'.format(fname))
    elif fname[fname.rfind('.')+1:].upper() == 'JSON':
        Data0 = load_json(fname)

    return _load_static_data(Data0)


def load_json(fname):
    with open(fname, 'r') as fp:
        toto = json.load(fp)
    Data0 = {}
    for k,v in toto.items():
        Data0[int(k)] = pd.read_json(v)
    return Data0

def _load_static_data(Data0):
    """See :func:`load_static_data`
    """

    Locations = list(Data0.keys())
    Locations.sort()

    # Extract data of temperature and elongation
    Data = {}
    for loc, X0 in Data0.items():
        X = X0.copy()

        # index of interpolated values
        Rblx = np.asarray(list(map(lambda x:np.mod(x>>0, 2), X['Type'])), dtype=bool)
        # index of missing data
        Nblx = np.asarray(list(map(lambda x:np.mod(x>>1, 2), X['Type'])), dtype=bool)
        # index of jumps
        Jblx = np.asarray(list(map(lambda x:np.mod(x>>2, 2), X['Type'])), dtype=bool)
        Jblx = np.logical_and(Jblx, np.logical_not(Nblx))  # remove the artificial jumps of missing data

        X[Nblx] = np.nan
        X['Missing'] = Nblx
        X['Jump'] = Jblx

        Data[loc] = X[Rblx].copy()# [['Temperature', 'Elongation', 'Missing', 'Jump']]
        del Data[loc]['Type']

    Tall = concat_mts_rm_index(Data, 'Temperature')  # temperature of all sensors
    Eall = concat_mts_rm_index(Data, 'Elongation')  # elongation of all sensors

    return Data, Tall, Eall, Locations


def concat_mts_rm_index(Data, fd):
    """
    Args:
        fd (str): for example, 'Temperature'
    """
    X0 = concat_mts(Data, fd)
    # remove the name 'index' in the index column
    return pd.DataFrame(data=np.asarray(X0), columns=X0.columns, index=list(X0.index))


# def load_static_data_mdb(fname):
#     """Load preprocessed static data from a given file.
#     """

#     Data0 = load_json(fname)

#     Locations = list(Data0.keys())
#     Locations.sort()

#     # Extract data of temperature and elongation
#     Data = {}
#     for loc, X0 in Data0.items():
#         X = X0.copy()

#         # index of interpolated values
#         Rblx = np.asarray(list(map(lambda x:np.mod(x>>0, 2), X['Type'])), dtype=bool)
#         # index of missing data
#         Nblx = np.asarray(list(map(lambda x:np.mod(x>>1, 2), X['Type'])), dtype=bool)
#         # index of jumps
#         Jblx = np.asarray(list(map(lambda x:np.mod(x>>2, 2), X['Type'])), dtype=bool)
#         Jblx = np.logical_and(Jblx, np.logical_not(Nblx))  # remove the artificial jumps of missing data

#         X[Nblx] = np.nan
#         X['Missing'] = Nblx
#         X['Jump'] = Jblx

#         Data[loc] = X[Rblx].copy()# [['Temperature', 'Elongation', 'Missing', 'Jump']]
#         del Data[loc]['Type']

#     Tall = concat_mts_rm_index(Data, 'Temperature')
#     Eall = concat_mts_rm_index(Data, 'Elongation')
#     Rall = concat_mts_rm_index(Data, 'Reference')


#     return Data, Tall, Eall, Rall, Locations


def concat_mts(Data, field):
    """Concatenate multiple time series.

    Args:
        Data (dict): dictionary of pandas datasheet
        field (str): name of the field to be concatenated, e.g., 'Temperature' or 'Elongation'

    Returns:
        Concatenated datasheet.
    """
    import pandas

    Locations = list(Data.keys())
    Locations.sort()
    # toto = [Data[loc][field] for loc in Locations]
    toto = []
    for loc in Locations:
        v = Data[loc][field].copy()
        # remove duplicate indexes
        v = v.reset_index().drop_duplicates(subset='index', keep='last').set_index('index')
        # strangely, v may still contain the field name, convert it to Series using iloc
        toto.append(v.iloc[:,0])
    return pandas.concat(toto, axis=1, keys=Locations)


def trend_seasonal_decomp(X0, mwsize=24, method='mean', kzord=1, causal=False, luwsize=0):
    """Decompose a time series into trend and seasonal components.

    Args:
        X0: pandas DataFrame or 1d numpy array
        mwsize...causal: see :func:`Tools.KZ_filter`
        luwsize (int): size of LU filter window
    Returns:
        ..
        - Xtrd, Xsnl: trend and seasonal components
    """
    # if not (isinstance(X0, pd.DataFrame)):
    #     raise TypeError('Input array must be a pandas DataFrame')
    if not (isinstance(X0, pd.DataFrame) or isinstance(X0, pd.Series)):
        raise TypeError('Input array must be a pandas DataFrame or Series')

    Xtrd = X0.copy(); Xsnl = X0.copy()

    if isinstance(X0, pd.DataFrame):
        for idx in range(X0.shape[1]):
            # Obtain the trend component by moving average or median (KZ filter)
            xtrd = Tools.KZ_filter(X0.iloc[:,idx], mwsize, kzord, method=method, causal=causal)
            # Obtain the seasonal component as the difference
            xsnl = X0.iloc[:,idx] - xtrd

            # LU filter on the components
            if luwsize>0:
                Xtrd.iloc[:,idx] = Tools.LU_mean(np.asarray(xtrd),luwsize)
                Xsnl.iloc[:,idx] = Tools.LU_mean(np.asarray(xsnl),luwsize)
            else:
                Xtrd.iloc[:,idx] = xtrd
                Xsnl.iloc[:,idx] = xsnl
    else:
        xtrd = Tools.KZ_filter(X0, mwsize, kzord, method=method, causal=causal)
        # Obtain the seasonal component as the difference
        xsnl = X0 - xtrd

        # LU filter on the components
        if luwsize>0:
            Xtrd.iloc[:] = Tools.LU_mean(np.asarray(xtrd),luwsize)
            Xsnl.iloc[:] = Tools.LU_mean(np.asarray(xsnl),luwsize)
        else:
            Xtrd = xtrd
            Xsnl = xsnl

    return Xtrd, Xsnl


def common_time_range(X0):
    """Find the common time range of multiple time series.

    Args:
        X0 (dict): X0[loc] is the pandas datasheet of data of the sensor loc.
    Returns:
        ..
        - t0, t1: timestamps of the real time range.
    """
    t0 = np.max([v.index[0] for k, v in X0.items()])
    t1 = np.min([v.index[-1] for k, v in X0.items()])

    # for n, (loc, data) in enumerate(X0.items()):
    #     Xt = data.index.to_pydatetime()
    #
    #     if n==0:
    #         t0, t1 = Xt[0], Xt[-1]
    #     else:
    #         t0 = min(tm, Xt[0])
    #         tmax = max(tmax, Xt[-1])

        # if np.isnan(tmin) or np.isnan(tmax):
        #     raise ValueError('Wrong time range')

    return t0, t1


def truncate_static_data(fname, timerange):
    """Load and truncate static data with a given time range.

    Args:
        fname (str): name of the pickle file containing preprocessed static data
        timerange (tuple of str): starting and ending timestamp of the data
    Returns:
        ..
        - Tall, Eall, Midx: truncated temperature, elongation and indicator of missing values
    """
    # Load preprocessed static data
    Data0, Tall0, Eall0, Locations = load_static_data(fname)
    # indicator of missing data, NaN: not defined, True: missing data
    Midx0 = concat_mts(Data0, 'Missing')

    tidx0, tidx1 = timerange     # beginning and ending timestamps

    Tall = Tall0[tidx0:tidx1]
    Eall = Eall0[tidx0:tidx1]
    # indicator of missing values, the nan values due forced alignment of concat_mts() are casted as True
    Midx = Midx0[tidx0:tidx1].astype(bool)

    return Tall, Eall, Midx


def prepare_static_data(fname, timerange=(None,None), mwsize=24, kzord=1, method='mean', causal=False):
    """Prepare static data for further analysis.

    This function does the same thing as truncate_static_data(), moreover, it
    make decomposition of data into trend and seasonal components.

    Args:
        fname (str): name of the pickle file containing preprocessed static data
        mwsize...method: see trend_seasonal_decomp()
        timerange (tuple of str): starting and ending timestamp of the data
    Returns:
        the following variables

        - (Tall, Tsnl, Ttrd): truncated temperature, its seasonal and trend components
        - (Eall, Esnl, Etrd): truncated elongation, its seasonal and trend components
        - Midx (pandas DataFrame): indicator of missing values

    """
    # beginning and ending timestamps
    tidx0, tidx1 = timerange

    # Load preprocessed static data
    Data0, Tall0, Eall0, Locations = load_static_data(fname)
    # indicator of missing data, NaN: not defined, True: missing data
    Midx0 = concat_mts(Data0, 'Missing')

    # Decomposition of signals
    Ttrd0, Tsnl0 = trend_seasonal_decomp(Tall0, mwsize=mwsize, kzord=kzord, method=method, causal=causal)
    Etrd0, Esnl0 = trend_seasonal_decomp(Eall0, mwsize=mwsize, kzord=kzord, method=method, causal=causal)

    # Data truncation
    Ttrd = Ttrd0[tidx0:tidx1]
    Tsnl = Tsnl0[tidx0:tidx1]
    Etrd = Etrd0[tidx0:tidx1]
    Esnl = Esnl0[tidx0:tidx1]
    Tall = Tall0[tidx0:tidx1]
    Eall = Eall0[tidx0:tidx1]
    # indicator of missing values, the nan values due forced alignment of concat_mts() are casted as True
    Midx = Midx0[tidx0:tidx1].astype(bool)

    # for loc, x in Tsnl.items():
    #     if x.std()<0.1:
    #         warnings.warn("Location {}: No significant seasonal component detected.".format(loc))
    #         # raise ValueError("No significant seasonal component detected.")

    return (Tall, Tsnl, Ttrd), (Eall, Esnl, Etrd), Midx


# def prepare_static_data_mdb(fname, timerange=(None,None)):
#     """Prepare static data of MongoDB for further analysis.

#     Args:
#         fname (str): name of the pickle file containing preprocessed static data
#         timerange (tuple of str): starting and ending timestamp of the data
#     Returns:
#         Tall, Eall, Rall, PLall, Midx: truncated temperature, elongation, reference, parameters and indicator of missing values
#     """
#     # beginning and ending timestamps
#     tidx0, tidx1 = timerange

#     # Load preprocessed static data
#     Data0, Tall0, Eall0, Rall0, Pall0, Locations = load_static_data_mdb(fname)

#     # indicator of missing data, NaN: not defined, True: missing data
#     Midx0 = concat_mts(Data0, 'Missing')

#     # Data truncation
#     Tall = Tall0[tidx0:tidx1]
#     Eall = Eall0[tidx0:tidx1]
#     Rall = Rall0[tidx0:tidx1]
#     Pall = {k:v[tidx0:tidx1] for k,v in Pall0.items()}
#     # indicator of missing values, the nan values due forced alignment of concat_mts() are casted as True
#     Midx = Midx0[tidx0:tidx1].astype(bool)

#     return Tall, Eall, Rall, Pall, Midx


#### MongoDB related ####
# transform table for temperature
tfAbaqueTemp = np.asarray([
    [243448, -7.97753526070585E-05, -20.5788499585168],
    [180772, -0.000110744424018251, -14.9805089813728],
    [135623, -0.00015210513506936, -9.3710452664882],
    [102751, -0.000206825232678387, -3.74850051706308],
    [78576, -0.000278504985239236, 1.88380772015819],
    [60623, -0.000371609067261241, 7.52805648457823],
    [47168, -0.000491497100167109, 13.1829352206822],
    [36995, -0.000644745325596389, 18.8523533204384],
    [29240, -0.000838926174496644, 24.5302013422819],
    [23280, -0.00108318890814558, 30.2166377816291],
    [18664, -0.00138888888888889, 35.9222222222222],
    [15064, -0.00176803394625177, 41.6336633663366],
    [12236, -0.00223613595706619, 47.3613595706619],
    [10000, -0.00280946226892173, 53.0946226892173],
    [8220.3, -0.00350852571749351, 58.8411339555119],
    [6795.2, -0.00435578012021953, 64.5983970729158],
    [5647.3, -0.00537750053775005, 70.3683587868359],
    [4717.5, -0.00660327522451136, 76.1509508716323],
    [3960.3, -0.00806581706726891, 81.9430553315051],
    [3340.4, -0.00980199960792002, 87.742599490296],
    [2830.3, -0.0118567702157932, 93.5582167417595],
    [2408.6, -0.0142775556824672, 99.3889206167904],
    [2058.4, -0.0171115674195756, 105.222450376454],
    [1766.2, -0.0204248366013072, 111.074346405229],
    [1521.4, -0.0242718446601942, 116.927184466019],
    [1315.4, -0.028735632183908, 122.798850574713],
    [1141.4, -0.033900603430741, 128.694148755848],
    [993.91, -0.0398215992354253, 134.579085696082],
    [868.35, -0.0466243938828795, 140.486292428198],
    [761.11, -0.0543951261966928, 146.400674499565],
    [669.19, -0.0632511068943706, 152.327008222644],
    [590.14, -0.0733137829912024, 158.265395894428],
    [521.94, -0.0847170450694679, 164.217214503558],
    [462.92, -0.0975800156128025, 170.171740827479],
    [411.68, -0.112107623318386, 176.152466367713],
    [367.08, -0.128402670775552, 182.13405238829],
    [328.14, -0.146670577882077, 188.128483426225],
    [294.05, -0.167056465085199, 194.122953558303],
    # [264.12, 0, 0],  # <--- this may create discontinuties
    ]
)

# def raw2celsuis(raw, T, round05_flag=True, nbdec=5):
#     """Convert raw temperature to celsuis.

#     Args:
#         raw (float): raw value
#         T (2d array): a table
#     """

#     v0 = 10000*raw/(1023-raw)

# #     idx = np.sum(v0 <= T[:,0])-1
# #     v1 = T[idx,1]*v0 + T[idx,2] if idx>=0 else 0
#     idx = max(np.sum(v0 <= T[:,0])-1, 0)
#     v1 = T[idx,1]*v0 + T[idx,2]

#     return np.floor(v1*2)/2 if round05_flag else np.round(v1, decimals=nbdec)


def _raw2celsuis_scalar(raw):
    T = tfAbaqueTemp
    try:
        v0 = 10000. * raw / (1023. - raw)
        idx = max(np.sum(v0 <= T[:,0])-1, 0)  # values outside the range [294.05, 243448] will be shrinked.
        v1 = T[idx,1]*v0 + T[idx,2]
    except ZeroDivisionError:
        v1 = np.nan
    return v1

_raw2celsuis = np.vectorize(_raw2celsuis_scalar)  # vectorized version

def raw2celsuis(V):
    """Convert raw temperature to celsuis.

    Args:
        V (pandas DataFrame): V[loc] is the raw temperature of loc
    Returns:
        a pandas DataFrame of temperature in celsuis
    """
    toto = _raw2celsuis(np.asarray(V).T)
    return pd.DataFrame(toto.T, columns=V.columns, index=V.index)


# def _raw2millimeters(x, a, b, c):
#     print(a,b,c)
#     return (a*x**2 + b*x + c)/1000. - 2.

def raw2millimeters(V, R, Parms):
    """Convert raw elongation to millimeters.

    Args:
        V (pandas DataFrame): V[loc] is the raw elongation of the location loc
        R (pandas DataFrame): R[loc] is the reference
        Parms (dict): Parms[loc] is the tuple of parameters (a, b, c)
    Returns:
        a pandas DataFrame of elongation in millimeters
    """
    toto={}
    X = V/R
    for loc in V.keys():  # iteration on V since Parms may contain more locations
        # print(loc)
        # print(X[loc].head())
        x = np.asarray(X[loc])
        a,b,c = Parms[loc]
        toto[loc] = (a*x**2 + b*x + c)/1000 - 2
    return pd.DataFrame(toto, index=V.index)


def _mongo_transform(X):
    """
    Args:
        X (dict): containing the fields `['data', 'uid', 'location', 'paramb', 'start', 'paramc', 'year', 'day', 'parama', 'month', '_id', 'newdoc', 'type']`. Typically this is one document in the list of documents returned by collection.find({'uid':?, 'type':?}).
    Return:
        a pandas DataFrame containing raw and transformed data.
    """
    # first put data in pandas format
    P = pd.DataFrame(X['data'])
    m0 = np.asarray(P['measure'])
    t0 = np.asarray(P['temperature'])
    # r0 = np.nan * np.zeros(len(P)) if not 'reference' in P else np.asarray(P['reference'])
    r0 = np.asarray(P['reference'])

    if np.any(np.abs(r0) == 0):
        raise ValueError("Zero in the reference values detected! Transformation of raw data failed.")

    # transform of 'temperature'
    temp = _raw2celsuis(t0)

    # transform of 'measure'
    a, b, c = X['parama'], X['paramb'], X['paramc']
    v0 = m0/r0
    elon = (a*v0**2 + b*v0 + c)/1000 - 2

#     # Old version: abandonned for simplification
#     # Singular case:  check parameters
#     a = np.nan if (not 'parama' in X) else X['parama']
#     b = np.nan if (not 'paramb' in X) else X['paramb']
#     c = np.nan if (not 'paramc' in X) else X['paramc']

#     if np.isnan(a) or np.isnan(b) or np.isnan(c):  # if any of these parameters is missing, use directly the field 'measure' without transform
#         elon = m0
# #                 temp = np.asarray(t0)
#     else:
#         if not 'reference' in P:  # raise error if 'reference' is missing
#             raise TypeError('location: {}, _id: {}: missing the field reference'.format(X['location'], X['_id']))

#         mb = np.abs(m0) > 0
#         rb = np.abs(r0) == 0

#         if np.any(np.logical_and(rb, mb)):  # if mesure/ref are not defined
#             raise ValueError('location: {}, _id: {}, reference value close to zero.'.format(X['location'], X['_id']))

#         v0 = np.zeros(len(P))
#         v0[~rb] = m0[~rb]/r0[~rb]
#         v0[rb] = m0[rb]  # on the entries that reference==0 we use 'measure' directly

#         v0 = m0/r0
#         elon = (a*v0**2 + b*v0 + c)/1000 - 2

    return pd.DataFrame({'ElongationTfm':elon, 'TemperatureTfm':temp, 'ElongationRaw':m0, 'TemperatureRaw':t0, 'Reference':r0, 'parama':a, 'paramb':b, 'paramc':c}, index=P['date']).sort_index()


def mongo_load_static(C, loc, dflag=True):
    """Extract raw static data from a collection of MongoDB and apply transformation

    The transformation applied on temperature and on elongation are defined in the function :func:raw2celsuis and :func:raw2millimeters.

    Args:
        C: collection of MongoDB
        loc (int): location key ID
        dflag (bool): if True remove duplicate entries
    Return:
        Pandas DataFrame with 'date' as index, and containing the fields 'Temperature' (in celsuis) and 'Elongation' (in millimeter).
    """
    # get raw data
    rawdata = C.find({'location':loc, 'type':1})
    # normally, rawdata is a list-like structure and rawdata[0] contains the fields
    # `['data', 'uid', 'location', 'paramb', 'start', 'paramc', 'year', 'day', 'parama', 'month', '_id', 'newdoc', 'type']`
    # but singular cases may exist and must be handled.

    L = []
    for X in rawdata:
        if len(X['data'])>0: # Singular case: check that data exist
            try:
                p0 = _mongo_transform(X)
            except Exception as msg:
                print(msg)
                continue
            # p0['newdoc'] = X['newdoc']
            # p0['location'] = X['location']
            L.append(p0)

    if len(L)>0:
        #         P = pd.concat(L).sort_values('date').reset_index(drop=True)
        P = pd.concat(L).sort_index() #.reset_index(drop=True)
        # remove duplicate time index
        return P.groupby(['date']).agg(np.mean) if dflag else P
    else:
        return []


def mongo_load_dynamic(C, uid):
    """Extract raw dynamic data from a collection of MongoDB and apply transformation

    The transformation applied on temperature and on elongation are defined in the function :func:raw2celsuis and :func:raw2millimeters.

    Args:
        C: collection of MongoDB
        uid (str): sensor ID
    Return:
        List of dynamic events in Pandas DataFrame format with 'date' as index, and containing the fields 'Temperature' (in celsuis) and 'Elongation' (in millimeter).
    """
    # get raw data
    rawdata = C.find({'uid':uid, 'type':2})
    # normally, rawdata is a list-like structure and rawdata[0] contains the fields
    # `['data', 'uid', 'location', 'paramb', 'start', 'paramc', 'year', 'day', 'parama', 'month', '_id', 'newdoc', 'type']`
    # but singular cases may exist and must be handled.

    L = {}
    for X in rawdata:
        if len(X['data'])>0: # Singular case: check that data exist
            p0 = _mongo_transform(X)
            p0['newdoc'] = X['newdoc']
            p0['location'] = X['location']
            L[X['start']] = p0

    # L is unordered. Sort in chronological order
    return [v for k, v in collections.OrderedDict(sorted(L.items())).items()]

