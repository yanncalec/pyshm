"""Collection of functions related to OSMOS data I/O
"""

import json
import dateutil, datetime
import os, glob, pickle

import numpy as np
# import numpy.linalg as la
import scipy

import pandas as pd
import colorama
# from colorama import Fore, Back, Style

from . import Tools
# import Tools

# Some nomenclature rules:
# dbdir: database directory
# projdir: project directory
# datadir: location directory

class LIRIS:
    """Class for describing a LIRIS object
    """
    def __init__(self, **entries):
        self.__dict__.update(entries)


def update_LIRIS_data_by_project(token, session, PID, projdir, endtime=None, verbose=0):
    """Update the database (by requesting from the OSMOS's server).

    Args:
        token, session: see Download_data.py.
        PID (int): project key id.
        projdir (str): project directory.
        endtime (str): fetch data til this time, if not given fetch til today.
        verbose (int): print message.
    Returns:
        A bool variable indicating new data.
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
                        with open(fname, 'wb') as fp:
                            pickle.dump({'LIRIS':record, 'Data':Measures0},  # record is the LIRIS information
                                        fp, protocol=pickle.HIGHEST_PROTOCOL)
        return nflag, Liris
    else:
        raise Exception('Failed response from the server or empty project.')


def to_valid_time_format(f):
    """Transform a datetime object to a string which can be used as a filename on Windows (and other systems).

    Example:
    '2016-12-11 17:14:30' is transformed to '2016-12-11-17H14M30S'
    """
    g = str(f).replace(' ', '-').replace(':','H', 1).replace(':','M', 1)+'S'#.replace('.','S', 1)
    return g


def assemble_to_pandas(projdir):
    """Assemble splitted data into a single pandas datasheet.

    Args:
        projdir (string): folder where the raw splitted OSMOS data are stored. This folder is organized in sub folders named by the location key id.
    Returns:
        A pickle file named Raw.pkl in the given projdir.
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


def raw2pandas(Rawdata):
    """Convert the raw OSMOS data to pandas format.

    Args:
        Rawdata: a list of tuples (Time, Temperature, Elongation , Type). These are the data obtained directly from OSMOS server, and Time is a string.

    Returns:
        Data: a pandas datasheet with the fields 'Temperature', 'Elongation', 'Type'.
    """

    toto=pd.DataFrame(data=Rawdata, columns=['Time', 'Temperature', 'Elongation', 'Type'])
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
    - remove wrong values due to the synchronisation problem (optional)
    - remove obvious outliers (optional)
    - resampling at a regular time step
    - detect step jumps (optional)

    Args:
        X0 (pandas DataFrame): containing the field 'Time', 'Temperature', 'Elongation'.
        sflag (bool): if True remove synchronization error
        oflag (bool): if True remove obvious outliers
        jflag (bool): if True apply step jumps detection
        tflag (bool): if True the temperature data will be processed also
        nh (int): gaps larger than nh hours will be marked as nan
    Returns:
        A pandas DataFrame sheet containing the field 'Temperature', 'Elongation', 'Type'.
        'Type' is a N-bit marker of the state of the current time-stamp:
            The first bit indicates if the value is original or interpolated
            The second bit indicates that no original data exist in 1 hour around
            The third bit indicates a jump of the value of deformation
        for example,
            Type = 000: means the value of the current time-stamp is original
            Type = 011: means the value of the current time-stamp is interpolated from points outside the 1 hour interval
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
        toto = pd.DataFrame(data = {'Temperature':Temp0, 'Elongation':Elon0}, index=X0.index)
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
        X (pandas DataFrame or Series): input time series, X.index must be object of DateTimeIndex
        rsp (string): step of resampling, by default use 'H' which stands for 'hour'.
        m (int): resampling is considered as invalid if no original data exist in an interval of m points
    Returns:
        S (pandas DataFrame or Series): resampled time series
        Rtsx (pandas DateTimeIndex): timestamps of resampled points
        Nidx: timestamps where no original observations exist on an interval of length m
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


def load_raw_data(fname, staticonly=True):
    """Load raw data from a given file.

    Raw data are in pandas format and containing only three fields: Temperature,
    Elongation, Type.

    Args:
        fname (string): pickle file name containing assembled Osmos data
    Returns:
        Rdata (dict): raw data
        Sdata (dict): static data
        Ddata (dict): dynamic data (elongation only, no temperature)
        Locations (list): location keyid of sensors

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
    for loc, Data in Rdata.items():
        Tidx1 = Data.Type==1
        toto = Data[Tidx1] if np.any(Tidx1) else []
        Sdata[loc] = toto.drop('Type',1)  # Drop the column 'Type'
        # Sdata[loc] = toto

        # Extraction of dynamic data
        if not staticonly:
            Tidx2 = Data.Type==2
            Ddata[loc] = []
            if np.any(Tidx2): # if containing dynamic data
                Didx = Tools.time_findgap(Data[Tidx2].index.to_pydatetime(), dtuple=(0,0,40000))
                Didx = np.int32(np.hstack([0, Didx, len(Tidx2)]))
                for s in range(len(Didx)-1):
                    Ddata[loc].append(Data[Tidx2]['Elongation'].iloc[Didx[s]:Didx[s+1]]) # add the first event

    return Rdata, Sdata, Ddata, Locations


def load_static_data(fname):
    """Load preprocessed static data from a given file.

    This function loads a pickle file that contains preprocessed static data in pandas format and
    extract the regulaly resampled values.

    Args:
        fname (string): name of the pickle file
    Returns:
        Data (dict): static data of all sensors
        Tall (pandas DataFrame): concatenated temperature of all sensors
        Eall (pandas DataFrame): concatenated elongation of all sensors
        Locations (list): location key IDs of sensors

    Remark:
        1. Preprocessed static data contain only three fields: [Temperature,
        Elongation, Type], which are identical to the raw data. In order to
        distinguish a pickle from of preprocessed data from that of raw data we
        can use the key 'LIRIS' which is present only in the raw data file (cf. assemble_to_pandas())
        as indicator.
        2. The outputs Tall, Eall may be longer than Data due to forced alignment.
    """

    with open(fname, 'rb') as fp:
        toto = pickle.load(fp)

    try:
        Data0 = toto['Data']
    except Exception:
        raise TypeError('{}: No preprocessed data found in the input file.'.format(fname))

    if len(toto.keys()) > 1:
        # Unlike the raw data file, the preprocessed data file contains only one field.
        raise TypeError('{}: Not a valid file of preprocessed data.'.format(fname))

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

        Data[loc] = X[Rblx][['Temperature', 'Elongation', 'Missing', 'Jump']]

    # Tall = concat_mts(Data, 'Temperature')  # temperature of all sensors
    # Eall = concat_mts(Data, 'Elongation')  # elongation of all sensors

    Tall0 = concat_mts(Data, 'Temperature')  # temperature of all sensors
    Eall0 = concat_mts(Data, 'Elongation')  # elongation of all sensors
    # remove the name 'index' in the index column
    Tall = pd.DataFrame(data=np.asarray(Tall0), columns=Tall0.columns, index=list(Tall0.index))
    Eall = pd.DataFrame(data=np.asarray(Eall0), columns=Eall0.columns, index=list(Eall0.index))

    return Data, Tall, Eall, Locations


def concat_mts(Data, field):
    """Concatenate multiple time series.

    Args:
        Data (dict): dictionary of pandas datasheet
        field (string): e.g., 'Temperature' or 'Elongation'

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
        mwsize...causal: see Tools.KZ_filter()
        luwsize (int): size of LU filter window
    Returns:
        Xtrd, Xsnl: trend and seasonal components
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
        t0, t1: timestamps of the real time range.
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
        Tall, Eall, Midx: truncated temperature, elongation and indicator of missing values
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
        (Tall, Tsnl, Ttrd): truncated temperature, its seasonal and trend components
        (Eall, Esnl, Etrd): truncated elongation, its seasonal and trend components
        Midx (pandas DataFrame): indicator of missing values

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


#### obsolete
def remove_close_samples(X, dT=20*1000):
    """ Remove from a time series the samples of timestamps closer than dT.

    Args:
        X (pandas DataFrame of Series): input time series, X.index must be object of DateTimeIndex
        dT (int): in micro second
    """
    Tidx = X.index.to_pydatetime() # convert to python date time
    cflag = np.diff(Tidx) > datetime.timedelta(0, 0, dT)
    Didx = np.where(cflag)[0]+1 # choice of threshold
#     Didx = np.hstack((Didx[0]-1, Didx))
    return X.iloc[Didx]

def pandas2list(X0):
    """Convert a pandas DataFrame to list.
    """
    if isinstance(X0, list):
        T0 = [x.Time.tolist() for x in X0]
        Xt0 = [x.Temperature.tolist() for x in X0]
        Xe0 = [x.Elongation.tolist() for x in X0]

        return Xt0, Xe0, T0
    else:
        return X0.Temperature.tolist(),  X0.Elongation.tolist(),  X0.Time.tolist()

def choose_component(Data0, cnames):
    """Select specific component from a pandas datasheet.

    This function applies on the data genenrated by OSMOS_pkg.Decomposition_of_static_data(), and is used by Analysis_of_static_data_ARX().
    """
    idx = cnames.find('-')
    if idx<0:
        raise Exception('Unrecognized component string')

    componentx = cnames[:idx]
    componenty = cnames[idx+1:]

    if componentx == 'All':
        Xraw = Data0['Temperature'].copy()
    elif componentx == 'AllDiff':
        Xraw = Data0['Temperature'].diff()
    elif componentx == 'Seasonal':
        Xraw = Data0['Temperature_seasonal'].copy()
    elif componentx == 'SeasonalDiff':
        Xraw = Data0['Temperature_seasonal'].diff()
    elif componentx == 'Trend':
        Xraw = Data0['Temperature_trend'].copy()
    elif componentx == 'TrendDiff':
        Xraw = Data0['Temperature_trend'].diff()
    else:
        raise NotImplementedError('Unknown type of component: {}'.format(componentx))

    if componenty == 'All':
        Yraw = Data0['Elongation'].copy()
    elif componenty == 'AllDiff':
        Yraw = Data0['Elongation'].diff()
    elif componenty == 'Seasonal':
        Yraw = Data0['Elongation_seasonal'].copy()
    elif componenty == 'SeasonalDiff':
        Yraw = Data0['Elongation_seasonal'].diff()
    elif componenty == 'Trend':
        Yraw = Data0['Elongation_trend'].copy()
    elif componenty == 'TrendDiff':
        Yraw = Data0['Elongation_trend'].diff()
    else:
        raise NotImplementedError('Unknown type of component: {}'.format(componenty))

    return Xraw, Yraw
