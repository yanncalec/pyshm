"""
.. module:: OSMOS
   :platform: Unix, Windows, Mac
   :synopsis: resume
"""

import json
import dateutil
import datetime
import os, glob, pickle

import numpy as np
from numpy import newaxis, mean, sqrt, zeros, zeros_like, squeeze, asarray, linspace
from numpy.linalg import norm, svd, inv

import scipy
from scipy import interpolate

import pandas as pd

from colorama import Fore, Back, Style

class LIRIS:
    """Class for describing a LIRIS object
    """
    def __init__(self, **entries):
        self.__dict__.update(entries)


def update_LIRIS_data_by_project(token, session, PID, pname, verbose=0):
    """
    Update the data downloaded from the OSMOS's server.

    Parameters
    ----------

    token, session : see Update_Osmos_data script
    PID : project key id
    pname : folder name for downloaded data
    verbose : print message
    """
    payload = {"token": token,
               "action": "getlirisfromproject",
               "data": {"projectkeyid": PID}}
    r = session.post("https://client.osmos-group.com/server/application.php",
                     data=json.dumps(payload))
    response0 = json.loads(r.text)

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

            if liris.locationkeyid is not None: # if the location is valid
                if verbose > 0:
                    print('Location ID: {}, Location name: {}, LIRIS ID: {}, LIRIS name: {}'.
                          format(liris.locationkeyid, liris.locationname, liris.liriskeyid, liris.name))

                loc = liris.locationkeyid  # location
                t_start0 = dateutil.parser.parse(liris.datestart)  # official starting time
                t_end0 = dateutil.parser.parse(liris.dateend)  # official ending time
                if verbose > 1:
                    print("Official operation period: from {} to {}".format(t_start0, t_end0))

                datadir = pname+'/{}'.format(loc) # Directory for saving the data of this location
                Flist = glob.glob(datadir+'/Raw_*.pkl'.format(liris.locationkeyid)) # already sorted in increasing order
                if len(Flist) > 0:
                    fname = Flist[-1]
                    idx = fname.rfind('[')
                    t_start0 = dateutil.parser.parse(fname[idx+5:idx+24])
                    t_start0 += datetime.timedelta(0, 1) # new start time is 1s after

                Measures0 = []
                t0 = t_start0

                while t0 < min(datetime.datetime.today(), t_end0): # for a valide time range
                    # increment by 30 days, which is the capacity of the Osmos server
                    timeinterval = [t0, t0+datetime.timedelta(30,0)]

                    if verbose > 1:
                        print("Downloading the data of the period from {} to {}".
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

                if len(Measures0) > 0:
                    # Retrieve the physical period of the new data
                    t_start1 = dateutil.parser.parse(Measures0[0][0])
                    t_end1 = dateutil.parser.parse(Measures0[-1][0])

                    if verbose > 1:
                        print('Updated data: from {} to {}'.format(t_start1, t_end1))

                    # Save data only when containing more than one day of data
                    if t_end1-t_start1 > datetime.timedelta(1):
                        try:
                            os.makedirs(datadir)
                        except OSError:
                            pass

                        fname = datadir+'/Raw_[Start_{}]_[End_{}]].pkl'.format(t_start1, t_end1)
                        with open(fname, 'wb') as fp:
                            pickle.dump({'LIRIS':record, 'Data':Measures0},  # record is the LIRIS information
                                            fp, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        raise Exception('Failed response from the server or empty project.')


def assemble_to_pandas(datadir):
    """
    Assemble splitted OSMOS data (of one location) into a single pandas data sheet.

    The output is a pickle file named Raw_Latest.pkl in the given datadir.

    Paramters
    ---------
    datadir : string
        name of the folder where the raw splitted OSMOS data are stored. This folder
        is organized in sub folders named by the location key id.
    """
    pnames = glob.glob(datadir+'/*')

    if len(pnames) > 0 :
        Data = {}  # for raw data of pandas format, a dictionary, referred by location key id
        Static_Data = {}  # for raw static data of pandas format, a dictionary, referred by location key id
        Liris = {}

        for p in pnames:
            if os.path.isdir(p):
                Sdic = []

                idx = p.rfind('/')
                try:
                    loc = int(p[idx+1:])  # location key ID
                except:
                    continue

                fnames = glob.glob(p+'/Raw_*.pkl')

                # assamble the splitted files
                if len(fnames)>0:
                    for f in fnames:
                        with open(f,'rb') as fp:
                            toto = pickle.load(fp)
                        Sdic += toto['Data']

                    Liris[loc] = LIRIS(**toto['LIRIS']) # object from dictionary
                    Data[loc] = raw2pandas(Sdic) # Convert to Pandas datasheet
                    Static_Data[loc] = Data[loc][Data[loc].Type==1]

        with open(datadir+'/Raw.pkl', 'wb') as fp:
            pickle.dump({'LIRIS':Liris, 'Data':Data, 'Static_Data':Static_Data},  # record is the LIRIS information
                        fp, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        raise Exception('Empty project.')


def raw2pandas(Rawdata):
    """
    Convert the raw data to pandas format.

    Parameters
    ----------
    Rawdata: a list of tuples (Time, Temperature, Elongation , Type)
        These are the data obtained directly from OSMOS server, and
        typically Time is a string.

    Return
    ------
    Data: a pandas datasheet with the fields 'Temperature', 'Elongation', 'Type'.
    """

    toto=pd.DataFrame(data=Rawdata, columns=['Time', 'Temperature', 'Elongation', 'Type'])
    ts=pd.to_datetime(toto.Time)  # convert from string to timestamps
    ts.name=None  # remove the name field

    x0=toto[['Temperature', 'Elongation', 'Type']]
    x0.index=ts  # change the index column to timestamps
    return x0.copy()


def real_time_range(X0):
    """
    Find the physical time range in a pandas datasheet containing multiple locations.

    Parameters
    ----------
    X0: a dictionary, X0[loc] is the pandas datasheet of the location loc.
    """
    for n, (loc, data) in enumerate(X0.items()):
        Xt = data.index.to_pydatetime()

        if n==0:
            tmin, tmax = Xt[0], Xt[-1]
        else:
            tmin = min(tmin, Xt[0])
            tmax = max(tmax, Xt[-1])

        # if np.isnan(tmin) or np.isnan(tmax):
        #     raise ValueError('Wrong time range')

    return tmin, tmax


def pandas2list(X0):
    """
    Convert a pandas DataFrame to list
    """
    if isinstance(X0, list):
        T0 = [x.Time.tolist() for x in X0]
        Xt0 = [x.Temperature.tolist() for x in X0]
        Xe0 = [x.Elongation.tolist() for x in X0]

        return Xt0, Xe0, T0
    else:
        return X0.Temperature.tolist(),  X0.Elongation.tolist(),  X0.Time.tolist()


def static_data_preprocessing(X0, dT=60*60,
                              dflag=True,
                              sflag=True,
                              oflag=True,
                              fflag=True,
                              rflag=True,
                              tflag=False,
                              jflag=True):
    """
    On a continuous bunch of static data, apply a series of pre-processing including:
    - remove (possible) presence of dynamic data
    - remove wrong values due to the synchronisation problem
    - remove obvious outliers
    - complete missing data
    - resample the time series at a regular time step
    - detect step jumps (optional)

    Parameters
    ----------
    X0 : pandas datasheet
        containing the field Time, Temperature, Elongation
    dT : integer
        regular sampling time step in second
    dflag : boolean
        if True remove possible dynamic data
    sflag : boolean
        if True remove synchronization error
    oflag : boolean
        if True remove obvious outliers
    fflag : boolean
        if True fill missing data of length < 12 hours
    rflag : boolean
        if True resampling with the step = 1h
    tflag : boolean
        if False the temperature data will not be processed
    jflag : boolean
        if True apply step jumps detection
    """

    # Index0 = X0.index.copy() # make a copy of timeseries index

    Time0 = X0.index.to_pydatetime().copy() # time-stamp
    Temp0 = np.asarray(X0.Temperature.tolist(), dtype=np.float64).copy() # temperature
    Elon0 = np.asarray(X0.Elongation.tolist(), dtype=np.float64).copy() # elongation
    nbTime = len(Time0) # number of observations

    dT_dynamic = 20000 # dynamic data sampling step in microsecond

    if Time0[-1]-Time0[0] >= datetime.timedelta(0,dT): # only for the length > sampling step
        # 1. Remove the presence of any possible dynamic data
        # The method we use here is quite simple and can be improved
        if dflag:
            Didx = np.where(np.diff(Time0) >= 2*datetime.timedelta(0, 0, dT_dynamic))[0]+1 # choice of threshold
            Didx = np.hstack((Didx[0]-1, Didx))
            Time0 = Time0[Didx] #[Time0[n] for n in Didx]
            Temp0 = Temp0[Didx]
            Elon0 = Elon0[Didx]

        # 2. Remove the wrong values due to synchronisation
        # The values wsize=1, thresh=5. have been tuned for OSMOS data
        # (see projets 36, 38)
        if sflag:
            Elon0 = Pyshm.Tools.remove_plateau_jumps(Elon0, wsize=1, thresh=5., dratio=0.5, bflag=False)
            if tflag:
                Temp0 = Pyshm.Tools.remove_plateau_jumps(Temp0, wsize=1, thresh=5., dratio=0.5, bflag=False)

        # 3. Remove obvious outliers
        # The values wsize=10, thresh=10. have been tuned for OSMOS data
        # (see projets 46/200, 44/192, 76/369)
        if oflag:
            Elon0 = Pyshm.Tools.remove_plateau_jumps(Elon0, wsize=10, thresh=10., dratio=0.5, bflag=False)
            if tflag:
                Temp0 = Pyshm.Tools.remove_plateau_jumps(Temp0, wsize=10, thresh=10., dratio=0.5, bflag=False)

        # 4. Completion of missing data (12h>= gaps > 2h in Time0) by Kriging
        if fflag:
            # cond1 = datetime.timedelta(0,12*60*60) >= np.diff(Time0) > datetime.timedelta(0,2*60*60)
            cond1 = np.diff(Time0) > datetime.timedelta(0,2*60*60)
            midx = np.where(cond1)[0]

            Time_list, Temp_list, Elon_list = [], [], []
            for n,p in enumerate(midx):
                # take the four days data around p
                # p0<pa<pb<p1
                p0, p1 = max(0,p-24*2), min(p+24*2,nbTime) # outter range
                pa, pb = max(0,p-2), min(p+2,nbTime) # inner range of missing data

                xobs = Pyshm.Tools.time2second(np.hstack([Time0[p0:pa], Time0[pb:p1]]), Time0[p0]) # position of observation
                N = max(int((Time0[pb]-Time0[pa])/datetime.timedelta(0,60*60,0)), 1) # number of points on the missing interval
                toto = Pyshm.Tools.time_linspace(Time0[pa], Time0[pb], N)
                xpred = Pyshm.Tools.time2second(toto, Time0[p0]) # position of prediction

                Time_list += toto

                # apply kriging
                tpred,_ = Pyshm.Tools.gp_interpl(xobs, np.hstack([Temp0[p0:pa], Temp0[pb:p1]]),
                                                xpred, nugget=1e-9)
                epred,_ = Pyshm.Tools.gp_interpl(xobs, np.hstack([Elon0[p0:pa], Elon0[pb:p1]]),
                                                xpred, nugget=1e-9)

                Temp_list += list(tpred)
                Elon_list += list(epred)

            Time0 = np.hstack([Time0, Time_list])
            Temp0 = np.hstack([Temp0, Temp_list])
            Elon0 = np.hstack([Elon0, Elon_list])
            sidx = np.argsort(Time0)
            Time0 = Time0[sidx]; Temp0 = Temp0[sidx]; Elon0 = Elon0[sidx]

        # 5. Re-sampling
        if rflag:
            Temp1, Time1 = Pyshm.Tools.interpl_timeseries(Time0, Temp0, dtuple=(0, dT, 0), method='spline', rounded=True)
            Elon1, _     = Pyshm.Tools.interpl_timeseries(Time0, Elon0, dtuple=(0, dT, 0), method='spline', rounded=True)
            Temp0, Elon0, Time0 = Temp1, Elon1, Time1

        # 6. Step jumps detection in elongation data
        Type0 = np.ones(len(Time0), dtype=int)*100
        if jflag:
            jidx = Pyshm.Tools.detect_step_jumps(Elon0, method='diff', thresh=20, mwsize=24, median=0.8)
            Type0[jidx] = 101

        return pd.DataFrame(data = np.c_[Temp0, Elon0, Type0],
                            columns = ['Temperature', 'Elongation', 'Type'],
                            index = pd.DatetimeIndex(Time0))
    else:
        raise Exception('Insufficient length of data (< {} seconds): from {} to {} ({} points)'.format(dT, Time0[0],Time0[-1], len(Time0)))


def Preprocessing_by_location(Data, MinDataLength=10*24,
                              dflag=True,
                              sflag=True,
                              oflag=True,
                              fflag=False,
                              rflag=True,
                              tflag=False,
                              jflag=True):
    """Apply preprocessing on the data of one sensor (location) and seperate the static and dynamic data.

    Parameters
    ----------
    Data : raw OSMOS data of a sensor
    MinDataLength : minimum length of static data (to be considered as meaningful)
    The other parameters : see static_data_preprocessing()

    Returns
    -------
    Sdata, Ddata : list of static and dynamic data in pandas format
    """

    Sdata, Ddata = [], []

    # Preprocessing of static data
    if len(Data[Data.Type==1]) > MinDataLength: # if containing static data
        X0 = Data[Data.Type==1] # retrieve data

        # Find the gap larger than the threshold (12 hours) in the time series
        gidx0 = Pyshm.Tools.time_findgap(X0.index.to_pydatetime(), dtuple=(0,12*3600,0)) # gap index
        gidx = np.hstack([0, gidx0, len(X0)])

        # Prepreocessing of continous bunch of static data
        Xlist = []
        for n in range(len(gidx)-1):
            try:
                Xlist.append(static_data_preprocessing(X0.iloc[gidx[n]:gidx[n+1]],
                                                       dflag=dflag, # remove dynamic data
                                                       sflag=sflag, # remove synchronization error
                                                       oflag=oflag, # remove obvious outliers
                                                       fflag=fflag, # fill missing data
                                                       rflag=rflag, # resampling
                                                       tflag=tflag, # filter only elongation
                                                       jflag=jflag # apply jump-detection
                                                      ))
            except Exception as msg:
                print(Fore.RED + 'Warning: ', msg)
                print(Style.RESET_ALL)

        Sdata = pd.concat(Xlist) # processed static data

    # Preprocessing of dynamic data
    if len(Data[Data.Type==2]) > 0: # if containing dynamic data
        Ddata = []
        Didx = Pyshm.Tools.find_block_true(np.asarray(Data.Type==2)) # blocks of dynamic events

        for rng in Didx:
            Ddata.append(Data.iloc[rng[0]:rng[1]])

    return Sdata, Ddata


def choose_component(Data0, cnames):
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


# #### Obsolete functions ####

def Preprocessing_by_project(RawData):
    """Apply preprocessing on data of a project.

    The preprocessing consists in first seperating the data into static and dynamic parts,
    then applying static_data_preprocessing on the static data.
    """

    Sdata = {}
    Ddata = {}

    for loc, Data in RawData.items():
        if Data is not None: # if containing data
            print('Processing location {}...'.format(loc))
            Sdata[loc], Ddata[loc] = Preprocessing_by_location(Data)

    return Sdata, Ddata


def dynamic_data_preprocessing(X0, dT=20000):
    """
    ----TODO----
    On a continuous bunch of dynamic data, apply a series of pre-processing including:
    - remove wrong values due to the synchronisation problem
    - remove obvious outliers (TODO)
    - resample the time series at a regular time step

    Parameters
    ----------
    X0 : pandas datasheet
        containing the field Time, Temperature, Elongation
    dT : integer
        regular sampling time step in microsecond
    """

    Time0 = X0.Time.tolist() # time-stamp
    Temp0 = np.asarray(X0.Temperature.tolist(), dtype=np.float64) # temperature
    Elon0 = np.asarray(X0.Elongation.tolist(), dtype=np.float64) # elongation

    if Time0[-1]-Time0[0] >= datetime.timedelta(0,0,dT): # only for the length > sampling step
        # 1. Remove the wrong values due to synchronisation
        Temp0 = Pyshm.Tools.remove_plateau_jumps(Temp0, wsize=1, thresh=5.) # threshold for synchronisation period
        Elon0 = Pyshm.Tools.remove_plateau_jumps(Elon0, wsize=1, thresh=5.) # threshold for synchronisation period

        # 2. Re-sampling
        Temp1, Time1 = Pyshm.Tools.interpl_timeseries(Time0, Temp0, dtuple=(0,0,dT), method='spline')
        Elon1, _     = Pyshm.Tools.interpl_timeseries(Time0, Elon0, dtuple=(0,0,dT), method='spline')

        return pd.DataFrame(data = np.c_[Time1, Temp1, Elon1],
                            columns=["Time", "Temperature", "Elongation"])
    else:
        raise Exception('Insufficient length of data (< {} microseconds): from {} to {}'.format(dT, Time0[0],Time0[-1]))


def detect_step_jumps_events(X0, gidx, jumps=True, **kwargs):
    # X0 = np.asarray(X0)
    # gidx = Gidx[loc]

    # print('Location {}'.format(loc))
    # print('Position index of interruptions (>12h): {}'.format(list(gidx)))

    pidx = []
    for n, x in enumerate(np.split(X0, gidx)):
        idx = Pyshm.Tools.detect_jumps(x, method='diff', **kwargs) #thresh=10, mwsize=24, median=0.75)
        rp = gidx[n-1] if n>0 else 0
        pidx += list(asarray(idx) + rp)

    return pidx
