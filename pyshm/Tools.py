import numpy as np
from numpy import newaxis, zeros, zeros_like, squeeze, asarray,\
     abs, linspace, fft, random, hstack, vstack, ones, log
from numpy.linalg import norm, inv

import scipy as sp
from scipy import special

import pandas as pd

import dateutil
import datetime
import copy

from sklearn.gaussian_process import GaussianProcess
from . import Stat

# #### Time series related ####


def KZ_filter(X0, mwsize=24, k=0, method='mean', causal=True):
    """Kolmogorov Zurbenko filter.

    The KZ filter is nothing but the recursive application of the moving
    average, it has length (mwsize-1)*k+1. Here we extend KZ filter using the
    moving median.

    Parameters
    ----------
    X0 : numpy array or pandas series/dataframe
        dimension of X0 <=2
    mwsize : integer
        size of moving window
    k : integer
        order of KZ filter
    method : 'mean' or 'median'
        operation taken on the moving window
    """
    X = asarray(X0).copy()

    for n in range(k):
        if method == 'mean':
            X = Stat.mwmean(X, mwsize, mode='hard', causal=causal)
        elif method == 'median':
            X = Stat.mwmedian(X, mwsize, mode='hard', causal=causal)

    # convert back to pandas format
    if isinstance(X0, pd.Series):
        return pd.Series(X, index=X0.index)
    elif isinstance(X0, pd.DataFrame):
        return pd.DataFrame(X, index=X0.index, columns=X0.columns)
    else:
        return X


def KZ_filter_pandas(X0, mwsize=24, k=0, method='mean', causal=True):
    """
    Kolmogorov Zurbenko filter for pandas data sheet.
    """
    X = X0.copy()
    for n in range(k):
        if method == 'mean':
            X = pd.Series.rolling(X, mwsize, min_periods=1, center=not causal).mean()
        elif method == 'median':
            X = pd.Series.rolling(X, mwsize, min_periods=1, center=not causal).median()

    X[np.isnan(X0)] = np.nan
    return X


def circular_convolution(x, y):
    """
    Circular convolution.

    The circular convolution between x and y is the convolution between
    periodized x and y. This function computes the circular convolution via
    fft.
    """
    assert(x.ndim == y.ndim == 1)  # only operate on 1d arrays

    N = max(len(x), len(y))
    res = fft.ifft(fft.fft(x, N) * fft.fft(y, N))  # convolution by fft

    if any(np.iscomplex(x)) or any(np.iscomplex(y)):  # take the real part
        return res
    else:
        return np.real(res)


def safe_slice(X0, tidx, wsize, mode='soft', causal=False):
    """
    Take a slice of a 1d or 2d array with safe behavior on the boundaries.

    Parameters
    ----------
    X0 : array
        1d or 2d, the column direction (axis 1) is considered as the time index
    tidx : integer
        position where the slice will be taken
    wsize : integer
        size of the slice
    mode : string
        'hard' (simple truncation at the boundary) or 'soft' (continuation by
        zero at the boundary).
    causal : boolean
        if True the window ends at tidx, otherwise the window is centered at
        tidx.

    Return
    ------
    A slice of X0.
    """

    if X0.ndim == 1:  # extend to 2d array
        X1 = X0[newaxis, :].copy()
    elif X0.ndim == 2:
        X1 = X0.copy()
    else:
        raise NotImplementedError('Only work for 1d or 2d array')

    # zero padding
    nrow, ncol = X1.shape
    X = zeros((nrow, ncol+2*(wsize-1)))
    X[:, wsize-1:(wsize-1+ncol)] = X1

    if causal:  # index of the slice
        tidx0 = tidx
        tidx1 = tidx0+wsize
    else:
        tidx0 = tidx-int(wsize/2)+wsize-1
        tidx1 = tidx0+wsize

    if mode == 'soft':  # soft mode, with zero-padding
        return X[:, tidx0:tidx1]
    else:  # hard mode, without zero-padding
        tidx0 = max(0, tidx0-(wsize-1))
        tidx1 = min(ncol, tidx1-(wsize-1))
        return X1[:, tidx0:tidx1]


def safe_norm(X0):
    """
    A custom version of numpy.linalg.norm that treats nan as zero.
    """
    X = X0.copy()
    X[np.isnan(X)] = 0
    return norm(X)

safenorm = safe_norm  # an alias


def time_range(t0, N, dtuple=(0, 60*60, 0)):
    """
    Generate a list of N datetime objects starting from the instant t0 with a
    given step dt.

    Parameters
    ----------
    t0 : datetime object
        starting instant
    N : integer
        length of the list
    dtuple : tuple
        (day, second, microsecond) that defines the sampling step dt

    Return
    ------
    A list of datetime objects with the n-th element given by t0+n*dt.
    """
    dt = datetime.timedelta(*dtuple)

    T = [t0 + dt*n for n in range(N)]

    return T


def time_linspace(t0, t1, N):
    """
    Generate N equally spaced points between two time instants t0 and t1.

    """
    dt = (t1-t0)/N
    return [t0 + dt*n for n in range(N)]


def time2second(T0, t0):
    """
    Convert a time series from datetime object to second.

    Parameters
    ----------
    T0 : list of datetime object
    t0 : datetime object
        the starting instant

    Return
    ------
    A numpy array with the n-th element given by T0[n]-t0 in seconds
    """
    dT0 = zeros(len(T0))  # relative time stamp wrt the T0[0], in seconds

    for i in range(len(T0)):
        toto = T0[i]-t0
        dT0[i] = toto.days*24*3600+toto.seconds  # +toto.microseconds/1000/1000

    return dT0


def time_ceil(t0, unit='hour'):
    """
    Return the instant next to t0 with rounded unit (hour or minute).
    """
    if isinstance(t0, str):
        t0 = dateutil.parser.parse(t0)

    if unit == 'hour':  # return the next instant with rounded hour
        flag = t0.minute > 0 or t0.second > 0 or t0.microsecond > 0
        t = t0 + datetime.timedelta(0, 60*60*flag)
        return t.replace(minute=0, second=0, microsecond=0)
    elif unit == 'minute':  # return the next instant with rounded minute
        flag = t0.second > 0 or t0.microsecond > 0
        t = t0 + datetime.timedelta(0, 60*flag)
        return t.replace(second=0, microsecond=0)
    else:  # return the next instant with rounded second
        flag = t0.microsecond > 0
        t = t0 + datetime.timedelta(0, 1*flag)
        return t.replace(microsecond=0)


def time_floor(t0, unit='hour'):
    """
    Return the instance just before t0 with rounded unit (hour or minute).
    """
    if isinstance(t0, str):
        t0 = dateutil.parser.parse(t0)

    if unit == 'hour':  # return the next instant with rounded hour
        return t0.replace(minute=0, second=0, microsecond=0)
    elif unit == 'minute':  # return the next instant with rounded minute
        return t0.replace(second=0, microsecond=0)
    else:  # return the next instant with rounded second
        return t0.replace(microsecond=0)


def str2time(t0):
    """
    Convert a string to a datetime object.
    """
    return [dateutil.parser.parse(t) for t in t0]


def time_findgap(T0, dtuple=(0, 60*60, 0)):
    """
    Find the gaps in a time-stamp series.

    Parameters
    ----------
    T0 : list of time-stamp
    dtuple : a tuple of (day, second, microsecond) that define the minimal size
    of the gap

    Return
    ------
    G : index of gaps
    """
    dt = datetime.timedelta(*dtuple)  # size of the gap

    G = []
    for n in range(len(T0)-1):
        if T0[n+1]-T0[n] >= dt:
            G.append(n+1)

    return np.asarray(G, dtype=int)

    # Equivalent to:
    # return np.asarray(np.where(np.diff(T0) > dt)[0]+1, dtype=int)


def timeseries_list2array(X0, Tl, dtuple=(0, 60*60, 0)):
    """
    Convert a list of arrays X0 to an array using the information of the given
    timeseries.

    Parameters
    ----------
    X0 : a list of arrays (of float), X0[n] has the same length as Tl[n]
    Tl : Tl[n] is a list of time-stamps, see timeseries_list2idx
    dtuple : see timeseries_list2idx
    """
    gidx, _ = timeseries_list2idx(Tl, dtuple=dtuple)
    X1 = rejoin_timeseries(X0, gidx)
    T1 = rejoin_timeseries(Tl, gidx)

    return np.asarray(X1, dtype=np.float64), T1


def timeseries_list2idx(Tl, dtuple=(0, 60*60, 0)):
    """
    Convert a list of time-stamp series into position indexes with respect to
    the first time-stamp and with a given sampling step which is only applied on the gap.

    Parameters
    ----------
    Tl: a list
        Tl[n] is a list of time-stamps aranged in increasing order. For m>n
        Tl[n] is older than Tl[m], and they are mutually non-interlapped.
    dtuple: a tuple of (day, second, microsecond) that define the sampling step on the gap

    Return
    ------
    A: a list. A[n] is the index of the time-stamp Tl[n][0]
    B: a list of indexes. B[n][k] is the index of Tl[n][k]
    """

    dt = datetime.timedelta(*dtuple)

    t_start, t_end = Tl[0][0], Tl[-1][-1]
    gidx = []

    n=0 # counter
    for idx, T in enumerate(Tl):
        gidx.append(list(range(n, n+len(T)))) # keep the non-gap part
        n += len(T)

        if idx<len(Tl)-1: # add the length of the gap
            n += int(np.floor((Tl[idx+1][0]-Tl[idx][-1])/dt))

    return [g[0] for g in gidx], gidx


def rejoin_timeseries(Y0, gidx, val=None):
    """
    Join a list of time series.

    Parameters
    ----------
    Y0 : list of numpy 1d array
    gidx : position index of time series
        gidx[n] correspond to the position index of Y0[n][0] in the time axis.
        gidx must be strictly increasing and gidx[0] must be 0.
    val :
        gaps between time series are filled by this value.
    ------
    """
    # Y = np.zeros(gidx[-1]+len(Y0[-1])); Y.fill(val)
    # for n, k in enumerate(gidx):
    #     Y[k:k+len(Y0[n])] = Y0[n]

    Y = [val] * (gidx[-1]+len(Y0[-1]))
    for n, k in enumerate(gidx):
        for m in range(k, k+len(Y0[n])):
            Y[m] = Y0[n][m-k]

    return Y


def construct_convolution_matrix(Xt, Nq, tflag=True):
    """
    Construct the matrix of the causal convolution operator from a given sequence.

    The matrix M in question is the linear operator corresponding to the following
    convolution equation:
        Y[t] = \sum_{i=0}^{Nq-1} a_i X[t-i]
    or,
        Y = M*a

    Parameters
    ----------
    Xt : array
        Input sequence
    Nq : integer
        Length of the convolution kernel
    tflag : boolean
        if False then the causal convolution stops at t-1, otherwise it stops at t

    Returns
    -------
    the matrix M
    """
    Nt = len(Xt)
    M = zeros([Nt, Nq])

    # causal convolution
    for t in range(Nt):
        t2 = t+1 if tflag else t  # if tflag=False then the value X[t] is dropped from the sum
        t1 = max(0, t2-Nq)

        if t2 > t1 :
            b = Xt[t1:t2][::-1].copy()  # b is a convolution kernel: from 0 to t2-t1
            b[np.isnan(b)] = 0 # replace nan by 0

            # taking average in the non full length case
            M[t, :(t2-t1)] = b*Nq/(t2-t1) if t2-t1 < Nq else b

    return M


def LS_estimation_4SS(Xtrn, Ytrn, Nq, mwsize=24*10, penal=True, dT=1, causal=False):
    """
    Least-square estimation of parameters for state-space model.
    """

    assert(len(Xtrn) == len(Ytrn))
    Ntrn = len(Xtrn)
    # Nidx = Xtrn[np.isnan(Xtrn)]

    # # Estimation of kernel, without or with penalization
    # if penal:
    #     pcoef = np.sqrt(Ntrn) * exponential_weight(Nq, w0=1e-1, w1=5e-2)
    #     _, kernel, *_ = ARX_fit(Ytrn, 0, Xtrn, Nq, tflag=True, bflag=True, pcoef=pcoef)
    # else:
    #     _, kernel, *_ = ARX_fit(Ytrn, 0, Xtrn, Nq, tflag=True, bflag=True, pcoef=None)

    # # Prediction with the estimated kernel
    # Yprd = ARX_simulation([], [], Ntrn, nv=0, X=Xtrn, g=kernel)
    # Yprd[Nidx] = np.nan

    # sigmar2 = np.var(Ytrn-Yprd)  # variance of the observation noise

    pcoef = np.sqrt(Ntrn) * exponential_weight(Nq, w0=1e-1) if penal else 1e-8 * ones(Nq)

    # Estimation of parameters
    V0 = []  # variance on a moving window
    G0 = []  # kernel on a moving window

    for t in range(0, Ntrn, dT):
        # data on a moving window
        Xt = squeeze(safe_slice(Xtrn, t, mwsize, mode='hard', causal=causal))
        Yt = squeeze(safe_slice(Ytrn, t, mwsize, mode='hard', causal=causal))
        # Xt = Xtrn[:t].copy()
        # Yt = Ytrn[:t].copy()

        Xt = np.atleast_1d(Xt)
        Yt = np.atleast_1d(Yt)

        # convert nan -> 0
        Xt[np.isnan(Xt)] = 0
        Yt[np.isnan(Yt)] = 0

        # ARX fit on the moving window
        try:
            _, g0, err, _ = ARX_fit(Yt, 0, Xt, Nq, tflag=True, bflag=True, pcoef=pcoef)
            V0.append(np.var(err))
            G0.append(g0)
        except Exception as msg:
            print(msg)
            V0.append(0)
            G0.append(zeros(Nq))

    # Drop the beginning of the result since it is not reliable
    Gm = asarray(G0)[mwsize:,]
    Vm = asarray(V0)[mwsize:]

    # Processing on the estimations
    sigmaq2 = np.var(np.diff(Gm, axis=0)/dT, axis=0)
    # toto = LU_filter(toto, 1) # smooth the estimation
    # toto[Nq//2:] = np.min(toto)  # force the tail to be small
    sigmar2 = np.mean(Vm)

    return sigmaq2, sigmar2


def ARX_simulation(Yinit, h, Nt0, nv=0., X=None, g=None, c=0):
    """
    Simulation of a AR series
        Y[t] = \sum_{j=1}^p h[j] * Y[t-j] + \sum_{i=0}^{q-1} g[i] * X[t-i] + c + u[t]

    The simulation starts with an initial sequence Yinit and generate the future value from the precedent simulations.

    Parameters
    ----------
    Yinit : array
        initialisation sequence
    h : array
        AR operator
    Nt0 : int
        length of simulation
    nv : float
        level (the standard deviation) of noise u
    X : array, optional
        exogeneous vector
    g : array, optional
        convolution operator
    c : float
        constant of trend

    Returns
    -------
    simulated sequence Y with the begining identical to Yinit
    """
    if X is not None:
        assert(g is not None)
        Nt = min(Nt0, len(X)-len(Yinit))
        # assert(len(X)-len(Yinit) >= Nt and (g is not None))

    Y = np.hstack([Yinit, zeros(Nt)])
    N0 = len(Yinit)

    for t in range(N0,len(Y)):
        s = 0
        for j in range(min(len(h), t)): # j <= t-1
            s += h[j] * Y[t-(j+1)]

        if X is not None:
            for j in range(min(len(g), t+1)): # j <= t
                s += g[j] * X[t-j] # if tflag else X[t-(j+1)])
        # for j in range(len(h)):
        #     if t-(j+1)>=0:
        #         s += h[j] * Y[t-(j+1)]

        # if X is not None:
        #     for j in range(len(g), t+1): # j <= t
        #         if t-j>=0:
        #             s += g[j] * (X[t-j] if tflag else X[t-(j+1)])

        Y[t] = s + c + random.randn() * nv

    return Y


def ARX_prediction(Ydata, h, X=None, g=None, c=0):
    """
    Unlike ARX_simulation, this function makes the prediction based on precedent observations Ydata.
    """

    if X is not None:
        assert(g is not None)
        Nt = min(len(X), len(Ydata))
    else:
        Nt = len(Ydata)

    Y = zeros(Nt)

    for t in range(1, Nt):
        s = 0
        for j in range(min(len(h), t)): # j <= t-1
            s += h[j] * Ydata[t-(j+1)]

        if X is not None:
            for j in range(min(len(g), t+1)): # j <= t
                s += g[j] * X[t-j] # if tflag else X[t-(j+1)])

        Y[t] = s + c

    return Y


def exponential_weight(Nq, w0=5e-1):
    w1 = 5e-2
    toto = np.exp(w1*np.arange(Nq))
    return w0*np.arange(Nq)/Nq*toto/np.max(toto)


def ARX_fit(Yt, Np, Xt, Nq, tflag=True, bflag=False, pcoef=None, gzflag=False, hoflag=False, cflag=True):
    """
    Least-square fitting of the ARX model.

    The ARX model considered here is

    .. math::

       y_t = \sum_{j=1}^p h_j y_{t-j} + \sum_{i=0}^{q-1} g_i x_{t-i} + c + u_t

    This function computes the coefficients ::math h_j and g_i as well as the noise variance
    of u_t.

    Parameters
    ----------
    Yt : array
    Np : integer
        length of the vector h
    Xt : array
    Nq : integer
        length of the vector g
    tflag : boolean
        if False then then X[t] is dropped from the convolution, ie force g_0 = 0
    bflag : boolean
        if True drop the first max(Nq,Np) row of the system matrix to avoid boundary effect
    pcoef : array
        penality coefficients
    gzflag : boolean
        if True force the sum of g_i to be zero
    hoflag : boolean
        if True force the sum of h_i to be one
    cflag : boolean
        if True add a constant trend c to the ARX model
    Returns
    -------
    H, G, C, err, A :
        the vector h, g, the constant trend c, the error of fitting and the system matrix A
    """

    assert(Xt.ndim == Yt.ndim == 1)  # must be 1d array
    assert(Xt.size == Yt.size)  # must have the same size

    # Construct the linear system corresponding to the convolutions operators
    Nt = len(Yt)

    Ay = zeros(Np)
    Ax = zeros(Nq)

    A = zeros([Nt, Nq+Np])

    for t in range(Nt):
        Ay.fill(0)
        t2 = t
        t1 = max(0, t2-Np)

        if t2 > t1:
            b = Yt[t1:t2][::-1].copy()
            b[np.isnan(b)] = 0  # replace nan by 0
            Ay[:(t2-t1)] = b*Np/(t2-t1) if t2-t1 < Np else b

        Ax.fill(0)
        t2 = t+1 if tflag else t  # if tflag=False then X[t] is dropped
        t1 = max(0, t2-Nq)

        if t2 > t1:
            b = Xt[t1:t2][::-1].copy()
            b[np.isnan(b)] = 0  # replace nan by 0
            Ax[:(t2-t1)] = b*Nq/(t2-t1) if t2-t1 < Nq else b

        A[t, :] = np.hstack([Ay, Ax])

    # Drop the beginning, boundary effect
    if bflag:
        nd = max(Nq, Np)
        if nd < A.shape[0]:
            A = A[nd:, :]
            Yt = Yt[nd:]

    # if hoflag and Np>0:
    #     toto = hstack([ones(Np), zeros(Nq)])
    #     A = vstack([A, toto])
    #     Yt = hstack([Yt, 1])
    #
    # if gzflag and Nq>0:
    #     toto = hstack([zeros(Np), ones(Nq)])
    #     A = vstack([A, A.shape[0]*toto])
    #     Yt = hstack([Yt, 0])

    if cflag: # add constant trend
        A = hstack([A, ones((A.shape[0],1))])

    if pcoef is None:
        AtA = A.T @ A
    else:
        AtA = A.T @ A + np.diag(pcoef**2)

    toto = inv(AtA) @ (A.T @ Yt)
    H = toto[:Np]
    G = toto[Np:-1] if cflag else toto[Np:]
    C = toto[-1] if cflag else 0
    err = squeeze(A @ toto - Yt)

    # rerr = norm(err) / norm(Yt)
    # if rerr > 1e3 or np.isnan(rerr):
    #     raise ValueError('ARX fitting failed.')

    return H, G, C, err, A


def ARX_fit_predict(Xdata, Ydata, nh, ng, twsize=4*30*24, cflag=True):
    """
    Least-square fit and prediction of ARX model.
    """

    dfct=1
    tidx0=0

    # Training data
    Xtrn = Xdata[tidx0:tidx0+twsize:dfct]
    Ytrn = Ydata[tidx0:tidx0+twsize:dfct]

    # penalization on the decay of kernel
    wh = exponential_weight(nh) if nh>0 else []
    wg = exponential_weight(ng) if ng>0 else []

    pcoef = hstack([wh, wg, 0]) if cflag else hstack([wh, wg])
    # pcoef = 1e-4*ones(nh+ng+1) if cflag else 1e-4*ones(nh+ng)

    h0, g0, c0, err, A = ARX_fit(Ytrn, nh, Xtrn, ng, bflag=True, pcoef=pcoef, cflag=cflag)
    rerr = safe_norm(err) / safe_norm(Ytrn)

    Yprd_all = ARX_prediction(Ydata, h0, Xdata, g0, c0)
#     Yprd[:max(len(h0), len(g0))] = np.nan  # remove the begining
    Yprd_X = ARX_prediction(Ydata, [], Xdata, g0)

    return (h0, g0, c0, rerr, A), Yprd_all, Yprd_X


def optimal_kernel_length_AR(Xtrn, Ytrn):
    """
    Determine the optimal length of the auto-regression kernel using information criteria.

    Xtrn : 1d array
        training data, input
    Ytrn : 1d array
        training data, output
    """

    assert(len(Xtrn)==len(Ytrn))
    Nt = len(Xtrn)

    # AR kernel
    AIC = []
    BIC = []

    for k in range(1,24):
        res = ARX_fit(Ytrn, k, Xtrn, 0, cflag=False)
        h0, g0, c0, err0, A0 = res
        nerr = norm(err0)**2/Nt

        AIC.append(log(nerr)+(Nt+2*k)/Nt)
        # AICc.append(log(nerr)+(Nt+ng)/(Nt-ng-2))
        BIC.append(log(nerr)+k*log(Nt)/Nt)

    return AIC, BIC


def optimal_kernel_length_conv(Xtrn, Ytrn):
    """
    Determine the optimal lenght of the convolution kernel using information criteria.
    """

    assert(len(Xtrn)==len(Ytrn))
    Nt = len(Xtrn)

    # Convolution kernel
    AIC = []
    BIC = []

    for k in range(1,24):
        res = ARX_fit(Ytrn, 0, Xtrn, k, cflag=False)
        h0, g0, c0, err0, A0 = res
        nerr = norm(err0)**2/Nt

        AIC.append(log(nerr)+(Nt+2*k)/Nt)
        BIC.append(log(nerr)+k*log(Nt)/Nt)

    return AIC, BIC


def equalization_variance(X0, wsize):
    """Equalization of variance of a time series X0 on a moving window of length wsize.
    """
#     assert(X0.ndim==1)
    X1 = np.zeros_like(X0)
    nval = np.zeros(len(X0))

    for t in range(len(X0)):
        toto = safe_slice(X0, t, wsize, mode='hard', causal=True)
#         nval[t] = norm(toto, ord=p)
        nval[t] = np.std(toto) if len(toto)>1 else norm(toto)
        X1[t,] = X0[t,]/nval[t]

    return X1, nval


# ### Interpolation ####

def interpl_timeseries(T0, Y0, dtuple=(0,60*60,0), method='spline', rounded=False, **kwargs):
    """Regular resampling of a time-series by interpolation.

    Parameters
    ----------
    T0 : list of datetime objects
        This is the equivalent of the sampling points in the usual interpolate function
    Y0 : numpy 1d array
        values of the function at T0
    dtuple: a tuple of (day, second, microsecond)
        that define the length of the resampling time step

    Return
    ------
    Y1, T1 : Interpolated values and resampled time-series
    """

    # time sampling step
    dt = datetime.timedelta(*dtuple)

    # Time range of interpolation
    if rounded: # beginning of the interpolation interval is the hour next to T0[0]
        t_start, t_end = time_ceil(T0[0]), time_floor(T0[-1])
    else:
        t_start, t_end = T0[0], T0[-1]

    nbTime = int(np.floor((t_end-t_start)/dt))+1
    if nbTime <=0 :
        raise Exception('Insufficient length of data: {}.'.format(nbTime))

    dT0 = np.asarray([(t-t_start)/dt for t in T0])
    dT1 = np.arange(0, nbTime)
    T1 = [t_start + n*dt for n in dT1]

    # print(dT0, dT1)
    # print(T1)
    # print(dT0[-1], dT1[-1])

    if method=='spline':
        Yfunc = sp.interpolate.interp1d(dT0, Y0, kind='nearest')  # use 'linear', 'slinear' can result in artefacts (a periodicity of 37 days due to accumulation of 97 seconds per hour)
    elif method=='gp':
        Yfunc = lambda T: gp_interpl(dT0, Y0, dT1, **kwargs)
    else:
        raise TypeError('Unknown interpolation method')

    return Yfunc(dT1), T1


def gp_interpl(x_obs, y_obs, x_pred, nugget=1e-9):
    """Interpolation of missing data using GP

    Parameters
    ----------
    x_obs : coordinates of observations
    y_obs : values of observations
    x_pred : coordinates of predictions
    """

    gp = GaussianProcess(corr='cubic', theta0=1e-2, thetaL=1e-4, thetaU=1e-1, nugget = nugget, random_start=100)
    # Fit to data using Maximum Likelihood Estimation of the parameters
    gp.fit(np.atleast_2d(x_obs).T, y_obs)

    # Mesh the input space for evaluations of the real function, the prediction and
    # its MSE

    y_pred, MSE = gp.predict(np.atleast_2d(x_pred).T, eval_MSE=True)
    sigma = np.sqrt(MSE)

    return y_pred, sigma


def normalize_2d_data(X0):
    """
    Remove the mean and normalize by the standard deviation.
    """
    if X0.ndim==1:
        X = X0[newaxis,:]
    else:
        X = X0.copy()

    mX = X.mean(axis=1)
    sX = X.std(axis=1)
    X = (X-mX[:,newaxis]) / sX[:,newaxis]

    return squeeze(X)


#### Filters and signal processing ####
UL_filter = lambda X,wsize: U_filter(L_filter(X,wsize),wsize)
LU_filter = lambda X,wsize: L_filter(U_filter(X,wsize),wsize)
LU_mean = lambda X,wsize: (L_filter(X,wsize) + U_filter(X,wsize))/2


def U_filter(X, wsize=1):
    """
    Non boundary-preserving upper filter
    """
    assert(X.ndim==1)

    nX = len(X)
    Y = zeros_like(X)

    for tidx in range(nX):
        v = zeros(wsize+1)

        for sidx in range(0, wsize+1):
            tmin = max(0, sidx-wsize+tidx)
            tmax = min(nX, tmin+wsize+1)
            v[sidx] = X[tmin:tmax].max()

        Y[tidx] = v.min()

    return Y


def U_filter_boundary(X, wsize=1):
    """
    Boundary-preserving upper filter
    """
    assert(X.ndim==1)

    nX = len(X)
    Y = zeros_like(X)

    for tidx in range(nX):
        v = zeros(wsize+1)

        for sidx in range(0, wsize+1):
            tmin0 = sidx-wsize+tidx
            tmax0 = tmin0+wsize+1
            tmin, tmax = max(0, tmin0), min(nX, max(0, tmax0))
            v[sidx] = X[tmin:tmax].max()

        Y[tidx] = v.min()

    return Y


def L_filter(X, wsize=1):
    """
    Non boundary-preserving lower filter.
    """
    assert(X.ndim==1)

    nX = len(X)
    Y = zeros_like(X)

    for tidx in range(nX):
        v = zeros(wsize+1)

        for sidx in range(0, wsize+1):
            tmin = max(0, sidx-wsize+tidx)
            tmax = min(nX, tmin+wsize+1)
            v[sidx] = X[tmin:tmax].min()

        Y[tidx] = v.max()

    return Y

def L_filter_boundary(X, wsize=1):
    """
    Boundary-preserving lower filter
    """
    assert(X.ndim==1)

    nX = len(X)
    Y = zeros_like(X)

    for tidx in range(nX):
        v = zeros(wsize+1)

        for sidx in range(0, wsize+1):
            tmin0 = sidx-wsize+tidx
            tmax0 = tmin0+wsize+1
            tmin, tmax = max(0, tmin0), min(nX, max(0, tmax0))
            v[sidx] = X[tmin:tmax].min()

        Y[tidx] = v.max()

    return Y


select_func = lambda x,y,c: y if abs(x-c)>abs(y-c) else x

def shrinkage(X, t, soft=True):
    """
    Shrinkage function.
    """
    assert(t>=0)
    Y=np.zeros_like(X)
    idx = np.abs(X)>=t
    Y[idx]=np.abs(X[idx])-t if soft else np.abs(X[idx])
    return Y*np.sign(X)


def shrinkage_by_percentage(X0, thresh, mode='hard'):
    """
    thresh: percentage of nonzeros to keep, eg. thresh=0.1 will keep the 10% largest coefficients (in magnitude)
    """
    assert(X0.ndim==1)
    assert(0<=thresh<=1)

    X1 = np.zeros_like(X0)
    idx=np.argsort(abs(X0))
    nz = int(len(idx)*(1-thresh))
    zidx = idx[nz:]

    if mode=='soft':
        toto = X0[zidx].copy()
        X1[zidx] = np.sign(toto)*(abs(toto)-abs(X0[nz]))
    else:
        X1[zidx]=X0[zidx]

    return X1, len(zidx)


def remove_plateau_jumps(y0, wsize=10, thresh=5., bflag=False, dratio=0.5):
    """
    Remove plateau-like or sawtooth-like jumps:

      |---|          /\  /\
    --|   |--  or --/  \/  \--

    Parameters
    ----------
    y0: 1d array, input signal
    wsize: plateau size
    thresh: threshold for detection of jumps
    bflag: if True apply boundary preserving filter
    dratio: ratio between the duration of a plateau jump and the size of y0

    Return
    ------
    Processed data.

    Note: This function may have no effect on the data which contain no regular part
    (eg, the whole signal is sawtooth-like).
    """

    # 1. Apply L and U filter to detect plateau jumps
    if bflag: # boundary preserving filter
        yu = U_filter_boundary(y0, wsize) # upper filter on the residual
        yl = L_filter_boundary(y0, wsize) # lower filter
    else: # non boundary preserving filter
        yu = U_filter(y0, wsize) # upper filter on the residual
        yl = L_filter(y0, wsize) # lower filter

    my = (yu+yl)/2 # mean filter
    dy = yu-yl # difference filter, for wsize=1 sawtooth jumps become plateau jumps here.

    # 2. Find significant plateau jumps
    # method 1: using the small values dy, this would require a larger threshold (eg > 10)
    # fy = np.sort(dy); vthresh = np.mean(fy[:int(len(y0)*dratio)]) * thresh
    # method 2: using all values of dy, this seems to be more stable
    vthresh = dy.mean()*thresh

    pflag = np.abs(shrinkage(dy, vthresh))>0 # might be all False if whole y0 is sawtooth-like
    pidx = find_block_true(pflag) # blocks of positions which are beyond the threshold
    if np.sum(pflag)<len(y0):
        mval = np.mean(my[np.logical_not(pflag)]) # take the mean value of positions below the threshold
    else:
        mval = y0.mean()

    # 3. Determine the action on each plateau jump
    mask = np.zeros_like(y0) # mask for plateau jumps
    yul =  np.zeros_like(y0) # values on the mask

    for (tmin, tmax) in pidx:
        # lt = (tmax-tmin); tmin = max(0, tmin-lt//2); tmax = min(len(y0), tmax+lt//2)
        # tmin = max(0, tmin-1); tmax = min(len(y0), tmax+1)
        if (tmax-tmin)/len(y0) < dratio:
            mask[tmin:tmax] = 1
            uflag = my[tmin:tmax].mean() >= mval # plateau is above the average
            yul[tmin:tmax] = yl[tmin:tmax] if uflag else yu[tmin:tmax]

    y1 = y0*(1-mask)+yul*mask # preserve the signal outside the mask
    # return y1

    # 4. There may remain sawteeth which are not removed in step 3.
    # We wipe out these residuals by detecting the alternation of � sign
    # in the derivative.
    mask.fill(0)
    wy = np.diff(np.r_[y1[0],y1]) #; wy = np.abs(wy0)
    mwy, swy = wy.mean(), wy.std()
    sidx = find_altsign(np.sign(np.diff(y1)), 40) # a list of [begin, end] of alternation patterns

    for (idx0, idx1) in sidx:
        # print(idx0, idx1, yu[-10:],yl[-10:])
        mask[idx0:idx1] = 1
        # # if on the alternation pattern the magnitude of oscillation is important:
        # if np.sqrt(np.mean((np.abs(np.diff(y1[idx0:idx1])) - mwy)**2))/swy > 1: # thresh:
        #     mask[idx0:idx1] = 1

    return y1*(1-mask)+my*mask


def remove_plateau_jumps_old(y0, wsize=10, thresh=5., bflag=False, dratio=0.5):
    """
    Remove plateau-like or sawtooth-like jumps:

      |---|          /\  /\
    --|   |--  or --/  \/  \--

    Parameters
    ----------
    y0: 1d array, input signal
    wsize: plateau size
    thresh: threshold for detection of jumps
    bflag: if True apply boundary preserving filter
    dratio: ratio between the duration of a plateau jump and the size of y0
    """

    # 1. Apply L and U filter to detect plateau jumps
    if bflag: # boundary preserving filter
        yu = U_filter_boundary(y0, wsize) # upper filter on the residual
        yl = L_filter_boundary(y0, wsize) # lower filter
    else: # non boundary preserving filter
        yu = U_filter(y0, wsize) # upper filter on the residual
        yl = L_filter(y0, wsize) # lower filter

    my = (yu+yl)/2 # mean filter
    dy = yu-yl # difference filter, the sawtooth jumps become plateau jumps here.

    # 2. Find significant plateau jumps
    fy0 = np.diff(np.r_[dy[0],dy]); fy = fy0.copy()
    # fy0 = np.diff(np.r_[dy[0],dy]); fy = np.abs(fy0)
    mfy, sfy = fy.mean(), fy.std()
    # jumps should be significantly deviated from the mean value
    fidx = np.where(abs(fy-mfy)/sfy > thresh)[0]

    # 3. Determine the action on each plateau jumps
    n = 0
    mask = np.zeros_like(y0) # mask for plateau jumps
    yul =  np.zeros_like(y0) # values on the mask

    while n<len(fidx):
        # detect plateau jumps
        if n==0 and np.sqrt(np.mean((dy[0:fidx[0]]-mfy)**2))/sfy > thresh and fidx[0]/len(y0) < dratio: # check the left boundary
            tmin, tmax = 0, fidx[0]
            dn = 1
        elif n==len(fidx)-1 and np.sqrt(np.mean((dy[fidx[n]:]-mfy)**2))/sfy > thresh and 1-fidx[n]/len(y0) < dratio: # check the right boundary
            tmin, tmax = fidx[n], len(y0)
            dn = 1
        elif n<len(fidx)-1 and fy0[fidx[n]] * fy0[fidx[n+1]] < 0 and np.sqrt(np.mean((dy[fidx[n]:fidx[n+1]]-mfy)**2))/sfy > thresh and (fidx[n+1]-fidx[n])/len(y0) < dratio: # check the inside
            tmin, tmax = fidx[n], fidx[n+1]
            dn = 2
        else:
            tmin, tmax = None, None

        if tmin is not None and tmax is not None:
            mask[tmin:tmax] = 1

            uflag = my[tmin:tmax].mean() >= y0.mean() # plateau is above the average
            yul[tmin:tmax] = yl[tmin:tmax] if uflag else yu[tmin:tmax]
            n+=dn
        else:
            n+=1

    return y0*(1-mask)+yul*mask # preserve the signal outside the mask


def detect_step_jumps(X0, method='diff', **kwargs):
    """
    Detect step jumps in a 1d array.

    Parameters
    ----------
    X0 : 1d array
    method : method of detection, by default use 'diff'
    keyword parameters :
    - thresh: factor of threshold, default value 10
    - mwsize: window size for the moving average, default value 24
    - median: location of the median position, default value 0.5

    Return
    ------
    A list of detected jumps
    """
#     X0 = np.asarray(X0)

    if method=='diff':
        thresh = kwargs['thresh'] if 'thresh' in kwargs.keys() else 10
        mwsize = kwargs['mwsize'] if 'mwsize' in kwargs.keys() else 24
        median = kwargs['median'] if 'median' in kwargs.keys() else 0.5

        # X1 = LU_filter(np.asarray(X0), 24) # Apply first a LU filter
        X1 = np.asarray(X0)
        mX = Stat.mwmean(X1, mwsize)
        dX0 = np.abs(np.hstack((0, np.diff(mX)))); dX=dX0[::mwsize].copy()

        sdX = np.sort(dX)
        # Take the modified median as threshold
        vthresh = thresh * sdX[int(len(sdX)*median)]
        pidx = np.where(dX>vthresh)[0]*mwsize
        # To do: add some aposteriori processing to adjust the location of jumps
        return list(pidx) #, dX, vthresh
    else:
        raise NotImplementedError('Unknown method: %s' % method)


# def find_true_cluster(X):
#     """
#     Find the clusters of True in a boolean array.

#     Example:
#     for X = [0, 1, 1, 1, 0, 1], the returned list is [[1,4], [5,6]]

#     Parameters
#     ----------
#     X : numpy 1d array of boolean type
#     """

#     idx = np.where(np.diff(np.asarray(X,dtype=bool))!=0)[0]+1
#     L = []

#     n = 0
#     while n < len(idx):
#         if n < len(idx)-1:
#             L.append([idx[n], idx[n+1]])
#         else:
#             L.append([idx[n], idx[n]+1])
#         n+=2

#     return L #,idx


def find_block_true(X):
    """
    Find the starting and ending positions of True blocks in a boolean array.

    Example:
    for X = [0, 1, 1, 1, 0, 1], the returned list is [[1,4], [5,6]]

    """
    n = 0
    sidx = []
    while n < len(X):
        if X[n]:
            a, b = n, n+1
            while b<len(X) and X[b]:
                b += 1
            sidx.append([a,b])
            n = b
        n += 1
    return sidx


def find_altsign(X, minlen=10):
    """
    Find the pattern of alternation of +/- sign in a 1d array.

    Parameters
    ----------
    X: 1d array
    minlen: minimal length of the pattern

    Return
    ------
    sidx: a list of starting and ending positions of patterns
    """

    dS = np.hstack([0, np.diff(np.sign(X))])
    sidx0 = find_block_true(np.abs(dS)==2) # positions of alternation patterns
    # a pattern is confirmed if the alternation is sufficient long:
    sidx = []
    for k in sidx0:
        # print(k)
        if k[1]-k[0]>minlen:
            sidx.append([k[0]-1, k[1]])

    return sidx


def stft(X, wsize, tsize, fsize):
    """
    Short-time Fourier transform
    """
    Xf=[]
    step = int(len(X)/tsize)

    for tidx in range(0, len(X), step):
        tidx0=max(tidx-int(wsize/2), 0)
        tidx1=min(tidx0+wsize, len(X))
        window = np.real(np.hanning(tidx1-tidx0))  # our half cosine window
        Xf.append(np.fft.rfft(window*X[tidx0:tidx1], n=fsize))

    return squeeze(np.asarray(Xf))


# def stft(x, fs, framesz, hop):
#     framesamp = int(framesz*fs) # window size
#     hopsamp = int(hop*fs)
#     w = scipy.hanning(framesamp)
#     X = scipy.array([scipy.fft(w*x[i:i+framesamp])
#                      for i in range(0, len(x)-framesamp, hopsamp)])
#     return X

# def istft(X, fs, T, hop):
#     x = scipy.zeros(T*fs)
#     framesamp = X.shape[1]
#     hopsamp = int(hop*fs)
#     for n,i in enumerate(range(0, len(x)-framesamp, hopsamp)):
#         x[i:i+framesamp] += scipy.real(scipy.ifft(X[n]))
#     return x


#### Combinatorials ####

def multiset(n, p):
    """
    Generate multisets of p tuples (non-negative values) with sum equals to n
        """
    def _mcomb(n, p, A):
        if n<0:
            return []
        else:
            L = []

            if p>0:
                for k in range(n+1):
                    A.append(k)
                    L += copy.deepcopy(_mcomb(n-k, p-1, A))
                    A.pop()
            else:
                if n==0:
                    return [A]
                else:
                    return []

            return L

    return _mcomb(n, p, [])


def mnomial(L):
    """
    Multinomial coefficient.
    For L=[l1, l2, ... lk], compute
    (l1+..lk)!/(l1!..lk!)
    """

    if len(L)>0:
        return special.binom(np.sum(L), L[0]) * mnomial(L[1:])
    else:
        return 1


#### Plots ####
from matplotlib import pyplot as plt
from numpy import ma

def _two_plots_func(ax1, ax2, X1, X2, T=None, L1='', L2='', S1='b', S2='r', nbYticks=7, XL='', title=None):
    # X1 = asarray(X1)
    # X2 = asarray(X2)

    T1 = T if T is not None else np.arange(len(X1))
    X1 = np.asarray(X1); X2 = np.asarray(X2)

    ax1.plot(T1, X1, color=S1, label=L1)
    dT = len(X1)*0.02
    xrng1 = [T[0]-dT, T[-1]+dT]

    idx = np.logical_not(np.isnan(X1))

    V1 = 0.8*(np.max(X1[idx]) - np.min(X1[idx]))
    yrng1 = [np.min(X1[idx])-V1,np.max(X1[idx])+V1]

    ax2.plot(T1, X2, color=S2, label=L2)
    idx = np.logical_not(np.isnan(X2))

    V2 = 0.2*(np.max(X2[idx]) - np.min(X2[idx]))
    yrng2 = [np.min(X2[idx])-V2,np.max(X2[idx])+V2]
    # ax1.legend(loc='upper left')
    # ax2.legend(loc='upper right')
    _=ax1.set_xlabel(XL)

    _=ax1.set_ylabel(L1, color=S1)
    _=ax2.set_ylabel(L2, color=S2, rotation=270)

    if title:
        ax1.set_title(title)

    return yrng1, yrng2, xrng1


def two_plots_func(X1, X2, fig=None, ax1=None, **kwargs):
    """
    Plot two curves in a same figure using twin X-axis.

    Parameters
    ----------
    X1, X2: numpy 1d arrays
        possibly containing None as value
    keyword arguments:
        L1, L2: labels for X1 and X2
        S1, S2: line styles for X1 and X2
        nbYticks: number of Y ticks
    """
    assert(len(X1)==len(X2))

    figsize = kwargs['figsize'] if 'figsize' in kwargs.keys() else (20,5)
    nbYticks = kwargs['nbYticks'] if 'nbYticks' in kwargs.keys() else 7
    T = kwargs['T'] if 'T' in kwargs.keys() else np.arange(len(X1))

    if fig is None or ax1 is None:
        fig, ax1 = plt.subplots(1,1,figsize=figsize)

    ax2 = ax1.twinx()

    yrng1, yrng2, xrng1 = _two_plots_func(ax1, ax2, X1, X2, **kwargs)
    # print(yrng1, yrng2, xrng1)
    _=ax1.set_yticks(linspace(yrng1[0],yrng1[1],nbYticks))
    _=ax2.set_yticks(linspace(yrng2[0],yrng2[1],nbYticks))
    ax1.set_xlim(xrng1)
    ax1.set_ylim(yrng1)
    ax2.set_ylim(yrng2)

    lenX = len(X1)
    if lenX//240<=1:
        _ = ax1.set_xticks(T[::60])
    elif lenX//240<10:
        _ = ax1.set_xticks(T[::120])
    else:
        _ = ax1.set_xticks(T[::240])

    return fig, ax1, ax2


def two_plots_list(X1_list, X2_list, T_list, **kwargs):
    """
    Plot two lists of data with time index given by T_list.
    The function two_plots_func should be prefered.
    """

    assert(len(X1_list)==len(X2_list)==len(T_list))

    figsize = kwargs['figsize'] if 'figsize' in kwargs.keys() else (20,5)
    nbYticks = kwargs['nbYticks'] if 'nbYticks' in kwargs.keys() else 7

    fig, ax1 = plt.subplots(1,1,figsize=figsize)
    ax2 = ax1.twinx()

    for idx, (X1, X2, T) in enumerate(zip(X1_list, X2_list, T_list)):
        yrng1, yrng2, xrng1 = _two_plots_func(ax1, ax2, X1, X2, T, **kwargs)

        yrng1_min = yrng1[0] if idx==0 else min(yrng1[0], yrng1_min)
        yrng1_max = yrng1[1] if idx==0 else max(yrng1[1], yrng1_max)
        yrng2_min = yrng2[0] if idx==0 else min(yrng2[0], yrng2_min)
        yrng2_max = yrng2[1] if idx==0 else max(yrng2[1], yrng2_max)

        if idx==0:
            ax1.legend(loc='upper left')
            ax2.legend(loc='upper right')

    lenX = T_list[-1][-1]
    ax1.set_xlim((-lenX*0.02, lenX*1.02))
    ax1.set_ylim((yrng1_min, yrng1_max))
    ax2.set_ylim((yrng2_min, yrng2_max))

    _ = ax1.set_yticks(linspace(yrng1_min,yrng1_max,nbYticks))
    # adjust the number of x ticks
    if lenX//240<=1:
        _ = ax1.set_xticks(np.arange(0,lenX+1,60))
    elif lenX//240<10:
        _ = ax1.set_xticks(np.arange(0,lenX+1,120))
    else:
        _ = ax1.set_xticks(np.arange(0,lenX+1,240))

    _ = ax2.set_yticks(linspace(yrng2_min,yrng2_max,nbYticks))

    ax1.set_xlabel('Time (hour)')
    ax1.set_ylabel(kwargs['L1']) if 'L1' in kwargs.keys() else 0
    ax2.set_ylabel(kwargs['L2'], {'rotation':270}) if 'L2' in kwargs.keys() else 0

    return fig, ax1, ax2


def TEplot(X, Y, idx=None, twinx=False, figsize=(20,5), colors=['r', 'b']):
    assert(len(X)==len(Y))
    fig, axes = plt.subplots(1,1,figsize=figsize)
    axa = axes
    if idx is None:
        idx = np.arange(len(X))

    axa.plot(idx, X, colors[0], alpha=0.5)
    axb = axa.twinx() if twinx else axa
    axb.plot(idx, Y, colors[1], alpha=0.5)
    return fig, axes

def printwarning(msg):
    print(Fore.RED + 'Warning: ', msg)
    print(Style.RESET_ALL)
