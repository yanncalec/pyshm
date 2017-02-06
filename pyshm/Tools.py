"""Collection of utility functions.
"""

import numpy as np
import numpy.linalg as la
# import numpy.ma as ma

import scipy as sp
import scipy.signal
# from scipy import special

import pandas as pd
import copy
import dateutil
import datetime
import inspect

#### Functional operators ####

# def along_axis(func):
#     """Decorator functional for applying a function on a nd array along a given axis.
#     """
#     def newfunc(X, *args, axis=-1, **kwargs):
#         np.apply_along_axis(func, axis, X, *args, **kwargs)
#     return newfunc


def nan_safe(func, endo=True):
    """Make a function safe to nan by replacing nan values.

    Args:
        func: a function
        endo (bool): if True the function is considered as an endomorphisme which preserves the dimension, i.e. the output has the same dimension as the input. The coefficients in the output corresponding to the nan in the input are marked as nan again.
    """
    def newfunc(X0, *args, nv=0, **kwargs):
        X = X0.copy()
        nidx = np.isnan(X0)
        X[nidx] = nv
        Y = func(X, *args, **kwargs)
        if endo and Y.shape==X.shape:
            Y[nidx] = np.nan
        return Y
    return newfunc


# def nan_safe_remove_nan_columns(func):
#     """Make a function safe to nan by removing all columns containing nan from the arguments.
#
#     Args:
#         func: a function that all positional arguments are 2d arrays of the same shape
#     """
#     def newfunc(*args, **kwargs):
#         X = [np.atleast_2d(x) for x in args]
#         Z0, nidx = remove_nan_columns(*X)
#         return func(*Z, **kwargs)
#     return newfunc


def along_axis(func):
    """Decorator functional for applying a function on a nd array along a given axis.
    """
    def newfunc(X, *args, axis=-1, **kwargs):
        Xm = np.moveaxis(X, axis, -1)  # move the target axis to the last dimension
        Xr = Xm.reshape((-1,Xm.shape[-1]))  # reshape to a 2d array
        Yr = [] # np.zeros_like(Xr)

        for n,x in enumerate(Xr):  # iteration on each row
            Yr.append(func(x, *args, **kwargs))

        ydim = list(Xm.shape)
        ydim[-1] = Yr[0].size  # dimension of result
        Ym = np.vstack(Yr).reshape(ydim)  # reshape back
        return np.moveaxis(Ym, -1, axis)  # move the axis back
    return newfunc


def rfunc(p, func, x, *args, **kwargs):
    """Recursive application of any function.

    Args:
        p (int): number of recursion
        func (function handle): func(func(x)) must be meaningful
        x, args, kwargs: input, arguments and keyword arguments of func
    """
    if p>0:
        return func(rfunc(p-1, func, x, *args, **kwargs), *args, **kwargs)
    else:
        return x #func(x, *args, **kwargs)


def get_actual_kwargs(func, **kwargs):
    """Generate a full dictionary of keyword arguments of a function from a given (partial) dictionary.

    Args:
        func: a callable function
        kwargs (dict): dictionary of some keyword arguments
    """
    allargs_name, _, _, kwargs_val = inspect.getargspec(func)
    d = {}
    for i,k in enumerate(allargs_name[-len(kwargs_val):]):
        if k in kwargs:
            d.update({k:kwargs[k]})
        else:
            d.update({k:kwargs_val[i]})
    return d
    # return {k:kwargs[k] if k in kwargs else kwargs_val[i] for i,k in enumerate(allargs_name[-len(kwargs_val):])}


#### Convolution and filters ####

def safe_convolve(X, kernel, mode="valid"):
    """FFT based nan-safe convolution.

    Unlike `convolve`, `fftconvolve` is not safe to nan values. This function is a wrapper of `fftconvolve` by applying a mask on the nan values.

    Args:
        X (1d array): input array.
        kernel (1d array): convolution kernel.
        mode (str): truncation mode of the convolution, can be ["valid", "full", "samel", "samer"].
    """
    assert(len(X)>=len(kernel))

    Xmask = np.isnan(X)
    if Xmask.any():
        Ymask = sp.signal.convolve(Xmask, np.ones(len(kernel),dtype=bool), mode="full")
        X1 = X.copy()
        X1[Xmask] = 0  # setting nan to 0 may result in high frequency artifacts.
        Y = sp.signal.fftconvolve(X1, kernel, mode="full")
        Y[Ymask] = np.nan
    else:
        Y = sp.signal.fftconvolve(X, kernel, mode="full")

    if mode=="valid":
        return Y[len(kernel)-1:1-len(kernel)]
    elif mode=="samel":  # left side is not valid
        return Y[:1-len(kernel)]
    elif mode=="samer":  # right side is not valid
        return Y[len(kernel)-1:]
    else: # mode=="full":
        return Y


def circular_convolution(x, y):
    """Circular convolution.

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


@along_axis
def convolve_fwd(X, psi, mode="valid"):
    """Forward operator of convolution.

    Args:
        X (nd array): input array.
        psi (1d array): convolution kernel.
        mode (str): truncation mode of the convolution.
        axis (int): axis along which to apply the 1d convolution.
    """
    return safe_convolve(X, psi, mode=mode)


@along_axis
def convolve_bwd(X, psi, mode="full"):
    """Backward operator of convolution.
    """
    # the following is equivalent to
    # convolve(X[::-1], psi, mode=mode)[::-1]
    return safe_convolve(X, psi[::-1], mode=mode)


def convolve_matrix(h,Nx,mode="valid"):
    """Get the matrix representation of the convolution operator.
    """
    Nh = len(h)
#     A = np.zeros((Nx+Nh-1, Nx))
    C = np.hstack([h, np.zeros(Nx-1)])
    R = np.zeros(Nx); R[0] = C[0]
    A = sp.linalg.toeplitz(C, R)

    n = Nh #min(Nh, Nx)
    if mode=="valid":
        return A[n-1:1-n]  # dim = (max(Nx,Nh)-min(Nx,Nh)+1, Nx)
    elif mode=="samel":
        return A[:1-n]
    elif mode=="samer":
        return A[n-1:]
    else:
        return A  # dim = (Nx+Nh-1, Nx)


def diff_kernel(deg, step=1):
    """Get the convolution kernel of a differential operator.

    The kernel is determined by its degree and step, for example, for degree 0 and step 1 the kernel is [1,-1], and for step 2 the kernel is [1, 0, -1]. For higher degree the kernel is obtained using auto-convolution of the kernel of degree 0.

    Args:
        deg, step (int): degree and step of the differential kernel.
    """
    assert(deg>=0)
    h0 = np.zeros(1+step)
    h0[0]=1; h0[-1]=-1; h0/=step

    h = h0
    for n in range(deg):
        h = sp.signal.convolve(h0, h0) if n==0 else sp.signal.convolve(h,h0)
    return h


def mts_cumview(X, N):
    """Cumulative view of a multivariate time series (mts).

    This function cumulates N past values of a mts:
        X[t] -> (X[t], X[t-1]...X[t-N+1])

    Args:
        X (2d array): each row is a variable, each column is an observation of all variables.
        N (int): number of cumulations, N>=1.
    Returns:
        a 2d array of row dimension N times the row dimension of the input array.
    """
    assert(N>=1)
    assert(X.ndim==2)

    Nv = X.shape[0]  # number of variables
    Y = np.zeros((N*Nv, X.shape[1]))

    for n in range(N):
        toto = np.roll(X, n, axis=1)  # positive roll: roll toward index-increasing direction
        toto[:, :n] = np.nan
        Y[n*Nv:(n+1)*Nv,:] = toto

    return Y


def KZ_filter(X0, mwsize, kzord, method="mean", causal=True):
    """Kolmogorov Zurbenko filter for pandas data sheet.

    The KZ filter is nothing but the recursive application of the moving
    average, it has length (mwsize-1)*k+1. Here we extend KZ filter using the
    moving median.

    Args:
        X0: input pandas DataFrame
        mwsize (int): size of moving window
        kzord (int): order of KZ filter
        method (string): "mean" or "median"
        causal (bool): if True use the causal filter

    Returns:
        Filtered datasheet
    """
    X = X0.copy()
    for n in range(kzord):
        if method == "mean":
            X = pd.Series.rolling(X, mwsize, min_periods=1, center=not causal).mean()
        elif method == "median":
            X = pd.Series.rolling(X, mwsize, min_periods=1, center=not causal).median()

    X[np.isnan(X0)] = np.nan

    return X


UL_filter = lambda X,wsize: U_filter(L_filter(X,wsize),wsize)
UL_filter_boundary = lambda X,wsize: U_filter_boundary(L_filter_boundary(X,wsize),wsize)
LU_filter = lambda X,wsize: L_filter(U_filter(X,wsize),wsize)
LU_filter_boundary = lambda X,wsize: L_filter_boundary(U_filter_boundary(X,wsize),wsize)
LU_mean = lambda X,wsize: (L_filter(X,wsize) + U_filter(X,wsize))/2


def U_filter(X, wsize=1):
    """
    Non boundary-preserving upper filter.
    """
    assert(X.ndim==1)

    nX = len(X)
    Y = np.zeros_like(X)

    for tidx in range(nX):
        v = np.zeros(wsize+1, X.dtype)

        for sidx in range(0, wsize+1):
            tmin = max(0, sidx-wsize+tidx)
            tmax = min(nX, tmin+wsize+1)
            v[sidx] = X[tmin:tmax].max()

        Y[tidx] = v.min()

    return Y


def U_filter_boundary(X, wsize=1):
    """
    Boundary-preserving upper filter.
    """
    assert(X.ndim==1)

    nX = len(X)
    Y = np.zeros_like(X)

    for tidx in range(nX):
        v = np.zeros(wsize+1, X.dtype)

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
    Y = np.zeros_like(X)

    for tidx in range(nX):
        v = np.zeros(wsize+1, X.dtype)

        for sidx in range(0, wsize+1):
            tmin = max(0, sidx-wsize+tidx)
            tmax = min(nX, tmin+wsize+1)
            v[sidx] = X[tmin:tmax].min()

        Y[tidx] = v.max()

    return Y


def L_filter_boundary(X, wsize=1):
    """
    Boundary-preserving lower filter.
    """
    assert(X.ndim==1)

    nX = len(X)
    Y = np.zeros_like(X)

    for tidx in range(nX):
        v = np.zeros(wsize+1, X.dtype)

        for sidx in range(0, wsize+1):
            tmin0 = sidx-wsize+tidx
            tmax0 = tmin0+wsize+1
            tmin, tmax = max(0, tmin0), min(nX, max(0, tmax0))
            v[sidx] = X[tmin:tmax].min()

        Y[tidx] = v.max()

    return Y


#### Transformation ####

def roll_fill(X0, s, val=np.nan):
    """A wrapper of numpy roll function by setting the cyclic head/tail to a given value.
    """
    X = np.roll(X0, s)
    if s>=0:
        X[:s] = val
    else:
        X[s:] = val
    return X


def sdiff(X, p, axis=-1):
    """Seasonal difference.

    At each time index t, the seasonal difference of lag p is defined as
        X[t] - X[t-p]

    Args:
        X (nd array): the last dimension is the time axis
        p (int): seasonality
    Return:
        an array of same dimension as X
    """
    if p==0:
        return X #np.np.zeros_like(xt)
    else:
        # roll(x,n): circulant rolling to right if n>0 (the left n elements are due to boundary effect)
        dX = X - np.roll(X, p, axis=axis)

        slc = [slice(None)] * X.ndim
        slc[axis] = slice(0,p)
        dX[slc] = np.nan  # remove the boundary effect

        return dX


def isdiff(dX, p, av=None, axis=-1):
    """Inverse seasonal difference (or seasonal integration).

    At each time index t, the seasonal difference of lag p is defined as
        X[t] + X[t-p]

    Args:
        dX (nd array): first dimension is the time axis
        p (int): seasonality
    """
    if av is None:
        av = np.zeros(p)
    else:
        assert(len(av)>=p)
    if p==0:
        return dX #np.np.zeros_like(xt)
    else:
        Nidx = np.isnan(dX)
        dX[np.isnan(dX)] = 0
        X = np.zeros_like(dX)

        for i in range(p):
            dX[i] += av[i]
            X[i::p] = np.cumsum(dX[i::p], axis=axis)
        X[Nidx] = np.NaN

        return X


def dsdt(X, sord=3, ssize=24, tord=1, tsize=1, **kwarg):
    """De-seasonal and de-trend operator.

    This function removes the seasonal and trend components by applying sdiff with different steps.
    """
    return rfunc(tord, sdiff, rfunc(sord, sdiff, np.asarray(X), ssize, **kwarg), tsize, **kwarg)


def idsdt(X, sord=3, ssize=24, tord=1, tsize=1, av=None):
    """Inverse de-seasonal and de-trend operator.
    """
    isdiff = lambda X, p: isdiff(X, p, av=av)
    return rfunc(sord, isdiff, rfunc(tord, isdiff, np.asarray(X), tsize), ssize)


def nmldiff_transform(X, v=None, center=False):
    """ Transform data by normalized difference.

        The normalized difference of a time series X(t) is the ratio between its derivative and its value:
            X'(t)/X(t)
        To prevent division-by-zero error, X(t) is adjusted by a constant v.

        Args:
            X (pandas Series or DataFrame): input time series
            v (float): a fail-proof value
            center (bool): if True use centered two points difference
    """
    if not (isinstance(X, pd.DataFrame) or isinstance(X, pd.Series)):
        raise TypeError("Input must be a pandas DataFrame or Series")

    if v is None:
        # transformed data have a fixed magnitude range.
        v = X.std()*100
    X = X - X.min() + v  # X must be strictly positive

    if center:
        dX = (X.shift(-1) - X.shift(1))/2
    else:
        dX = X.diff()

    return dX/X


#### Interpolation and projection ####

def interpl_nans(X0):
    """Interpolation of nan values in a 1d array.
    """
    assert(X0.ndim==1)
    X = X0.copy()
    nidc = np.isnan(X)  # indicator of nan values
    nidx = nidc.nonzero()[0]; vidx = (~nidc).nonzero()[0]  # index of nan values and normal values
    X[nidc] = np.interp(nidx, vidx, X[~nidc])
    return X


def interpl_timeseries(T0, Y0, dtuple=(0,60*60,0), method="spline", rounded=False, **kwargs):
    """Regular resampling of a time-series by interpolation.

    Args:
        T0 (list of datetime objects): the equivalent of the sampling points in the usual interpolate function
        Y0 (1d array): values of the function at T0
        dtuple: a tuple of (day, second, microsecond) that define the length of the resampling time step
    Returns:
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
        raise Exception("Insufficient length of data: {}.".format(nbTime))

    dT0 = np.asarray([(t-t_start)/dt for t in T0])
    dT1 = np.arange(0, nbTime)
    T1 = [t_start + n*dt for n in dT1]

    # print(dT0, dT1)
    # print(T1)
    # print(dT0[-1], dT1[-1])

    if method=="spline":
        Yfunc = sp.interpolate.interp1d(dT0, Y0, kind="slinear")  # use "linear", "slinear" can result in artefacts (a periodicity of 37 days due to accumulation of 97 seconds per hour)
    elif method=="gp":
        Yfunc = lambda T: gp_interpl(dT0, Y0, dT1, **kwargs)
    else:
        raise TypeError("Unknown interpolation method")

    return Yfunc(dT1), T1


def gp_interpl(x_obs, y_obs, x_pred, nugget=1e-9):
    """Interpolation of missing data using GP

    Args:
        x_obs : coordinates of observations
        y_obs : values of observations
        x_pred : coordinates of predictions
    """
    from sklearn.gaussian_process import GaussianProcess

    gp = GaussianProcess(corr="cubic", theta0=1e-2, thetaL=1e-4, thetaU=1e-1, nugget = nugget, random_start=100)
    # Fit to data using Maximum Likelihood Estimation of the parameters
    gp.fit(np.atleast_2d(x_obs).T, y_obs)

    # Mesh the input space for evaluations of the real function, the prediction and
    # its MSE

    y_pred, MSE = gp.predict(np.atleast_2d(x_pred).T, eval_MSE=True)
    sigma = np.sqrt(MSE)

    return y_pred, sigma

@along_axis
def polyprojection(X, deg=1):
    """Projection onto the linear space generated by polynomial of a given degree.

    Args:
        X (nd array): input array.
        deg (int): degree of polynomials
        axis (int): axis along which to apply the 1d convolution.
    """
    assert(deg>=0)
    assert(X.ndim==1)
    Nt = len(X)

    xrg = np.arange(Nt)/Nt
    nidx = np.isnan(X)
    coef = np.polyfit(xrg[~nidx], X[~nidx], deg)  # polynomial coefficients
    V = np.vander(xrg, deg+1)  # Vandermond matrix

    return V @ coef

select_func = lambda x,y,c: y if abs(x-c)>abs(y-c) else x

def shrinkage(X, t, soft=True):
    """Shrinkage by a given threshold.

    Args:
        X (array): input
        t (float): threshold value
        soft (bool): hard or soft shrinkage
    Returns:
        Thresholded array
    """
    assert(t>=0)
    Y = np.zeros_like(X)
    idx = np.abs(X)>=t
    Y[idx] = np.abs(X[idx])-t if soft else np.abs(X[idx])
    return Y*np.sign(X)


def shrinkage_by_percentage(X0, thresh, soft=True):
    """Shrinkage by a given percentage of nonzeros.

    Args:
        X0 (1d array): input
        thresh (float): percentage of nonzeros to keep, eg. thresh=0.1 will keep the 10\% largest coefficients (in magnitude)
        soft (bool): hard or soft shrinkage
    Returns:
        Thresholded array and the number of kept nonzeros.
    """
    assert(X0.ndim==1)
    assert(0<=thresh<=1)

    if thresh == 1.0:
        return X0, len(X0)
    elif thresh == 0.0:
        return np.zeros_like(X0), 0
    else:
        X = X0.copy()
        X[np.isnan(X)] = 0  # remove all nans

        idx = np.argsort(abs(X))
        nz0 = int(np.floor(len(idx)*(1-thresh)))
        nz1 = int(np.ceil(len(idx)*(1-thresh)))

        zidx = idx[nz0:nz1+1]
        v = np.mean(np.abs(X[zidx]))
        return shrinkage(X, v, soft=soft), len(zidx)
        # X1 = np.zeros_like(X)
        # if soft:
        #     toto = X[zidx].copy()
        #     X1[zidx] = np.sign(toto)*(abs(toto)-abs(X[nz]))
        # else:
        #     X1[zidx]=X[zidx]
        #
        # return X1, len(zidx)


#### Outliers and detection ####

def remove_plateau_jumps(y0, wsize=10, thresh=5., bflag=False, dratio=0.5):
    """Remove plateau-like or sawtooth-like jumps.

    Plateau-like and sawtooth-like jumps:
          |---|          /\  /\
        --|   |--  or --/  \/  \--

    Args:
        y0 (1d array): input signal
        wsize (int): plateau size
        thresh (float): threshold for detection of jumps
        bflag (bool): if True apply boundary preserving filter
        dratio (float): ratio between the duration of a plateau jump and the size of y0
    Return:
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
    # method 1: using the small values of dy, this would require a larger threshold (eg > 10)
    fy = np.sort(dy); vthresh = np.mean(fy[:int(len(y0)*dratio)]) * thresh
    # method 2: using all values of dy, this seems to be more stable
    # vthresh = dy.mean()*thresh

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
    # We wipe out these residuals by detecting the alternation of ± sign
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


def detect_step_jumps(X0, method="diff", **kwargs):
    """Detect step jumps in a 1d array.

    Args:
        X0 (1d array): input array
        method (string): method of detection, by default use "diff"
    keyword parameters :
        thresh: factor of threshold, default value 10
        mwsize: window size for the moving average, default value 24
        median: location of the median position, default value 0.5
    Return:
        A list of detected jumps
    """
    assert(X0.ndim==1)

    if method=="diff":
        thresh = kwargs["thresh"] if "thresh" in kwargs.keys() else 10
        mwsize = kwargs["mwsize"] if "mwsize" in kwargs.keys() else 24
        median = kwargs["median"] if "median" in kwargs.keys() else 0.5

        # X1 = LU_filter(np.asarray(X0), 24) # Apply first a LU filter
        X1 = pd.Series(X0)
        mX = X1.rolling(mwsize, min_periods=1, center=True).mean()
        # mX = Stat.mwmean(X1, mwsize)
        dX = np.abs(np.hstack((0, np.diff(mX))))[::mwsize]

        sdX = np.sort(dX)
        # Take the modified median as threshold
        vthresh = thresh * sdX[int(len(sdX)*median)]
        pidx = np.where(dX>vthresh)[0] * mwsize
        # To do: add some aposteriori processing to adjust the location of jumps
        return list(pidx) #, dX, vthresh
    else:
        raise NotImplementedError("Unknown method: %s" % method)


def find_block_true(X):
    """Find the starting and ending positions of True blocks in a boolean array.

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
            sidx.append((a,b))
            n = b
        n += 1
    return sidx


def find_altsign(X, minlen=10):
    """Find the pattern of alternation of +/- sign in a 1d array.

    Args:
        X (1d array): input array
        minlen (int): minimal length of the pattern
    Return:
        a list of starting and ending positions of patterns
    """

    dS = np.hstack([0, np.diff(np.sign(X))])
    sidx0 = find_block_true(np.abs(dS)==2) # positions of alternation patterns
    # a pattern is confirmed if the alternation is sufficient long:
    sidx = []
    for k in sidx0:
        # print(k)
        if k[1]-k[0]>minlen:
            sidx.append((k[0]-1, k[1]))

    return sidx


def remove_nan_columns(*args):
    """Remove (common) nan columns from a (list of) 2d array.

    Args:
        *args: a 2d array (or a list of 2d arrays with the same number of columns).
    Returns:
        processed array(s) and a bool array containing the indexes of removed nan columns.
    """

    assert(len(args)>0 and args[0].ndim==2)
    Nt = args[0].shape[1]
    V = np.zeros(Nt, dtype=int)
    for Y in args:
        assert(Y.ndim==2 and Y.shape[1]==Nt)
        V += np.sum(np.isnan(Y), axis=0)
    cidx = ~(V>0)  # columns do not containing nan
    Z = [Y[:,cidx] for Y in args]

    if len(args)>1:
        return Z, cidx
    else:
        return Z[0], cidx


#### Datetime and time-series related ####

def time_findgap(T0, dtuple=(0, 60*60, 0)):
    """
    Find gaps in a time-stamp series.

    Args:
        T0: list of time-stamp
        dtuple: a tuple of (day, second, microsecond) that define the minimal size of the gap
    Returns:
        G: index of gaps
    """
    dt = datetime.timedelta(*dtuple)  # size of the gap

    G = np.asarray(np.where(np.diff(T0) > dt)[0]+1, dtype=int)
    return G
    # # G = np.int32(np.hstack([0, G, len(T0)]))
    #
    # sidx = []
    # for n in range(1,len(G)):
    #     sidx.append((G[n-1], G[n]))
    #
    # return sidx


def time_ceil(t0, unit="hour"):
    """
    Return the instant next to t0 with rounded unit (hour or minute).
    """
    if isinstance(t0, str):
        t0 = dateutil.parser.parse(t0)

    if unit == "hour":  # return the next instant with rounded hour
        flag = t0.minute > 0 or t0.second > 0 or t0.microsecond > 0
        t = t0 + datetime.timedelta(0, 60*60*flag)
        return t.replace(minute=0, second=0, microsecond=0)
    elif unit == "minute":  # return the next instant with rounded minute
        flag = t0.second > 0 or t0.microsecond > 0
        t = t0 + datetime.timedelta(0, 60*flag)
        return t.replace(second=0, microsecond=0)
    else:  # return the next instant with rounded second
        flag = t0.microsecond > 0
        t = t0 + datetime.timedelta(0, 1*flag)
        return t.replace(microsecond=0)


def time_floor(t0, unit="hour"):
    """
    Return the instance just before t0 with rounded unit (hour or minute).
    """
    if isinstance(t0, str):
        t0 = dateutil.parser.parse(t0)

    if unit == "hour":  # return the next instant with rounded hour
        return t0.replace(minute=0, second=0, microsecond=0)
    elif unit == "minute":  # return the next instant with rounded minute
        return t0.replace(second=0, microsecond=0)
    else:  # return the next instant with rounded second
        return t0.replace(microsecond=0)


def time_range(t0, N, dtuple=(0, 60*60, 0)):
    """
    Generate a list of N datetime objects starting from the instant t0 with a
    given step dt.

    Args:
        t0 (datetime object): starting instant
        N (int): length of the list
        dtuple (tuple): (day, second, microsecond) that defines the sampling step dt
    Return:
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

    Args:
        T0 : list of datetime object
        t0 : datetime object
            the starting instant
    Return:
        An array with the n-th element given by T0[n]-t0 in seconds
    """
    dT0 = np.zeros(len(T0))  # relative time stamp wrt the T0[0], in seconds

    for i in range(len(T0)):
        toto = T0[i]-t0
        dT0[i] = toto.days*24*3600+toto.seconds  # +toto.microseconds/1000/1000

    return dT0


def str2time(t0):
    """
    Convert a string to a datetime object.
    """
    return [dateutil.parser.parse(t) for t in t0]


# def timeseries_list2array(X0, Tl, dtuple=(0, 60*60, 0)):
#     """
#     Convert a list of arrays X0 to an array using the information of the given
#     timeseries.
#
#     Args:
#         X0: a list of arrays (of float), X0[n] has the same length as Tl[n]
#         Tl: Tl[n] is a list of time-stamps, see timeseries_list2idx
#         dtuple : see timeseries_list2idx
#     """
#     gidx, _ = timeseries_list2idx(Tl, dtuple=dtuple)
#     X1 = rejoin_timeseries(X0, gidx)
#     T1 = rejoin_timeseries(Tl, gidx)
#
#     return np.asarray(X1, dtype=np.float64), T1


def timeseries_list2idx(Tl, dtuple=(0, 60*60, 0)):
    """
    Convert a list of time-stamp series into position indexes with respect to
    the first time-stamp and with a given sampling step which is only applied on the gap.

    Args:
        Tl (list): Tl[n] is a list of time-stamps aranged in increasing order. For m>n Tl[n] is older than Tl[m], and they are mutually non-interlapped.
        dtuple (tuple): a tuple of (day, second, microsecond) that define the sampling step on the gap.
    Returns:
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


#### Linear algebra related ####

def issymmetric(A):
    """Check if a matrix is symmetric.
    """
    if A.ndim!=2 or A.shape[0]!=A.shape[1]:
        return False
    else:
        return np.allclose(A.T, A)


def ispositivedefinite(A):
    """Check if a matrix is positive definite.
    """
    if A.ndim!=2:
        return False
    else:
        return np.all(np.linalg.eigvals(A) > 0)


def mat_transpose_op(dimr, dimc):
    ovec = np.zeros(dimc); ovec[0]=1
    Cmat = np.kron(np.eye(dimr), ovec)
    Lmat = np.zeros((dimr*dimc, dimr*dimc))

    for n in range(dimc):
        Lmat[n*dimr:(n+1)*dimr,:] = np.roll(Cmat, n, axis=1)

    return np.asarray(Lmat, dtype=bool)  # convert to bool to save memory

# # test
# dimr, dimc = 100, 50
# Top = mat_transpose_op(dimr,dimc)
# X = rand(dimr, dimc)
# norm(dot(Top, X.flatten()).reshape(dimc, dimr)-X.T)  # must be 0

def c2f_op(dimr, dimc):
    """C-order(row) array representation to F-order(column) representation
    """
    return mat_transpose_op(dimr, dimc)


def f2c_op(dimr, dimc):
    """F-order array representation to C-order representation
    """
    return mat_transpose_op(dimc, dimr)


def matprod_op_right(A, dimXr):
    """Linear operator representation of the matrix product at right (i.e. L(X) = X*A)
    """
    dimXc = A.shape[0]
    TopX = mat_transpose_op(dimXr, dimXc)
    LopT = matprod_op_left(A.T, dimXr)
    TopL = mat_transpose_op(A.shape[1], dimXr)

    return TopL @ matprod_op_left(A.T, dimXr) @ TopX


def matprod_op_left(A, dimXc):
    """Linear operator representation of the matrix product at left (i.e. L(X) = A*X)
    """
    dimXr = A.shape[1]
    Tl = f2c_op(A.shape[0], dimXc)
    Tr = c2f_op(dimXr, dimXc)
    return Tl @ np.kron(np.eye(dimXc), A) @ Tr


@along_axis
def safe_slice(X0, tidx, wsize, mode="soft", causal=False):
    """Take a slice of a nd array along given axis with safe behavior on the boundaries.

    Args:
        X0 (1d or 2d array): each column is one observation
        tidx (int): position where the slice will be taken
        wsize (int): size of the slice
        axis (int): axis along which the slice will be taken
        mode (str): "hard" (simple truncation at the boundary) or "soft" (continuation by zero at the boundary).
        causal (bool): if True the window ends at tidx, otherwise the window is centered at tidx.
    Return:
        A slice of X0.
    """

    # zero padding
    X = np.zeros(X0.size+2*(wsize-1))
    X[wsize-1:(wsize-1+X0.size)] = X0

    if causal:  # index of the slice
        tidx0 = tidx
        tidx1 = tidx0+wsize
    else:
        tidx0 = tidx-int(wsize/2)+wsize-1
        tidx1 = tidx0+wsize

    if mode == "soft":  # soft mode, with zero-padding
        return X[tidx0:tidx1]
    else:  # hard mode, without zero-padding
        tidx0 = max(0, tidx0-(wsize-1))
        tidx1 = min(X0.size, tidx1-(wsize-1))
        return X0[tidx0:tidx1]


safe_norm = nan_safe(la.norm, endo=False)

