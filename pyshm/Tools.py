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
from functools import wraps

from . import Stat

#### Functional operators ####

def nan_safe(func):
    """Decorator that make a function safe to nan by replacing nan values."""
    @wraps(func)
    def newfunc(X0, *args, nv=0, **kwargs):
        X = X0.copy()
        nidx = np.isnan(X0)
        X[nidx] = nv
        return func(X, *args, **kwargs)
    return newfunc

# def along_axis(func):
#     """Decorator functional for applying a function on a nd array along a given axis.
#     """
#     def newfunc(X, *args, axis=-1, **kwargs):
#         np.apply_along_axis(func, axis, X, *args, **kwargs)
#     return newfunc

def along_axis(func):
    """Decorator for applying a function on a nd-array along a given axis.
    """
    @wraps(func)
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


def cumop(func):
    """Decorator of cumulative operation.
    """
    @wraps(func)
    def newfunc(X, *args, **kwargs):
        Y = np.zeros(len(X))
        for t in range(len(X)):
            Y[t] = func(X[:t+1], *args, **kwargs)
        return Y
    return newfunc


def recursive(niter):
    """Decorator for recursion.
    """
    def _inner(func):
        def _recursive(niter, x, *args, **kwargs):
            if niter>0:
                return func(_recursive(niter-1, x, *args, **kwargs), *args, **kwargs)
            return x
        return _recursive
    return _inner


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

@along_axis
def safe_convolve(X, kernel, mode="valid"):
    """FFT based nan-safe convolution.

    Unlike ``convolve``, the numpy function ``fftconvolve`` is not safe to nan values. This function is a wrapper of ``fftconvolve`` by applying a mask on the nan values.

    Args:
        X (1d array): input array.
        kernel (1d array): convolution kernel.
        mode (str): truncation mode of the convolution, can be ["valid", "full", "samel", "samer"], where "samel" means the leftmost end is not valid, and "samer" means the rightmost end is not valid.
    Returns:
        Result of convolution.
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

    The circular convolution between the 1d arrays `x` and `y` is the convolution between periodized version of `x` and `y`. This function computes the circular convolution via fft.
    """
    assert(x.ndim == y.ndim == 1)  # only operate on 1d arrays

    N = max(len(x), len(y))
    res = np.fft.ifft(np.fft.fft(x, N) * np.fft.fft(y, N))  # convolution by fft

    if any(np.iscomplex(x)) or any(np.iscomplex(y)):  # take the real part
        return res
    else:
        return np.real(res)


@along_axis
def convolve_fwd(X, psi, mode="valid"):
    """Forward operator of the convolution considered as a linear operator.

    Args:
        X (nd array): input array.
        psi (1d array): convolution kernel.
        mode (str): truncation mode of the convolution.
        axis (int): axis along which to apply the 1d convolution.
    """
    return safe_convolve(X, psi, mode=mode)


@along_axis
def convolve_bwd(X, psi, mode="full"):
    """Backward operator of the convolution considered as a linear operator.
    """
    # the following is equivalent to
    # convolve(X[::-1], psi, mode=mode)[::-1]
    return safe_convolve(X, psi[::-1], mode=mode)


def convolve_matrix(h,Nx,mode="valid"):
    """Matrix representation of the convolution operator.

    Args:
        h (1d array): convolution kernel
        Nx (int): length of the input signal
        mode (str): mode of truncation, ["valid", "samel", "samer", "all"]
    Returns:
        A matrix.
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
    """Convolution kernel of a differential operator.

    The kernel is determined by its degree and step, for example, for degree 0 and step 1 the kernel is :math:`[1,-1]`, and for step 2 the kernel is :math:`[1, 0, -1]`. For higher degree the kernel is obtained using auto-convolution of the kernel of degree 0.

    Args:
        deg (int): degree of the differential kernel.
        step (int): step of the differential kernel.
    Returns:
        The kernel convolution.
    """
    assert(deg>=0)
    h0 = np.zeros(1+step)
    h0[0]=1; h0[-1]=-1; h0/=step

    h = h0
    for n in range(deg):
        h = sp.signal.convolve(h0, h0) if n==0 else sp.signal.convolve(h,h0)
    return h


def mts_cumview(X, N):
    """Cumulative view of a multivariate time series (*mts*).

    Args:
        X (2d array): each row is a variable, each column is an observation of all variables.
        N (int): size of the cumulation, N>=1.
    Returns:
        A 2d array that the :math:`t`-th column is the concatenation of the N past values :math:`(X[t], X[t-1]...X[t-N+1])`.
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

    The KZ filter is nothing but the recursive application of the moving average, it has length (mwsize-1)*k+1. Here we extend KZ filter using the moving median.

    Args:
        X0: input pandas DataFrame
        mwsize (int): size of moving window
        kzord (int): order of KZ filter
        method (str): "mean" or "median"
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
    """Non-linear upper filter.

    Args:
        X (1d array): input
        wsize (int): size of the moving window
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
    """Boundary-preserving upper filter.
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
    """Non-linear lower filter.
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
    """Boundary-preserving lower filter.
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


#### Datetime and time-series related ####

def time_findgap(T0, dtuple=(0, 60*60, 0)):
    """Find gaps in a list of timestamps.

    Args:
        T0: pandas DatetimeIndex
        dtuple: a tuple of (day, second, microsecond) that define the size of the gap
    Returns:
        G: index of gaps
    """
    dt = datetime.timedelta(*dtuple)  # size of the gap

    G = np.asarray(np.where(np.diff(T0.to_pydatetime()) > dt)[0]+1, dtype=int)
    return G
    # # G = np.int32(np.hstack([0, G, len(T0)]))
    #
    # sidx = []
    # for n in range(1,len(G)):
    #     sidx.append((G[n-1], G[n]))
    #
    # return sidx


#### Transformation ####

def roll_fill(X0, shift, val=np.nan, axis=-1):
    """A wrapper of numpy roll function by setting the cyclic head/tail to a given value.
    """
    X = np.roll(X0, shift, axis=axis)
    slc=[slice(None,None,None)]*X0.ndim
    if shift>=0:
        slc[axis] = slice(0,shift,None)
    else:
        slc[axis] = slice(shift,None,None)
    X[slc] = val
    return X


def sdiff(X, p, axis=-1):
    """Seasonal difference :math:`X[t] - X[t-p]`.

    Args:
        X (nd array): the last dimension is the time axis
        p (int): seasonality
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


#### Interpolation and projection ####

def interpl_nans(X0):
    """Interpolation of nan values in a 1d array using :func:`numpy.interp`.
    """
    assert(X0.ndim==1)
    X = X0.copy()
    nidc = np.isnan(X)  # indicator of nan values
    nidx = nidc.nonzero()[0]; vidx = (~nidc).nonzero()[0]  # index of nan values and normal values
    X[nidc] = np.interp(nidx, vidx, X[~nidc])
    return X


def gp_interpl(x_obs, y_obs, x_pred, nugget=1e-9):
    """Interpolation of missing data using :class:`sklearn.gaussian_process.GaussianProcess`

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
    """Projection onto the linear space generated by polynomials of a given degree using :func:`numpy.polyfit`.

    Args:
        X (nd array): input array.
        deg (int): degree of polynomials.
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


def shrinkage(X0, t, soft=True):
    """Shrinkage by a given threshold.

    Args:
        X0 (array): input
        t (float): threshold value
        soft (bool): hard or soft shrinkage
    Returns:
        Thresholded array
    """
    assert(t>=0)
    X = X0.copy()
    X[np.isnan(X)] = 0  # remove all nans
    Y = np.zeros_like(X)
    idx = np.abs(X)>=t
    Y[idx] = np.abs(X[idx])-t if soft else np.abs(X[idx])
    return Y*np.sign(X), idx


def shrinkage_percentile(X0, thresh, soft=True):
    """Shrinkage by a given percentage of nonzeros.

    Args:
        X0 (1d array): input
        thresh (float): percentage of nonzeros to keep, eg. thresh=0.1 will keep the 10\% largest coefficients (in magnitude)
        soft (bool): hard or soft shrinkage
    Returns:
        Thresholded array
    """
    assert(X0.ndim==1)
    assert(0<=thresh<=1)

    if thresh == 1.0:
        return X0
    elif thresh == 0.0:
        return np.zeros_like(X0)
    else:
        v = Stat.percentile(np.abs(X0), thresh)
        return shrinkage(X0, v, soft=soft)


#### Outliers and detection ####

def remove_plateau_jumps(y0, wsize=10, bflag=False, thresh=5., dratio=0.5):
    """Remove plateau-like or sawtooth-like jumps.

    This function applies first a non-linear LU filter to transform sawtooth-like jumps to plateau jumps, then detect significant plateau jumps by computing the threshold from given arguments. For each jump exceeding the threshold value, an appropriate action is determined by a histogram based analysis.

    Args:
        y0 (1d array): input signal
        wsize (int): size of the moving window of the LU-filter
        bflag (bool): if True apply boundary preserving LU-filter
        thresh (float): threshold for detection of jumps
        dratio (float): ratio between the duration of a plateau jump and the size of y0
    """
    assert y0.ndim == 1

    # 1. Apply L and U filter to transform sawtooth-like jumps to plateau jumps
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

    _, pflag = shrinkage(dy, vthresh)
    # pflag = np.abs(sdy)>0 # might be all False if whole y0 is sawtooth-like
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
        for X=[0, 1, 1, 1, 0, 1], the return is [[1,4], [5,6]]
    """
    n = 0
    sidx = []
    while n < len(X):
        # print(X.shape, X[n])
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

    Example:
        for X=[0,1,-1,1,-1,0,0], the return is [1,5]
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

#### Linear algebra related ####

def cprod(a,b):
    """Compute the product a*(a+1)...*b with correct handling of singular cases.
    """
    assert b >= 0 and a >= 0
    v = 1
    for t in range(a, b+1):
        v *= t
    return v


def dpvander(v, pord, dord):
    """Modified Vandermond matrix of the derivative of a polynomial.
    """
    Vmat = np.rot90(np.vander(v, pord-dord+1))[::-1,:]  # decreasing order
    fcof = np.asarray([cprod(s, s+dord-1) for s in range(1, pord-dord+2)])[::-1] # factorial coefficents due to derivative
    return np.diag(fcof) @ Vmat


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
    """Matrix-transpose operator.
    """
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


safe_norm = nan_safe(la.norm)
cummax = along_axis(cumop(nan_safe(np.max)))
cummin = along_axis(cumop(nan_safe(np.min)))
cumstd = along_axis(cumop(nan_safe(np.std)))

#### Plot ####
def plot(X0, T=None, mask=None):
    assert X0.ndim == 1
    X = X0.copy()
    if mask is not None:
        X[mask] = np.nan
    from matplotlib import pyplot as plt
    fig, axa = plt.subplots(1,1,figsize=(20,5))
    if T is not None:
        axa.plot(T, X)
    else:
        axa.plot(X)
    return fig, axa