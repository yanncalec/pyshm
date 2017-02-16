"""Statistics related functions.
"""

import numpy as np
import numpy.linalg as la
import pandas as pd
# from numpy import newaxis, mean, sqrt, zeros, ones, squeeze,\
#     asarray, abs
# from numpy.linalg import norm, svd, inv, pinv

# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
# from sklearn.gaussian_process import GaussianProcess
from scipy import signal

from . import Tools


#### Regression analysis ####

def MLR_split_safe_call(func):
    """Decorator functional for multiple linear regression routines.

    This decorator applies dimension check and remove nan values (from inputs), and split the output of MLS routine into list of regressors that each correspongds to an explanary variable.
    """
    def newfunc(Y0, X0, *args, **kwargs):
        # check of dimensions
        assert(Y0.ndim==2 and X0.ndim==2)
        assert(Y0.shape[1] == X0.shape[1])  # X and Y must have same number of observations
        Nt0 = X0.shape[1]  # original length of time series

        # if more than one exogeneous input exist, save extra dimensional informations
        Xs = [X0]
        dimXs = [X0.shape[0]]
        for X in args:
            assert(X.ndim==2 and X.shape[1] == Nt0)
            Xs.append(X)
            dimXs.append(X.shape[0])
        dimXs = np.cumsum(dimXs)  # cumulative dimensions
        Xs = np.vstack(Xs)

        # Remove simultaneously from X and Y all columns containing nan
        (X,Y),_ = Tools.remove_nan_columns(Xs, Y0)

        L, Cvec = func(Y, X, **kwargs)

        # Processing on the residuals
        Err = Y0 - (np.dot(L, Xs) + Cvec)  # Err has the same length as Y0
        Ern,_ = Tools.remove_nan_columns(Err)
        # Sig = la.norm(Err,"fro")**2 / (Y.size - Y.shape[0]*X.shape[0])
        Sig = np.dot(Ern, Ern.T) / (Y.size - Y.shape[0]*X.shape[0])  # covariance matrix

        # the second argument tells hsplit where to split the columns of L
        return np.hsplit(L, dimXs[:-1]), Cvec, Err, Sig

    newfunc.__doc__ = func.__doc__
    # newfunc.__name__ = func.__name__
    return newfunc


@MLR_split_safe_call
def multi_linear_regression_ls(Y, X, constflag=False, penal=0.):
    """Multiple linear regression by least-square method.

    It looks for the matrices Li and the vector C such that
        the Frobenius norm ||L0*X0 + L1*X1 + ... + C - Y||  (* denotes the matrix product)
    is minimized.

    Args:
        Y (2d array): observations variables, each row is a variable and each column a single observation.
        X (2d array): explanatory variables, same form as Y. X must have the same number of columns as Y.
        *args (2d arrays): other groups of explanatory variables.
        constflag (bool): if True fit the model with the constant vector C.
        penal (float): penality
    Returns:
        L: list of estimated matrix
        C: constant vector
        Err: residual error matrix
        Sig: estimation of the noise covariance matrix
    """
    Nt = X.shape[1]  # number of non nan oservations
    dimX, dimY = X.shape[0], Y.shape[0]

    # shorthands
    sX = np.sum(X, axis=1)[:, np.newaxis]  # column vector
    sY = np.sum(Y, axis=1)[:, np.newaxis]  # column vector
    YXt = np.dot(Y, X.T)
    XXt = np.dot(X, X.T)
    # PM = 1e-2*penal*np.eye(dimX)+penal*np.ones((dimX,dimX))
    PM = penal*np.ones((dimX,dimX))
    iXXt = la.inv(XXt + PM)
    sXtiXXt = np.dot(sX.T, iXXt)
    L0 = np.dot(YXt, iXXt)  # of dimension dimY by dimX

    if constflag:
        v1 = sY - np.dot(L0, sX)
        v2 = Nt - np.dot(sXtiXXt, sX)
        Cvec = v1 / v2  # this formula conforms with the scalar linear regression case
    else:
        Cvec = np.zeros((dimY,1))
    L = L0 - np.dot(Cvec, sXtiXXt) #np.dot(np.diag(C), np.dot(XUt.T, iXXt))

    return L, Cvec


@MLR_split_safe_call
def multi_linear_regression_corr(Y, X, constflag=False):
    """Multiple linear regression by correlation method.

    This function solve the linear regression problem using the analytical formula
        L = cov(Y, X) * cov(X,X)^-1
        C = mean(Y) - L * mean(X)
    It has the same interface as multi_linear_regression_ls().
    """
    dimY = Y.shape[0]
    dimX = X.shape[0]

    # covariance matrices
    Sm = np.cov(Y, X)
    Syx = Sm[:dimY, dimY:]
    Sxx = Sm[dimY:, dimY:]

    # column mean vectors
    mX = np.atleast_2d(np.mean(X, axis=1)).T
    mY = np.atleast_2d(np.mean(Y, axis=1)).T

    L = np.dot(Syx, la.inv(Sxx))
    Cvec = mY - np.dot(L, mX) if constflag else np.zeros((dimY, 1))

    return L, Cvec


# def linear_regression(Y, X, constflag=False):
#     """Linear regression of the model Y = aX + b + e
#
#     Args:
#         Y (1d array): the observation variable
#         X (1d array): the explanatory variable
#
#     Returns:
#         a, b: the least square solution (using scipy.sparse.linalg.cgs)
#         err: residual error
#         sigma2: estimation of the noise variance
#     """
#     L, Cvec, Err, Sig = multi_linear_regression_ls(np.atleast_2d(Y), np.atleast_2d(X), constflag=constflag)
#     return L[0][0], Cvec[0], Err[0], Sig[0]


def linear_regression(Y, X, nanmode="remove"):
    """
    Linear regression of the model:
        Y = aX + b

    Args:
        Y (1d array): the observation variable
        X (1d array): the explanatory variable
        nanmode (str): what to do with nan values. "interpl": interpolation, "remove": remove, "zero": replace by 0
    Returns:
        a, b: the least square solution (using scipy.sparse.linalg.cgs)
        err: residual error
        sigma2: estimation of the noise variance
        S: variance of the estimate a and b (as random variables)
    """
    assert(Y.ndim==X.ndim==1)
    assert(len(Y)==len(X))

    if nanmode == "remove":
        (X0, Y0), nidx = Tools.remove_nan_columns(np.atleast_2d(X), np.atleast_2d(Y))  # the output is a 2d array
        X0 = X0[0]; Y0 = Y0[0]  # convert to 1d array
    elif nanmode == "interpl":
        X0 = Tools.interpl_nans(X)
        Y0 = Tools.interpl_nans(Y)
    else:
        X0 = X.copy()
        Y0 = Y.copy()

    # final clean-up
    X0[np.isnan(X0)] = 0; Y0[np.isnan(Y0)] = 0

    if len(X0)>0 and len(Y0)>0:
        A = np.vstack([X0, np.ones(len(X0))]).T
        a, b = np.dot(la.pinv(A), Y0)

        err = a*X + b - Y         # residual
        # non-biased estimation of noise's variance
        A_rank = la.matrix_rank(A)
        if A.shape[0] > A_rank:
            sigma2 = Tools.safe_norm(err)**2 / (A.shape[0] - A_rank)
        else:
            sigma2 = np.nan
        # assert(sigma2 >= 0 or np.isnan(simga2))
        S = sigma2*np.diag(la.pinv(A.T @ A))
    else:
        raise ValueError("Linear regression failed due to lack of enough meaningful values in the inputs.")
        # a, b, err, sigma2, S = np.nan, np.nan, np.nan*np.zeros_like(Y), np.nan, np.nan*np.zeros(2)

    return a, b, err, sigma2, S


#### Moving window estimation ####

def optimal_delay(X, Y, tidx, dlrange):
    """Estimate the local optimal delay of a time series X wrt another Y using linear regression.

    Given the time index tidx and the range dlrange of validate delay, the
    optimal delay on X is determined by selecting a subsequence of size of Y
    on a moving window centered around tidx, such that the error of linear
    regression between the windowed X and Y is minimized.

    Returns:
    dt: estimated optimal delay
    res: result of linear regression
    corr: optimal correlation
    Xd: optimal delayed slice of X
    """
    res = []

    for n in range(*dlrange):
        Xn = Tools.safe_slice(X, tidx-n, Y.size, mode="soft")
        try:
           res.append(linear_regression(Y, Xn))
        except Exception:
            res.append(None)

    # the optimal delay is taken as the one minimizing the residual of LS
    toto = np.asarray([np.nan if r is None else la.norm(r[2]) for r in res])
    if np.isnan(toto).any():
        # at least one of the linear regression failed, we cannot use
        # argmin/argmax which always return the index of the first nan if the
        # array contains nan

        # nidx = None
        raise ValueError("Optimal delay cannot be determined due to nan(s).")
    else:
        # all linear regressions are successful
        nidx = np.argmin(toto)

        dt = dlrange[0] + nidx
        Xd = Tools.safe_slice(X, tidx-dt, Y.size, mode="soft")
        # print("corr = {}".format(corr(Xd,Y)))
        return dt, res[nidx], corr(Xd, Y), Xd
    # else:
    #     return np.nan, None, np.nan, None
        # return None, None, None, None


def mw_linear_regression_with_delay(Y0, X0, D0=None, wsize=24*10, dlrange=(-6,6)):
    """Moving window linear regression of two time series with delay.

    We suppose that X is related to Y through:
        Y[t] = K[t]*X[t-D[t]] + B[t] + error
    Provided the delay D, the linear regression estimates the scalars K[t], B[t]
    on a moving window centered around t.  If D is not given, the function has
    to estimate the optimal delay D[t] of X by solving a series of linear
    regression problems with a range of trial values and select the one
    minimizing the residual.

    Args:
        Y0 (1d array): observation variable
        X0 (1d array): explanatory variable
        D0 (1d array of int): delay, if provided then D0 is used as delay for linear regression and dlrange will be ignored (no estimation of delay)
        wsize (int): size of the moving window
        dlrange (tuple of int):  range in which the optimal delay is searched
    Returns:
        D, C, K, B: estimated delay (if D0 not given), correlation, slope, intercept

    """
    assert(X0.ndim == Y0.ndim == 1)
    assert(len(X0) == len(Y0))

    # quantities after compensation of thermal delay
    D = np.zeros(len(X0)) if D0 is None else D0
    C = np.zeros(len(X0))
    K = np.zeros(len(X0))
    B = np.zeros(len(X0))

    for tidx in range(len(X0)):
        # select Y on a window
        y = Tools.safe_slice(Y0, tidx, wsize, mode="soft")
        if D0 is None:
            # if delay is not provided, estimate the optimal delay
            try:
                D[tidx], res, C[tidx], _ = optimal_delay(X0, y, tidx, dlrange)
                K[tidx], B[tidx] = res[0], res[1]
            except Exception as msg:
                # print(msg)
                D[tidx], C[tidx], K[tidx], B[tidx] = np.nan, np.nan, np.nan, np.nan
        else:
            Xd = Tools.safe_slice(X0, tidx-D0[tidx], y.size, mode="soft")
            try:
                K[tidx], B[tidx], *_ = linear_regression(y, Xd)
                # K[tidx], B[tidx] = res[0], res[1]
                C[tidx] = corr(Xd, y)
            except Exception as msg:
                # print(msg)
                K[tidx], B[tidx], C[tidx] = np.nan, np.nan, np.nan

    return D, C, K, B

    # for tidx in range(len(X0)):
    #     yidx0 = max(0, tidx-wsize//2)
    #     yidx1 = min(yidx0 + wsize, len(X0))
    #
    #     res = []  # results of linear regression
    #     xidxs = []  # to keep the index range of delayed X
    #
    #     for t in range(*dlrange):
    #         xidx0 = max(0, tidx+t-wsize//2)
    #         xidx1 = min(xidx0 + wsize, len(X0))
    #         res.append(linear_regression(Y0[yidx0:yidx1], X0[xidx0:xidx1]))
    #         xidxs.append((xidx0, xidx1))
    #
    #     midx = np.argmin([r[2] for r in res])  # index correpsonding to the minimum residual
    #     xidx0, xidx1 = xidxs[midx]
    #     D[tidx] = dlrange[0] + midx  #
    #     C[tidx] = corr(X0[xidx0:xidx1], Y0[yidx0:yidx1])
    #     K[tidx] = res[midx][0]
    #     B[tidx] = res[midx][1]


def local_statistics(X, mwsize, mad=False, causal=False, drop=True):
    """Local mean and standard deviation estimation using pandas library.
    """
    if isinstance(X, pd.Series) or isinstance(X, pd.DataFrame):
        Err = X
    else:
        if X.ndim == 1:
            Err = pd.Series(X)
        else:
            Err = pd.DataFrame(X.T)

    if mad:  # use median-based estimator
        mErr = Err.rolling(window=mwsize, min_periods=1, center=not causal).median() #.bfill()
        # sErr = 1.4826 * (Err-mErr).abs().rolling(window=mwsize, min_periods=1, center=not causal).median() #.bfill()
        sErr = (Err-mErr).abs().rolling(window=mwsize, min_periods=1, center=not causal).median() #.bfill()
    else:
        mErr = Err.rolling(window=mwsize, min_periods=1, center=not causal).mean() #.bfill()
        sErr = Err.rolling(window=mwsize, min_periods=1, center=not causal).std() #.bfill()

    # drop the begining
    if drop:
        mErr.iloc[:int(mwsize*1.1)]=np.nan
        sErr.iloc[:int(mwsize*1.1)]=np.nan

    if isinstance(X, pd.Series) or isinstance(X, pd.DataFrame):
        return mErr, sErr
    else:
        return np.asarray(mErr), np.asarray(sErr)


def Hurst(data, mwsize, sclrng=None, wvlname="haar"):
    """Estimate the Hurst exponent of a time series using wavelet transform.

    Args:
        data (1d array): input time series
        mwsize (int): size of the smoothing window
        sclrng (tuple): index of scale range used for linear regression
    Returns:
        H (1d array): estimate of Hurst exponent
        B (1d array): estimate of intercept
        V (2d array): estimate of variance of H and B
    """
    import pywt
    import pandas as pd

    Nt = len(data)  # length of data
    wvl = pywt.Wavelet(wvlname)  # wavelet object
    maxlvl = pywt.dwt_max_level(Nt, wvl)  # maximum level of decomposition
    if sclrng is None:  # range of scale for estimation
        sclrng = (0, maxlvl)
    else:
        sclrng = (max(0,sclrng[0]), min(maxlvl, sclrng[1]+1))
    # Compute the continuous wavelet transform
    C0 = []
    for n in range(1, maxlvl+1):
        phi, psi, x = wvl.wavefun(level=n) # x is the grid for wavelet
        # C0.append(scipy.signal.fftconvolve(data, psi/2**((n-1)/2), mode="same"))
        C0.append(Tools.safe_convolve(data, psi/2**((n-1)/2), mode="samel"))
    C = np.asarray(C0)  # matrix of wavelet coefficients, each column is a vector of coefficients

    # Compute the wavelet spectrum
    S = np.asarray(pd.DataFrame((C**2).T).rolling(window=mwsize, center=True, min_periods=1).mean()).T  # of dimension maxlvl-by-Nt
    # S = C**2
    # S0 = []
    # for n in range(0, maxlvl-1): #sclrng):
    #     Cs = pd.Series(C[n, :]**2)  # wavelet spectrum in pandas format
    #     S0.append(np.asarray(Cs.rolling(window=mwsize, center=True, min_periods=1).mean())) #, win_type="boxcar").mean())
    # S = np.asarray(S0)

    # Linear regression
    H, B, V = np.zeros(Nt), np.zeros(Nt), np.zeros((2, Nt))
    xvar = np.arange(*sclrng)  # explanatory variable
    for t in range(Nt):
        yvar = np.log2(S[sclrng[0]:sclrng[1],t])
        try:
            a, b, err, sigma2, v = linear_regression(yvar, xvar, nanmode="interpl")
        except:
            a, b, err, sigma2, v = np.nan, np.nan, None, np.nan, np.nan * np.ones(2)

        H[t] = max(min((a-1)/2,1), 0)
        B[t] = b
        V[:,t] = v

    # roll to get a causal result
    sc = mwsize // 2
    H = np.roll(H, -sc); H[-sc:] = np.nan
    B = np.roll(B, -sc); B[-sc:] = np.nan
    V = np.roll(V, -sc); V[:,-sc:] = np.nan

    # drop the begining
    # H[:mwsize] = np.nan
    # B[:mwsize] = np.nan
    # V[:,:mwsize] = np.nan
    return H, B, V


# #### Generic ####

def sign_safe_svd(A):
    """SVD with coherent sign pattern.
    """
    U, S, V0 = la.svd(A)
    V = V0.T # U @ diag(S) @ V.T = A
    N = len(S)

    sl = np.zeros(N)
    sr = np.zeros(N)

    for n in range(N):
        # toto = U[:,n] @ A
        # sl[n] = np.sign(toto) @ (toto**2)
        # toto = A @ V[:,n]
        # sr[n] = np.sign(toto) @ (toto**2)

        toto = U[:,n] @ (A / la.norm(A, axis=0)[np.newaxis,:])
        sl[n] = np.sum(toto)

        toto = (A / la.norm(A, axis=1)[:,np.newaxis]) @ V[:,n]
        sr[n] = np.sum(toto)

        if sl[n] * sr[n] < 0:
            if sl[n] < sr[n]:
                sl[n] = -sl[n]
            else:
                sr[n] = -sr[n]

    U[:,:N] = U[:,:N] @ np.diag(np.sign(sl))
    V[:,:N] = V[:,:N] @ np.diag(np.sign(sr))

    return U, S, V.T


def pca(X0, nc=None, sflag=False):
    """
    Principal Component Analysis.

    Args:
        X0 (2d array): each row represents a variable and each column represents an observation
        nc (int): number of components to hold
        sflag (bool): if True apply sign correction to the principal vectors
    Returns:
        C, U : coefficients and corresponded principal directions
    """

    X0 = normalize(X0, std=False) # remove the mean
    # U0, S, _ = sign_safe_svd(np.cov(X0))
    U0, S, _ = la.svd(np.cov(X0))
    U = U0.copy()
    # sign correction:
    if sflag:
        X1 = X0/la.norm(X0, axis=0)[np.newaxis,:]
        for n in range(U.shape[1]):
            toto = U[:,n] @ X1

            # method 1:
            # toto = toto[np.abs(toto)>0.3]
            # method 2:
            # idx0 = np.argsort(np.abs(toto))[::-1]
            # idx = idx0[:max(1,int(len(idx0)/4))]

            # if np.sign(toto) @ (toto**4) < 0:
            #     U[:,n] *= -1

            if np.mean(toto) < 0:
                U[:,n] *= -1
    C  = U.T @ X0

    if nc is None:
        return C, U
    else:
        return C[:nc,:], U[:,:nc]

@Tools.nan_safe
def corr(x0, y0):
    x = np.atleast_2d(x0)
    y = np.atleast_2d(y0)
    toto = np.corrcoef(x, y)[:x.shape[0], x.shape[0]:]
    # print(len(x), len(y), x.shape, toto0)
    return np.squeeze(toto)*1.

# def corr(x, y):
#     """Compute the correlation matrix of two multi-variate random variables.

#     Similar to the numpy function corrcoef but is safe to nan (treat as zero) and complex variables.

#     Args:
#         x (1d or 2d array):  each column of x is a sample from the first variable.
#         y (1d or 2d array):  each column of y is a sample from the second variable. y must have the same number of columns as x.
#     Return:
#         The correlation matrix, with the (i,j) element being corr(x_i, y_j)
#     """
#     if x.ndim==1:
#         x = x[np.newaxis,:]
#         x[np.isnan(x)] = 0
#     if y.ndim==1:
#         y = y[np.newaxis,:]
#         y[np.isnan(y)] = 0

#     assert(x.shape[1]==y.shape[1])

#     mx = np.mean(x, axis=1); my = np.mean(y, axis=1)
#     xmx = x-mx[:, np.newaxis]; ymy = y-my[:,np.newaxis]
#     dx = np.sqrt(np.mean(np.abs(xmx)**2, axis=1)) # standard deviation of X
#     dy = np.sqrt(np.mean(np.abs(ymy)**2, axis=1)) # standard deviation of Y
#     # vx = mean(abs(xmx)**2, axis=1); vy = mean(abs(ymy)**2, axis=1)

#     return np.squeeze((xmx/dx[:,np.newaxis]) @ (np.conj(ymy.T)/dy[np.newaxis,:])) / x.shape[1]


# def safe_corr(X0, Y0):
#     X = ma.masked_invalid(X0)
#     Y = ma.masked_invalid(Y0)
#     toto = ma.corrcoef(X, Y).data
#     Cxy = toto[:X.shape[0], X.shape[0]:]
#     return Cxy

@Tools.nan_safe
@Tools.along_axis
def normalize(X, std=True):
    """Centralize an array by its mean and reduce it by its standard deviation along a given axis.

    This function is safe to nan values.
    """
    if std:
        return (X - np.mean(X)) / np.std(X)
    else:  # remove mean only
        return X - np.mean(X)
