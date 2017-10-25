"""Statistics related functions.
"""

import numpy as np
import numpy.linalg as la
import scipy
import scipy.special
import pandas as pd
from functools import wraps
import warnings
# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
# from sklearn.gaussian_process import GaussianProcess

from . import Tools, Kalman
# from pyshm import Tools, Kalman

# Convention:
# for 2d array of observations of a random vector: each row represents a variable and each column represents an observation.

def mean(X0, W=None):
    """Reweighted mean.

    Args:
        X0 (2d array): each row represents a variable and each column represents an observation
        W : symmetric positive-definite weigth matrix
    """
    assert X0.ndim == 2
    N = X0.shape[1]
    X = np.ma.masked_invalid(X0)

    if W is None:
        mX = X.mean(axis=-1)
    else:
        Wo = W @ np.ones(N)
        oWo = np.ones(N) @ Wo
        mX = X.dot(Wo) / oWo
    return np.asarray(mX)


def cov(X0, Y0, W=None):
    """Reweighted covariance matrix.
    """
    assert X0.ndim == Y0.ndim == 2
    assert X0.shape[1] == Y0.shape[1]

    N = X0.shape[1]
    X = np.ma.masked_invalid(X0)
    Y = np.ma.masked_invalid(Y0)

    if W is None:
        mX = X.mean(axis=-1)
        mY = Y.mean(axis=-1)
        C = (X-mX[:,np.newaxis]).dot((Y-mY[:,np.newaxis]).T) / N
    else:
        Wo = W @ np.ones(N)
        oWo = np.ones(N) @ Wo
        mX = X.dot(Wo) / oWo
        mY = Y.dot(Wo) / oWo
        C = ((X-mX[:,np.newaxis]).dot(W)).dot((Y-mY[:,np.newaxis]).T) / oWo
    return np.asarray(C)


def robust_std(x0, ratio=3.):
    """Robust std.
    """
    x1 = x0.flatten() - np.mean(x0)
    idx = np.abs(x1) < ratio * np.std(x1)
#     idx = np.abs(x1) > ratio
#     return np.std(x1), []
#     return np.std(x1[idx]), np.where(~idx)[0]
    oidx = np.where(~idx)[0]  # index of outliers
    v = np.std(x1[idx]) if np.any(idx) else np.std(x1)
    return v, oidx


def corr(X0, Y0, W=None):
    """Reweighted correlation matrix.
    """
    Cxy = cov(X0, Y0, W=W)
    Cxx = cov(X0, X0, W=W)
    Cyy = cov(Y0, Y0, W=W)

    return np.diag(1/np.sqrt(np.diag(Cxx))) @ Cxy @ np.diag(1/np.sqrt(np.diag(Cyy)))


def centralize(X0, W=None):
    """Centralize a random vector so that its mean becomes 0.
    """
    mX = mean(X0, W=W)
    return X0 - mX[:,np.newaxis]


def normalize(X0, W=None):
    """Normalize a random vector so that its covariance matrix becomes identity.
    """
    Xc = centralize(X0, W=W)
    Cm = cov(X0, X0, W=W)
    if X0.shape[0]>1:
        U, S, _ = la.svd(Cm)
        Xn = np.diag(1/np.sqrt(S)) @ U.T @ Xc
    else:
        Xn = 1/np.sqrt(Cm) * Xc
    return Xn


def diag_normalize(X0, W=None):
    """Per dimension normalization, safe wrt singular dimension.
    """
    Xc = centralize(X0, W=W)
    Cm = cov(X0, X0, W=W)
    v = np.sqrt(np.diag(Cm))
    idx = np.isclose(v,0)
    u = np.zeros(Cm.shape[0])
    u[~idx] = 1/v[~idx]
    return np.diag(u) @ Xc


def pca(X, W=None, nc=None, corrflag=False, centerflag=False):
    """Principal Component Analysis.

    Args:
        X (2d array): each row represents a variable and each column represents an observation
        nc (int): number of components to hold
    Returns:
        C, U : coefficients and corresponded principal directions
    """
    if corrflag:
        U, S, _ = la.svd(corr(X, X, W=W))
    else:
        U, S, _ = la.svd(cov(X, X, W=W))

    if centerflag:
        C = U.T @ (X - mean(X)[:, np.newaxis])
    else:
        C = U.T @ X
    return C[:nc, :], U[:, :nc], S[:nc]

    # X_masked = np.ma.masked_invalid(X)
    # Xcov = np.ma.cov(X_masked, X_masked).data[:X.shape[0], :X.shape[0]]
    # if corrflag:
    #     # Xcor = np.ma.corrcoef(X_masked, X_masked).data  # extremely slow
    #     dv = np.diag(1/np.sqrt(np.diag(Xcov)))
    #     Xcor = dv @ Xcov @ dv
    #     U, S, _ = la.svd(Xcor)
    # else:
    #     U, S, _ = la.svd(Xcov)


def cca(X0, Y0, W=None):
    """Canonical Correlation Analysis.
    """
    # X = X0 - mean(X0)[:,np.newaxis]
    # Y = Y0 - mean(Y0)[:,np.newaxis]
    X, Y = X0, Y0

    Cxy = cov(X,Y, W=W); Cyx = Cxy.T
    Cxx = cov(X,X, W=W); iCxx = la.inv(Cxx)
    Cyy = cov(Y,Y, W=W); iCyy = la.inv(Cyy)
    Cxxsq =  scipy.linalg.sqrtm(Cxx)
    Cyysq =  scipy.linalg.sqrtm(Cyy)
    iCxxsq = la.inv(Cxxsq)
    iCyysq = la.inv(Cyysq)
    # iCxxsq = scipy.linalg.sqrtm(iCxx)
    # iCyysq = scipy.linalg.sqrtm(iCyy)

    A0, Sa, _ = la.svd(iCxxsq @ Cxy @ iCyy @ Cyx @ iCxxsq)
    A = iCxxsq @ A0; iA = A0.T @ Cxxsq
    B0, Sb, _ = la.svd(iCyysq @ Cyx @ iCxx @ Cxy @ iCyysq)
    B = iCyysq @ B0; iB = B0.T @ Cyysq

    return (A, Sa, iA), (B, Sb, iB), corr(A.T @ X, B.T @ Y)


def percentile(X0, p):
    """Compute the value corresponding to a percentile (increasing order) in an array.

    Args:
        X0 (nd array): input array
        p (float): percentile, between 0. and 1.
    Return:
        value corresponding to the percentile p
    """
    assert 0. <= p <= 1.

    X = X0.copy().flatten()
    X = X[~np.isnan(X)]  # remove all nans
    # X[np.isnan(X)] = 0   # fill all nans by zero

    N = len(X)
    idx = np.argsort(X)  # increasing order sort
    nz0 = int(np.floor(N * p))
    nz1 = int(np.ceil(N * p))

    if nz0==nz1==0:
        v = X[idx[0]]
    elif nz0==nz1==N:
        v = X[idx[-1]]
    else:
        v = np.mean(X[idx[nz0:nz1]])
    return v


def local_statistics(X, mwsize, mad=False, causal=False, win_type='boxcar', drop=True):
    """Local mean and standard deviation estimation using pandas library.

    Args:
        X (pandas Series/DataFrame or numpy array)
        mwsize (int): size of the moving window
        mad (bool): use median based estimator
        causal (bool): use causal estimator
        drop (bool): drop the begining of estimation to avoid side effect
    Returns:
        local mean and standard deviation of X
    """
    if isinstance(X, pd.Series) or isinstance(X, pd.DataFrame):
        Err = X
    else:
        if X.ndim == 1:
            Err = pd.Series(X)
        elif X.ndim == 2:
            Err = pd.DataFrame(X.T)
        else:
            raise TypeError("Input must be 1d or 2d array.")

    if mad:  # use median-based estimator
        mErr = Err.rolling(window=mwsize, center=not causal).median() #.bfill()
        # sErr = 1.4826 * (Err-mErr).abs().rolling(window=mwsize, min_periods=1, center=not causal).median() #.bfill()
        sErr = (Err-mErr).abs().rolling(window=mwsize, min_periods=1, center=not causal).median() #.bfill()
    else:
        mErr = Err.rolling(window=mwsize, win_type=win_type, center=not causal).mean() #.bfill()
        sErr = Err.rolling(window=mwsize, center=not causal).std() #.bfill()

    # drop the begining
    if drop:
        mErr.iloc[:int(mwsize*1.1)]=np.nan
        sErr.iloc[:int(mwsize*1.1)]=np.nan

    if isinstance(X, pd.Series) or isinstance(X, pd.DataFrame):
        return mErr, sErr
    else:
        return np.asarray(mErr), np.asarray(sErr)


def linear_regression(Y, X):
    """Scalar linear regression of the model Y = aX + b.

    Args:
        Y (1d array): the observation variable
        X (1d array): the explanatory variable
    Returns:
        a, b: the least square solution (using scipy.sparse.linalg.cgs)
        err: residual error
        sigma2: estimation of the noise variance
        S: variance of the estimate a and b (as random variables)
    """
    assert Y.ndim==X.ndim==1
    assert len(Y)==len(X)

    (X0, Y0), nidx = Tools.remove_nan_columns(np.atleast_2d(X), np.atleast_2d(Y))  # the output is a 2d array
    X0 = X0[0]; Y0 = Y0[0]  # convert to 1d array

    # X0[np.isnan(X0)] = 0; Y0[np.isnan(Y0)] = 0

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


def Hurst(data, mwsize, sclrng=None, wvlname="haar"):
    """Estimate the Hurst exponent of a time series using wavelet transform.

    Args:
        data (1d array): input time series
        mwsize (int): size of the smoothing window
        sclrng (tuple): index of scale range used for linear regression
        wvlname (str): wavelet name
    Returns:
        ..
        - H (1d array): estimate of Hurst exponent
        - B (1d array): estimate of intercept
        - V (2d array): estimate of variance of H and B
    """
    def scalar_linear_regression(yvar0, xvar0):
        mx, my = np.mean(xvar0), np.mean(yvar0)
        xvar = xvar0 - mx
        yvar = yvar0 - my
        a = np.dot(yvar, xvar) / np.dot(xvar, xvar)
        b = my - a * mx
        v = np.sqrt(np.mean((a * xvar0 + b - yvar0)**2))
        return a, b, v

    import pywt
    import pandas as pd

    Nt = len(data)  # length of data
    wvl = pywt.Wavelet(wvlname)  # wavelet object
    maxlvl = pywt.dwt_max_level(Nt, wvl)  # maximum level of decomposition
    if sclrng is None:  # range of scale for estimation
        sclrng = (1, maxlvl)
    else:
        sclrng = (max(1,sclrng[0]), min(maxlvl, sclrng[1]))

    # Compute the continuous wavelet transform
    C0 = []
    for n in range(sclrng[0], sclrng[1]+1): # (1, maxlvl+1):
        phi, psi, x = wvl.wavefun(level=n) # x is the grid for wavelet
        # C0.append(Tools.safe_convolve(data, psi[::-1]/2**((n-1)/2.), mode="samel"))
        C0.append(scipy.signal.convolve(data, psi[::-1]/2**((n-1)/2.), mode="full", method="direct")[:1-len(psi)])
    C = np.asarray(C0)  # matrix of wavelet coefficients, each column is a vector of coefficients

    # Compute the wavelet spectrum
    kernel = scipy.signal.triang(mwsize)
    # kernel = np.ones(mwsize)/mwsize
    S = Tools.safe_convolve(C**2, kernel, mode="samel", axis=-1)

    # Linear regression
    # H, B = np.zeros(Nt), np.zeros(Nt)
    H, B, V = np.zeros(Nt), np.zeros(Nt), np.zeros(Nt)
    xvar = np.arange(sclrng[0], sclrng[1]+1)  # explanatory variable

    eps_val = 1e-10
    # print(eps_val)
    for t in range(Nt):
        toto = S[:,t].copy()
        # avoid divide by zero warnings
        toto[np.isnan(toto)] = eps_val
        toto[toto<eps_val] = eps_val
        yvar = np.log2(toto)
        # yvar = np.log2(S[sclrng[0]:sclrng[1],t])
        a, b, v = scalar_linear_regression(yvar, xvar)
        H[t] = max(min((a-1)/2., 1.), 0.)  # since log2(S_j) = (2H+1)*j + e
        B[t] = b
        V[t] = v

    # due to the smoothing in the compute of the spectrum, we have to roll to get a causal result
    sc = mwsize # // 2
    H = np.roll(H, -sc); H[-sc:] = np.nan
    B = np.roll(B, -sc); B[-sc:] = np.nan
    V = np.roll(V, -sc); V[-sc:] = np.nan

    # drop the begining
    # H[:mwsize] = np.nan
    # B[:mwsize] = np.nan
    # V[:,:mwsize] = np.nan
    return H, B, V


def Hurst_rs(X0, nrng=None, alc=False):
    """Estimate the Hurst exponent of a time series using the definition.

    Note:
        This method seems highly sensitive to the parameters and computationally inefficient compared to the wavelet based method.
    """
    def power_regression(yvar0, n=1):
        yvar = np.log(yvar0)
        xvar = np.log(np.arange(len(yvar))+n)  # log(n)...log(n+N)
        xvarc = xvar - np.mean(xvar)
        yvarc = yvar - np.mean(yvar)
        return np.dot(yvarc, xvarc) / np.dot(xvarc, xvarc)

    def rs(X):  # rescaled range
        Ycum = np.cumsum(X - np.mean(X))
        return (np.max(Ycum) - np.min(Ycum)) / np.std(X)

    assert X0.ndim == 1
    Nt = len(X0)
    X1 = X0.copy(); X1[np.isnan(X1)] = 0
    if not isinstance(X1, pd.Series):
        X = pd.Series(X1)
    else:
        X = X1
    if nrng is None:
        nrng = (3*Nt//10, 7*Nt//10)
    yvar0 = []
    for n in range(*nrng):
        toto = pd.Series.rolling(X, window=n, min_periods=1, center=True).apply(rs)
        yvar0.append(np.mean(toto[~np.isnan(toto)]))
    # if alc:
    #     if nrng[1]-nrng[0] <= 340:
    #         toto = scipy.special.gamma((Nt-1)/2) / scipy.special.gamma(Nt/2) / np.sqrt(np.pi)
    #     else:
    #         toto = 1 / np.sqrt(n*np.pi/2)
    #     Alc = toto * np.sum(np.sqrt((Nt-np.arange(1,Nt))/np.arange(1,Nt)))
    #     return 0.5 + power_regression(np.asarray(yvar0) - Alc, n=nrng[0])
    # else:
    return power_regression(np.asarray(yvar0), n=nrng[0])


def global_delay(xobs, yobs, dlrng=(-12,12)):
    """Estimation of the global delay of one time series with respect to another.

    A delay t is applied on the 1d time series xobs by rolling it, the delayed xobs is then feeded to the linear regression together with yobs, and the value of t minimizing the residual of linear regression is the optimal delay.

    Args:
        xobs, yobs (pandas Series): observation of input and output time-series.
        dlrng (tuple of int): range in which the optimal delay will be searched.
    Return:
        estimated optimal delay
    """
    res = []
    for r in range(*dlrng):
        xvar, yvar = np.roll(xobs,r), yobs
        res.append(linear_regression(yvar, xvar))

    # return np.argmin([v[4][0] for v in res])+dlrng[0]
    return np.argmin([v[3] for v in res])+dlrng[0]


########## Regression analysis ##########

def training_period(Nt, tidx0=None, Ntrn=None):
    """Compute the valid training period from given parameters.

    Args:
        Nt (int): total length of the data
        tidx0 (int): starting index of the training period.
        Ntrn (int): length of the training period.
    Returns:
        ..
        - (tidx0, tidx1): tuple of valid starting and ending index.
        - Ntrn: length of valid training period.
    """
    tidx0 = 0 if tidx0 is None else (tidx0 % Nt)
    tidx1 = Nt if Ntrn is None else min(tidx0+Ntrn, Nt)
    Ntrn = tidx1 - tidx0

    # tidx1 = 0 if tidx0 is None else min(max(0, tidx0), self.Nt)
    # Ntrn = self.Nt if Ntrn is None else min(max(0, Ntrn), self.Nt)
    # tidx0 = 0 if tidx0 is None else min(max(0, tidx0), self.Nt)
    # # tidx0 = np.max(self.lag)-1
    # tidx1 = tidx0+Ntrn
    return (tidx0, tidx1), Ntrn


def percentile_subset(func):
    @wraps(func)
    def newfunc(Y0, X0, W0, *args, pthresh=0.5, **kwargs):
        """
        Args:
            pthresh (float): select the original dataset by this percentile
        """
        if pthresh > 0:
            # selection on the original dataset
            nXY = Tools.safe_norm(np.vstack([X0, Y0]), axis=0)
            sidx = np.where(nXY > percentile(nXY, pthresh))[0]  # index of largest values
            X, Y = X0[:,sidx], Y0[:,sidx]
            W = W0[sidx,:][:,sidx] if W0 is not None else None
        else:
            X, Y, W = X0, Y0, W0
        # regression on the selected dataset
        L, Cvec, *_ = func(Y, X, W, *args, **kwargs)
        # error and covariance on the whole dataset
        Err = Y0 - (np.dot(L, X0) + Cvec)
        Sig = cov(Err, Err)
        return L, Cvec, Err, Sig
    return newfunc


def dim_reduction_cca(func):
    @wraps(func)
    def newfunc(Y0, X0, W0, *args, vthresh=1e-3, cdim=None, **kwargs):
        """Dimension reduction in multivariate linear regression by CCA.

        Args:
            vthresh (float): see dim_reduction_pca
        """
        # dimension reduction: either by vthresh or by cdim
        if vthresh > 0 or cdim is not None:
            # (U, Sy, iU), (V, Sx, iV), Ryx = cca(Y0, X0)
            _, (V, S, _), Ryx = cca(Y0, X0, W=W0)

            if cdim is None:  # cdim not given, use vthresh to compute cdim
                assert 0. <= vthresh <=1.
                # two possible strategies:
                # 1. by relative value of sv
                # toto = S/S[0]
                # cdim = np.sum(toto >= vthresh)
                # 2. by cumulation of sv
                toto = np.cumsum(S) / np.sum(S)
                cdim = np.sum(toto <= 1-vthresh)
            # else:  # if cdim is given, vthresh has no effect
            #     pass
            # assert cdim>0  # check cdim
            # raise ValueError("dim_reduction: output dimension is zero, relax the threshold.")
            cdim = max(1, cdim) # force the minimal dimension

            # Xcof = V.T @ (X0 - mean(X0)[:, np.newaxis])
            # Ycof = U.T @ (Y0 - mean(Y0)[:, np.newaxis])
            Xcof = V.T @ X0
            # Ycof = U.T @ Y0

            Lc, Cvec, Err, Sig = func(Y0, Xcof[:cdim,:], W0, *args, **kwargs)
            L = Lc @ (V[:, :cdim].T)
            # Cvecc,Errc,Sigc = None,None,None
            # Lc, Cvecc, Errc, Sigc = func(Ycof[:cdimy,:], Xcof[:cdimx,:], W0, *args, **kwargs)
            # L = iU[:, :cdimy] @ Lc @ (V[:, :cdimx].T)
            # Cvec = iU[:, :cdimy] @ Cvecc
            # Err = iU[:, :cdimy] @ Errc
            # Sig = iU[:, :cdimy] @ Sigc @ iU[:, :cdimy].T
        else:
            L, Cvec, Err, Sig = func(Y0, X0, W0, *args, **kwargs)
            Lc, V, S, cdim = None, None, None, None
        return (L, Cvec, Err, Sig), (Lc, V, S) # , (Lc, Cvecc, Errc, Sigc)
        # the first tuple in the returned values is the same as in multi_linear_regression and it is computed from the dimension-reduced solution, the second is (the reduced matrix, the pca basis, the singular values)
    return newfunc


def dim_reduction_pca(func):
    @wraps(func)
    def newfunc(Y0, X0, W0, *args, vthresh=1e-3, cdim=None, corrflag=False, **kwargs):
        """Dimension reduction in multivariate linear regression.

        Args:
            vthresh (float): relative threshold in the (SVD based) dimension reduction, between 0 and 1. 1-vthresh is the percentage of information kept, i.e. 90 percent of information is kept if vthresh=0.1. Have no effect if cdim is set.
            cdim (int): number of dimension to be kept.
            corrflag (bool): if True use the correlation matrix instead of the covariance matrix
        """
        # dimension reduction: either by vthresh or by cdim
        if vthresh > 0 or cdim is not None:
            # use covariance/correlation matrix:
            # if corrflag:
            #     Cyx = corr(Y0, X0, W=W0)  # correlation matrix
            # else:
            #     Cyx = cov(Y0, X0, W=W0)  # covariance matrix
            # U, S, V=la.svd(Cyx); V=V.T
            #
            # or use PCA:
            _, V, S = pca(X0, W=W0, nc=None, corrflag=corrflag)  # pca coefficients

            if cdim is None:  # cdim not given, use vthresh to compute cdim
                assert 0. <= vthresh <=1.
                # two possible strategies:
                # 1. by relative value of sv
                # toto = S/S[0]
                # cdim = np.sum(toto >= vthresh)
                # 2. by cumulation of sv
                toto = np.cumsum(S) / np.sum(S)
                cdim = np.sum(toto <= 1-vthresh)
            # else:  # if cdim is given, vthresh has no effect
            #     pass
            # assert cdim>0  # check cdim
            # raise ValueError("dim_reduction: output dimension is zero, relax the threshold.")
            cdim = max(1, cdim) # force the minimal dimension

            # Xcof = V.T @ (X0 - mean(X0)[:, np.newaxis])
            Xcof = V.T @ X0
            # Ycof = U.T @ (Y0 - mean(Y0)[:, np.newaxis])
            # Ycof = U.T @ Y0

            Lc, Cvec, Err, Sig = func(Y0, Xcof[:cdim,:], W0, *args, **kwargs)
            L = Lc @ V[:, :cdim].T
        else:
            L, Cvec, Err, Sig = func(Y0, X0, W0, *args, **kwargs)
            Lc, Cvecc, Errc, Sigc = L, Cvec, Err, Sig
            U, V, S, cdim = None, None, None, None
        return (L, Cvec, Err, Sig), (Lc, V, S)
        # the first tuple in the returned values is the same as in multi_linear_regression and it is computed from the dimension-reduced solution, the second is (the reduced matrix, the pca basis, the singular values)
    return newfunc



def dim_reduction_bm(func):
    """Dimension reduction in Brownian Motion model multivariate linear regression.
    """
    @wraps(func)
    def newfunc(Yvar0, Xvar0, sigmaq2, sigmar2, x0, p0, smooth=False, sidx=0, Ntrn=None, vthresh=0., cdim=None, covflag=True, rescale=True):
        Nt = Xvar0.shape[1]  # length of observations
        # training data for dim reduction
        (tidx0, tidx1), _ = training_period(Nt, tidx0=sidx, Ntrn=Ntrn)  # valid training period

        # Rescaling makes sigmaq2, sigmar2 insensible to the numerical amplitude of Yvar0, Xvar0.
        def _normalize(X0):
            mX = mean(X0)[:, np.newaxis]
            Cm = cov(X0, X0)
            if X0.shape[0]>1:
                U, S, _ = la.svd(Cm)
                B = U @ np.diag(np.sqrt(S))
                Xn = np.diag(1/np.sqrt(S)) @ U.T @ (X0 - mX)
            else:
                B = 1/np.sqrt(Cm)
                Xn = B * (X0 - mX)
            return Xn, mX, B

        if rescale:
            Yscl = np.sqrt(np.trace(cov(Yvar0[:,tidx0:tidx1], Yvar0[:,tidx0:tidx1])))
            Yvar = Yvar0 / Yscl
            Xscl = np.sqrt(np.trace(cov(Xvar0[:,tidx0:tidx1], Xvar0[:,tidx0:tidx1])))
            Xvar = Xvar0 / Xscl

            # _, Ym, Yscl = _normalize(Yvar0[:,tidx0:tidx1])
            # Yvar = la.inv(Yscl) @ (Yvar0)
            # _, Xm, Xscl = _normalize(Xvar0[:,tidx0:tidx1])
            # Xvar = Xvar0
            # Xscl_inv = la.inv(Xscl)
            # Xvar = Xscl_inv @ (Xvar0)

            # # rescaling won't work if the mean is substracted!
            # Xvar = la.inv(Xscl) @ (Xvar0 - Xm)
            # Yvar = la.inv(Yscl) @ (Yvar0 - Ym)
        else:
            # Xscl = np.eye(Xvar0.shape[0])
            # Yscl = np.eye(Yvar0.shape[0])
            Xvar, Yvar = Xvar0, Yvar0

        # Xvar, Yvar = Xvar0, Yvar0

        # dimension reduction: either by vthresh or by cdim
        if vthresh > 0 or cdim is not None:
            # # by covariance
            # corrflag=False
            # if corrflag:
            #     Cyx = corr(Yvar[:,tidx0:tidx1], Xvar[:,tidx0:tidx1], W=None)  # correlation matrix
            # else:
            #     Cyx = cov(Yvar[:,tidx0:tidx1], Xvar[:,tidx0:tidx1], W=None)  # covariance matrix
            # _, S, U=la.svd(Cyx); U=U.T
            _, U, S = pca(Xvar[:,tidx0:tidx1], nc=None, corrflag=False)  # pca coefficients
            if cdim is None:  # cdim not given, use vthresh to compute cdim
                assert 0. <= vthresh <=1.
                # two possible strategies:
                # 1. by relative value of sv
                # toto = S/S[0]
                # cdim = np.sum(toto >= vthresh)
                # 2. by cumulation of sv
                toto = np.cumsum(S) / np.sum(S)
                cdim = np.sum(toto <= 1-vthresh)
            # else:  # if cdim is given, vthresh has no effect
            #     pass
            # assert cdim>0  # check cdim
            # raise ValueError("dim_reduction: output dimension is zero, relax the threshold.")
            cdim = max(1, cdim) # force the minimal dimension

            # # by cca:
            # _, (U, S, _), Ryx = cca(Yvar[:,tidx0:tidx1], Xvar[:,tidx0:tidx1], W=None)
            # print(S, Ryx)
            # if cdim is None:  # cdim not given, use vthresh to compute cdim
            #     cdim = np.sum(np.abs(np.diag(Ryx)) > vthresh)
            #     # print(np.abs(Ryx))

            if cdim == 0:  # check cdim
                warnings.warn("cdim==0: the reduced dimension is set to 1.")
                cdim = 1
                # raise ValueError("dim_reduction: output dimension is zero, relax the threshold.")

            Xcof = U.T @ Xvar  # coefficients of Xvar
            (Amatc, Acovc), (Cvec, Ccov), Err, Sig = func(Yvar, Xcof[:cdim, :], sigmaq2, sigmar2, x0=x0, p0=p0, smooth=smooth)

            # Recover the kernel matrices in the full shape
            W = U[:, :cdim]  # analysis matrix of subspace basis
            Wop = Tools.matprod_op_right(W.T, Amatc.shape[1])
            Amat = np.asarray([Amatc[t,] @ W.T for t in range(Nt)])
            Acov = np.asarray([Wop @ Acovc[t,] @ Wop.T for t in range(Nt)]) # if covflag else None
        else:
            (Amat, Acov), (Cvec, Ccov), Err, Sig = func(Yvar, Xvar, sigmaq2, sigmar2, x0=x0, p0=p0, smooth=smooth)
            U, S = None, None
            Amatc, Acovc = Amat, Acov

        if rescale:  # inverse rescaling
            for t in range(Nt):
                Amat[t,] = Yscl / Xscl * Amat[t,]
                if Acov is not None:
                    Acov[t,] = (Yscl / Xscl)**2 * Acov[t,]
                Cvec[t,] = Yscl * Cvec[t,]
                Ccov[t,] = Yscl**2 * Ccov[t,]

        # if rescale:  # inverse rescaling
        #     # print(len(Amat), Nt)
        #     for t in range(Nt):
        #         Amat[t,] = Yscl @ Amat[t,] @ Xscl_inv
        #         if Acov is not None:
        #             # Tmat0 = []
        #             # for i in range(Yvar.shape[0]):
        #             #     for j in range(Xvar.shape[0]):
        #             #         Tmat0.append(np.kron(Yscl[i,:], Xscl[:,j]))
        #             # Tmat = np.asarray(Tmat0)
        #             # Tmat0 = np.atleast_2d(Yscl.flatten('C')).T @ np.atleast_2d(Xscl.flatten('F'))
        #             # Tmat = Tmat0.reshape((Yscl.shape[0],-1), order='C')
        #             Tmat = np.kron(Yscl, Xscl_inv.T)
        #             Acov[t,] = Tmat @ Acov[t,] @ Tmat.T
        #         Cvec[t,] = Yscl @ Cvec[t,] # + Ym - Amat[t,] @ Xm
        #         Ccov[t,] = Yscl @ Ccov[t,] @ Yscl.T

        return ((Amat, Acov), (Cvec, Ccov), Err, Sig), ((Amatc, Acovc), U, S)
        # The returned variables are similar to those in dim_reduction_pca
    return newfunc


# def dim_reduction_bm(func):
#     """Dimension reduction in Brownian Motion model multivariate linear regression.
#     """
#     @wraps(func)
#     def newfunc(Yvar, Xvar, sigmaq2, sigmar2, x0, p0, smooth=False, sidx=0, Ntrn=None, vthresh=0., corrflag=False):
#         Nt = Xvar.shape[1]  # length of observations
#         # training data for dim reduction
#         (tidx0, tidx1), _ = training_period(Nt, tidx0=sidx, Ntrn=Ntrn)  # valid training period
#         # dimension reduction
#         if vthresh > 0:
#             _, U, S = pca(Xvar[:, tidx0:tidx1], nc=None, corrflag=corrflag)  # pca coefficients
#             cdim = np.sum(S/S[0] > vthresh)  # reduced dimension
#             Xcof = U.T @ Xvar  # coefficients of Xvar
#             (A, Acov), (Cvec, Ccov), Err, Sig = func(Yvar, Xcof[:cdim, :], sigmaq2, sigmar2, x0=x0, p0=p0, smooth=smooth)
#             L0, C0 = [], []
#             # print(A.shape, Acov.shape)
#             W = U[:, :cdim]  # analysis matrix of subspace basis
#             Wop = Tools.matprod_op_right(W.T, A.shape[1])
#             for t in range(Nt):
#                 L0.append(A[t,] @ W.T)
#                 C0.append(Wop @ Acov[t,] @ Wop.T)
#                 # C0.append(U[:, :cdim] @ Acov[t,] @ U[:, :cdim].T)
#             L = np.asarray(L0)
#             Lcov = np.asarray(C0)
#         else:
#             (L,Lcov), (Cvec,Ccov), Err, Sig = func(Yvar, Xvar, sigmaq2, sigmar2, x0=x0, p0=p0, smooth=smooth)

#         return (L,Lcov), (Cvec,Ccov), Err, Sig
#     return newfunc


def random_subset(func):
    @wraps(func)
    def newfunc(Y, X, W, *args, Nexp=100, method="median", **kwargs):
        if Nexp > 0:
            Nt = X.shape[1]
            Ns = min(Nt, X.shape[0]*Y.shape[0])
            # print(Nt, Ns)
            res = []
            for n in range(Nexp):
                # regression on a random subset
                idx = np.random.choice(Nt, Ns, replace=False)
                Ysub = Y[:,idx]
                Xsub = X[:,idx]
                Wsub = W[:,idx][idx,:] if W is not None else None
                res.append(func(Ysub, Xsub, Wsub, *args, **kwargs))
                # res.append([toto[0], toto[1]])  # L, Cvec
            # final voting procedure
            if method=="mean":
                L = np.mean([x[0] for x in res], axis=0)
                Cvec = np.mean([x[1] for x in res], axis=0)
            elif method=="median":
                L = np.median([x[0] for x in res], axis=0)
                Cvec = np.median([x[1] for x in res], axis=0)
            else:
                raise NameError("Unknown method {}".format(method))
            Err = Y - (np.dot(L, X) + Cvec)  # error
            Sig = cov(Err, Err)  # covariance matrix of error
        else:
            L, Cvec, Err, Sig = func(Y, X, W, *args, **kwargs)
        return L, Cvec, Err, Sig
    return newfunc


def ransac(func):
    def inliers_detection(E, v):
        return Tools.safe_norm(centralize(E), axis=0)**2 < v

    @wraps(func)
    def newfunc(Y, X, W, *args, Nexp=100, method="best", **kwargs):
        """
        Args:
        """
        if Nexp > 0:
            ifct, ithresh = 0.75, 0.5  # ransac parameters
            Nt = X.shape[1]
            Ns = X.shape[0]*Y.shape[0]
            res = []
            for n in range(Nexp):
                # regression on a random subset
                idx = np.random.choice(Nt, 2*Ns, replace=False)
                Ysub = Y[:,idx]
                Xsub = X[:,idx]
                Wsub = W[:,idx][idx,:] if W is not None else None
                L, Cvec, *_ = func(Ysub, Xsub, Wsub, *args, **kwargs)
                Err = Y - (np.dot(L, X) + Cvec)  # error of the whole dataset
                Sig = cov(Err, Err)  # covariance matrix of error
                # inliers detection
                idx_in = inliers_detection(Err, ifct * np.trace(Sig))
                # print(np.sum(idx_in)/Nt)
                # de-biaising using inliers if their proportion is more than ithresh
                if np.sum(idx_in)/Nt > ithresh and np.sum(idx_in) > Ns:
                    toto = func(Y[:,idx_in], X[:,idx_in], W[:, idx_in][idx_in,:], *args, **kwargs)
                    res.append([toto[0], toto[1], idx_in])  # L, Cvec, and idx_in
            if len(res)==0:
                raise RuntimeError("Ransac failed.")
            # final voting procedure
            if method=="best":
                # policy 1: choose the one with the largest inliers set
                bidx = np.argmax([np.sum(x[-1]) for x in res])
                L, Cvec = res[bidx][:2]
            elif method=="mean":
                L = np.mean([x[0] for x in res], axis=0)
                Cvec = np.mean([x[1] for x in res], axis=0)
            elif method=="median":
                L = np.median([x[0] for x in res], axis=0)
                Cvec = np.median([x[1] for x in res], axis=0)
            else:
                raise NameError("Unknown method {}".format(method))
            Err = Y - (np.dot(L, X) + Cvec)  # error
            Sig = cov(Err, Err)  # covariance matrix of error
        else:
            L, Cvec, Err, Sig = func(Y, X, W, *args, **kwargs)
        return L, Cvec, Err, Sig
    return newfunc


def multi_linear_regression(Y, X, W, vreg=0):
    """Multivariate linear regression by generalized least square (GLS).

    GLS looks for the matrices L and the vector C such that the reweighted norm

        ||L*X + C - Y||_W

     is minimized. Analytical formula of the solutions are given by

        L = cov_W(Y, X) * cov_W(X,X)^-1,  C = mean_W(Y) - L * mean_W(X)
    where cov_W and mean_w are the W-modified covariance matrix / mean vector.

    Args:
        Y (2d array): response variables
        X (2d array): explanatory variables
        W (2d matrix): symmetric and positive definite
        vreg (float): regularization parameter
    Returns:
        L, C, E, S: the matrix and the constant vector, the residual and its covariance matrix
    """
    if W is not None:
        assert Tools.issymmetric(W)  # check symmetry
        # assert Tools.ispositivedefinite(W)  # check positiveness

    # covariance matrices
    Syx = cov(Y, X, W=W)
    Sxx = cov(X, X, W=W)
    # column mean vectors
    mX = np.atleast_2d(mean(X, W=W)).T
    mY = np.atleast_2d(mean(Y, W=W)).T

    # L = np.dot(Syx, la.pinv(Sxx))  # pseudo inverse
    # print('vreg',vreg)
    L = np.dot(Syx, la.inv(Sxx + vreg * np.eye(Sxx.shape[0])))  # with regularization
    Cvec = mY - np.dot(L, mX) # if constflag else np.zeros((dimY, 1))

    Err = Y - (np.dot(L, X) + Cvec)  # Err has the same length as Y0
    Sig = cov(Err, Err)

    return L, Cvec, Err, Sig


def multi_linear_regression_bm(Y, X, sigmaq2, sigmar2, x0=0., p0=1., kftype='smoother'):
    """Multivariate linear regression by Brownian motion model.

    Args:
        Y (2d array): response variables
        X (2d array): explanatory variables
        sigmaq2 (float): variance of innovation noise
        sigmar2 (float): variance of observation noise
        x0 (float): initial state (a constant vector)
        p0 (float): variance of the initial state
        kftype (str): type of Kalman filter: 'smoother' or 'filter'
    Returns:
        ..
        - (A, covA): estimation of the operator and its covariance matrix, time-dependent
        - (C, covC): estimation of the bias and its covariance matrix, time-dependent
        - Err: residual Y - A*X, time-dependent
        - Sig: covariance matrix of Err, time-independent
    """
    assert Y.shape[1] == X.shape[1]

    dimobs, Nt = Y.shape  # dimension of the observation vector and duration
    dimsys = X.shape[0] * dimobs + dimobs  # dimension of the system vector

    A = np.eye(dimsys)  # the transition matrix: time-independent
    # construct the observation matrices: time-dependent
    B = np.zeros((Nt, dimobs, dimsys))
    for t in range(Nt):
        toto = X[:,t].copy()
        toto[np.isnan(toto)] = 0
        B[t,] = np.hstack([np.kron(np.eye(dimobs), toto), np.eye(dimobs)])

    # initialize the kalman filter
    _Kalman = Kalman.Kalman(Y, A, B, G=None, Q=sigmaq2, R=sigmar2, X0=x0, P0=p0)

    if kftype.upper()=='SMOOTHER':
        LXt, LPt, *_ = _Kalman.smoother()
    else:
        LXt, LPt, *_ = _Kalman.filter()

    Xflt = np.asarray(LXt)  # state vector
    Pflt = np.asarray(LPt)  # cov matrix
    # filtered / smoothed observations
    Yflt = np.sum(B * Xflt.transpose(0,2,1), axis=-1).T  # transpose(0,2,1) is for the computation of matrix-vector product, and .T is to make the second axis the time axis.
    Err = Y - Yflt
    Sig = cov(Err, Err)

    Amat0 = []
    Cvec0 = []
    for t in range(Nt):
        Amat0.append(np.reshape(Xflt[t,:-dimobs], (dimobs,-1)))
        Cvec0.append(Xflt[t,-dimobs:])

    Cvec = np.asarray(Cvec0)
    Amat = np.asarray(Amat0)  # Ka has shape Nt * dimobs * (dimsys-dimobs)

    return (Amat, Pflt[:,:-dimobs,:-dimobs]), (Cvec, Pflt[:,-dimobs:,-dimobs:]), Err, Sig


###### Alarms #####
def detect_periods_of_instability(hexp, hthresh, hgap=0, mask=None):
    # hexp = np.asarray(hexp0).copy(); hexp[np.isnan(hexp0)] = -np.inf
    # if hgap > 0:
    #     # with post-processing
    #     hidc = Tools.L_filter(np.int32(hexp>hthresh), wsize=hgap)>0  # non linear filter, slower
    #     # hidc = scipy.signal.convolve(hexp>ithresh, np.ones(options.hgap, dtype=bool), mode="same")>0
    # else:
    #     # no post-processing
    hidc = hexp>hthresh
    # apply the mask
    if mask is not None:
        hidc[np.where(mask)[0]] = False
    blk = Tools.find_block_true(hidc)  # starting-ending indexes of blocks of instability
    return [b for b in blk if b[1]-b[0] > hgap]


def detect_oscillations(x0, order=1, wsize=1, minlen=20, ratio=2.):
    # index of local extrema
    idx0 = scipy.signal.argrelmax(x0,order=order)[0]
    idx1 = scipy.signal.argrelmin(x0,order=order)[0]
    idx = np.sort(np.concatenate([idx0, idx1]))
    P = []

    if len(idx)>0:
        # separate the indexes into groups by testing outliers
        didx = np.hstack([0, np.diff(idx)])
        uu, ll = Tools.U_and_L_filter(didx, wsize=wsize)
        dd = uu-ll
    #     tt = ratio*np.median(dd)  # this can be 0!
    #     sd = np.where(dd > tt)[0]
        tt, sd = robust_std(dd, ratio=ratio)
#         print(dd, sd)
#         sd = []
        sidx = np.split(idx, sd)
#         print(sidx)
        for s in sidx:
#             print(s)
            if len(s) > minlen:
                se = np.sign(x0[s])  # sign of local extrema
                ia = Tools.find_altsign(se, minlen=minlen)  # interval of alternative-signs
                ip = [s[t0:t1] for t0,t1 in ia]
                P += ip  # period of oscillation of each interval
    #             P.append(ip)
#     return P, idx, sidx, sd, dd
    return P


def detect_outliers(x0, ratio=5.):
    x1 = x0.flatten() - np.mean(x0)
    return np.abs(x1) > ratio * robust_std(x1, ratio=2.)
