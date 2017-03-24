"""Statistics related functions.
"""

import numpy as np
import numpy.linalg as la
import scipy
import scipy.special
import pandas as pd
from functools import wraps

# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
# from sklearn.gaussian_process import GaussianProcess

from . import Tools, Kalman

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


def pca(X, nc=None, corrflag=False):
    """
    Principal Component Analysis.

    Args:
        X (2d array): each row represents a variable and each column represents an observation
        nc (int): number of components to hold
        sflag (bool): if True apply sign correction to the principal vectors
    Returns:
        C, U : coefficients and corresponded principal directions

    """
    if corrflag:
        U, S, _ = la.svd(corr(X,X))
    else:
        U, S, _ = la.svd(cov(X,X))

    C = U.T @ (X - mean(X)[:, np.newaxis])
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


def percentile(X0, ratio):
    """Compute the value corresponding to a percentile in an array.

    Args:
        X0 (nd array): input array
        ratio (float): percentile
    """
    assert 0. <= ratio <= 1.

    X = X0.copy().flatten()
    X[np.isnan(X)] = 0  # remove all nans

    idx = np.argsort(X)  # increasing order sort
    nz0 = int(np.floor(len(idx) * ratio))
    nz1 = int(np.ceil(len(idx) * ratio))
    if nz0==nz1==0:
        return X[idx[0]]
    if nz0==nz1==len(idx):
        return X[idx[-1]]
    else:
        return np.mean(X[idx[nz0:nz1+1]])


def local_statistics(X, mwsize, mad=False, causal=False, drop=True):
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


# def linear_regression(Y, X, nanmode="remove"):
#     """Scalar linear regression of the model Y = aX + b.

#     Args:
#         Y (1d array): the observation variable
#         X (1d array): the explanatory variable
#         nanmode (str): what to do with nan values. "interpl": interpolation, "remove": remove, "zero": replace by 0
#     Returns:
#         a, b: the least square solution (using scipy.sparse.linalg.cgs)
#         err: residual error
#         sigma2: estimation of the noise variance
#         S: variance of the estimate a and b (as random variables)
#     """
#     assert(Y.ndim==X.ndim==1)
#     assert(len(Y)==len(X))

#     if nanmode == "remove":
#         (X0, Y0), nidx = Tools.remove_nan_columns(np.atleast_2d(X), np.atleast_2d(Y))  # the output is a 2d array
#         X0 = X0[0]; Y0 = Y0[0]  # convert to 1d array
#     elif nanmode == "interpl":
#         X0 = Tools.interpl_nans(X)
#         Y0 = Tools.interpl_nans(Y)
#     else:
#         X0 = X.copy()
#         Y0 = Y.copy()

#     # final clean-up
#     X0[np.isnan(X0)] = 0; Y0[np.isnan(Y0)] = 0

#     if len(X0)>0 and len(Y0)>0:
#         A = np.vstack([X0, np.ones(len(X0))]).T
#         a, b = np.dot(la.pinv(A), Y0)

#         err = a*X + b - Y         # residual
#         # non-biased estimation of noise's variance
#         A_rank = la.matrix_rank(A)
#         if A.shape[0] > A_rank:
#             sigma2 = Tools.safe_norm(err)**2 / (A.shape[0] - A_rank)
#         else:
#             sigma2 = np.nan
#         # assert(sigma2 >= 0 or np.isnan(simga2))
#         S = sigma2*np.diag(la.pinv(A.T @ A))
#     else:
#         raise ValueError("Linear regression failed due to lack of enough meaningful values in the inputs.")
#         # a, b, err, sigma2, S = np.nan, np.nan, np.nan*np.zeros_like(Y), np.nan, np.nan*np.zeros(2)

#     return a, b, err, sigma2, S


def Hurst(data, mwsize, sclrng=None, wvlname="haar"):
    """Estimate the Hurst exponent of a time series using wavelet transform.

    Args:
        data (1d array): input time series
        mwsize (int): size of the smoothing window
        sclrng (tuple): index of scale range used for linear regression
        wvlname (str): wavelet name
    Returns:
        H (1d array): estimate of Hurst exponent
        B (1d array): estimate of intercept
        V (2d array): estimate of variance of H and B
    """
    def linear_regression(yvar0, xvar0):
        mx, my = np.mean(xvar0), np.mean(yvar0)
        xvar = xvar0 - mx
        yvar = yvar0 - my
        a = np.dot(yvar, xvar) / np.dot(xvar, xvar)
        b = my - a * mx
        return a, b

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
    H, B = np.zeros(Nt), np.zeros(Nt)
    # H, B, V = np.zeros(Nt), np.zeros(Nt), np.zeros((2, Nt))
    xvar = np.arange(*sclrng)  # explanatory variable

    for t in range(Nt):
        yvar = np.log2(S[sclrng[0]:sclrng[1],t])
        # a0, b0, *_ = multi_linear_regression(np.atleast_2d(yvar), np.atleast_2d(xvar), None)
        # a, b = float(a0), float(b0)
        a, b = linear_regression(yvar, xvar)
        # try:
        #     a, b, err, sigma2, v = linear_regression(yvar, xvar, nanmode="interpl")
        # except:
        #     a, b, err, sigma2, v = np.nan, np.nan, None, np.nan, np.nan * np.ones(2)
        H[t] = max(min((a-1)/2,1), 0)
        B[t] = b
        # V[:,t] = v

    # roll to get a causal result
    sc = mwsize // 2
    H = np.roll(H, -sc); H[-sc:] = np.nan
    B = np.roll(B, -sc); B[-sc:] = np.nan
    # V = np.roll(V, -sc); V[:,-sc:] = np.nan

    # drop the begining
    # H[:mwsize] = np.nan
    # B[:mwsize] = np.nan
    # V[:,:mwsize] = np.nan
    return H, B


def Hurst_rs(X0, nrng=None, alc=False):
    """Estimate the Hurst exponent of a time series using the definition.

    This method seems highly sensitive to the parameters and computationally slow.
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
        (tidx0, tidx1): tuple of valid starting and ending index.
        Ntrn: length of valid training period.
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


def dim_reduction(func):
    @wraps(func)
    def newfunc(Y0, X0, W0, *args, vthresh=1e-3, corrflag=False, **kwargs):
        """Dimension reduction in multivariate linear regression by PCA.

        Args:
            vthresh (float): threshold on the singular values S[0], S[1]... Only those such that S[n]/S[0]>vthresh will be kept after the dimension reduction
            corrflag (bool): if True use the correlation matrix instead of the covariance matrix
        """
        # dimension reduction
        if vthresh > 0:
            Xcof, U, S = pca(X0, nc=None, corrflag=corrflag)  # pca coefficients
            cdim = np.sum(S/S[0] > vthresh)  # reduced dimension
            Lc, Cvec, Err, Sig = func(Y0, Xcof[:cdim, :], W0, *args, **kwargs)
            L = Lc @ U[:, :cdim].T
        else:
            L, Cvec, Err, Sig = func(Y0, X0, W0, *args, **kwargs)
        return L, Cvec, Err, Sig
    return newfunc


def dim_reduction_bm(func):
    """Dimension reduction in Brownian Motion model multivariate linear regression.
    """
    @wraps(func)
    def newfunc(Yvar, Xvar, sigmaq2, sigmar2, x0, p0, smooth=False, sidx=0, Ntrn=None, vthresh=0., corrflag=False):
        Nt = Xvar.shape[1]  # length of observations
        # training data for dim reduction
        (tidx0, tidx1), _ = training_period(Nt, tidx0=sidx, Ntrn=Ntrn)  # valid training period
        # dimension reduction
        if vthresh > 0:
            _, U, S = pca(Xvar[:, tidx0:tidx1], nc=None, corrflag=corrflag)  # pca coefficients
            cdim = np.sum(S/S[0] > vthresh)  # reduced dimension
            Xcof = U.T @ Xvar  # coefficients of Xvar
            (A, Acov), (Cvec, Ccov), Err, Sig = func(Yvar, Xcof[:cdim, :], sigmaq2, sigmar2, x0=x0, p0=p0, smooth=smooth)
            L0, C0 = [], []
            # print(A.shape, Acov.shape)
            W = U[:, :cdim]  # analysis matrix of subspace basis
            Wop = Tools.matprod_op_right(W.T, A.shape[1])
            for t in range(Nt):
                L0.append(A[t,] @ W.T)
                C0.append(Wop @ Acov[t,] @ Wop.T)
                # C0.append(U[:, :cdim] @ Acov[t,] @ U[:, :cdim].T)
            L = np.asarray(L0)
            Lcov = np.asarray(C0)
        else:
            (L,Lcov), (Cvec,Ccov), Err, Sig = func(Yvar, Xvar, sigmaq2, sigmar2, x0=x0, p0=p0, smooth=smooth)

        return (L,Lcov), (Cvec,Ccov), Err, Sig
    return newfunc


def random_subset(func):
    @wraps(func)
    def newfunc(Y, X, W, *args, Nexp=100, method="median", **kwargs):
        if Nexp > 0:
            Nt = X.shape[1]
            Ns = X.shape[0]*Y.shape[0]
            res = []
            for n in range(Nexp):
                # regression on a random subset
                idx = np.random.choice(Nt, 2*Ns, replace=False)
                Ysub = Y[:,idx]
                Xsub = X[:,idx]
                Wsub = W[:,idx][idx,:] if W is not None else None
                toto = func(Ysub, Xsub, Wsub, *args, **kwargs)
                res.append([toto[0], toto[1]])  # L, Cvec
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


def multi_linear_regression(Y, X, W):
    """Multivariate linear regression by generalized least square (GLS).

    GLS looks for the matrices L and the vector C such that the reweighted norm
        ||L*X + C - Y||_W  (* denotes the matrix product)
    is minimized. Analytical formula of the solutions:
        L = cov_W(Y, X) * cov_W(X,X)^-1
        C = mean_W(Y) - L * mean_W(X)
    where cov_W and mean_w are the W-modified covariance matrix / mean vector.

    Args:
        Y (2d array): response variables
        X (2d array): explanatory variables
        W (2d matrix): symmetric and positive definite
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

    # L = np.dot(Syx, la.pinv(Sxx))
    vreg = 0  # regularization
    L = np.dot(Syx, la.inv(Sxx + vreg * np.eye(Sxx.shape[0])))
    Cvec = mY - np.dot(L, mX) # if constflag else np.zeros((dimY, 1))

    Err = Y - (np.dot(L, X) + Cvec)  # Err has the same length as Y0
    Sig = cov(Err, Err)

    return L, Cvec, Err, Sig


def multi_linear_regression_bm(Y, X, sigmaq2, sigmar2, x0=0., p0=1., smooth=False):
    """Multivariate linear regression by Brownian motion model.
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

    if smooth:
        LXt, LPt, *_ = _Kalman.smoother()
    else:
        LXt, LPt, *_ = _Kalman.filter()

    Xflt = np.asarray(LXt)
    Pflt = np.asarray(LPt)
    # print(Xflt.shape, Pflt.shape)
    # filtered / smoothed observations
    Yflt = np.sum(B * Xflt.transpose(0,2,1), axis=-1).T  # transpose(0,2,1) is for the computation of matrix-vector product, and .T is to make the second axis the time axis.
    Err = Y - Yflt
    Sig = cov(Err, Err)

    Ka0 = []
    Cvec0 = []
    for t in range(Nt):
        V = np.reshape(Xflt[t,], (dimobs, -1))  # state vector reshaped as a matrix
        # Cvec0.append(np.atleast_2d(V[:,-1]))
        Cvec0.append(V[:,-1])
        Ka0.append(V[:,:-1])
    Cvec = np.asarray(Cvec0)
    Ka = np.asarray(Ka0)  # Ka has shape Nt * dimobs * (dimsys-dimobs)

    # # filtered / smoothed observations without the polynomial trend
    # B0 = B[:, :, :-dimobs]
    # Xflt0 = Xflt[:, :-dimobs, :]
    # Yflt0 = np.sum(B0 * Xflt0.transpose(0,2,1), axis=-1).T

    # Xflt = np.squeeze(Xflt).T  # take transpose to make the second axis the time axis
    # Xflt0 = np.squeeze(Xflt0).T

    return (Ka, Pflt[:,:-dimobs,:-dimobs]), (Cvec, Pflt[:,-dimobs:,-dimobs:]), Err, Sig
    # return Ka, Cvec, Err, Sig


def _gp_cov_matrix(Nt, snr2, clen2):
    """Construct the covariance matrix of a Gaussian process of covariance function
        f(x)=exp(-a*x**2)
    """
    f = lambda x: np.exp(-(x**2)/clen2)
    C = snr2 * f(np.arange(Nt))
    C[0] += 1  # noise
    return scipy.linalg.toeplitz(C)


def _dgp_cov_matrix(Nt, snr2=100, clen2=1):
    """construct covariance matrix of a differential Gaussian process.

    Args:
        snr2: squared SNR
        clen2: squared correlation length
    Returns:
        W, Winv: the covariance matrix and its inverse
    """
    ddf = lambda x: (-2/clen2 + (2*x/clen2)**2) * np.exp(-(x**2)/clen2) # second derivative of f(x)=exp(-(x**2)/clen2)
    C = -snr2 * ddf(np.arange(Nt))
    C[0] += 2 + 0.01  # noise, add a small number to regularize
    C[1] += -1
    return scipy.linalg.toeplitz(C)


########## Interfaces ##########

def deconv(Y0, X0, lag, dord=1, pord=1, snr2=None, clen2=None, dspl=1, sidx=0, Ntrn=None, vthresh=0., corrflag=False, Nexp=0):
    """Deconvolution of multivariate time series using a vectorial FIR filter by GLS.

    We look for the kernel convolution matrices A of the model
        Y_t = A (*) X_t + P_t
    where A = [A_0... A_{p-1}] are matrices, (*) denotes the convolution, P_t is a polynomial Gaussian process of order pord. We estimate A by Generalized Least-Square (GLS) by differentiating the data dord times.

    Args:
        Y0 (2d array): observations, each row is a variable and each column is an observation
        X0 (2d array): inputs, each row is a variable and each column is an observation
        dord (int): order of derivative
        pord (int): order of polynomial trend
        sidx (int): starting index of the training period
        Ntrn (int): length of the training period
        dspl (int): down-sampling rate
        snr2, clen2 (float): signal to noise ratio and correlation length of the polynomial Gaussian process
        vthresh (float): threshold in dimension reduction
        corrflag (bool): if True use the correlation matrix for dimension reduction
        Nexp (int): number of experiments in the RANSAC algorithm
    """
    assert X0.ndim == Y0.ndim == 2
    assert X0.shape[1] == Y0.shape[1]
    assert pord >= dord

    Nt = X0.shape[1]  # length of observations

    # the part of external input
    dX = np.zeros_like(X0) * np.nan; dX[:,dord:] = np.diff(X0, dord, axis=-1)
    Xvar0 = Tools.mts_cumview(dX, lag)  # cumulative view for convolution
    # the part of polynominal trend
    Xvar1 = Tools.dpvander(np.arange(Nt)/Nt, pord, dord)  # division by Nt: normalization for numerical stability
    Xvar = np.vstack([Xvar0, Xvar1[:-1,:]])  #[:-1,:] removes the constant trend which may cause non-invertible covariance matrix. If the constant trend is kept here, Yprd at the end of this function should be modified accordingly like this:
    # Amat0 = Amat[:, :-(pord-dord+1)] ...

    dY = np.zeros_like(Y0) * np.nan; dY[:,dord:] = np.diff(Y0, dord, axis=-1)
    Yvar = dY

    # construct the covariance matrix of the Gaussian process
    if clen2 is not None and clen2 > 0 and snr2 is not None and snr2 >= 0:
        W0 = _dgp_cov_matrix(Nt, snr2, clen2)
        if dord > 1:
            warnings.warn("The current implementation of the GP covariance matrix is not exact for dord>1.")
    else:
        W0 = None # invalid parameters, equivalent to W0=np.eye(Nt)

    # prepare regressor
    regressor = random_subset(dim_reduction(multi_linear_regression))
    # regressor = random_subset(dim_reduction(percentile_subset(multi_linear_regression)))

    # training data
    (tidx0, tidx1), _ = training_period(Nt, tidx0=sidx, Ntrn=Ntrn)  # valid training period
    Xtrn, Ytrn = Xvar[:,tidx0:tidx1:dspl], Yvar[:,tidx0:tidx1:dspl]  # down-sampling of training data
    # GLS matrix
    if W0 is not None :
        Winv = la.inv(W0[tidx0:tidx1:dspl,:][:,tidx0:tidx1:dspl])
    else:
        Winv = None # equivalent to np.eye(Xtrn.shape[1])

    # regresion
    # method ("mean" or "median") used in random_subset is active only when Nexp>0
    Amat, Cvec, *_ = regressor(Ytrn, Xtrn, Winv, vthresh=vthresh, corrflag=corrflag, Nexp=Nexp, method="mean")
    Err = Yvar - (Amat @ Xvar + Cvec)  # differential residual
    Sig = cov(Err, Err)  # covariance matrix
    Amat0 = Amat[:, :Amat.shape[-1]-(pord-dord)]
    # Amat0 = Amat[:, :-(pord-dord)] if pord-dord > 0 else Amat
    # if kthresh>0:
    #     Amat[np.abs(Amat)<kthresh] = 0

    # prediction
    Xcmv = Tools.mts_cumview(X0, lag)
    # Xcmv[np.isnan(Xcmv)] = 0  # Remove nans will introduce large values around discontinuties
    Yflt = Amat0 @ Xcmv
    if dord > 0:
        Yprd = Yflt - Tools.polyprojection(Yflt, deg=dord-1, axis=-1)  # projection \Psi^\dagger \Psi
    else:
        Yprd = Yflt
    return Yprd, (Amat, Cvec, Err, Sig)


def deconv_bm(Y0, X0, lag, dord=1, pord=1, sigmaq2=10**-6, sigmar2=10**-1, x0=0., p0=1., smooth=False, sidx=0, Ntrn=None, vthresh=0., corrflag=False):
    """Deconvolution of multivariate time series using a vectorial FIR filter by Kalman filter.
    """
    assert X0.ndim == Y0.ndim == 2
    assert X0.shape[1] == Y0.shape[1]

    assert pord >= dord

    Nt = X0.shape[1]  # length of observations

    # the part of external input
    dX = np.zeros_like(X0) * np.nan; dX[:,dord:] = np.diff(X0, dord, axis=-1)
    Xvar0 = Tools.mts_cumview(dX, lag)  # cumulative view for convolution
    # the part of polynominal trend
    Xvar1 = Tools.dpvander(np.arange(Nt)/Nt, pord, dord)  # division by Nt: normalization for numerical stability
    Xvar = np.vstack([Xvar0, Xvar1[:-1,:]])  #[:-1,:] removes the constant trend which may cause non-invertible covariance matrix. If the constant trend is kept here, Yprd at the end of this function should be modified accordingly like this:
    # Amat0 = Amat[:, :-(pord-dord+1)] ...

    dY = np.zeros_like(Y0) * np.nan; dY[:,dord:] = np.diff(Y0, dord, axis=-1)
    Yvar = dY

    # prepare regressor
    regressor = dim_reduction_bm(multi_linear_regression_bm)

    # regression
    (Amat, Acov), (Cvec,Ccov), Err, Sig = regressor(Yvar, Xvar, sigmaq2, sigmar2, x0, p0, smooth=smooth, sidx=sidx, Ntrn=Ntrn, vthresh=vthresh, corrflag=corrflag)
    # (Amat, Acov): kernel matrices and covariance matrix
    Amat0 = Amat[:, :, :Amat.shape[-1]-(pord-dord)]  # kernel matrices without polynomial trend

    # # prediction method 1: apply kernel matrices directly on raw inputs
    # # this method is theoretically non exact but numerically stable
    # Xcmv = Tools.mts_cumview(X0, lag) # DO NOT remove nans: Xcmv[np.isnan(Xcmv)] = 0, see comments in deconv()
    # Yflt = np.zeros_like(Y0)
    # for t in range(Nt):
    #     Yflt[:,t] = Amat0[t,] @ Xcmv[:,t]

    # prediction method 2: apply kernel matrices on differentiated inputs
    # then re-integrate. This method is theoretically exact but numerically unstable when dord>=2
    Xcmv = Xvar0
    Yflt = np.zeros_like(Y0)
    for t in range(Nt):
        # Yflt[:,t] = Amat0[t,] @ Xcmv[:,t] + Cvec[t]
        Yflt[:,t] = Amat0[t,] @ Xcmv[:,t]
    # integration to obtain the final result
    if dord > 0:
        Yflt[np.isnan(Yflt)] = 0
        for n in range(dord):
            Yflt = np.cumsum(Yflt,axis=-1)

    # prediction: projection \Psi^\dagger \Psi
    # Remark: Yprd has shape Y0.shape[1]*Nt
    Yprd = Yflt - Tools.polyprojection(Yflt, deg=dord-1, axis=-1) if dord > 0 else Yflt

    # covariance matrix
    Ycov = np.zeros((Nt,Y0.shape[0],Y0.shape[0]))
    for t in range(Nt):
        M = np.kron(np.eye(Y0.shape[0]), Xcmv[:,t])
        Ycov[t,:,:] = M @ Acov[t,] @ M.T

    return (Yprd,Ycov), ((Amat,Acov), (Cvec,Ccov), Err, Sig)


def ssproj(X0, cdim=1, vthresh=None, corrflag=False, drophead=0, percent=1.):
    """Projection of a multivariate time series onto a subspace.

    Args:
        X0 (2d array): input
        cdim (int): dimension of the subspace, if cdim==0 return zero
        vthresh (float): relative threshold, if given cdim will be ignored
        corrflag (bool): if True use correlation matrix for PCA
    Returns:
        Xprj: projection
        U,S: PCA basis and singular values
        cdim: true dimension of the subspace
    """
    assert not ((cdim is None) and (vthresh is None))
    # take derivative to transform to a stationary time series
    X1 = np.diff(X0,axis=-1)
    X1[:,:drophead] = np.nan  # remove the begining
    # another option is to centralize X1 = centralize(X0)

    # PCA of transformed time series
    # percentile regularization
    if percent < 1.:
        nX = Tools.safe_norm(X1, axis=0)
        sidx = np.where(nX < percentile(nX, percent))[0]  # index of largest values
        X2 = X1[:,sidx]
    else:
        X2 = X1

    _, U, S = pca(X2, corrflag=corrflag)
    # subspace dimension
    if cdim is None: # if cdim is not given, use vthresh to determine it
        # toto = S/S[0]
        # cdim = np.sum(toto > vthresh)
        toto = 1-np.cumsum(S/np.sum(S))
        cdim = np.where(toto < vthresh)[0][0]+1
        print(cdim,toto)
        # cdim = np.sum(S/S[0] > vthresh)
        # cdim = np.sum(S/np.sum(S) > vthresh)
    else:  # if cdim is given, vthresh has no effect
        pass

    # projection
    if cdim > 0:
        Xprj = U[:,:cdim] @ U[:,:cdim].T @ X0
        # # or by integration
        # dXprj = U[:,:cdim] @ U[:,:cdim].T @ X1
        # dXprj [np.isnan(dXprj)] = 0
        # Xprj = np.zeros_like(X0)
        # Xprj[:,1:] = np.cumsum(dXprj, axis=-1)
    else:
        Xprj = np.zeros_like(X0)
    return Xprj, (U,S), cdim


def mutdecorr(Y0, lag, vthresh=1e-3, corrflag=False): #, sidx=0, Ntrn=None):
    """Dimension-wise mutual decorrelation."""
    Yprd = []
    for n in range(Y0.shape[0]):
        locs = list(range(Y0.shape[0]))  # list of row index
        locs.pop(n)  # exclude the current row
        # Regression of the current row by the other rows
        toto, *_ = deconv(Y0[[n],:], Y0[locs,:], lag, dord=1, pord=1, clen2=None, dspl=1, vthresh=vthresh, corrflag=corrflag)
        Yprd.append(toto[0])  # toto is 1-by-? 2d array
    return np.asarray(Yprd)

