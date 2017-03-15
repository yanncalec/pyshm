"""Statistics related functions.
"""

import numpy as np
import numpy.linalg as la
import scipy
import pandas as pd
from functools import wraps

# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
# from sklearn.gaussian_process import GaussianProcess

from . import Tools, Kalman


def cov(X0, Y0, W=None):
    """Reweighted covariance matrix.
    """
    assert X0.ndim == Y0.ndim ==2
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
    """Reweighted correlation matrix"""
    Cxy = cov(X0, Y0, W=W)
    Cxx = cov(X0, X0, W=W)
    Cyy = cov(Y0, Y0, W=W)

    return np.diag(1/np.sqrt(np.diag(Cxx))) @ Cxy @ np.diag(1/np.sqrt(np.diag(Cyy)))


def mean(X0, W=None):
    """Reweighted mean"""

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


def centralize(X0, W=None):
    mX = mean(X0, W=W)
    return X0 - mX[:,np.newaxis]


def normalize(X0, W=None):
    Xc = centralize(X0, W=W)
    Cm = cov(X0, X0, W=W)
    if X0.shape[0]>1:
        U, S, _ = la.svd(Cm)
        Xn = np.diag(1/np.sqrt(S)) @ U.T @ Xc
    else:
        Xn = 1/np.sqrt(Cm) * Xc
    return Xn


# @Tools.nan_safe
# @Tools.along_axis
# def normalize(X, std=True):
#     """Centralize an array by its mean and reduce it by its standard deviation along a given axis.

#     This function is safe to nan values.
#     """
#     if std:
#         return (X - np.mean(X)) / np.std(X)
#     else:  # remove mean only
#         return X - np.mean(X)


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
    """Compute the value corresponding to a percentile from an array.

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


#### Regression analysis ####

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
            vthresh (float): threshold on the singular values S[0], S[1]... Only those such that S[n]/S[0]>vthresh will be kept in the dimension reduction
            corrflag (bool): if True the correlation matrix instead of the covariance matrix
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
            Lc, Cvec, Err, Sig = func(Yvar, Xcof[:cdim, :], sigmaq2, sigmar2, x0=x0, p0=p0, smooth=smooth)
            L0 = []
            for t in range(Nt):
                L0.append(Lc[t,] @ U[:, :cdim].T)
            L = np.asarray(L0)
        else:
            L, Cvec, Err, Sig = func(Yvar, Xvar, sigmaq2, sigmar2, x0=x0, p0=p0, smooth=smooth)

        return L, Cvec, Err, Sig
    return newfunc


def random_subset(func):
    @wraps(func)
    def newfunc(Y, X, W, *args, Nexp=100, method="median", **kwargs):
        """
        Args:
        """
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


def mlr_split_call(func):
    """Decorator functional for multiple linear regression routines.

    This decorator applies dimension check and remove nan values (from inputs), and split the output of MLS routine into list of regressors that each corresponds to an explanary variable.
    """
    @wraps(func)
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

        L, Cvec, Err, Sig = func(Y0, Xs, *args, **kwargs)

        # the second argument tells hsplit where to split the columns of L
        return np.hsplit(L, dimXs[:-1]), Cvec, Err, Sig
    return newfunc


def multi_linear_regression(Y, X, W):
    """Multiple linear regression by generalized least square (GLS).

    GLS looks for the matrices L and the vector C such that the reweighted norm
        ||L*X + C - Y||_W  (* denotes the matrix product)
    is minimized. Analytical formula of the solutions:
        L = cov_W(Y, X) * cov_W(X,X)^-1
        C = mean_W(Y) - L * mean_W(X)
    where cov_W and mean_w are the W-modified covariance matrix / mean vector.

    Args:
        Y (2d array): response variables, each row is a variable and each column an observation
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
    """Brownian motion model.
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
        # LXtn, LPtn, LJt, res = _Kalman.smoother()
        toto, *_ = _Kalman.smoother()
    else:
        # LXtt, LPtt, LXtm, LPtm, LEt, LSt, LKt, LmXt, *_ = _Kalman.filter()
        toto, *_ = _Kalman.filter()

    Xflt = np.asarray(toto)
    # filtered / smoothed observations
    Yflt = np.sum(B * Xflt.transpose(0,2,1), axis=-1).T  # transpose(0,2,1) is for the computation of matrix-vector product, and .T is to make the second axis the time axis.
    Err = Y - Yflt
    Sig = cov(Err, Err)

    Ka0 = []
    Cvec0 = []
    for t in range(Nt):
        V = np.reshape(Xflt[t,], (dimobs, -1))  # state vector reshaped as a matrix
        Cvec0.append(np.atleast_2d(V[:,-1]))
        Ka0.append(V[:,:-1])
    Cvec = np.squeeze(np.asarray(Cvec0)).T
    Ka = np.asarray(Ka0)

    # # filtered / smoothed observations without the polynomial trend
    # B0 = B[:, :, :-dimobs]
    # Xflt0 = Xflt[:, :-dimobs, :]
    # Yflt0 = np.sum(B0 * Xflt0.transpose(0,2,1), axis=-1).T

    # Xflt = np.squeeze(Xflt).T  # take transpose to make the second axis the time axis
    # Xflt0 = np.squeeze(Xflt0).T

    return Ka, Cvec, Err, Sig
    # return (Xflt, Ka, Cvec), Yflt0, Err, Sig


# def multi_linear_regression_bm(Y, X, sigmaq2, sigmar2, x0=0., p0=1., smooth=False):
#     """Brownian motion model.
#     """
#     assert Y.shape[1] == X.shape[1]
#     const_trend = True

#     dimobs, Nt = Y.shape  # dimension of the observation vector and duration
#     dimsys = X.shape[0] * dimobs  # dimension of the system vector
#     dimsys += dimobs if const_trend else 0

#     A = np.eye(dimsys)  # the transition matrix: time-independent
#     # construct the observation matrices: time-dependent
#     B = np.zeros((Nt, dimobs, dimsys))
#     for t in range(Nt):
#         toto = X[:,t].copy()
#         toto[np.isnan(toto)] = 0
#         if const_trend:
#             B[t,] = np.hstack([np.kron(np.eye(dimobs), toto), np.eye(dimobs)])
#         else:
#             B[t,] = np.kron(np.eye(dimobs), toto)

#     # initialize the kalman filter
#     _Kalman = Kalman.Kalman(Y, A, B, G=None, Q=sigmaq2, R=sigmar2, X0=x0, P0=p0)

#     if smooth:
#         LXtn, LPtn, LJt, res = _Kalman.smoother()
#         Xflt = np.asarray(LXtn)
#         # Xflt = np.transpose(np.asarray(LXtn), (0,2,1))
#     else:
#         LXtt, LPtt, LXtm, LPtm, LEt, LSt, LKt, LmXt, *_ = _Kalman.filter()
#         Xflt = np.asarray(LXtt)
#         # Xflt = np.transpose(np.asarray(LXtt), (0,2,1))

#     Yflt = np.sum(B * Xflt.transpose(0,2,1), axis=-1).T  # transpose(0,2,1) is for the computation of matrix-vector product, and .T is to make the second axis the time axis.
#     Err = Y - Yflt
#     Sig = cov(Err, Err)

#     # filtered / smoothed observations without the constant trend
#     B0 = B[:, :, :-dimobs] if const_trend else B
#     Xflt0 = Xflt[:, :-dimobs, :] if const_trend else Xflt
#     Yflt0 = np.sum(B0 * Xflt0.transpose(0,2,1), axis=-1).T

#     Xflt = np.squeeze(Xflt).T
#     Xflt0 = np.squeeze(Xflt0).T

#     return (Xflt, Yflt, B), (Xflt0, Yflt0, B0), Err, Sig


#### Interfaces ####
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


def deconv(Y0, X0, lag, dord=1, pord=1, sidx=0, Ntrn=None, dspl=1, snr2=None, clen2=None, vthresh=0., corrflag=False, Nexp=0):
    """Deconvolution of multivariate time series using a vectorial FIR filter.

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

    Remarks:
        snr2=10**4, clen2=1, dspl=2 seems to work well on the trend component with full training data
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
    Amat, Cvec, Err, Sig = regressor(Yvar, Xvar, sigmaq2, sigmar2, x0, p0, smooth=smooth, sidx=sidx, Ntrn=Ntrn, vthresh=vthresh, corrflag=corrflag)
    Amat0 = Amat[:, :, :Amat.shape[-1]-(pord-dord)]  # kernel matrices without polynomial trend

    # prediction method 1: apply kernel matrices directly on raw inputs
    # this method is theoretically non exact but numerically stable
    Xcmv = Tools.mts_cumview(X0, lag) # DO NOT remove nans: Xcmv[np.isnan(Xcmv)] = 0, see comments in deconv()
    Yflt = np.zeros_like(Y0)
    for t in range(Nt):
        Yflt[:,t] = Amat0[t,] @ Xcmv[:,t]

    # # prediction method 2: apply kernel matrices on differentiated inputs
    # # then re-integrate. This method is theoretically exact but numerically unstable when dord>=2
    # Xcmv = Xvar0
    # Yflt = np.zeros_like(Y0)
    # for t in range(Nt):
    #     Yflt[:,t] = Amat0[t,] @ Xcmv[:,t]
    # # integration to obtain the final result
    # if dord > 0:
    #     Yflt[np.isnan(Yflt)] = 0
    #     for n in range(dord):
    #         Yflt = np.cumsum(Yflt,axis=-1)

    if dord > 0:
        Yprd = Yflt - Tools.polyprojection(Yflt, deg=dord-1, axis=-1)  # projection \Psi^\dagger \Psi
    else:
        Yprd = Yflt

    return Yprd, (Amat, Cvec, Err, Sig)


def mutdecorr(Y0, lag=1, vthresh=1e-3, corrflag=False): #, sidx=0, Ntrn=None):
    """Dimension-wise mutual decorrelation."""
    Yprd = []
    for n in range(Y0.shape[0]):
        locs = list(range(Y0.shape[0]))  # list of row index
        locs.pop(n)  # exclude the current row
        # Regression of the current row by the other rows
        toto, *_ = diffdeconv(Y0[[n],:], Y0[locs,:], lag=lag, vthresh=vthresh, corrflag=corrflag) #, sidx=sidx, Ntrn=Ntrn)
        Yprd.append(toto[0])  # toto is 1-by-? 2d array
    return np.asarray(Yprd)


# def safe_dot(X,Y):
#     return np.asarray(np.ma.dot(np.ma.masked_invalid(X), np.ma.masked_invalid(Y)))

def ssproj(X0, cdim=None, vthresh=1e-1, corrflag=False):
    """Projection of a multivariate time series onto a subspace.
    """
    X1 = np.diff(X0,axis=-1)
    _, U, S = pca(X1, corrflag=corrflag)
    if cdim is None:
        cdim = np.sum(S/S[0] > vthresh)
    Xprj = U[:,:cdim] @ U[:,:cdim].T @ X0
    # C = U[:,:cdim].T @ X0  # compressed
    # toto = np.sqrt(np.diag(safe_dot(C, C.T)))
    # C = C/toto[:,newaxis]
    # Xcof = np.asarray(np.ma.dot(np.ma.masked_invalid(X0), np.ma.masked_invalid(C).T))
    return Xprj, U, S #Xcof, C


#### Moving window estimation ####

def local_thermal_delay(X, Y, tidx, dlrng):
    """Estimate the local thermal delay of a time series X wrt another Y using linear regression.

    Given the time index tidx and the range dlrng of validate delay, the
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

    for n in range(*dlrng):
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

        dt = dlrng[0] + nidx
        Xd = Tools.safe_slice(X, tidx-dt, Y.size, mode="soft")
        # print("corr = {}".format(corr(Xd,Y)))
        return dt, res[nidx], corr(Xd, Y), Xd
    # else:
    #     return np.nan, None, np.nan, None
        # return None, None, None, None


def mw_linear_regression_with_delay(Y0, X0, D0=None, wsize=24*10, dlrng=(-6,6)):
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
        D0 (1d array of int): delay, if provided then D0 is used as delay for linear regression and dlrng will be ignored (no estimation of delay)
        wsize (int): size of the moving window
        dlrng (tuple of int):  range in which the optimal delay is searched
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
                D[tidx], res, C[tidx], _ = local_optimal_delay(X0, y, tidx, dlrng)
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
    #     for t in range(*dlrng):
    #         xidx0 = max(0, tidx+t-wsize//2)
    #         xidx1 = min(xidx0 + wsize, len(X0))
    #         res.append(linear_regression(Y0[yidx0:yidx1], X0[xidx0:xidx1]))
    #         xidxs.append((xidx0, xidx1))
    #
    #     midx = np.argmin([r[2] for r in res])  # index correpsonding to the minimum residual
    #     xidx0, xidx1 = xidxs[midx]
    #     D[tidx] = dlrng[0] + midx  #
    #     C[tidx] = corr(X0[xidx0:xidx1], Y0[yidx0:yidx1])
    #     K[tidx] = res[midx][0]
    #     B[tidx] = res[midx][1]


def local_statistics(X, mwsize, mad=False, causal=False, drop=True):
    """Local mean and standard deviation estimation using pandas library.

    Args:
        X (pandas Series/DataFrame or numpy array)
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


def estimate_thermal_coefficient(xobs, yobs, delay=0, dstep=2, robust=False, shrink=None):
    """Estimate the thermal coefficient.

    Args:
        xobs, yobs (pandas Series): observation of input and output time-series.
        delay (int): delay of xobs
        dstep (int): period for differential
        shrink (float): threshold value for the estimated thermal coefficient
    Returns:
        a, b: thermal coefficients
        yprd: prediction
    """
    xvar = np.roll(xobs.diff(dstep), delay)
    yvar = np.asarray(yobs.diff(dstep))

    #     ynrm = Tools.safe_norm(yvar)
    #     res = pyshm.Stat.linear_regression(yvar/ynrm, xvar/ynrm)
    if robust:
        a0, b, *_ = robust_linear_regression(yvar, xvar, Ns=100, Ng=1000)
    else:
        a0, b, *_ = linear_regression(yvar, xvar)

    if shrink is not None:
        # a = np.sign(a0)*(np.abs(a0)-shrink) if np.abs(a0) > shrink else 0
        a = a0 if np.abs(a0) > shrink else 0
    else:
        a = a0

    yprd = pd.Series(a * Tools.roll_fill(xobs, delay), index=xobs.index)

    return a, b, yprd


def global_thermal_delay(xobs, yobs, dlrng=(-12,12)):
    """Estimation of the global thermal delay.

    A delay t is applied on the 1d time series xobs by rolling, the delayed xobs is then feeded to the linear regression with yobs, and the optimal delay is the value of t minimizing the residual of linear regression.

    Args:
        xobs, yobs (pandas Series): observation of input and output time-series.
        dlrng (tuple of int): range in which the optimal delay will be searched.
    Return:
        estimated optimal thermal delay
    """
    res = []
    for r in range(*dlrng):
        xvar, yvar = np.roll(xobs,r), yobs
        res.append(linear_regression(yvar, xvar))

    # return np.argmin([v[4][0] for v in res])+dlrng[0]
    return np.argmin([v[3] for v in res])+dlrng[0]


# def optimal_diff_period(xobs, yobs, optimdelay, dprange=(1,25)):
#     res = []
#
#     for d in range(*dprange):
#         xvar, yvar = np.roll(xobs.diff(d),optimdelay), yobs.diff(d)
# #         xx, yy = xvar[optimdelay:], yvar[optimdelay:]
#         ynrm = Tools.safe_norm(yvar)
#         res.append(pyshm.Stat.linear_regression(yvar/ynrm, xvar/ynrm))
#
#     return np.argmin([v[3] for v in res]) + 1
#


########### BELOW is OBSELETED #############


def robust_multi_linear_regression(Y, X, W=None, constflag=False, Ng=None, Ns=1000):
    """Robust linear regression in the spirit of RANSAC
    """
    Nt = X.shape[1]
    if Ng is None:
        Ng = max(1, Nt//10)
    res = []
    for n in range(Ng):
        idx = np.random.choice(Nt, Ns, replace=False)
        xsub = X[:,idx]
        ysub = Y[:,idx]
        wsub = W[:,idx][idx,:]
        #     xx, yy = xvar[optimdelay:], yvar[optimdelay:]
        #         ynrm = Tools.safe_norm(yvar)
        toto = multi_linear_regression_corr(ysub, xsub, W=wsub, constflag=constflag)
        res.append([toto[0], toto[1], toto[3]])

    A = np.asarray([k[0][0] for k in res])
    B = np.asarray([k[1] for k in res])
    L, Cvec = np.mean(A, axis=0), np.mean(B, axis=0)

    # Processing on the residuals
    Err = Y - (np.dot(L, X) + Cvec)  # Err has the same length as Y0
    Ern,_ = Tools.remove_nan_columns(Err)
    # Sig = la.norm(Err,"fro")**2 / (Y.size - Y.shape[0]*X.shape[0])
    Sig = np.dot(Ern, Ern.T) / (Y.size - Y.shape[0]*X.shape[0])  # covariance matrix

    return [L], Cvec, Err, Sig



@mlr_split_call
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


# #### Generic ####

# def sign_safe_svd(A):
#     """SVD with coherent sign pattern.
#     """
#     U, S, V0 = la.svd(A)
#     V = V0.T # U @ diag(S) @ V.T = A
#     N = len(S)

#     sl = np.zeros(N)
#     sr = np.zeros(N)

#     for n in range(N):
#         # toto = U[:,n] @ A
#         # sl[n] = np.sign(toto) @ (toto**2)
#         # toto = A @ V[:,n]
#         # sr[n] = np.sign(toto) @ (toto**2)

#         toto = U[:,n] @ (A / la.norm(A, axis=0)[np.newaxis,:])
#         sl[n] = np.sum(toto)

#         toto = (A / la.norm(A, axis=1)[:,np.newaxis]) @ V[:,n]
#         sr[n] = np.sum(toto)

#         if sl[n] * sr[n] < 0:
#             if sl[n] < sr[n]:
#                 sl[n] = -sl[n]
#             else:
#                 sr[n] = -sr[n]

#     U[:,:N] = U[:,:N] @ np.diag(np.sign(sl))
#     V[:,:N] = V[:,:N] @ np.diag(np.sign(sr))

#     return U, S, V.T


# def pca(X0, nc=None, sflag=False):
#     """
#     Principal Component Analysis.

#     Args:
#         X0 (2d array): each row represents a variable and each column represents an observation
#         nc (int): number of components to hold
#         sflag (bool): if True apply sign correction to the principal vectors
#     Returns:
#         C, U : coefficients and corresponded principal directions
#     """

#     X0 = normalize(X0, std=False) # remove the mean
#     # U0, S, _ = sign_safe_svd(np.cov(X0))
#     U0, S, _ = la.svd(np.cov(X0))
#     U = U0.copy()
#     # sign correction:
#     if sflag:
#         X1 = X0/la.norm(X0, axis=0)[np.newaxis,:]
#         for n in range(U.shape[1]):
#             toto = U[:,n] @ X1

#             # method 1:
#             # toto = toto[np.abs(toto)>0.3]
#             # method 2:
#             # idx0 = np.argsort(np.abs(toto))[::-1]
#             # idx = idx0[:max(1,int(len(idx0)/4))]

#             # if np.sign(toto) @ (toto**4) < 0:
#             #     U[:,n] *= -1

#             if np.mean(toto) < 0:
#                 U[:,n] *= -1
#     C  = U.T @ X0

#     if nc is None:
#         return C, U
#     else:
#         return C[:nc,:], U[:,:nc]

# @Tools.nan_safe
# def corr(x0, y0):
#     x = np.atleast_2d(x0)
#     y = np.atleast_2d(y0)
#     toto = np.corrcoef(x, y)[:x.shape[0], x.shape[0]:]
#     # print(len(x), len(y), x.shape, toto0)
#     return np.squeeze(toto)*1.

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


# def corr(X0, Y0):
#     # X = np.ma.masked_invalid(X0)
#     # Y = np.ma.masked_invalid(Y0)
#     # return np.ma.corrcoef(X, Y).data[:X.shape[0], X.shape[0]:]
#     X = X0.copy(); X[np.isnan(X0)] = 0
#     Y = Y0.copy(); Y[np.isnan(Y0)] = 0
#     return np.corrcoef(X, Y)[:X.shape[0], X.shape[0]:]

# def cov(X0, Y0):
#     X = X0.copy(); X[np.isnan(X0)] = 0
#     Y = Y0.copy(); Y[np.isnan(Y0)] = 0
#     return np.cov(X, Y)[:X0.shape[0], X0.shape[0]:]
#     # print(toto.shape)
#     # Cxy = toto[:X0.shape[0], X0.shape[0]:]
#     # return Cxy
