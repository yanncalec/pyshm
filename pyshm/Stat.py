"""Statistics related functions.
"""

import numpy as np
import numpy.linalg as la
# from numpy import newaxis, mean, sqrt, zeros, ones, squeeze,\
#     asarray, abs
# from numpy.linalg import norm, svd, inv, pinv

# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
# from sklearn.gaussian_process import GaussianProcess

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
        # Sig = la.norm(Err,'fro')**2 / (Y.size - Y.shape[0]*X.shape[0])
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


def linear_regression(Y, X):
    """
    Linear regression of the model:
        Y = aX + b

    Args:
        Y (1d array): the observation variable
        X (1d array): the explanatory variable
    Returns:
        a,b: the least square solution (using scipy.sparse.linalg.cgs)
        err: residual error
        sigma2: estimation of the noise variance
        S: variance of the estimate a and b (as random variables)
    """
    assert(Y.ndim==X.ndim==1)
    assert(len(Y)==len(X))

    (X0, Y0), nidx = Tools.remove_nan_columns(np.atleast_2d(X), np.atleast_2d(Y))  # the output is a 2d array
    X0 = X0[0]; Y0 = Y0[0]  # convert to 1d array
    A = np.vstack([X0, np.ones(len(X0))]).T
    a, b = np.dot(la.pinv(A), Y0)

    err = a*X + b - Y # residual
    sigma2 =  Tools.safe_norm(err)**2 / (A.shape[0] - la.matrix_rank(A)) # non-biased estimation of noise's variance
    # assert(sigma2 >= 0 or np.isnan(simga2))

    return a, b, err, sigma2, sigma2*np.diag(la.pinv(A.T @ A))


#### Moving window estimation ####

def mw_linear_regression_with_delay(Y0, X0, D0=None, wsize=24*10, dlrange=(-12,12)):
    """Moving window linear regression of two time series with delay.

    We suppose that X is related to Y through:
        Y[t] = K[t]*X[t-D[t]] + B[t] + error
    Provided D, the linear regression estimates the scalars K[t], B[t] on a moving window centered around t.
    If D is not given, the function estimates the optimal delay D[t] on X by solving a series of linear regression problems using different values and selecting the one minimizing the residual.

    Args:
        Y0 (1d array): observation variable
        X0 (1d array): explanatory variable
        D0 (1d array): delay, if provided then D0 is used as delay for linear regression and dlrange will be ignored (no estimation of delay)
        wsize (int): size of the moving window
        dlrange (tuple of int):  range in which the optimal delay will be searched
    Returns:
        D, C, K, B: estimated delay, correlation, slope, intercept
    """
    def optimal_delay(X, Y, tidx, dlrange):
        """Estimate the optimal delay of a long time series X wrt another short time series Y using linear regression. The optimal delay on X is determined by selecting a subsequence of size of Y on a moving window centered around tidx, such that the error of linear regression between the windowed X and Y is minimized."""
        res = []
        for n in range(*dlrange):
            Xn = Tools.safe_slice(X, tidx-n, Y.size, mode='soft')
            res.append(linear_regression(Y, Xn))

        # the optimal delay is taken as the one minimizing the residual of LS
        nidx = np.argmin([la.norm(r[2]) for r in res]) # argmin/argmax always return 0 if data contain nan
        dt = dlrange[0] + nidx
        Xd = Tools.safe_slice(X, tidx-dt, Y.size, mode='soft')
        return dt, res[nidx], corr(Xd, Y), Xd

    assert(X0.ndim == Y0.ndim == 1)
    assert(len(X0) == len(Y0))

    # quantities after compensation of thermal delay
    D = np.zeros(len(X0), dtype=int) if D0 is None else D0
    C = np.zeros(len(X0))
    K = np.zeros(len(X0))
    B = np.zeros(len(X0))

    for tidx in range(len(X0)):
        y = Tools.safe_slice(Y0, tidx, wsize, mode='soft')
        if D0 is None:
            D[tidx], res, C[tidx], _ = optimal_delay(X0, y, tidx, dlrange)
            K[tidx], B[tidx] = res[0], res[1]
        else:
            Xd = Tools.safe_slice(X0, tidx-D0[tidx], y.size, mode='soft')
            res = linear_regression(y, Xd)
            K[tidx], B[tidx] = res[0], res[1]
            C[tidx] = corr(Xd, y)

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

# @Tools.nan_safe
# def corr(x,y):
#     return np.corrcoef(x,y)[:x.shape[0], x.shape[0]:]

def corr(x, y):
    """Compute the correlation matrix of two multi-variate random variables.

    Similar to the numpy function corrcoef but is safe to nan (treat as zero) and complex variables.

    Args:
        x (1d or 2d array):  each column of x is a sample from the first variable.
        y (1d or 2d array):  each column of y is a sample from the second variable. y must have the same number of columns as x.
    Return:
        The correlation matrix, with the (i,j) element being corr(x_i, y_j)
    """
    if x.ndim==1:
        x = x[np.newaxis,:]
        x[np.isnan(x)] = 0
    if y.ndim==1:
        y = y[np.newaxis,:]
        y[np.isnan(y)] = 0

    assert(x.shape[1]==y.shape[1])

    mx = np.mean(x, axis=1); my = np.mean(y, axis=1)
    xmx = x-mx[:, np.newaxis]; ymy = y-my[:,np.newaxis]
    dx = np.sqrt(np.mean(np.abs(xmx)**2, axis=1)) # standard deviation of X
    dy = np.sqrt(np.mean(np.abs(ymy)**2, axis=1)) # standard deviation of Y
    # vx = mean(abs(xmx)**2, axis=1); vy = mean(abs(ymy)**2, axis=1)

    return np.squeeze((xmx/dx[:,np.newaxis]) @ (np.conj(ymy.T)/dy[np.newaxis,:])) / x.shape[1]


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
