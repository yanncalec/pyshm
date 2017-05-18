"""Collection of models for SHM analysis.
"""

import numpy as np
import numpy.linalg as la
from numpy import ma
import scipy.linalg
import numbers
import warnings

from . import Tools, Stat, Kalman
# from .Kalman import Kalman


########## Utility functions ##########

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

def deconv(Y0, X0, lag, dord=1, pord=1, snr2=None, clen2=None, dspl=1, sidx=0, Ntrn=None, vthresh=0., cdim=None, Nexp=0):
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
        Nexp (int): number of experiments in the RANSAC algorithm, no RANSAC if Nexp==0
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
    regressor = Stat.dim_reduction_pca(Stat.random_subset(Stat.multi_linear_regression))
    # regressor = dim_reduction_cca(random_subset(multi_linear_regression))  # not recommended
    # regressor = random_subset(dim_reduction_pca(multi_linear_regression))
    # regressor = dim_reduction_pca(random_subset(percentile_subset(multi_linear_regression)))

    # training data
    (tidx0, tidx1), _ = Stat.training_period(Nt, tidx0=sidx, Ntrn=Ntrn)  # valid training period
    Xtrn, Ytrn = Xvar[:,tidx0:tidx1:dspl], Yvar[:,tidx0:tidx1:dspl]  # down-sampling of training data
    # GLS matrix
    if W0 is not None :
        Winv = la.inv(W0[tidx0:tidx1:dspl,:][:,tidx0:tidx1:dspl])
    else:
        Winv = None # equivalent to np.eye(Xtrn.shape[1])

    # regresion
    # method ("mean" or "median") used in random_subset is active only when Nexp>0
    # corrflag=False
    # corrflag (bool): if True use the correlation matrix for dimension reduction
    # ((Amat,Amatc), Cvec, _, _), toto = regressor(Ytrn, Xtrn, Winv, vthresh=vthresh, corrflag=corrflag, Nexp=Nexp, method="mean")
    (Amat, Cvec, *_), (Amatc, *_) = regressor(Ytrn, Xtrn, Winv, vthresh=vthresh, cdim=cdim, Nexp=Nexp, method="mean")
    Err = Yvar - (Amat @ Xvar + Cvec)  # differential residual
    Sig = Stat.cov(Err, Err)  # covariance matrix
    Amat0 = Amat[:, :Amat.shape[-1]-(pord-dord)]  # kernel matrix corresponding to the external input
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

    # return Yprd, (Amat, Cvec, Err, Sig), Amatc #(U, Sy, cdimy, V, Sx, cdimx)
    return Yprd, Amat, Amatc


def deconv_bm(Y0, X0, lag, dord=1, pord=1, sigmaq2=10**-6, sigmar2=10**-3, x0=0., p0=1., smooth=False, sidx=0, Ntrn=None, vthresh=0., cdim=None): # rescale=True
    """Deconvolution of multivariate time series using a vectorial FIR filter by Kalman filter.
    """
    assert X0.ndim == Y0.ndim == 2
    assert X0.shape[1] == Y0.shape[1]

    assert pord >= dord

    Nt = X0.shape[1]  # length of observations

    # the part of external input
    dX0 = np.zeros_like(X0) * np.nan; dX0[:,dord:] = np.diff(X0, dord, axis=-1)
    dY0 = np.zeros_like(Y0) * np.nan; dY0[:,dord:] = np.diff(Y0, dord, axis=-1)
    dX, dY = dX0, dY0

    Xvar0 = Tools.mts_cumview(dX, lag)  # cumulative view for convolution
    # the part of polynominal trend
    Xvar1 = Tools.dpvander(np.arange(Nt)/Nt, pord, dord)  # division by Nt: normalization for numerical stability
    # Xvar and Yvar are the variables passed to the Kalman filter
    Xvar = np.vstack([Xvar0, Xvar1[:-1,:]])  #[:-1,:] removes the constant trend which may cause non-invertible covariance matrix. If the constant trend is kept here, Yprd at the end of this function should be modified accordingly like this:
    # Amat0 = Amat[:, :-(pord-dord+1)] ...
    Yvar = dY

    # prepare regressor
    regressor = Stat.dim_reduction_bm(Stat.multi_linear_regression_bm)

    # regression
    ((Amat, Acov), (Cvec, Ccov), Err, Sig), ((Amatc, Acovc), *_) = regressor(Yvar, Xvar, sigmaq2, sigmar2, x0, p0, smooth=smooth, sidx=sidx, Ntrn=Ntrn, vthresh=vthresh, cdim=cdim, rescale=True)

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
        # Remark: Yprd has shape Y0.shape[1]*Nt
        Yprd = Yflt - Tools.polyprojection(Yflt, deg=dord-1, axis=-1)  # prediction: projection \Psi^\dagger \Psi
    else:
        Yprd = Yflt

    # # covariance matrix: abandonned
    # Ycov = np.zeros((Nt,Y0.shape[0],Y0.shape[0]))
    # for t in range(Nt):
    #     M = np.kron(np.eye(Y0.shape[0]), Xcmv[:,t])
    #     Ycov[t,:,:] = M @ Acov[t,] @ M.T

    # return Yprd, ((Amat, Acov), (Cvec, Ccov), Err, Sig), (Amatc, Acovc)
    return Yprd, (Amat, Acov), (Amatc, Acovc)


def ssproj(X0, cdim=1, vthresh=None, corrflag=False, drophead=0):
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
    # percent = 1.
    # if percent < 1.:
    #     nX = Tools.safe_norm(X1, axis=0)
    #     sidx = np.where(nX < Stat.percentile(nX, percent))[0]  # index of largest values
    #     X2 = X1[:,sidx]
    # else:
    #     X2 = X1
    # _, U, S = Stat.pca(X2, corrflag=corrflag)

    _, U, S = Stat.pca(X1, corrflag=corrflag)

    # subspace dimension
    if cdim is None: # if cdim is not given, use vthresh to determine it
        assert 0 < vthresh <=1.
        # toto = S/S[0]
        # cdim = np.sum(toto > vthresh)
        toto = np.cumsum(S/np.sum(S))
        cdim = np.where(toto >= vthresh)[0][0]+1
        # print(cdim,toto)
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


def mutdecorr(Y0, lag, vthresh=1e-3): #, sidx=0, Ntrn=None):
    """Dimension-wise mutual decorrelation."""
    Yprd = []
    for n in range(Y0.shape[0]):
        locs = list(range(Y0.shape[0]))  # list of row index
        locs.pop(n)  # exclude the current row
        # Regression of the current row by the other rows
        toto, *_ = Stat.deconv(Y0[[n],:], Y0[locs,:], lag, dord=1, pord=1, clen2=None, dspl=1, vthresh=vthresh)
        Yprd.append(toto[0])  # toto is 1-by-? 2d array
    return np.asarray(Yprd)

