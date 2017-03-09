"""Collection of models for SHM analysis.
"""

import numpy as np
import numpy.linalg as la
from numpy import ma
import scipy.linalg
import numbers

from . import Tools, Stat, Kalman
# from .Kalman import Kalman


def dgp_cov_matrix(Nt, snr2=100, clen2=1):
    """construct covariance matrix of a differential Gaussian process.

    Args:
        snr2: squared SNR
        clen2: squared correlation length
    Returns:
        W, Winv: the covariance matrix and its inverse
    """
    assert snr2>=0 and clen2>0

    a = 1/clen2
    f = lambda x: (-2*a + (2*a*x)**2) * np.exp(-a * (x**2)) # second derivative of f(x)=exp(-a*x**2)

    C=np.zeros(Nt)
    C[0] = 2 + 10**-2
    C[1] = -1
    for t in range(0,Nt):
        C[t] += (-snr2 * f(t)) #if np.log10(np.abs(v)) > -8 else 0

    W = scipy.linalg.toeplitz(C)
    return W


def diffdeconv(Y0, X0, dord=1, lag=1, snr2=None, clen2=None, dspl=1, sidx=0, Ntrn=None, vthresh=0., corrflag=False, Nexp=0, method="mean"):
    """Deconvolution of multivariate time series using a vectorial FIR filter.

    Args:
        Y0 ():
        X0 ():
        dord (int):
    snr2=100, clen2=0.1, dspl=5 seems to work well on the trend component with full training data
    """
    assert X0.ndim == Y0.ndim == 2
    assert X0.shape[1] == Y0.shape[1]

    dX = np.zeros_like(X0) * np.nan; dX[:,dord:] = np.diff(X0, dord, axis=-1)
    dY = np.zeros_like(Y0) * np.nan; dY[:,dord:] = np.diff(Y0, dord, axis=-1)
    Xvar = Tools.mts_cumview(dX, lag); Yvar = dY

    # prepare regressor
    regressor = Stat.random_subset(Stat.dim_reduction(Stat.multi_linear_regression))
    # regressor = Stat.random_subset(Stat.dim_reduction(Stat.percentile_subset(Stat.multi_linear_regression)))

    # training data
    (tidx0, tidx1), _ = Stat.training_period(X0.shape[1], tidx0=sidx, Ntrn=Ntrn)  # valid training period
    Xtrn, Ytrn = Xvar[:,tidx0:tidx1:dspl], Yvar[:,tidx0:tidx1:dspl]  # down-sampling of training data
    Ntrn = Xtrn.shape[1]
    # GLS matrix
    if snr2 is not None and clen2 is not None and dord > 0:
        # use GLS matrix only for differential process and with valid parameters
        W0 = dgp_cov_matrix(Xvar.shape[1], snr2=snr2, clen2=clen2)
        W = W0[tidx0:tidx1:dspl,:][:,tidx0:tidx1:dspl]
        Winv = la.inv(W)
    else:
        Winv = None # np.eye(Ntrn)

    # estimation of kernel
    Amat, Cvec, *_ = regressor(Ytrn, Xtrn, Winv, vthresh=vthresh, corrflag=corrflag, Nexp=Nexp, method=method)
    Err = Yvar - (Amat @ Xvar + Cvec)  # differential residual
    Sig = Stat.cov(Err, Err)  # covariance matrix
    # if kthresh>0:
    #     Amat[np.abs(Amat)<kthresh] = 0

    toto = Amat @ Tools.mts_cumview(X0, lag)  # cumulative view for convolution
    if dord > 0:
        Yprd = toto - Tools.polyprojection(toto, deg=0, axis=-1)  # projection \Psi^\dagger \Psi
    else:
        Yprd = toto
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
    _, U, S = Stat.pca(X1, corrflag=corrflag)
    if cdim is None:
        cdim = np.sum(S/S[0] > vthresh)
    Xprj = U[:,:cdim] @ U[:,:cdim].T @ X0
    # C = U[:,:cdim].T @ X0  # compressed
    # toto = np.sqrt(np.diag(safe_dot(C, C.T)))
    # C = C/toto[:,newaxis]
    # Xcof = np.asarray(np.ma.dot(np.ma.masked_invalid(X0), np.ma.masked_invalid(C).T))
    return Xprj, U, S #Xcof, C


