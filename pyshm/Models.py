"""Collection of models for SHM analysis.
"""

import numpy as np
import numpy.linalg as la
from numpy import ma
import scipy.linalg
from functools import wraps
import numbers
import warnings
import pandas as pd

from . import Tools, Stat, Kalman
# from pyshm import Tools, Stat, Kalman


########## Utility functions ##########

def _gp_cov_matrix(Nt, snr2, clen2):
    """Construct the covariance matrix of a Gaussian process of covariance function f(x)=exp(-a*x**2)
    """
    f = lambda x: np.exp(-(x**2)/clen2)
    C = snr2 * f(np.arange(Nt))
    C[0] += 1 # noise
    return scipy.linalg.toeplitz(C)


def _dgp_cov_matrix_wrong(Nt, snr2=100, clen2=1):
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


def _dgp_cov_matrix(Nt, snr2=100, clen2=1):
    """construct covariance matrix of a differential Gaussian process.

    Args:
        snr2: squared SNR
        clen2: squared correlation length
    Returns:
        W, Winv: the covariance matrix and its inverse
    """
    f = lambda x: np.exp(-(x**2)/clen2)
    C = snr2 * (2*f(np.arange(Nt)) - f(1+np.arange(Nt))- f(-1+np.arange(Nt)))
    C[0] += 2 + 0.01  # noise, add a small number to regularize
    C[1] += -1
    return scipy.linalg.toeplitz(C)


########## Class Interfaces to deconv and deconv_bm (not implemented) ##########

# def func_mparms_mean(func):
#     """
#     """

#     @wraps(func)
#     def newfunc(*args, prng=(1,24*3), **kwargs):
#         Y0 = []
#         A0 = []
#         C0 = []
#         for p in range(*prng):
#             y, a, c = func(*args, dsp=p, **kwargs)
#             Y0.append(y)
#             A0.append(a)
#             C0.append(c)
#         Y = np.asarray(Y0)
#         A = np.asarray(A0)
#         C = np.asarray(C0)
#         return np.mean(Y, axis=0), np.mean(A, axis=0), np.mean(C, axis=0)
#     return newfunc

class MxDeconv:
    def __init__(self, Y0, X0, lag, pord, dord, smth):
        assert X0.ndim == Y0.ndim == 2
        assert X0.shape[1] == Y0.shape[1]
        assert pord >= dord
        assert lag >= 1

        self.X0 = X0.copy()
        self.Y0 = Y0.copy()
        self.lag = lag
        self.Nt = self.X0.shape[1]  # length of observations

        self.pord = pord
        self.dord = dord
        self.smth = smth

    def _prepare_data(self, mwsize=24, kzord=1, method='mean', causal=False):
        # smoothed derivatives
        if self.dord>0:
            if self.smth:
                X1 = Tools.KZ_filter(self.X0.T, mwsize, kzord, method=method, causal=causal).T
                Y1 = Tools.KZ_filter(self.Y0.T, mwsize, kzord, method=method, causal=causal).T
            else:
                X1, Y1 = self.X0, self.Y0

            dX = np.zeros_like(self.X0) * np.nan; dX[:,self.dord:] = np.diff(X1, self.dord, axis=-1)
            dY = np.zeros_like(self.Y0) * np.nan; dY[:,self.dord:] = np.diff(Y1, self.dord, axis=-1)
            # or:
            # dX = Tools.sdiff(X0, dsp, axis=-1)
            # dY = Tools.sdiff(Y0, dsp, axis=-1)
        else:
            dX, dY = self.X0, self.Y0

        Xvar0 = Tools.mts_cumview(dX, self.lag)  # cumulative view for convolution
        # polynominal trend
        Xvar1 = Tools.dpvander(np.arange(self.Nt)/self.Nt, self.pord, self.dord)  # division by Nt: normalization for numerical stability
        Xvar = np.vstack([Xvar0, Xvar1[:-1,:]])  #[:-1,:] removes the constant trend which may cause non-invertible covariance matrix. If the constant trend is kept here, Yprd at the end of this function should be modified accordingly like this:
        # Amat0 = Amat[:, :-(pord-dord+1)] ...
        Yvar = dY

        return Xvar, Yvar

    def fit(self):
        pass

    def predict(self):
        pass


class MxDeconv_LS(MxDeconv):
    def __init__(self, Y0, X0, lag, pord=1, dord=0, smth=False, snr2=None, clen2=None, dspl=1):
        super().__init__(Y0, X0, lag, pord, dord, smth)
        self.snr2 = snr2
        self.clen2 = clen2
        self.dspl = dspl

        # construct the covariance matrix of the Gaussian process
        if self.clen2 is not None and self.clen2 > 0 and self.snr2 is not None and self.snr2 >= 0:
            if self.dord > 0:
                self.W0 = _dgp_cov_matrix(self.Nt, self.snr2, self.clen2)
                if self.dord > 1:
                    warnings.warn("The current implementation of the GP covariance matrix is not exact for dord>1.")
            else:
                self.W0 = _gp_cov_matrix(self.Nt, self.snr2, self.clen2)
        else:
            self.W0 = None # invalid parameters, equivalent to W0=np.eye(Nt)

    def fit(self, sidx=0, Ntrn=None, vthresh=0., cdim=None, Nexp=0, vreg=1e-8):

        Xvar, Yvar = self._prepare_data(mwsize=24, kzord=1)

        # training data
        (tidx0, tidx1), _ = Stat.training_period(self.Nt, tidx0=sidx, Ntrn=Ntrn)  # valid training period
        Xtrn, Ytrn = Xvar[:,tidx0:tidx1:self.dspl], Yvar[:,tidx0:tidx1:self.dspl]  # down-sampling of training data
        # GLS matrix
        if self.W0 is not None:
            Winv = la.inv(self.W0[tidx0:tidx1:self.dspl,:][:,tidx0:tidx1:self.dspl])
        else:
            Winv = None # equivalent to np.eye(Xtrn.shape[1])

        # prepare regressor
        regressor = Stat.dim_reduction_pca(Stat.random_subset(Stat.multi_linear_regression))
        # regressor = dim_reduction_cca(random_subset(multi_linear_regression))  # not recommended
        # regressor = random_subset(dim_reduction_pca(multi_linear_regression))
        # regressor = dim_reduction_pca(random_subset(percentile_subset(multi_linear_regression)))

        # regresion
        # method ("mean" or "median") used in random_subset is active only when Nexp>0
        # corrflag=False
        # corrflag (bool): if True use the correlation matrix for dimension reduction
        # ((Amat,Amatc), Cvec, _, _), toto = regressor(Ytrn, Xtrn, Winv, vthresh=vthresh, corrflag=corrflag, Nexp=Nexp, method="mean")
        (Amat, Cvec, *_), (Amatc, *_) = regressor(Ytrn, Xtrn, Winv, vthresh=vthresh, cdim=cdim, Nexp=Nexp, method="mean", vreg=vreg)
        Err = Yvar - (Amat @ Xvar + Cvec)  # differential residual
        Sig = Stat.cov(Err, Err)  # covariance matrix
        # if kthresh>0:
        #     Amat[np.abs(Amat)<kthresh] = 0

        # kernel matrix corresponding to the external input, without the polynomial trend
        Amat0 = Amat[:, :Amat.shape[-1]-(self.pord-self.dord)]

        self._fit_results = {'tidx0':tidx0, 'tidx1':tidx1, 'vthresh':vthresh, 'cdim':cdim, 'Nexp':Nexp, 'vreg':vreg, 'Amat':Amat, 'Cvec':Cvec, 'Err':Err, 'Sig':Sig, 'Amat0':Amat0}
        # self._fit_results = (Amat, Cvec, Amat0, Err, Sig)

        return self._fit_results

    def predict(self, Yvar=None, Xvar=None, polytrend=False):
        if Yvar is None:
            Yvar = self.Y0
        if Xvar is None:
            Xvar = self.X0

        assert Xvar.shape[1] == Yvar.shape[1]
        assert Xvar.shape[0] == self.X0.shape[0] and Yvar.shape[0] == self.Y0.shape[0]

        # prediction
        Xcmv0 = Tools.mts_cumview(Xvar, self.lag)
        if polytrend: # with the polynomial trend, ie: return A*X(t) + P(t)
            Amat, Cvec = self._fit_results['Amat'], self._fit_results['Cvec']
            Xcmv1 = Tools.dpvander(np.arange(self.Nt)/self.Nt, self.pord, 0)  # polynominal trend
            Xcmv = np.vstack([Xcmv0, Xcmv1[:(self.pord-self.dord+1),:]])  # input with polynomial trend
            # Xcmv[np.isnan(Xcmv)] = 0  # Remove nans will introduce large values around discontinuties
            Yflt = np.hstack([Amat, Cvec]) @ Xcmv
        else: # without the polynomial trend, ie, return A*X(t)
            Amat0 = self._fit_results['Amat0']
            Yflt = Amat0 @ Xcmv0

        Yprd = Yflt
        # if self.dord > 0:
        #     Yprd = Yflt - Tools.polyprojection(Yflt, deg=self.dord-1, axis=-1)  # projection \Psi^\dagger \Psi
        # else:
        #     Yprd = Yflt

        Err = Yvar - Yprd
        Sig = Stat.cov(Err, Err)  # covariance matrix
        self._predict_results = {'Yprd':Yprd, 'Err':Err, 'Sig':Sig}

        return self._predict_results


##########  functional implementation #########
def deconv(Y0, X0, lag, pord=1, dord=0, snr2=None, clen2=None, dspl=1, sidx=0, Ntrn=None, vthresh=0., cdim=None, Nexp=0, vreg=1e-8, polytrend=False, smth=True):
    """Deconvolution of multivariate time series using a vectorial FIR filter by GLS.

    We look for the kernel convolution matrices A of the model

        Y_t = A (*) X_t + P_t

    where A=[A_0... A_{p-1}] are matrices, (*) denotes the convolution, P_t is a polynomial Gaussian process of order `pord`. We estimate A by Generalized Least-Square (GLS) by differentiating the data dord times.

    Args:
        Y0 (2d array): observations, each row is a variable and each column is an observation
        X0 (2d array): inputs, each row is a variable and each column is an observation
        lag (int): length of the convolution matrices A
        dord (int): order of derivative
        pord (int): order of polynomial trend
        sidx (int): starting index of the training period
        Ntrn (int): length of the training period
        dspl (int): down-sampling rate
        snr2 (float): signal to noise ratio of the polynomial Gaussian process
        clen2 (float): correlation length of the polynomial Gaussian process
        vthresh (float): relative threshold in dimension reduction, between 0 and 1. The dimension corresponding to the percentage of (1-vthresh) is kept, i.e. 10 percent of information is dropped if vthresh=0.1. No reduction if set to 0.
        cdim (int): desired dimension, same effect as vthresh, no reduction if set to None.
        Nexp (int): number of experiments in the RANSAC algorithm, no RANSAC if Nexp==0
        vreg (floag):
        polytrend (bool):
        smth (bool):
    Returns:
        Yprd, Amat, Amatc: prediction, estimation of the matrix and estimation of the dimension-reduced matrix
    """
    assert X0.ndim == Y0.ndim == 2
    assert X0.shape[1] == Y0.shape[1]
    assert pord >= dord
    # if pord>1 or dord>1:
    #     raise ValueError('pord>1 or dord>1 not supported!')

    Nt = X0.shape[1]  # length of observations

    # external input
    if dord>0:
        if smth:
            X1 = Tools.KZ_filter(X0.T, 24, 1, method="mean", causal=False).T
            Y1 = Tools.KZ_filter(Y0.T, 24, 1, method="mean", causal=False).T
        else:
            X1, Y1 = X0, Y0

        dX = np.zeros_like(X0) * np.nan; dX[:,dord:] = np.diff(X1, dord, axis=-1)
        dY = np.zeros_like(Y0) * np.nan; dY[:,dord:] = np.diff(Y1, dord, axis=-1)
        # or:
        # dX = Tools.sdiff(X0, dsp, axis=-1)
        # dY = Tools.sdiff(Y0, dsp, axis=-1)
    else:
        dX, dY = X0, Y0

    Xvar0 = Tools.mts_cumview(dX, lag)  # cumulative view for convolution
    # polynominal trend
    # division by Nt and multiplication by 10: normalization for numerical stability
    # *100 or *1 numerically works worse
    Xvar1 = Tools.dpvander(np.arange(Nt)/Nt*10, pord, dord)
    Xvar = np.vstack([Xvar0, Xvar1[:-1,:]])  #[:-1,:] removes the constant trend which may cause non-invertible covariance matrix. If the constant trend is kept here, Yprd at the end of this function should be modified accordingly like this:
    # Amat0 = Amat[:, :-(pord-dord+1)] ...
    Yvar = dY

    # construct the covariance matrix of the Gaussian process
    if clen2 is not None and clen2 > 0 and snr2 is not None and snr2 >= 0:
        if dord > 0:
            W0 = _dgp_cov_matrix(Nt, snr2, clen2)
            if dord > 1:
                warnings.warn("The current implementation of the GP covariance matrix is not exact for dord>1.")
        else:
            W0 = _gp_cov_matrix(Nt, snr2, clen2)
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
    (Amat, Cvec, *_), (Amatc, *_) = regressor(Ytrn, Xtrn, Winv, vthresh=vthresh, cdim=cdim, Nexp=Nexp, method="mean", vreg=vreg)
    Err = Yvar - (Amat @ Xvar + Cvec)  # differential residual
    Sig = Stat.cov(Err, Err)  # covariance matrix
    Amat0 = Amat[:, :Amat.shape[-1]-(pord-dord)]  # kernel matrix corresponding to the external input
    # Amat0 = Amat[:, :-(pord-dord)] if pord-dord > 0 else Amat
    # if kthresh>0:
    #     Amat[np.abs(Amat)<kthresh] = 0

    # prediction
    Xcmv0 = Tools.mts_cumview(X0, lag)
    if polytrend: # with the polynomial trend, ie: return A*X(t) + P(t)
        # polynominal trend
        Xcmv1 = Tools.dpvander(np.arange(Nt)/Nt, pord, 0)
        Xcmv = np.vstack([Xcmv0, Xcmv1[:(pord-dord+1),:]])
        # Xcmv[np.isnan(Xcmv)] = 0  # Remove nans will introduce large values around discontinuties
        Yflt = np.hstack([Amat, Cvec]) @ Xcmv
    else: # without the polynomial trend, ie: return A*X(t)
        Yflt = Amat0 @ Xcmv0

    Yprd = Yflt
    # if dord > 0:
    #     Yprd = Yflt - Tools.polyprojection(Yflt, deg=dord-1, axis=-1)  # projection \Psi^\dagger \Psi
    # else:
    #     Yprd = Yflt

    return Yprd, Amat, Amatc

# deconv = func_mparms_mean(_deconv)


def deconv_bm(Y0, X0, lag, pord=1, dord=0, sigmaq2=10**-6, sigmar2=10**-3, x0=0., p0=1., smooth=False, sidx=0, Ntrn=None, vthresh=0., cdim=None, polytrend=False, rescale=True, smth=True):
    """Deconvolution of multivariate time series using a vectorial FIR filter by Kalman filter.

    Args:
        Y0 (2d array): observations, each row is a variable and each column is an observation
        X0 (2d array): inputs, each row is a variable and each column is an observation
        lag (int): length of the convolution matrices A
        dord (int): order of derivative
        pord (int): order of polynomial trend
        sidx (int): starting index of the training period
        Ntrn (int): length of the training period
        vthresh (float): see deconv()
        cdim (int): desired dimension, same effect as vthresh, no reduction if set to None.
        sigmaq2 (float): variance of innovation noise
        sigmar2 (float): variance of observation noise
        x0 (float): initial state (a constant vector)
        p0 (float): variance of the initial state
        smooth (bool): if True apply Kalman smoother
    Returns:
        Yprd, Amat, Amatc: prediction, estimation of the matrix and estimation of the dimension-reduced matrix
    """
    assert X0.ndim == Y0.ndim == 2
    assert X0.shape[1] == Y0.shape[1]

    assert pord >= dord

    Nt = X0.shape[1]  # length of observations

    # external input
    if dord>0:
        if smth:
            X1 = Tools.KZ_filter(X0.T, 24, 1, method="mean", causal=False).T
            Y1 = Tools.KZ_filter(Y0.T, 24, 1, method="mean", causal=False).T
        else:
            X1, Y1 = X0, Y0

        dX = np.zeros_like(X0) * np.nan; dX[:,dord:] = np.diff(X0, dord, axis=-1)
        dY = np.zeros_like(Y0) * np.nan; dY[:,dord:] = np.diff(Y0, dord, axis=-1)
        # or:
        # dX = Tools.sdiff(X0, dsp, axis=-1)
        # dY = Tools.sdiff(Y0, dsp, axis=-1)
    else:
        dX, dY = X0, Y0

    Xvar0 = Tools.mts_cumview(dX, lag)  # cumulative view for convolution
    # the part of polynominal trend
    Xvar1 = Tools.dpvander(np.arange(Nt)/Nt*10, pord, dord)  # division by Nt: normalization for numerical stability
    # Xvar and Yvar are the variables passed to the Kalman filter
    Xvar = np.vstack([Xvar0, Xvar1[:-1,:]])  #[:-1,:] removes the constant trend which may cause non-invertible covariance matrix. If the constant trend is kept here, Yprd at the end of this function should be modified accordingly like this:
    # Amat0 = Amat[:, :-(pord-dord+1)] ...
    Yvar = dY

    # prepare regressor
    regressor = Stat.dim_reduction_bm(Stat.multi_linear_regression_bm)

    # regression
    ((Amat, Acov), (Cvec, Ccov), Err, Sig), ((Amatc, Acovc), *_) = regressor(Yvar, Xvar, sigmaq2, sigmar2, x0, p0, smooth=smooth, sidx=sidx, Ntrn=Ntrn, vthresh=vthresh, cdim=cdim, rescale=rescale)
    Amat0 = Amat[:, :, :Amat.shape[-1]-(pord-dord)]  # kernel matrices without polynomial trend

    # prediction
    # # method 1: apply kernel matrices directly on raw inputs
    # # this method is theoretically non exact but numerically stable
    # Xcmv = Tools.mts_cumview(X0, lag) # DO NOT remove nans: Xcmv[np.isnan(Xcmv)] = 0, see comments in deconv()
    # Yflt = np.zeros_like(Y0)
    # for t in range(Nt):
    #     Yflt[:,t] = Amat0[t,] @ Xcmv[:,t]
    #
    # # method 2: apply kernel matrices on differentiated inputs
    # then re-integrate. This method is theoretically exact but numerically unstable when dord>=2
    if polytrend:
        Xcmv = Xvar
        Yflt = np.zeros_like(Y0)
        for t in range(Nt):
            Yflt[:,t] = Amat[t,] @ Xcmv[:,t] + Cvec[t]
        # # integration to obtain the final result
        # if dord > 0:
        #     Yflt[np.isnan(Yflt)] = 0
        #     for n in range(dord):
        #         Yflt = np.cumsum(Yflt,axis=-1)
        #     # Remark: Yprd has shape Y0.shape[1]*Nt
        #     Yprd = Yflt - Tools.polyprojection(Yflt, deg=dord-1, axis=-1)  # prediction: projection \Psi^\dagger \Psi
        # else:
        #     Yprd = Yflt
    else:
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
    # Yprd = Yflt - Tools.polyprojection(Yflt, deg=dord-1, axis=-1)  # prediction: projection \Psi^\dagger \Psi
    Yprd = Yflt

    # # covariance matrix: abandonned
    # Ycov = np.zeros((Nt,Y0.shape[0],Y0.shape[0]))
    # for t in range(Nt):
    #     M = np.kron(np.eye(Y0.shape[0]), Xcmv[:,t])
    #     Ycov[t,:,:] = M @ Acov[t,] @ M.T

    # return Yprd, ((Amat, Acov), (Cvec, Ccov), Err, Sig), (Amatc, Acovc)
    return Yprd, (Amat, Acov), (Amatc, Acovc)


def ssproj(X0, cdim=1, vthresh=None, corrflag=False, sidx=0, Ntrn=None, dflag=False):
    """Projection of a multivariate time series onto a subspace.

    Args:
        X0 (2d array): input
        cdim (int): dimension of the subspace, if cdim==0 return zero
        vthresh (float): see deconv(), if cdim is set vthresh will be ignored.
        corrflag (bool): if True use correlation matrix for PCA
        dflag (bool): if True take the derivative of the input
        sidx (int): starting index of the training period
        Ntrn (int): length of the training period
    Returns:
        Xprj: projection
        U,S: PCA basis and singular values
        cdim: true dimension of the subspace
    """
    assert not ((cdim is None) and (vthresh is None))
    assert sidx >= 0

    if Ntrn is None:
        tidx0, tidx1 = sidx, None
    else:
        tidx0, tidx1 = sidx, sidx+Ntrn
    # print(tidx0, tidx1)
    if dflag:
        # take derivative to transform to a stationary time series
        X1 = np.diff(X0[:,tidx0:tidx1], axis=-1)
    else:
        # another option is to centralize
        X1 = Stat.centralize(X0)

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
        assert 0. <= vthresh <=1.
        # two possible strategies:
        # 1. by relative value of sv
        # toto = S/S[0]
        # cdim = np.sum(toto >= vthresh)
        # 2. by cumulation of sv
        toto = np.cumsum(S) / np.sum(S)
        cdim = np.sum(toto <= 1-vthresh)
    # else:  # if cdim is given, vthresh has no effect
    cdim = min(max(1, cdim), len(S))

    # projection
    if cdim > 0:
        Xprj = U[:,:cdim] @ U[:,:cdim].T @ Stat.centralize(X0)
        # Xprj = U[:,:cdim] @ U[:,:cdim].T @ X0

        # # or by integration, not working well in practice
        # dXprj = U[:,:cdim] @ U[:,:cdim].T @ np.diff(X0, axis=-1)
        # dXprj[np.isnan(dXprj)] = 0
        # Xprj = np.zeros_like(X0)
        # Xprj[:,1:] = np.cumsum(dXprj, axis=-1)
    else:
        Xprj = np.zeros_like(X0)
    return Xprj, (U,S), cdim


# def mutdecorr(Y0, lag, vthresh=1e-3): #, sidx=0, Ntrn=None):
#     """Dimension-wise mutual decorrelation."""
#     Yprd = []
#     for n in range(Y0.shape[0]):
#         locs = list(range(Y0.shape[0]))  # list of row index
#         locs.pop(n)  # exclude the current row
#         # Regression of the current row by the other rows
#         toto, *_ = deconv(Y0[[n],:], Y0[locs,:], lag, dord=1, pord=1, clen2=None, dspl=1, vthresh=vthresh)
#         Yprd.append(toto[0])  # toto is 1-by-? 2d array
#     return np.asarray(Yprd)

