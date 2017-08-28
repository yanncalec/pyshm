"""Collection of models for SHM analysis.
"""

import numpy as np
import numpy.linalg as la
from numpy import ma
import scipy.linalg
import scipy.signal
from functools import wraps
import numbers
import warnings
import pandas as pd
import pykalman

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
        self.linear_trend = np.arange(self.Nt)/self.Nt*10

    def _prepare_data(self, mwsize=24, kzord=1, method='mean', causal=False):
        """Preparation of data.
        """
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
        Xvar1 = Tools.dpvander(self.linear_trend, self.pord, self.dord)  # division by Nt: normalization for numerical stability
        Xvar = np.vstack([Xvar0, Xvar1[:-1,:]])  #[:-1,:] removes the constant trend which may cause non-invertible covariance matrix. If the constant trend is kept here, Yprd at the end of this function should be modified accordingly like this:
        # Amat0 = Amat[:, :-(pord-dord+1)] ...
        Yvar = dY

        return Xvar0, Xvar, Yvar

    def fit(self):
        pass

    def predict(self):
        pass


class MxDeconv_LS(MxDeconv):
    def __init__(self, Y0, X0, lag, pord=1, dord=1, smth=False, snr2=None, clen2=None, dspl=1):
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

        _, Xvar, Yvar = self._prepare_data()

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

        # regresion
        (Amat, Cvec, *_), (Amatc, *_) = regressor(Ytrn, Xtrn, Winv, vthresh=vthresh, cdim=cdim, Nexp=Nexp, method="mean", vreg=vreg)
        Err = Yvar - (Amat @ Xvar + Cvec)  # differential residual
        Sig = Stat.cov(Err, Err)  # covariance matrix

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
            Xcmv1 = Tools.dpvander(self.linear_trend, self.pord, 0)  # polynominal trend
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


class MxDeconv_BM(MxDeconv):
    def __init__(self, Y0, X0, lag, pord=1, dord=1):
        super().__init__(Y0, X0, lag, pord, dord, smth=False)

    def fit(self, sidx=0, Ntrn=None, vthresh=0., cdim=None, sigmaq2=None, sigmar2=None, x0=None, p0=None):
        # self.sigmaq2 = sigmaq2
        # self.sigmar2 = sigmar2

        Xvar0, Xvar, Yvar = self._prepare_data()

        # make a copy of data
        self._Xvar = Xvar.copy()
        self._Xvar0 = Xvar0.copy()
        self._Yvar = Yvar.copy()

        # training data
        (tidx0, tidx1), _ = Stat.training_period(self.Nt, tidx0=sidx, Ntrn=Ntrn)  # valid training period
        Xtrn, Ytrn = Xvar[:,tidx0:tidx1], Yvar[:,tidx0:tidx1]

        # dimension reduction: either by vthresh or by cdim
        self.dim_reduction = (vthresh > 0) or (cdim is not None)
        if self.dim_reduction:
            _, U, S = Stat.pca(Xtrn, nc=None, corrflag=False)  # pca coefficients
            if cdim is None:  # cdim not given, use vthresh to compute cdim
                assert 0. <= vthresh <=1.
                toto = np.cumsum(S) / np.sum(S)
                cdim = np.sum(toto <= 1-vthresh)
            cdim = max(1, cdim) # force the minimal dimension

            Xcof = (U.T @ Xvar)[:cdim,:]  # coefficients of Xvar
            self._U = U  # save the basis matrix
            self._cdim = cdim  # save the reduced dimension
        else:
            Xcof = Xvar
            self._U = None # np.eye(Xvar.shape[0])
            self._cdim = None

        dimobs = Yvar.shape[0]  # dimension of the observation vector and duration
        dimsys = Xcof.shape[0] * dimobs  # dimension of the system vector

        # construct the transition matrix: time-independent
        A = np.eye(dimsys)  # the transition matrix: time-independent
        # construct the observation matrices: time-dependent
        B = np.zeros((self.Nt, dimobs, dimsys))
        for t in range(self.Nt):
            toto = Xcof[:,t].copy()
            toto[np.isnan(toto)] = 0
            B[t,] = np.kron(np.eye(dimobs), toto)

        em_vars=[]
        # Covariance matrix of the innovation noise, time-independent
        if isinstance(sigmaq2, numbers.Number) and sigmaq2>0:
            Q = np.eye(dimsys) * sigmaq2
        else:
            Q = None
            em_vars.append('transition_covariance')

        # Covariance matrix of the observation noise
        if isinstance(sigmar2, numbers.Number) and sigmar2>0:
            R = np.eye(dimobs) * sigmar2
        else:
            R = None
            em_vars.append('observation_covariance')

        if x0 is None:
            em_vars.append('initial_state_mean')
            mu0 = None
        elif isinstance(x0, numbers.Number):
            mu0 = np.ones(dimsys) * x0
        elif isinstance(x0, np.array):
            assert(len(x0)==dimsys)
            mu0 = x0
        else:
            raise TypeError('x0 must be a number or a vector or None')

        if p0 is None:
            em_vars.append('initial_state_covariance')
            P0 = None
        elif isinstance(p0, numbers.Number):
            P0 = np.eye(dimsys) * p0
        elif isinstance(p0, np.array):
            assert(p0.shape[0]==p0.shape[1]==dimsys)
            # assert Tools.ispositivedefinite(P0)
            P0 = p0
        else:
            raise TypeError('P0 must be a number or a symmetric matrix or None')

        KF = pykalman.KalmanFilter(transition_matrices=A,
                                   observation_matrices=B,
                                   transition_covariance=Q, observation_covariance=R,
                                   # transition_offsets=None, observation_offsets=None,
                                   initial_state_mean=mu0, initial_state_covariance=P0,
                                   # random_state=None,
                                   em_vars=em_vars)

        # parameter tuning by EM algorithm
        if len(em_vars)>0:
            # print('Runing EM...')
            # print(Ymas.shape)
            # print(KF.n_dim_obs, KF.n_dim_state)

            # mask invalid values, taking transpose due to the convention of pykalman
            # see: https://pykalman.github.io/#pykalman.KalmanFilter.em
            Ymas = ma.masked_invalid(Ytrn.T)
            KF = KF.em(Ymas)
            # KF.transition_covariance /= 2  # not helpful

        self._fit_results = {'KF': KF, 'tidx0':tidx0, 'tidx1':tidx1, 'vthresh':vthresh, 'cdim':cdim}
        return self._fit_results


    def predict(self, kftype='smoother', polytrend=False):

        KF = self._fit_results['KF']
        # dimsys = KF.n_dim_state  # dimension of the observation vector
        dimobs = KF.n_dim_obs  # dimension of the observation vector
        Ymas = ma.masked_invalid(self._Yvar.T)

        if kftype.upper()=='SMOOTHER':
            Xflt, Pflt = KF.smooth(Ymas)
        else:
            Xflt, Pflt = KF.filter(Ymas)
        # print(Xflt.shape, Pflt.shape)
        Amatc0 = []
        for t in range(self.Nt):
            Amatc0.append(np.reshape(Xflt[t,], (dimobs,-1)))

        Amatc = np.asarray(Amatc0)  # Amatc has shape Nt * dimobs * (dimsys-dimobs)
        # Cvec = np.asarray(Cvec0)
        Acovc = Pflt
        # Acovc = Pflt[:,:-dimobs,:-dimobs]
        # Ccov = Pflt[:,-dimobs:,-dimobs:]

        if self.dim_reduction:
            # Recover the kernel matrices in the full shape
            W = self._U[:, :self._cdim]  # analysis matrix of subspace basis
            Wop = Tools.matprod_op_right(W.T, Amatc.shape[1])
            Amat = np.asarray([Amatc[t,] @ W.T for t in range(self.Nt)])
            Acov = np.asarray([Wop @ Acovc[t,] @ Wop.T for t in range(self.Nt)])
        else:
            Amat, Acov = Amatc, Acovc

        # kernel matrix corresponding to the external input X(t)
        Amat0 = Amat[:, :, :Amat.shape[-1]-(self.pord-self.dord)]

        # Prediction
        Yflt = np.zeros_like(self.Y0)
        if polytrend:
            Xcmv = self._Xvar
            for t in range(self.Nt):
                Yflt[:,t] = Amat[t,] @ Xcmv[:,t]
        else:
            Xcmv = self._Xvar0
            for t in range(self.Nt):
                # Yflt[:,t] = Amat0[t,] @ Xcmv[:,t] + Cvec[t]
                Yflt[:,t] = Amat0[t,] @ Xcmv[:,t]
        # integration to obtain the final result
        if self.dord > 0:
            Yflt[np.isnan(Yflt)] = 0
            for n in range(self.dord):
                Yflt = np.cumsum(Yflt,axis=-1)

        # # an inexact method
        # Xcmv = Tools.mts_cumview(self.X0, self.lag)  # cumulative view for convolution
        # # Xcmv = self.X0
        # for t in range(self.Nt):
        #     Yflt[:,t] = Amat0[t,] @ Xcmv[:,t]

        Yprd = Yflt

        Err = self.Y0 - Yprd
        Sig = Stat.cov(Err, Err)  # covariance matrix
        self._predict_results = {'Amat':Amat, 'Acov':Acov, 'Amatc':Amatc, 'Acovc':Acovc,'Amat0':Amat0, 'Yprd':Yprd, 'Err':Err, 'Sig':Sig}

        return self._predict_results


##########  functional implementation #########
def deconv(Y0, X0, lag, pord=1, dord=1, snr2=None, clen2=None, dspl=1, sidx=0, Ntrn=None, vthresh=0., cdim=None, Nexp=0, vreg=1e-8, polytrend=False, smth=False):
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
    Amat0 = Amat[:, :Amat.shape[-1]-(pord-dord)]  # kernel matrix corresponding to the external input X(t) only, without polynomial trend
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

    # Yprd = Yflt
    if dord > 0:
        Yprd = Yflt - Tools.polyprojection(Yflt, deg=dord-1, axis=-1)  # projection \Psi^\dagger \Psi
    else:
        Yprd = Yflt

    return Yprd, Amat, Amatc

# deconv = func_mparms_mean(_deconv)


def deconv_bm(Y0, X0, lag, pord=1, dord=0, sigmaq2=10**-6, sigmar2=10**-3, x0=0., p0=1., kftype='smoother', sidx=0, Ntrn=None, vthresh=0., cdim=None, polytrend=False, rescale=True, smth=True):
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

        dX = np.zeros_like(X0) * np.nan; dX[:,dord:] = np.diff(X1, dord, axis=-1)
        dY = np.zeros_like(Y0) * np.nan; dY[:,dord:] = np.diff(Y1, dord, axis=-1)
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
    ((Amat, Acov), (Cvec, Ccov), Err, Sig), ((Amatc, Acovc), *_) = regressor(Yvar, Xvar, sigmaq2, sigmar2, x0, p0, kftype=kftype, sidx=sidx, Ntrn=Ntrn, vthresh=vthresh, cdim=cdim, rescale=rescale)
    Amat0 = Amat[:, :, :Amat.shape[-1]-(pord-dord)]  # kernel matrix corresponding to the external input X(t) only, without polynomial trend

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


class MRA_DCT(object):
    """Multi-resolution analysis DCT.
    """
    def __init__(self, M, p=2):
        """
        Args:
            M (int): number of frequencies
            p (int): scaling factor, p>1 and p must divide M

        Remarks:
            M is also the length of the convolution kernels. M=2 gives a Haar-like tight frame.
            Increase M to have more frequencies. The Gibbs effect is proportional to M and will
            appear at the head and the tail of the signal of synthesis.
        """
        assert isinstance(p, int) and p > 1 and (M%p == 0)

        self.nfreq = M
        self.scaling = p  # numerically the scaling factor has to be 2^n to have a tight-frame, error in the original paper?
        self.kernels = MRA_DCT.dct_kernels(M)  # kernel matrix, each row corresponds to a kernel, from low to high frequency

    def analysis(self, X0, lvl=None):
        C0, _ = MRA_DCT._analysis(X0, self.scaling, self.kernels, lvl=lvl)
        Cv, cdims = MRA_DCT.coeff_list2vec(C0)
        return C0, (Cv, cdims)

    def synthesis(self, C0, lvl=None, ns=None):
        if isinstance(C0, tuple):  # coefficients are in form of vector
            Cv, cdims = C0   # dimension information must be provided
            Cl = MRA_DCT.coeff_vec2list(Cv, cdims)
        else:  # coefficients are in form of list
            Cl = C0

        X0, Xlist = MRA_DCT._synthesis(Cl, self.scaling, self.kernels, lvl=lvl, ns=ns)
        return X0, Xlist

    def analysis_tat(self, X0, lvl=None):
        C0, _, self._kl_tat = MRA_DCT._analysis_tat(X0, self.scaling, self.kernels, lvl=lvl)
        Cv, cdims = MRA_DCT.coeff_list2vec(C0)
        self._kn_tat = np.cumsum([l if n==0 else l-1 for n, l in enumerate(self._kl_tat)][::-1])[::-1]
        return C0, (Cv, cdims)

    def synthesis_tat(self, C0, lvl=None, ns=None):
        if isinstance(C0, tuple):  # coefficients are in form of vector
            Cv, cdims = C0   # dimension information must be provided
            Cl = MRA_DCT.coeff_vec2list(Cv, cdims)
        else:  # coefficients are in form of list
            Cl = C0

        X0, Xlist = MRA_DCT._synthesis_tat(Cl, self.scaling, self.kernels, lvl=lvl, ns=ns)
        return X0, Xlist

    def full2valid_tat(self, C0):
        return [M[:, l:-l] for M,l in zip(C0, self._kn_tat)]

    def full2validr_tat(self, C0):
        return [M[:, :-l] for M,l in zip(C0, self._kn_tat)]

    def full2validl_tat(self, C0):
        return [M[:, l:] for M,l in zip(C0, self._kn_tat)]

    @staticmethod
    def coeff_list2vec(Cl):
        Cv = np.concatenate([c.flatten() for c in Cl])
        cdims = [c.shape for c in Cl]  # dimension information of coefficients of all scales
        return Cv, cdims

    @staticmethod
    def coeff_vec2list(Cv, cdims):
        cidx = np.cumsum([np.prod(d) for d in cdims])
        Cl = [A.reshape(d) for A, d in zip(np.split(Cv, cidx[:-1]), cdims)]
        return Cl

    @staticmethod
    def shrinkage(C0, percentage, softmode=False):
        assert 0 < percentage <= 1

        if isinstance(C0, tuple): # coefficients are in form of vector
            Cv, cdims = C0   # dimension information must be provided
            Cl = MRA_DCT.coeff_vec2list(*C0)
        else: # coefficients are in form of list
            Cv, cdims = MRA_DCT.coeff_list2vec(C0)
            Cl = C0

        idx = max(0, int(len(Cv)*percentage)-1)
        vth = np.sort(np.abs(Cv))[::-1][idx]

        Ct = []
        for c in Cl:
            s, _ = Tools.shrinkage(c, vth, soft=softmode)
            Ct.append(s)

        return Ct

    @staticmethod
    def dct_kernels(M):
        toto = []
        for l in range(M):
            cst = M if l==0 else M/np.sqrt(2)  # <- or /np.sqrt(scaling) ?
            toto.append(np.cos(np.pi*(np.arange(M)+0.5)*l/M)/cst)
        return np.asarray(toto)

    @staticmethod
    def _analysis(X0, scl, gs, lvl=None):
        """
        Args:
            X0 (1d array): input signal
            scl (int): scaling factor, >=2
            gs (2d array): square matrix of kernels, each row corresponds to a kernel
            lvl (int): level of decomposition, if not given the maximal level is computed from the length of X0
            mode (str): mode of convolution, {'full', 'same', 'valid'}. Only 'full' gives a tight frame
        """
        # assert lvl>0
        # assert gs.shape[0] == gs.shape[1]

        # zero-padding
        #     x0 = np.zeros(2**int(np.ceil(np.log2(len(X0)))))
        #     x0[:len(X0)] = X0
        #         if len(X0)%2==1:
        #             x0 = np.concatenate([X0,[0]])
        #         else:
        #             x0 = X0

        if lvl is None:
            lvl = int(np.floor(np.log(len(X0))/np.log(scl)))

        res = [[np.asarray(X0)]]
        for n in range(lvl):
            c0 = res[-1][0]  # approximation coeffs of the last level
            toto = []

            for knl in gs:
                # for online application, use convolve instead of fftconvolve
                cx = np.sqrt(scl) * scipy.signal.fftconvolve(c0, knl[::-1], mode='full')
                toto.append(cx)
            c1 = np.asarray(toto)[:, ::scl]  # down-sampling
            res.append(c1)

        # full coefficients, res[1:] to drop the original signal, and
        # [::-1] to arange the scale from coarse to to fine
        coeff_full = res[1:][::-1]
        coeff = [coeff_full[0]]  # no approximation coefficients except for the coarsest level
        coeff += [c[1:] for c in coeff_full[1:]]

        return coeff, coeff_full

    @staticmethod
    def _synthesis(C0, scl, gs, lvl=None, ns=None):
        """
        Args:
            C0 (list of 2d arrays): coefficients of analysis from coarse to fine, must have the same dimension as the first value returned by _analysis()
            scl (int): scaling factor, >=2
            gs (2d array): square matrix of kernels, each row corresponds to a kernel
            lvl (int): level of synthesis, if not given all levels in C0 will be used
            ns (int): truncation length of the final output
        """
        if lvl is None:
            lvl = len(C0)
    #     assert 0 < order <= len(C0)
    #     assert gs.shape[0] == gs.shape[1]

        M = gs.shape[1]
        Xlist = []  # approximation coefficients

        for n in range(lvl):
            # get the full coefficient matrix of the current level
            if n==0:  # the coarsest level
                cmat = C0[0]
            else:  # for other levels
                L = C0[n].shape[1]
                cmat = np.vstack([Xlist[-1][:L], C0[n]])

            toto = []
            cmau = np.zeros((cmat.shape[0], scl*cmat.shape[1]))
            cmau[:,::scl] = cmat
            for knl, c1 in zip(gs, cmau):
                # c1 = np.zeros(scl*len(c0))
                # c1[::scl] = c0  # up-sampling
                # for online application, use convolve instead of fftconvolve
                cx = np.sqrt(scl) * scipy.signal.fftconvolve(c1, knl, mode='full')
                toto.append(cx)

            # truncate the first M-1 coefficients which cause border effect
            # The offset M-1 below in necessay to have perfect reconstruction and it is determined by experiments, although theoretical analysis does not seem to need such an offset.
            Xlist.append(np.sum(np.asarray(toto), axis=0)[M-1:])

        # truncate the last M-1 coefficients which cause border effect. It seems that when the length of the original signal is even, the last M coefficients should be truncated to keep the resynthesis the same size as the original. Reason?
        X0 = Xlist[-1][:-(M-1)] if ns is None else Xlist[-1][:ns]

        return X0, Xlist

    @staticmethod
    def _analysis_tat(X0, scl, gs, lvl=None):
        """
        Args:
            X0 (1d array): input signal
            scl (int): scaling factor, >=2
            gs (2d array): square matrix of kernels, each row corresponds to a kernel
            lvl (int): level of decomposition, if not given the maximal level is computed from the length of X0
            mode (str): mode of convolution, {'full', 'same', 'valid'}. Only 'full' gives a tight frame
        """
        # assert lvl>0
        # assert gs.shape[0] == gs.shape[1]

        # zero-padding
        #     x0 = np.zeros(2**int(np.ceil(np.log2(len(X0)))))
        #     x0[:len(X0)] = X0
        #         if len(X0)%2==1:
        #             x0 = np.concatenate([X0,[0]])
        #         else:
        #             x0 = X0

        if lvl is None:
            lvl = int(np.floor(np.log(len(X0))/np.log(scl)))

        res = [[np.asarray(X0)]]
        # rev = [[np.asarray(X0)]]
        ker_len = []

        for n in range(lvl):
            c0 = res[-1][0]  # approximation coeffs of the last level
            # c1 = rev[-1][0]  # approximation coeffs of the last level
            toto = []
            # fofo = []

            fct = scl**n  # up-sampling factor
            for knl0 in gs:
                # up-sampling of the kernels
                knl = np.zeros(fct*(len(knl0)-1)+1)
                # knl = np.zeros(fct*(len(knl0)))
                knl[::fct] = knl0
                # for online application, use convolve instead of fftconvolve
                cx = np.sqrt(scl) * scipy.signal.fftconvolve(c0, knl[::-1], mode='full')
                # cy = np.sqrt(scl) * scipy.signal.fftconvolve(c1, knl[::-1], mode='full')
                toto.append(cx)
                # fofo.append(cy)

            ker_len.append(len(knl))
            res.append(np.asarray(toto))
            # rev.append(np.asarray(fofo)[:, :-(len(knl)-1)])
            # ker_len.append(len(knl))

        # full coefficients, res[1:] to drop the original signal, and
        # [::-1] to arange the scale from coarse to to fine
        coeff_full = res[1:][::-1]
        coeff = [coeff_full[0]]  # no approximation coefficients except for the coarsest level
        coeff += [c[1:] for c in coeff_full[1:]]

        return coeff, coeff_full, ker_len[::-1]

    @staticmethod
    def _synthesis_tat(C0, scl, gs, lvl=None, ns=None):
        """
        Transforme à trous (TAT)
        Args:
            C0 (list of 2d arrays): coefficients of analysis from coarse to fine, must have the same dimension as the first value returned by _analysis()
            scl (int): scaling factor, >=2
            gs (2d array): square matrix of kernels, each row corresponds to a kernel
            lvl (int): level of synthesis, if not given all levels in C0 will be used
            ns (int): truncation length of the final output
        """
        if lvl is None:
            lvl = len(C0)
    #     assert 0 < order <= len(C0)
    #     assert gs.shape[0] == gs.shape[1]

        M = gs.shape[1]
        Xlist = []  # approximation coefficients

        for n in range(lvl):
            # get the full coefficient matrix of the current level
            if n==0:  # the coarsest level
                cmat = C0[0]
            else:  # for other levels
                L = C0[n].shape[1]
                cmat = np.vstack([Xlist[-1][:L], C0[n]])

            fct = scl**(lvl-n)  # up-sampling factor for coefficient vector
            fctk = scl**(lvl-n-1)  # up-sampling factor for kernel vector
            mask = np.zeros(cmat.shape[1])
            mask[::fct] = 1

            toto = []
            for knl0, c0 in zip(gs, cmat):
                # # down-sampling the coefficient vector
                # c1 = c0[::fct]  # c1 is identical to the coefficient vector of wavelet transform
                # # up-sampling the down-sampled vector
                # c2 = np.zeros(scl*len(c1))
                # c2[::scl] = c1

                # mask the coefficient vector
                c1 = mask * c0
                # up-sampling of the kernels
                knl = np.zeros(fctk*(len(knl0)-1)+1)
                # knl = np.zeros(fctk*(len(knl0)))
                knl[::fctk] = knl0
                # for online application, use convolve instead of fftconvolve
                cx = np.sqrt(scl) * scipy.signal.fftconvolve(c1, knl, mode='full')
                toto.append(cx[len(knl)-1:])  # drop head

            Xlist.append(np.sum(np.asarray(toto), axis=0))

        # truncate the last M-1 coefficients which cause border effect. It seems that when the length of the original signal is even, the last M coefficients should be truncated to keep the resynthesis the same size as the original. Reason?
        X0 = Xlist[-1] #[::scl]
        X0 = X0[:-(M-1)] if ns is None else X0[:ns]

        return X0, Xlist

