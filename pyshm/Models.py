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
import pywt

# # import sklearn
# # from sklearn import linear_model, decomposition, pipeline, cross_validation
# # from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
# from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, RANSACRegressor, TheilSenRegressor, Lasso, LassoCV, LassoLars, LassoLarsCV, OrthogonalMatchingPursuit, OrthogonalMatchingPursuitCV
# # from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
# # from sklearn.preprocessing import StandardScaler, RobustScaler
# # from sklearn.gaussian_process import GaussianProcess
# # from sklearn.pipeline import Pipeline

from . import Tools, Stat, Kalman
# from pyshm import Tools, Stat, Kalman


########## Utility functions ##########

def prepare_data(Xobs, Yobs, lag, dord=0, tflag=True):
    """
    """
    assert Xobs.ndim == Yobs.ndim == 2

    Xcmv = Tools.mts_cumview(Xobs, lag)
    Xcmv = np.diff(Xcmv, dord, axis=-1)
    Ycmv = np.diff(Yobs, dord, axis=-1)

    Zoo, _ = Tools.remove_nan_columns(Xcmv, Ycmv)
    if tflag:
        Xvar, Yvar = Zoo[0].T, Zoo[1].T
    else:
        Xvar, Yvar = Zoo[0], Zoo[1]

    return Xvar, Yvar


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
        # or to centralize
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


########## Class Interfaces ##########

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
        self.n_coefs = None  # number of coefficients

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
        self.n_coefs = Amat0.size

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
            # print(Xcmv1)
            # print(Cvec)
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




class MRA_Regression:
    """MRA (multi-resolution analysis) regression.
    """

    def __init__(self, lag, wvlname, maxlvl, mode='acdc', reg_name='lasso', loss=1e-3, n_components=None):
        """
        Args:
            mode: 'full', 'acdc'
            loss (float or int): tolerance of infomation loss in PCR
        """

        self.lag = lag
        self.wvlname = wvlname
        self.maxlvl = maxlvl
        self.mode = mode  # mode of regression
        self._regs = None
        self._reg_name = reg_name  # name of regressor
        self.loss = loss
        self.n_components = n_components

        self._dimx = None  # number of features of the input variable
        self.n_coefs_ = None  # number of coefficients per scale (depending on the mode)
        self.n_coefs = None  # total number of coefficients

        # self._pywt_mode = 'smooth'  # boundary extension modes for pywt
        self._pywt_mode = 'constant'  # boundary extension modes for pywt
        self._dc_score_thresh = 0.7

    def fit(self, X, y, **kwargs):
        """
        Args:
            X (2d array): input variables, n_samples by n_features
            y (1d array): output variable, must have same number of observations as X
        """
        assert X.ndim==2 and y.ndim==1
        assert X.shape[0] == y.shape[-1]

        self._dimx = X.shape[-1]

        # wavelet decomposition
        Xcmv = Tools.mts_cumview(X.T, self.lag).T  # cumulative view
        # raw coefficients
        Xcof0 = pywt.wavedec(Xcmv, self.wvlname, level=self.maxlvl, axis=0, mode=self._pywt_mode)
        ycof0 = pywt.wavedec(y, self.wvlname, level=self.maxlvl, mode=self._pywt_mode)
        # remove nan values
        Xcof = []; ycof = []
        for cX, cy in zip(Xcof0, ycof0):
            cnt = np.sum(np.isnan(cX), axis=-1) + np.isnan(cy)
            Xcof.append(cX[cnt==0,:])
            ycof.append(cy[cnt==0])

        self._regs = []
        self.n_coefs_ = []

        # regression of the dc component
        #
        # # equivalent implementation using sklearn.pipeline:
        # base_estimator = sklearn.linear_model.LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=-1)
        # estimators = [('reduce_dim', sklearn.decomposition.PCA(n_components=loss)), ('reg', base_estimator)]
        # regressor = sklearn.pipeline.Pipeline(estimators)
        #
        # reg = Stat.PCRegression(loss=self.loss, n_components=self.n_components, reg_name=self._reg_name, **kwargs)
        # reg = Stat.PCRegression(loss=1e-4, n_components=None, reg_name='lasso', **kwargs)
        reg = Stat.PCRegression(loss=1e-4, n_components=None, reg_name='ridge', **kwargs)
        # reg = Stat.PCRegression(loss=1e-4, n_components=None, reg_name='ransac', nexp=100, **kwargs)

        # combining data with its derivative doesn't work well
        # Xvar = np.vstack([Xcof[0], np.diff(Xcof[0],axis=0)])
        # yvar = np.hstack([ycof[0], np.diff(ycof[0])])
        # reg.fit(Xvar, yvar)

        score1 = reg.score(Xcof[0], ycof[0])
        score2 = reg.score(np.diff(Xcof[0],axis=0), np.diff(ycof[0]))
        # print(score1, score2)
        self._dc_score = min(score1, score2)
        if self._dc_score > self._dc_score_thresh:
            reg.fit(Xcof[0], ycof[0])
        else:
            reg.fit(np.diff(Xcof[0],axis=0), np.diff(ycof[0]))
            reg.adjust_intercept(Xcof[0], ycof[0])
        self._regs.append(reg)
        self.n_coefs_.append(reg.n_components)

        # regression of the ac components
        if self.mode=='full':
            for n, (cX, cy) in enumerate(zip(Xcof[1:], ycof[1:])):
                reg = Stat.PCRegression(loss=self.loss, n_components=self.n_components, reg_name=self._reg_name, **kwargs)
                reg.fit(cX, cy)
                self.n_coefs_.append(reg.n_components)
                self._regs.append(reg)
        elif self.mode=='acdc':
            # regression of all detail coeffs
            if len(Xcof) > 1:
                Xdcf = np.concatenate(Xcof[1:], axis=0)
                ydcf = np.concatenate(ycof[1:])
                # print(Xdcf.shape, ydcf.shape)
                reg = Stat.PCRegression(loss=self.loss, n_components=self.n_components, reg_name=self._reg_name, **kwargs)
                reg.fit(Xdcf, ydcf)
                self.n_coefs_.append(reg.n_components)
                self._regs.append(reg)
        else:
            raise TypeError('Unknown mode: {}'.format(self.mode))

        self.n_coefs = np.sum(self.n_coefs_)

    def predict(self, X):
        """
        Args:
            X (2d array): input variables, n_samples by n_features
        """
        assert X.ndim==2 and X.shape[-1] == self._dimx
        if self._regs is None:
            raise ValueError('Run self.fit() first!')

        # preparation of data
        Xcmv = Tools.mts_cumview(X.T, self.lag).T

        # wavelet decomposition
        Xcof = pywt.wavedec(Xcmv, self.wvlname, level=self.maxlvl, axis=0, mode=self._pywt_mode)
        # if self.nflag:
        #     # Xcof = [scaler.transform(cX) for scaler, cX in zip(self._scalers, Xcof0)]
        #     Xcof = [cX @ np.diag(1/scaler.scale_) for scaler, cX in zip(self._scalers, Xcof0)]
        # else:
        #     Xcof = Xcof0
        yprc = []  # predicted coefficients

        if self.mode=='full':
            for cX, reg in zip(Xcof, self._regs):
                yprc.append(reg.predict(cX))
        elif self.mode=='acdc':
            # toto = self._regs[0].predict(Xcof[0]); yprc.append(np.zeros_like(toto))
            yprc.append(self._regs[0].predict(Xcof[0]))

            if len(Xcof) > 1:
                Xdcf = np.concatenate(Xcof[1:], axis=0)
                # toto = self._regs[1].predict(Xdcf); yprc0 = np.zeros_like(toto)
                yprc0 = self._regs[1].predict(Xdcf)
                cdims = np.cumsum([C.shape[0] for C in Xcof[1:]])[:-1]
                yprc += np.split(yprc0, cdims)
            # print(cdims, [y.shape for y in yprc])
        # elif self.mode=='whole':
        #     Xall = np.concatenate(Xcof, axis=0)
        #     yprc0 = self._regs[0].predict(Xall)
        #     cdims = np.cumsum([C.shape[0] for C in Xcof])[:-1]
        #     yprc = np.split(yprc0, cdims)
        #     # print(cdims, [y.shape for y in yprc])
        else:
            raise TypeError('Unknown mode: {}'.format(self.mode))

        # # fill nans of the predicted coefficients by zero
        # for y in yprc:
        #     y[np.isnan(y)] = 0.

        yprd = pywt.waverec(yprc, self.wvlname, mode=self._pywt_mode)
        # yprd[-100:] = 0
        return yprd[:X.shape[0]]


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


########### Wavelet-like transform ###########

class MRA_DCT(object):
    """1D Multi-resolution analysis DCT.
    """
    def __init__(self, M, p=2, method='fft'):
        """
        Args:
            M (int): number of frequencies
            p (int): scaling factor, p>1 and p must divide M
            method (str): method of convolution, 'fft' or 'direct', cf. `scipy.signal.convolve()`
        Remarks:
            M is also the length of the convolution kernels. M=2 gives a Haar-like tight frame.
            Increase M to have more frequencies. The Gibbs effect is proportional to M and will
            appear at the head and the tail of the signal of synthesis.
        """
        assert isinstance(p, int) and p > 1 and (M%p == 0)

        self.nfreq = M
        self.scaling = p
        self.kernels = self.dct_kernels(M)  # kernel matrix, each row corresponds to a kernel, from low to high frequency

        # dynamic informations
        self._dimx = None  # dimension of the input vector
        self._dimc = None  # dimension of the output vector
        self._lvl = None  # number of scales
        self._bn = None  # information of boundary coefficients
        self._method = method  # method of convolution
        self._masks = None  # masks of restriction / extension

    @staticmethod
    def dct_kernels(M):
        toto = []
        for l in range(M):
            cst = M if l==0 else M/np.sqrt(2)  # <- or /np.sqrt(scaling) ?
            toto.append(np.cos(np.pi*(np.arange(M)+0.5)*l/M)/cst)  # DCT-II as in the original paper
            # toto.append(np.cos(np.pi*(np.arange(M)+0.5)*(l+0.5)/M)/cst)  # DCT-IV
        return np.asarray(toto)

    def maxlevel(self, N, c=1):
        """Determine the maximum level of decomposition J so that
            N/scl^J >= c*nfreq
        """
        return int(np.floor(np.log(N/self.nfreq/c)/np.log(self.scaling))) + 1

    def analysis(self, X0, lvl=None):
        """Analysis operator.
        Args:
            X0 (1d or 2d array): input array with the last dimension corresponds to time
            lvl (int): number of levels of analysis
        Returns:
            cA_n, cD_n, cD_n-1...cD_1: approximation coefficients (cA_n) and detail coefficients (cD_x) from coarse to fine with n being the number of levels. Let rdim being the row dimension of X0 (i.e., rdim == np.atleast_2d(X0).shape[0]), then cA_n is a rdim-by-? 2d array, and cD_x is a nfreq-by-rdim-by-? 3d array, with ? being the undetermined length of the time axis.
        Remark:
            information of the dimension of coefficients, the number of boundary-crossing elements and the masks are updated after each call
        """
        assert X0.ndim == 1 or X0.ndim == 2

        if lvl is None:
            lvl = self.maxlevel(len(X0))
        assert lvl > 0

        res = [[np.asarray(X0)]]

        for n in range(lvl):
            c0 = res[-1][0]  # approximation coeffs of the last level
            toto = []

            for g in self.kernels:
                h = np.atleast_2d(g[::-1])
                cx = np.sqrt(self.scaling) * scipy.signal.convolve(np.atleast_2d(c0), h, mode='full', method=self._method)
                toto.append(cx)
            # the first dimension of c1 corresponds to frequency, the second dimension corresponds to X0's first dimension, the third dimension corresponds to time
            c1 = np.asarray(toto)[:, :, ::self.scaling]  # down-sampling
            res.append(c1)

        # full coefficients, res[1:] to drop the original signal, and
        # [::-1] to arange the scale from coarse to fine
        Cf = res[1:][::-1]
        Cl = [Cf[0][0,:,:]]  # approximation coefficients
        Cl += [c[1:,:,:] for c in Cf]  # detail coefficients

        # update informations
        # on the first application or on change of lvl
        if self._lvl is None or self._lvl != lvl or self._dimx != X0.shape:
            self._dimx = X0.shape
            self._dimc = [c.shape for c in Cl]
            self._lvl = lvl
            self._bn = self._compute_bn(self._lvl)
            self._masks = self._make_mask(self._bn, self._dimc)

        return Cl

    def synthesis(self, Cl, lvl=None, ns=None):
        """Synthesis operator.

        Args:
            Cl (list of ndarray): list of coefficients of analysis from coarse to fine as returned by the method `self.analysis()`
            lvl (int): number of level of synthesis, if not given all levels presented in Cl will be used
            ns (int): truncation length of the final output
        Return:
            signal synthesized from the coefficients
        """
        Ca, Cd = Cl[0], Cl[1:]  # approximation and detail coefficients
        lvl = len(Cd) if lvl is None else min(lvl, len(Cd))  # number of levels of synthesis

        # dimension check
        assert Ca.ndim == 2  # approximation coefficient must be 2d
        for c in Cd:
            assert c.ndim == 3  # detail coefficient must be 3d
        assert lvl > 0

        Xl = [Ca]  # reconstructed approximation coefficients
        for n in range(lvl):
            # get the full coefficient matrix of the current level
            if n==0:  # the coarsest level
                cmat = np.concatenate([Ca[np.newaxis,:,:], Cd[0]])
            else:  # for other levels
                cmat = np.concatenate([Xl[-1][np.newaxis,:,:Cd[n].shape[-1]], Cd[n]])

            toto = []
            cmau = np.zeros((cmat.shape[0], cmat.shape[1], self.scaling*cmat.shape[2]))
            cmau[:,:,::self.scaling] = cmat
            for c1, g in zip(cmau, self.kernels):
                # c1 = np.zeros(scl*len(c0))
                # c1[::scl] = c0  # up-sampling
                cx = np.sqrt(self.scaling) * scipy.signal.convolve(c1, np.atleast_2d(g), mode='full', method=self._method)
                toto.append(cx)

            # Truncate the first M-1 coefficients which cause border effect, with M being the length of kernel (or number of frequency).
            # The offset M-1 below is necessay to have perfect reconstruction and it is determined by experiments, although theoretical analysis does not seem to need such an offset.
            Xl.append(np.sum(np.asarray(toto), axis=0)[:,self.nfreq-1:])

        # truncate the last M-1 coefficients which cause border effect. It seems that when the length of the original signal is even, the last M coefficients should be truncated to keep the resynthesis the same size as the original. Reason?
        X0 = Xl[-1][:,:-(self.nfreq-1)] if ns is None else Xl[-1][:,:ns]

        return np.squeeze(X0)

    def _compute_bn(self, lvl):
        """Compute the number of boundary-crossing samples in the convolution.

        Args:
            lvl (int): level of analysis
        Return:
            number of boundary-crossing samples from coarse to fine scale
        """
        bn = [0]  # number of samples crossing the left/right boundary
        for n in range(lvl):
            # 1. down-sampling of N samples by the factor scl gives (N-1)//scl + 1 samples
            # 2. bn[-1]+M-1 is the number of samples acrossing the left/right boundary, with M being the number of freqeuncies
            # => hence after the downsampling the number of boundary crossing samples is:
            bn.append((bn[-1]+self.nfreq-2)//self.scaling+1)
        bn.append(bn[-1])  # repeat the value of the coarsest scale for the approximation coefficient
        return bn[1:][::-1]

    def _make_mask(self, bn, dimc):
        assert len(bn) == len(dimc)
        vmask = []; lmask = []; rmask = []

        for b, dim in zip(bn, dimc):
            V = np.zeros(dim, dtype=bool); V[...,b:-b] = True; vmask.append(V)
            L = np.zeros(dim, dtype=bool); L[...,b:] = True; lmask.append(L)
            R = np.zeros(dim, dtype=bool); R[...,:-b] = True; rmask.append(R)
        return {'valid':vmask, 'left':lmask, 'right':rmask}

    def restriction(self, Cl, mode='valid', nflag=False):
        """Restrict the coefficients of analysis by removing boundary-crossing samples.
        """
        assert len(Cl) == len(self._dimc)
        # for c, dim in zip(Cl, self._dimc):
        #     assert c.shape == dim
        Cr = []
        for C, mask0 in zip(Cl, self._masks[mode]):
            mask = np.logical_not(mask0) if nflag else mask0
            if C.ndim == 2:  # approximation coefficients
                Cr.append(C[mask].reshape(C.shape[0], -1))
            else:
                Cr.append(C[mask].reshape(C.shape[0], C.shape[1], -1))
        return Cr
        # return [C[mask] for C,mask in zip(Cl, self._masks[mode])]

    def extension(self, Cr, mode='valid', nflag=False):
        """Extend the coefficients of analysis by zero-padding boundary-crossing samples.
        """
        assert len(Cr) == len(self._dimc)
        Cl = []
        for C, dim, mask0 in zip(Cr, self._dimc, self._masks[mode]):
            mask = np.logical_not(mask0) if nflag else mask0
            V = np.zeros(dim); V[mask] = C.flatten()
            Cl.append(V)
        return Cl

    def clear_boundary(self, Cl):
        return self.extension(self.restriction(Cl, mode='valid'), mode='valid')

    def clear_interior(self, Cl):
        return self.extension(self.restriction(Cl, mode='valid', nflag=True), mode='valid', nflag=True)

    def shrinkage(self, Cl, p, soft=False, keepdc=True):
        """
        """
        assert Cl[0].shape[0] == 1  # shrinkage can be applied only on coefficients of 1d array

        Cr = self.clear_boundary(Cl)  # interior samples
        Cb = self.clear_interior(Cl)  # boundary samples

        if keepdc:
            Ca = Cr[0].copy()  # save the approximation coefficients
            Cr[0].fill(0)
            Cv = self.coeff_list2vec(Cr)
            Cv = Tools.shrinkage_percentile(Cv, p, soft=soft)
            Cw = self.coeff_vec2list(Cv)
            Cw[0] = Ca  # restore
        else:
            Cv = self.coeff_list2vec(Cr)
            Cv1 = Tools.shrinkage_percentile(Cv, p, soft=soft)
            Cw = self.coeff_vec2list(Cv1)

        # Cs = self.coeff_list2vec(Cw) + self.coeff_list2vec(Cb)
        # return self.coeff_vec2list(Cs)
        Cs = [w+b for w, b in zip(Cw, Cb)]
        return Cs

    def denoising(self, X0, p, lvl=None, soft=False, keepdc=True, cb=False):
        if lvl is None:
            lvl = self.maxlevel(X0.shape[-1], c=4)

        Xs = []; Sv = []; C0 = []; Cs = []
        for x0 in np.atleast_2d(X0):
            xs, (c0, cs), v = self._denoising(x0, p, lvl=lvl, soft=soft, keepdc=keepdc, cb=cb)
            Xs.append(xs)
            Sv.append(v)
            C0.append(c0)
            Cs.append(cs)

        # # snr = Sv[0] if len(Sv)==1 else Sv
        # for Cl in C0:
        #     for c in Cl:

        return np.squeeze(np.asarray(Xs)), Sv

    def _denoising(self, x0, p, lvl=None, soft=False, keepdc=True, cb=True):
        Nt = x0.shape[-1]

        c0 = self.analysis(x0, lvl=lvl)
        cs = self.shrinkage(c0, p, soft=soft, keepdc=keepdc)
        if cb:
            cs = self.clear_boundary(cs)
        xs = self.synthesis(cs, ns=Nt)
        err = x0 - xs

        # Synthesis may contain Gibbs effect => truncation of result
        # Remark: the length of truncation comes from observation
        ss = scipy.var(xs[self.nfreq:-self.nfreq])
        sn = scipy.var(err[self.nfreq:-self.nfreq])
        snr = 10*np.log10(ss/sn)

        return xs, (c0, cs), snr

    def coeff_list2vec(self, Cl):
        return np.concatenate([C.flatten() for C in Cl])

    def coeff_vec2list(self, V):
        cidx = np.cumsum([np.prod(d) for d in self._dimc])
        Cl = []
        for dim, C in zip(self._dimc, np.split(V, cidx)):
            Cl.append(C.reshape(dim))
        return Cl

    def post_processing(self, Cv):
        """post-processing of coefficients"""
        Ca, Cd = self.coeff_vec2list(self.clear_boundary(Cv))  # clear the boundary coefficients
        Ra, Rd = self.restriction_list(Ca, Cd, mode='right')  # make causal
        return Ra, Rd

    # def sys_invariant(self, C0, eps=1e-7, method='diff'):
    #     """Transform the feature to make it invariant to the underlying linear system"""
    #     if method=='diff':
    #         C1 = [np.diff(c, axis=1) for c in C0]
    #     elif method=='mean':
    #         C1 = [c - np.mean(c, axis=1)[:,np.newaxis] for c in C0]
    #     else:
    #         raise ValueError('Unknown method.')

    #     if isinstance(C0, np.ndarray)
    #         return np.asarray(C1)
    #     else:
    #         return C1

    def logscale_tw(self, C0, nbins=3, aflag=False):
        """Log-scale triangle windowing.

        Args:
            C0: list of matrices of detail coefficients
            nbins (int): number of bins in log scale
        """
        C1 = []
        for c0 in C0:
            c1 = np.abs(c0) if aflag else c0
            toto = Tools.logscale_triangle_windowing(c1, nbins, self.scaling, axis=0)
            C1.append(toto)

        if isinstance(C0, np.ndarray):
            return np.asarray(C1)
        else:
            return C1

    def feature_extraction(self, Cf, Ne=5, cord=True):
        """Feature extration.

        Args:
            Cf: list of matrices of detail coefficients
            Ne (int): number of largest extrema to keep
            cord (bool): if True the result will be in chronological order, otherwise it will be in a decreasing order of magnitude
        """
        W = []
        for cf in Cf:
            # # log-scale triangle windowing in the frequency space
            # cf = Tools.logscale_triangle_windowing(c, nbins, self.scaling, axis=0)
            # find the N largest local extrema of the windowed coefficients in the time space (in chronological order)
            # cf = c[:nbins, :]
            cidx = Tools.find_N_local_extrema(cf, Ne, cord=cord, axis=1)
            W0 = []
            for v, t0 in zip(cf, cidx):
                w = np.zeros(Ne)
                toto = v[t0[t0!=-1]]  # -1 in index means no extrema
                w[:len(toto)] = toto
                W0.append(w)
            W.append(W0)
        return np.asarray(W)

    # def feature_extraction(self, C0, nbins=3, netrm=5):
    #     W = []
    #     for c in C0:
    #         # log-scale triangle windowing in the frequency space
    #         cf = Tools.logscale_triangle_windowing(c, nbins, self.scaling)
    #         # find the N largest local extrema of the windowed coefficients in the time space (in chronological order)
    #         # cf = c[:nbins, :]
    #         cidx = Tools.local_extrema(cf, N=netrm)
    #         W0 = []
    #         for v, t0 in zip(cf, cidx):
    #             w = np.zeros(netrm)
    #             toto = v[t0[t0!=-1]]  # -1 in index means no extrema
    #             w[:len(toto)] = toto
    #             W0.append(w)
    #         W.append(np.asarray(W0).flatten())
    #     return np.asarray(W)


class MRA_DCT_TAT(MRA_DCT):
    """Multi-resolution analysis DCT using transform à trous.
    """

    def analysis(self, X0, lvl):
        """
        Args:
            X0 (1d array): input signal
            lvl (int): level of decomposition, if not given the maximal level is computed from the length of X0
        """
        assert X0.ndim == 1 or X0.ndim == 2

        if lvl is None:
            lvl = self.maxlevel(len(X0))
        assert lvl > 0

        res = [[np.asarray(X0)]]
        # M = self.nfreq  # kernel length

        for n in range(lvl):
            c0 = res[-1][0]  # approximation coeffs of the last level
            toto = []

            fct = self.scaling**n  # up-sampling factor
            g = np.zeros(fct*(self.nfreq-1)+1)
            # knl = np.zeros(fct*(len(knl0)))

            for g0 in self.kernels:
                # up-sampling of the kernels
                g.fill(0); g[::fct] = g0
                h = np.atleast_2d(g[::-1])
                cx = np.sqrt(self.scaling) * scipy.signal.convolve(np.atleast_2d(c0), h, mode='full', method=self._method)
                toto.append(cx)
            res.append(np.asarray(toto))

        # full coefficients, res[1:] to drop the original signal, and
        # [::-1] to arrange the scale from coarse to to fine
        Cf = res[1:][::-1]
        Cl = [Cf[0][0,:,:]]  # approximation coefficients
        Cl += [c[1:,:,:] for c in Cf]  # detail coefficients

        # update informations
        self._dimx = len(X0)
        # on the first application or on change of lvl
        if self._lvl is None or self._lvl != lvl:
            self._dimc = [c.shape for c in Cl]
            self._lvl = lvl
            self._kl = self._compute_kl(self._lvl)
            self._bn = self._compute_bn(self._lvl)
            self._masks = self._make_mask(self._bn, self._dimc)

        return Cl

    def synthesis(self, Cl, lvl=None, ns=None):
        Ca, Cd = Cl[0], Cl[1:]  # approximation and detail coefficients
        lvl = len(Cd) if lvl is None else min(lvl, len(Cd))  # number of levels of synthesis

        # dimension check
        assert Ca.ndim == 2  # approximation coefficient must be 2d
        for c in Cd:
            assert c.ndim == 3  # detail coefficient must be 3d
        assert lvl > 0

        Xl = [Ca]  # reconstructed approximation coefficients
        for n in range(lvl):
            # get the full coefficient matrix of the current level
            if n==0:  # the coarsest level
                cmat = np.concatenate([Ca[np.newaxis,:,:], Cd[0]])
            else:  # for other levels
                cmat = np.concatenate([Xl[-1][np.newaxis,:,:Cd[n].shape[-1]], Cd[n]])

            mask = np.zeros_like(cmat) #((cmat.shape[-2], cmat.shape[-1]))
            mask[:, :, ::self.scaling**(lvl-n)] = 1 # up-sampling factor for coefficient vector
            fct = self.scaling**(lvl-n-1)  # up-sampling factor for kernel vector
            toto = []

            for c1, g0 in zip(mask * cmat, self.kernels):
                # up-sampling of the kernels
                g = np.zeros(fct*(self.nfreq-1)+1); g[::fct] = g0
                # g = np.zeros(fctk*(len(g0)))
                cx = np.sqrt(self.scaling) * scipy.signal.convolve(c1, np.atleast_2d(g), mode='full', method=self._method)
                toto.append(cx)  # drop head

            # Truncate the first M-1 coefficients which cause border effect, with M being the length of kernel of the current level.
            Xl.append(np.sum(np.asarray(toto), axis=0)[:, fct*(self.nfreq-1):])

        X0 = Xl[-1][:,:-(self.nfreq-1)] if ns is None else Xl[-1][:,:ns]

        return np.squeeze(X0)

    def _compute_kl(self, lvl):
        """Compute the length of convolution kernel of each scale.

        Args:
            lvl (int): level of analysis
        Return:
            length of kernel from coarse to fine scale
        """
        kl = []  # kernal length
        for n in range(lvl):
            fct = self.scaling**n  # up-sampling factor
            kl.append(fct*(self.nfreq-1)+1)
        kl.append(kl[-1])  # repeat the value of the coarsest scale for the approximation coefficient
        return kl[::-1]

    def _compute_bn(self, lvl):
        """Compute the number of boundary-crossing samples in the convolution.

        Args:
            lvl (int): level of analysis
        Return:
            number of boundary-crossing samples from coarse to fine scale
        """
        kl = self._compute_kl(lvl)
        return np.cumsum([l-1 for l in kl][::-1])[::-1]


    # def full2valid(self, C0, fz=True):
    #     return [M[:, l:-l] for M,l in zip(C0, self._kn)]

    # def full2validr(self, C0, fz=False):
    #     L = []
    #     for M,l in zip(C0, self._kn):
    #         toto = M[:, :-l].copy()
    #         if fz:
    #             toto[:, :l] = 0
    #         L.append(toto)
    #     return L

    # def full2validl(self, C0, fz=False):
    #     L = []
    #     for M,l in zip(C0, self._kn):
    #         toto = M[:, l:].copy()
    #         if fz:
    #             toto[:, :-l] = 0
    #         L.append(toto)
    #     return L

    def post_processing(self, Cv, centered=False):
        """post-processing of coefficients"""
        Ca, Cd = self.coeff_vec2list(self.clear_boundary(Cv))  # clear the boundary coefficients
        Ra, Rd = self.restriction_list(Ca, Cd, mode='right')  # make causal

        # if restricted:
        #     Ra, Rd = self.restriction_list(Ca, Cd, mode='right')  # make causal
        # else:
        #     Ra, Rd = Ca, Cd

        if centered:  # correct on signal with jumps
            # Alignement of coefficients
            Ra = np.roll(Ra, -self._kl[0])
            Rd = [np.roll(c, -self._kl[n], axis=1) for n, c in enumerate(Rd)]  # aligned coefficient
        # if centered:
        #     # Alignement of coefficients
        #     Ra = np.roll(Ra, -self._kl[0]//2)
        #     Rd = [np.roll(c, -self._kl[n]//2, axis=1) for n, c in enumerate(Rd)]  # aligned coefficient
        return Ra, np.asarray(Rd)

    # def detect_event_discrete(self, Cd, wsize=50, thresh=1):
    #     """Detection of event using detail coefficients
    #     Args:
    #         Cd (list of 2d array): list of detail coefficients
    #     Returns:
    #         Erng: list of index range of events
    #         mask: indicator array of events
    #         W: probability of events
    #     """
    #     W = np.zeros(Cd[0].shape[1])
    #     for c in Cd:
    #         v = Tools.U_filter(np.sqrt(np.sum(c**2, axis=0))>0, wsize=wsize) * 1.
    #         W += v

    #     mask = Tools.U_filter(W>thresh, wsize=wsize)
    #     # mask = W > thresh
    #     Erng = Tools.find_block_true(mask)
    #     return Erng, mask, W

    def detect_event(self, Cd, wsize=100, thresh=1e-5):
        """Detection of event using detail coefficients
        Args:
            Cd (list of 2d array): list of detail coefficients
        Returns:
            Erng: list of index range of events
            mask: indicator array of events
            W: probability of events
        """
        # This is the continuous version of the method, which works better than the discrete version

        # version 1:
        W = np.sqrt(np.sum(np.sum(np.asarray(Cd)**2, axis=0), axis=0))

        # version 2: per scale normalization
        # W = np.zeros(Cd[0].shape[1])
        # for c in Cd:
        #     # v = Tools.U_filter(np.sqrt(np.sum(c**2, axis=0)), wsize=wsize)
        #     n = np.sum(c**2)
        #     v = np.sum(c**2, axis=0)
        #     W += (v/n if n>0 else v)

        W /= np.sum(W)
        mask = Tools.LU_filter(W > thresh, wsize=wsize)
        Erng = Tools.find_block_true(mask)
        return Erng, mask, W

    def feature_extraction_pca(self, Ca, Cd, nbins=4, keepdc=True, absflag=True, logflag=False, dflag=True, ddflag=True, cdim=3, vthresh=None):
        """
        Args:
            Ca (1d array): vector of approximation coefficients
            Cd (list of 2d array): detail coefficients
            nbins (int): number of the first bins to use
        """
        if cdim is None:
            assert 0. <= vthresh <=1.

        # Concatenation of Ca and Cd
        X0 = [c.copy() for c in Cd]
        X0[0] = np.vstack([Ca, Cd[0]])

        # check if all scales have the same length
        toto = np.asarray([x.shape[1] for x in X0])
        sflag = np.all(toto==toto[0])

        # Mel scale windowing
        # bins = Tools.logscale_bin(nfreq, scaling=scaling, nbins=nbins+1)[:-1]  # nbins+1 and [:-1] to remove the last frequency from the scale
        bins = Tools.logscale_bin(self.nfreq, scaling=self.scaling)[:nbins]
        func = lambda x:Tools.triangle_windowing(x, bins)

        M0 = []
        for n, x in enumerate(X0):
            if keepdc:
                c = x.copy()
                if n==0:
                    c[0,:] = Stat.centralize(np.atleast_2d(c[0,:]))
                    # c[0,:] = Stat.normalize(np.atleast_2d(c[0,:]))
            else:
                c = x[1:,:].copy() if n==0 else x.copy()  # drop the DC component which is the first row of X0[0]
            if absflag:
                v = np.apply_along_axis(func, 0, np.abs(c))
                # dv = np.abs(np.diff(v, axis=1))
                # ddv = np.abs(np.diff(v, 2, axis=1))
                dv = np.apply_along_axis(func, 0, np.abs(np.diff(c, axis=1)))
                ddv = np.apply_along_axis(func, 0, np.abs(np.diff(c, 2, axis=1)))
                if logflag:
                    v = np.log(v + 1e-10)
                    dv = np.log(dv + 1e-10)
                    ddv = np.log(ddv + 1e-10)
            else:
                v = np.apply_along_axis(func, 0, c)
                dv = np.apply_along_axis(func, 0, np.diff(c, axis=1))
                ddv = np.apply_along_axis(func, 0, np.diff(c, 2, axis=1))

            h = [v[:,:-2]]
            if dflag:
                h.append(dv[:,:-1])
            if ddflag:
                h.append(ddv)
            M0.append(np.vstack(h)*(self.scaling**n))

        # feature
        if sflag:
            # all scales have the same length, apply pca overall
            M = np.vstack(M0)

            # drop all zero column
            # idx = np.mean(np.abs(M), axis=0) < 1e-8
            # M = M[:, ~idx]

            _, U, S = Stat.pca(M, nc=None, corrflag=False)
            # F = np.hstack([S[:cdim], U[:,:cdim].flatten(order='F')])  # combine the singular values with the vectors
            F = U[:,:cdim].flatten(order='F')  # combine the singular values with the vectors
        else:
            # scales have different length, apply pca per scale
            F0 = []
            for M in M0:
                # drop all zero column
                idx = np.mean(np.abs(M), axis=0) < 1e-8
                M = M[:, ~idx]
                _, U, S = Stat.pca(M, nc=None, corrflag=False)

                # if cdim is None: # if cdim is not given, use vthresh to determine it
                #     toto = np.cumsum(S) / np.sum(S)
                #     cdim = find(cumsum(S/np.sum(S))>1-vthresh)[0]
                # toto = np.hstack([S, U.flatten(order='F')])
                toto = U[:,:cdim].flatten(order='F')
                F0.append(toto)
            F = np.hstack(F0)  # combine the singular values with the vectors
        return F, M0
