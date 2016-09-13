import numpy as np
from numpy import newaxis, mean, sqrt, zeros, ones, squeeze,\
    asarray, abs
from numpy.linalg import norm, svd, inv, pinv

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
# from sklearn.gaussian_process import GaussianProcess

from . import Tools

# #### Generic ####


def sroll(X0, shift):
    """Per dimension roll of a nd-array.

    Parameters
    ----------
    X0: ndarray
    shift: 1d array of integer.
        The n-th dimension of X0 will be rolled by shift[n].

    Return: a transformed ndarray
    """
    X = X0.copy()

    for n in range(X0.ndim):
        X = np.roll(X, shift[n], axis=n)

    return X


def croll(X0, shift):
    """
    Per row roll of a 2d array.

    Parameters
    ----------
    X0: 2d array
    shift: 1d array of integer.
        The n-th row of X0 will be rolled by shift[n].

    Return: a transformed array
    """
    if X0.ndim == 1:
        X = X0[np.newaxis, :]
    else:
        X = X0.copy()

    assert(len(shift) == X.shape[0])

    for n in range(X.shape[0]):
        X[n, :] = np.roll(X[n, :], shift[n])

    return X


# #### Moving window ####
# For the single variate moving window functions, prefer the implementation of
# the pandas library, for example pandas.rolling_mean for the moving average.

def mwmean(X0, wsize, mode='hard', causal=False):
    """
    Moving average.

    Parameters
    ----------
    X0: data set
    wsize: size of the window
    """
    # if X0.ndim>1:
    #     myfunc = lambda X: np.mean(X, axis=-1)
    # else:
    #     myfunc = lambda X: np.mean(X)

    mX = mw_estimator(np.mean, X0, wsize, mode=mode, axis=-1, causal=causal)
    mX = np.asarray(mX).T

    return squeeze(mX)


def mwmedian(X0, wsize, mode='hard', causal=False):
    """
    Moving average.

    Parameters
    ----------
    X0: data set
    wsize: size of the window
    """
    # if X0.ndim>1:
    #     myfunc = lambda X: np.median(X, axis=-1)
    # else:
    #     myfunc = lambda X: np.median(X)

    mX = mw_estimator(np.median, X0, wsize, mode=mode, axis=-1, causal=causal)
    mX = np.asarray(mX).T

    return squeeze(mX)


def mwstd(X0, wsize, causal=False):
    """
    Moving standard deviation.

    Parameters
    ----------
    X0: data set
    wsize: size of the window
    """
    sX = mw_estimator(np.std, X0, wsize, mode='hard', axis=1, causal=causal)
    sX = np.asarray(sX).T

    return squeeze(sX)


def mw_estimator(func, X0, wsize, mode='soft', causal=False, tfct=1,
                 nflag=False, **kwargs):
    """
    General moving window estimator.
    This function is similar to rolling_apply of the Pandas library.

    Parameters
    ----------
    func : callable function
        this function computes the estimator (on some window).
    X0 : 2d array
        data set, each column is an observation.
    wsize : integer
        size of the window.
    mode : string
        'soft' or 'hard', slicing mode of the moving window,
        see Tools.safe_slice.
    causal : boolean
        see Tools.safe_slice
    tfct : integer
        X0 will be under-sampled by a factor tfct in the time axis
    nflag : boolean
        if True the data of each window are normalized by normalize_2d_data
    """

    if X0.ndim == 1:
        X = X0[np.newaxis, :].copy()
    else:
        X = X0.copy()

    nbTime = X.shape[1]  # column corresponds to time
    res = []

    for tidx in range(0, nbTime, tfct):  # downsampling by tfct
        Xw0 = Tools.safe_slice(X, tidx, wsize, mode=mode, causal=causal)

        Xw = normalize_2d_data(Xw0, std=True) if nflag else Xw0

        if kwargs:
            res.append(func(Xw, **kwargs))
        else:
            res.append(func(Xw))

    return res


def mw_estimator_twovar(func, X0, Y0, wsize, D0=None, mode='soft', causal=False, tfct=1, nflag=False, **kwargs):
    """
    General moving window estimator with two variates.

    Parameters
    ----------
    func : callable function
        the estimator to be applied on each window
    X0, Y0 : array-like
        data set, each column is an observation
    wsize : integer
        size of the window
    D0 : array-like, has the same dimension as X0
        at each time index t, a delay D0[n,t] is applied to X0[n,:]
        when calling Tools.safe_slice function to take a slice around t.
    mode, causal, tfct, nflag : same as in the function mw_estimator

    Returns
    -------
    A list of results returned by the function func.
    """

    if X0.ndim==1:
        X = X0[np.newaxis,:]
    else:
        X = X0.copy()

    if Y0.ndim==1:
        Y = Y0[np.newaxis,:]
    else:
        Y = Y0.copy()

    assert(X.shape[1] == Y.shape[1])
    if D0 is not None:
        assert(X.shape == D0.shape)

    nbTime = X.shape[1]
    res = []

    for tidx in range(0, nbTime, tfct):
        if D0 is None:
            Xw0 = Tools.safe_slice(X, tidx, wsize, mode=mode, causal=causal)
        else: # if a delay is given
            toto = []
            for n, D in enumerate(D0):
                toto.append(squeeze(Tools.safe_slice(X[n,:], tidx-D[tidx], wsize, mode=mode, causal=causal)))
            Xw0 = np.asarray(toto)

        Yw0 = Tools.safe_slice(Y, tidx, wsize, mode=mode, causal=causal)

        Xw = normalize_2d_data(Xw0, std=True) if nflag else Xw0
        Yw = normalize_2d_data(Yw0, std=True) if nflag else Yw0

        res.append(func(Xw, Yw, **kwargs))
        # if kwargs:
        #     res.append(func(Xw, Yw, **kwargs))
        # else:
        #     res.append(func(Xw, Yw))

    return res


def mw_estimate_delay(X0, Y0, wsize=24*10, dlrange=[-12,12], method='regression'):
    """
    Moving window estimation of delay.

    Parameters
    ----------
    X0: 2d array
        each column is an input vector
    Y0: 2d array
        each column is an observation vector
    wsize: size of the moving window
    dlrange: range of delay for test

    Returns
    -------
    (k, b, e, er, s): returned by linear_regression. k the slope and b the intercept
        in the relation yt = k*xt + b, e and er are the residual and the relative
        residual e/norm(yt), s is the noise variance vector.
    C: correlations on the moving window after compensation of the optimal delay
    D: optimal delay on the moving window
    """
    assert(X0.shape == Y0.shape)

    if X0.ndim==1:
        X = X0[np.newaxis,:]
    else:
        X = X0.copy()

    if Y0.ndim==1:
        Y = Y0[np.newaxis,:]
    else:
        Y = Y0.copy()

    nbLoc,nbTime = X.shape

    # quantities after compensation of thermal delay
    D = zeros((nbLoc, nbTime)) # thermal delay
    yt = zeros((nbLoc, wsize))

    for tidx in range(nbTime):
        yt.fill(0) # reused for each tidx
        # estimation of the optimal delay and linear regression
        for n in range(nbLoc):
            yt[n,:] = squeeze(Tools.safe_slice(Y[n,:], tidx, wsize, mode='soft'))

            D[n,tidx], *_\
              = linear_regression_with_delay(X[n,:], yt[n,:], tidx, dlrange=dlrange)
            # if method=='regression':
            #     D[n,tidx], *_\
            #       = linear_regression_with_delay(X[n,:], yt[n,:], tidx, dlrange=dlrange)
            # else:
            #     D[n,tidx], *_\
            #       = correlation_with_delay(X[n,:], yt[n,:], tidx, dlrange=dlrange)
    return D


#### Regression ####

def linear_regression(A, Y):
    """
    Linear regression of the model:
        Y = Ax + e

    Parameters
    ----------
    A: 2d array
        the system operator
    Y: 1d array
        the observation vector

    Returns
    -------
    X: the least square solution (using scipy.sparse.linalg.cgs)
    E: residual error
    Er: relative residual error
    sigma2: estimation of the noise variance
    """
    assert(A.ndim == 2)
    assert(Y.size == A.shape[0])

    # X, *_ = sparse.linalg.lsqr(A, Y) # Least-square estimation of X
    # X, *_ = sparse.linalg.cgs(A.T @ A, A.T @ Y) # Least-square estimation of X
    X = pinv(A) @ Y # Least-square estimation of X

    err = A @ X - Y # residual
    nerr = norm(err)
    sigma2 =  nerr**2 / (A.shape[0] - np.linalg.matrix_rank(A)) # non-biased estimation of noise's variance
    # assert(sigma2 >= 0 or np.isnan(simga2))

    return np.hstack([np.squeeze(X), asarray([nerr, nerr/norm(Y), sigma2])])


def op_linear_regression(X0, Y0):
    """
    Estimate the linear operator y = K*x + b between
    two vectors x and y using linear regression.

    Parameters
    ----------
    X0: 1d (or 2d array, each column is an input vector)
    Y0: 1d (or 2d array, each column is an observation vector)

    Returns
    -------
    K, b, err, err_rel: the estimated matrix K (slope), vector b (intercept) and
        residual and relative residual
    """
    assert(0<X0.ndim<=2)
    assert(0<Y0.ndim<=2)

    if X0.ndim==1:
        X = X0[np.newaxis,:]
    else:
        X = X0.copy()

    if Y0.ndim==1:
        Y = Y0[np.newaxis,:]
    else:
        Y = Y0.copy()

    assert(X.shape[1] == Y.shape[1])

    mX = mean(X, axis=1)
    mY = mean(Y, axis=1)

    XmX = X-mX[:, newaxis]
    YmY = Y-mY[:, newaxis]

    K = np.zeros((Y.shape[0], X.shape[0]))
    b = np.zeros_like(mY)
    err = np.zeros(Y.shape[0])
    err_rel = np.zeros(Y.shape[0])

    if np.max(np.isnan(X0.flatten())) or np.max(np.isnan(Y0.flatten())):
        K.fill(np.nan)
        b.fill(np.nan)
        err.fill(np.nan)
        err_rel.fill(np.nan)
    else:
        try:
            K = (YmY @ XmX.T) @ inv(XmX @ XmX.T)
            b = mY - K @ mX
            # err = sqrt(mean((K @ X + b[:,newaxis] - Y)**2, axis=1)) # error
            err = norm(K @ X + b[:,newaxis] - Y, axis=1)
            err_rel = err / norm(Y, axis=1) # relative error
        except Exception:
            # in case that X is a constant vector
            K.fill(np.nan)
            b.fill(np.nan)
            err.fill(np.nan)
            err_rel.fill(np.nan)

    # Instead of squeeze(K), we use squeeze(K)[()] which handles the 1d case correctly
    return squeeze(K)[()], squeeze(b)[()], squeeze(err)[()], squeeze(err_rel)[()]


def linear_regression_with_delay(X, Y, tidx=None, dlrange=[-12,12]):
    """Estimate the optimal delay of the sequence X wrt Y, such that the least
    squared error between the delayed X and Y is maximized, up to some slope and
    intercept (estimated by linear regression).

    X is related to Y through:
        Y[n] = k*X[n-d] + b + error
    This function estimate the optimal k, d, and b (all scalars) by solving a series of least
    square problem and choosing the optimal delay as the one minimizing the residual.

    The delay of X here is realized via either a moving window on X if
    tidx is given, otherwise by rolling X.

    Parameters
    ----------
    X, Y: 1darray, satisfying len(X) >= len(Y)
    tidx: center of the window on X, optional
    dlrange: seek the estimated delay in this range

    Returns
    -------
    D: estimated delay
    K, B, E: slope, intercept, and residual between X and Y after delaying X
    C : correlation between X and Y after delaying X
    Xn: delayed X
    """

    assert(X.ndim==Y.ndim==1)
    assert(X.size >= Y.size) # X and Y must have the same length
    assert(dlrange[1]>dlrange[0])

    wsize = Y.size # moving window size is that of Y

    # res[0,1]:slope and intercept
    # res[2,3]:residual error and relative error,
    # res[4]: noise variance
    # res[5,6]: variance of slope and intercept
    res = zeros((7, dlrange[1] - dlrange[0]))

    # if data contain nan then all outputs will be nan
    if np.max(np.isnan(Y)):
        return np.nan, ones(res.shape[0])*np.nan, np.nan, ones(wsize)*np.nan

    for n in range(dlrange[0], dlrange[1]):
        if tidx is None:
            Xn = np.roll(X[:wsize], n)
            # n>0 (eg n=1): [1,2,3,4] -> [4,1,2,3]
            # n<0 (eg n=-1): [1,2,3,4] -> [2,3,4,1]
        else:
            Xn = squeeze(Tools.safe_slice(X, tidx-n, wsize, mode='soft'))
            # the -n here produces the similar effect as with np.roll

        A = np.vstack([Xn, ones(wsize)]).T
        res[:5, n-dlrange[0]] = linear_regression(A, Y)
        sigma2 = res[4, n-dlrange[0]]
        res[5:, n-dlrange[0]] = np.diag(sigma2 * pinv(A.T @ A)) # variance of the estimators (K, B)

    # the optimal delay is taken as the one minimizing the residual of LS
    nidx = np.argmin(res[2,:]) # argmin/argmax always return 0 if data contain nan

    dt = dlrange[0]+nidx

    if tidx is None:
        Xn = np.roll(X[:wsize], dt)
    else:
        Xn = squeeze(Tools.safe_slice(X, tidx-dt, wsize, mode='soft'))

    return dt, res[:, nidx], corr(Xn, Y), Xn


def correlation_with_delay(X, Y, tidx=None, dlrange=[-12,12]):

    assert(X.ndim==Y.ndim==1)
    assert(X.size >= Y.size) # X and Y must have the same length
    assert(dlrange[1]>dlrange[0])

    wsize = Y.size # moving window size is that of Y
    res = zeros(dlrange[1] - dlrange[0])

    # if data contain nan then all outputs will be nan
    if np.max(np.isnan(Y)):
        return np.nan, np.nan, np.nan, ones(wsize)*np.nan

    for n in range(dlrange[0], dlrange[1]):
        if tidx is None:
            Xn = np.roll(X[:wsize], n)
            # n>0 (eg n=1): [1,2,3,4] -> [4,1,2,3]
            # n<0 (eg n=-1): [1,2,3,4] -> [2,3,4,1]
        else:
            Xn = squeeze(Tools.safe_slice(X, tidx-n, wsize, mode='soft'))
            # the -n here produces the similar effect as with np.roll

        res[n] = corr(Xn, Y)

    # the optimal delay is taken as the one minimizing the residual of LS
    nidx = np.argmax(np.abs(res)) # argmin/argmax always return 0 if data contain nan

    dt = dlrange[0]+nidx

    if tidx is None:
        Xn = np.roll(X[:wsize], dt)
    else:
        Xn = squeeze(Tools.safe_slice(X, tidx-dt, wsize, mode='soft'))

    return dt, res[nidx], corr(Xn, Y), Xn


def block_SIC(X0, Y0):
    """
    Blockwise estimation of the Slope, Intercept and Correlation.
    For each row n in the block, apply linear regression:
        Y0[n,:] = k*X0[n,:] + b

    This function is typically used with mw_estimator_twovars().

    Parameters
    ----------
    X0, Y0: 2d arrays of the same dimension

    Returns
    -------
    K: slope
    B: intercept
    E: residual error
    Er: relative residual error
    C: correlation
    S: variance of estimators
    """

    nbLoc, nbTime = X0.shape

    A = [] # global linear regression result

    K = zeros(nbLoc) # slope
    B = zeros(nbLoc) # intercept
    E = zeros(nbLoc) # residual error
    Er = zeros(nbLoc) # relative residual error
    C = zeros(nbLoc) # correlation
    S = zeros((nbLoc, 3)) # variance of estimators

    for n in range(nbLoc):
        C[n] = corr(X0[n,:], Y0[n,:]) # correlation
        A = np.vstack([X0[n,:], np.ones(nbTime)]).T # system matrix
        toto = linear_regression(A, Y0[n,:]) # linear regression

        K[n], B[n] = toto[0], toto[1] # estimator of slope and intercept
        E[n], Er[n], sigma2 = toto[2], toto[3], toto[4]

        S[n,0] = sigma2 # variance of the noise
        S[n,1:] = np.diag(sigma2 * pinv(A.T @ A)) # variance of the estimators (K, B)

    return K, B, E, Er, C, S


#### Statistics ####

def sign_safe_svd(A):
    """
    A: a real matrix
    """
    U, S, V0 = svd(A)
    V = V0.T # U @ diag(S) @ V.T = A
    N = len(S)

    sl = zeros(N)
    sr = zeros(N)

    for n in range(N):
        # toto = U[:,n] @ A
        # sl[n] = np.sign(toto) @ (toto**2)
        # toto = A @ V[:,n]
        # sr[n] = np.sign(toto) @ (toto**2)

        toto = U[:,n] @ (A / norm(A, axis=0)[newaxis,:])
        sl[n] = np.sum(toto)

        toto = (A / norm(A, axis=1)[:,newaxis]) @ V[:,n]
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

    Parameters
    ----------
    X0 : array
        2 dimensional, each column represents an observation
    nc : integer
        number of components to hold
    sflag : boolean
        if True apply sign correction to the principal vectors

    Returns
    -------
    C, U : coefficients and corresponded principal directions
    """

    X0 = normalize_2d_data(X0, std=False) # remove the mean

    # U0, S, _ = sign_safe_svd(np.cov(X0))
    U0, S, _ = svd(np.cov(X0))

    U = U0.copy()

    # sign correction:
    if sflag:
        X1 = X0/norm(X0, axis=0)[newaxis,:]

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


def corr(x, y):
    """
    Compute the correlation matrix of two multi-variate random variables. Complex variables are correctly handled. This is similar to the numpy function corrcoef but is safe to nan values (treat as zero).

    Parameters
    ----------
    x : 1d or 2d array
        each column of x is a sample from the first variable.
    y : 1d or 2d array
        each column of y is a sample from the second variable. y must have the same number of columns as x.

    Returns
    -------
    The correlation matrix, with the (i,j) element being corr(x_i, y_j)
    """
    if x.ndim==1:
        x = x[np.newaxis,:]
        x[np.isnan(x)] = 0
    if y.ndim==1:
        y = y[np.newaxis,:]
        y[np.isnan(y)] = 0

    assert(x.shape[1]==y.shape[1])

    mx = mean(x, axis=1); my = mean(y, axis=1)
    xmx = x-mx[:, np.newaxis]; ymy = y-my[:,np.newaxis]
    dx = sqrt(mean(abs(xmx)**2, axis=1)) # standard deviation of X
    dy = sqrt(mean(abs(ymy)**2, axis=1)) # standard deviation of Y
    # vx = mean(abs(xmx)**2, axis=1); vy = mean(abs(ymy)**2, axis=1)

    return squeeze((xmx/dx[:,newaxis]) @ (np.conj(ymy.T)/dy[newaxis,:])) / x.shape[1]


#### Sensor clustering ####

def normalize_2d_data(X0, std=True):
    """
    Remove the mean and normalize by the standard deviation.

    Parameters
    ----------
    X0: 1d or 2d array, each column is an observation.
    """
    if X0.ndim==1:
        X = X0[np.newaxis,:]
    else:
        X = X0.copy()

    mX = X.mean(axis=1)
    sX = X.std(axis=1)

    if std:
        X = (X-mX[:,newaxis]) / sX[:,newaxis]
    else:
        X = (X-mX[:,newaxis])

    return squeeze(X)
    # return np.reshape(X, X0.shape)



def sensor_similarity(X0, sfct=1, nflag=False):
    """
    Parameters
    ----------
    X0 : 2d array. Each row corresponds to one sensor, and each column corresponds
        to one observation. Each sensor is treated as a sample of some (high dimensional)
        random variable and apply PCA on the covariance matrix of the transposed X0.
    sfct : integer. Down-sampling factor for speeding-up the computation.

    Returns
    -------
    V, U : coefficients and principal vectors.
    """
    if nflag:
        # We first normalize the record of each individual sensor
        X = normalize_2d_data(X0[:,::sfct])
    else:
        X = X0[:,::sfct]


    # Treat each sensor as the sample of some (high dimensional)
    # random variable and apply PCA
    V, U = pca(X.T)

    return V, U


def sensor_clustering(X0, nbCluster=5, cdim=3, wflag=True, aflag=False):
    """
    Cluster the sensors by the similarities between their records.

    Parameters
    ----------
    X0 : 2d array, each row represents the record of one sensor.
    nbCluster : number of cluster desired
    cdim : integer. The first cdim principal vectors coefficients are used as features
        in the KMeans fitting. If None all coefficients are used.
    wflag: if True the input data will be normalized by normalize_2d_data
    aflag: if True the KMeans clustering algorithm is applied on the absolute values of the PCA coefficients
        to counter the +/- sign problem
    """

    assert(X0.ndim==2)

    X = normalize_2d_data(X0) if wflag else X0 # pre-whitening data

    V = PCA(n_components=cdim).fit_transform(X)

    if aflag: # use absolute value
        V = np.abs(V)

    W = KMeans(n_clusters=nbCluster).fit(V)

    idxg = []
    for n in range(nbCluster):
        g = np.where(W.labels_==n)[0]
        idxg.append(list(g))

    return idxg, V, W


def sensor_correlate(X0, nflag=False, mode='same'):
    assert(X0.ndim==2)

    nbSensor, nbTime = X0.shape

    Y = []
    # Y = zeros((nbSensor, nbSensor, 2*nbTime-1))
    # Y = zeros((nbSensor*nbSensor, 2*nbTime-1))

    for n in range(nbSensor):
        for m in range(nbSensor):
            toto = np.correlate(X0[n,:], X0[m,:], mode=mode)

            if nflag: # with normalization the output coefficients are bounded in [-1, 1]
                toto /= (norm(X0[n,:]) * norm(X0[m,:]))

            Y.append(toto)
            # Y[n*nbSensor+m,:] = toto
            # Y[n,m,:] = toto

            # if m!=n:
            #     Y.append(toto)

    return np.asarray(Y)
