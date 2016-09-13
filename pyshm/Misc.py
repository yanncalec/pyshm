"""
Collection of some old/deprecated functions.
"""

def smoothing_data(X, mwSize):
    """Smooth the data (X) using a moving average and return the smoothed data.
    """
    if type(mwSize) is np.ndarray and len(mwSize) == X.shape[0]:
        mX = zeros_like(X)

        for n in range(X.shape[0]):
            mX[n,:] = mwmean(X[n,:], mwSize[n])
    else:
        mX = mwmean(X, mwSize)

    return mX


def op_linear_regression(X0, Y0):
    if X0.ndim==1:
        X = X0[np.newaxis,:]
    else:
        X = X0.copy()

    if Y0.ndim==1:
        Y = Y0[np.newaxis,:]
    else:
        Y = Y0.copy()

    assert(X.shape[1] == Y.shape[1])

    m = X.shape[0]

    S = np.cov(X, Y)
    SXX, SXY = S[:m,:m], S[:m, m:]

    Lmat = (inv(SXX) @ SXY).T

    if Lmat.size==1: # singular case
        return (Lmat.squeeze().take(0), X.mean(axis=1), Y.mean(axis=1))
    else:
        return (Lmat.squeeze(), X.mean(axis=1), Y.mean(axis=1))


def op_linear_regression_residual(X0, Y0):
    Lmat, mX, mY = op_linear_regression(X0,Y0)
    Y1 = Lmat @ (X0 - mX[:, newaxis]) + mY[:, newaxis]
    return Lmat, Y1, norm(Y0-Y1, axis=1), norm(Y0-Y1, axis=1)/norm(Y0, axis=1)


def mw_linear_regression_predictor(X0, Y0, wsize, lwratio=0.2):
    """

    Parameters
    ----------
    X0, Y0: array-like
        data set, each column is an observation
    wsize: integer
        size of the window
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

    nbTime = X.shape[1]
    res = []
    Ya = zeros_like(Y0)
    Ct = zeros_like(Y0)

    lwsize = int(wsize*lwratio)
    # LM = []

    for tidx in range(nbTime):
        Xw,xrng = Tools.safe_slice(X0, tidx, wsize, mode='hard', returnidx=True)
        Yw = Tools.safe_slice(Y0, tidx, wsize, mode='hard', returnidx=False)

        Lmat, mX, mY = op_linear_regression(Xw[:,:lwsize],Yw[:,:lwsize])
        Y1 = Lmat @ (Xw - mX[:, newaxis]) + mY[:, newaxis]
        Ya[:, xrng[0]:xrng[1]] += Y1
        Ct[:, xrng[0]:xrng[1]] += 1
        # LM.append(Lmat)

    return Ya/Ct #, LM



def mw_estimate_linear_relation(X, Y, wsize=24*10, delay=True):
    assert(X.shape == Y.shape)

    nbLoc, nbTime = X.shape
    K = zeros_like(X) # slope
    B = zeros_like(X) # intercept
    C = zeros_like(X) # correlation
    D = zeros_like(X) # delay

    for n in range(nbLoc):
        if delay:
            for tidx in range(nbTime):
                # estimate the best delay of y relative to x
                D[n,tidx], C[n,tidx], xt, yt = estimate_delay(X[n,:], Y[n,:], tidx, wsize)
                xt, yt = squeeze(xt), squeeze(yt)

                # The following part has no effect on the results
                # dxt = np.diff(xt); dxt *= norm(xt)/norm(dxt)
                # dyt = np.diff(yt); dyt *= norm(yt)/norm(dyt)
                # K[n,tidx], B[n,tidx] = estimate_linear_relation_reg(np.r_[xt, dxt], np.r_[yt,dyt])

                K[n,tidx], B[n,tidx] = estimate_linear_relation_reg(xt, yt)
                # G.append(sensor_clustering())
        else:
            for tidx in range(nbTime):
                xt = Tools.safe_slice(X[n,:], tidx, wsize)
                yt = Tools.safe_slice(Y[n,:], tidx, wsize)
                C[n,tidx] = corr(xt, yt)
                K[n,tidx], B[n,tidx] = estimate_linear_relation_reg(xt, yt)

    return K, B, C, D


def estimate_linear_relation_pca(X, Y):
    """
    Estimate the linear relation Y = k*X + b between
    two vectors X and Y using PCA.
    """
    assert(X.ndim == Y.ndim == 1)

    UU, ss, _ = svd(np.vstack((X,Y)))
    k = UU[1,0]/UU[0,0]
    b = mean(Y) - k*mean(X)

    return k, b


# def estimate_delay_fixed_pd(Xr, dlrange=[0,24], negcorr=False):
#     """
#     Wrapper of estimate_delay_fixed for pandas datasheet
#     """

#     # X = np.asarray(Xr.iloc[:,0])
#     # Y = np.asarray(Xr.iloc[:,1])
#     X, Y = Xr[:,0], Xr[:,1]
#     val, _ = estimate_delay_fixed(X, Y, tidx, wsize, dlrange, negcorr)

#     return val

def estimate_delay_old(X, Y, tidx, wsize, window=True, dlrange=[0,24], negcorr=False):
    """Estimate the optimal delay of the sequence Y wrt X around a given
    index and on a given time window, such that the correlation
    between the delayed Y and X is maximized.

    Parameters
    ----------
    X, Y: 1darray of the same length
    tidx: center of the window
    wsize: window size
    dlrange: seek the estimated delay in this range
    negcorr: if True the negative correlation value is tolerated

    Returns
    -------
    t0: estimated delay
    val: correlation between X and Y after delaying Y
    x0: windowed sequence of X
    y0: windowed sequence of Y

    """

    assert(X.ndim==Y.ndim==1)
    assert(X.size==Y.size)
    assert(dlrange[1]>dlrange[0])
    # assert(max(abs(Nt[0], abs(Nt[1]))) < int(wsize/2))
    # assert(X.size >= wsize + abs(Nt[1])-Nt[0])

    nbTime = len(X)

    if window:
        wd = signal.windows.gaussian(wsize, wsize * 0.1) # window function
    else:
        wd = np.ones(wsize)

    x0 = Tools.safe_slice(X, tidx, wsize, mode='soft')

    val = zeros(dlrange[1]-dlrange[0])
    val0 = zeros(dlrange[1]-dlrange[0])

    for n in range(dlrange[0], dlrange[1]):
        y = Tools.safe_slice(Y, tidx+n, wsize, mode='soft')

        # with pre-whitenning
        # val[n-dlrange[0]] = corr(x0*wd, y*wd)
        # val0[n-dlrange[0]] = corr(x0, y)
        #
        # without pre-whitenning
        val[n-dlrange[0]] = (x0*wd) @ (y*wd).T / norm(x0*wd)/norm(y*wd)
        val0[n-dlrange[0]] = (x0 @ y.T) / norm(x0)/norm(y)

    if negcorr:
        nidx = np.argmax(abs(val))
    else:
        nidx = np.argmax(val)

    y0 = Tools.safe_slice(Y, tidx+dlrange[0]+nidx, wsize, mode='soft')
    # y0 = Tools.safe_slice(Y, tidx, wsize, mode='soft')

    # A simple way to estimate the absolute value of the slope
    # xf = fft.fft(squeeze(Tools.safe_slice(X, tidx, wsize, mode='soft')))
    # yf = fft.fft(squeeze(Tools.safe_slice(Y, tidx, wsize, mode='soft')))
    # kabs = np.sum(abs(yf[1:]))/np.sum(abs(xf[1:]))

    return dlrange[0]+nidx, val0[nidx], x0, y0 #, kabs


def estimate_delay_fixed(X, Y, dlrange=[0,24], negcorr=False):
    """Estimate the optimal delay of the sequence Y wrt X, such that the correlation
    between the delayed Y and X (delay is implemented via np.roll) is maximized.
    """

    X = np.squeeze(X)
    Y = np.squeeze(Y)

    assert(X.ndim==Y.ndim==1)
    assert(X.size==Y.size)
    assert(dlrange[1]>dlrange[0])

    val = zeros(dlrange[1]-dlrange[0])

    for tt in range(dlrange[0], dlrange[1]):
        y = np.roll(Y, -tt)

        if tt==0: # case for non -delay
            val[tt] = corr(X, y)
        else:
            val[tt] = corr(X[0:-tt], y[0:-tt])
            # val[tt] = corr(xx, yt)

    if negcorr:
        n0 = np.argmax(abs(val))
    else:
        n0 = np.argmax(val)

    return dlrange[0]+n0, val[n0]



def linear_regression_with_delay(X, Y, dlrange=[0,24]):
    """Estimate the optimal delay of the sequence X wrt Y, such that the least squared error
    between the delayed X and Y is maximized, up to some slope and intercept (estimated by linear regression).

    X is related to Y through:
        Y[n] = k*X[n-d] + b + error
    This function estimate the optimal k, d, and b by solving a series of least
    square problem and choosing the optimal delay as the one minimizing the residual.

    Parameters
    ----------
    X, Y: 1darray of the same length
    dlrange: seek the estimated delay in this range

    Returns
    -------
    D: estimated delay
    K, B, E: slope, intercept, and residual between X and Y after delaying X
    C : correlation between X and Y after delaying X
    X0, Y: delayed X and the original Y
    """

    assert(X.ndim==Y.ndim==1)
    assert(X.size==Y.size)
    assert(dlrange[1]>dlrange[0])

    res = zeros((3, dlrange[1] - dlrange[0])) # slope and intercept and residual error
    x0, y0 = X,Y

    for n in range(dlrange[0], dlrange[1]):
        xn = np.roll(x0, n)
        A = np.vstack([xn, ones(wsize)]).T
        toto = inv(A.T @ A) @ (A.T @ y0)

        res[:2, n-dlrange[0]] = toto
        res[2, n-dlrange[0]] = norm(y0-A @ toto)

    nidx = np.argmin(res[2,:])
    xn = np.roll(x0, nidx)

    return dlrange[0]+nidx, res[:, nidx], corr(xn, y0), (xn, y0)





def mw_estimate_DCOLR(X0, Y0, wsize=24*10, dlrange=[-12,12], **kwargs):
    """
    Moving window estimation of Delay, Correlation, and (Multivariate)
    Linear Regression (DCOLR).

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
    (M, I, E, Er): M the matrix and I the vector of intercept in the relation Yt = M*Xt + I,
        E and Er are the residual and the relative residual.
    (k, b, e, er): k the slope and b the intercept in the relation yt = k*xt + b,
        e is the residual, and er is the relative residual e/norm(yt)
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

    regweight = kwargs['regweight'] if 'regweight' in kwargs else 1e-3

    nbLoc,nbTime = X.shape

    A = [] # global linear regression result

    K0 = zeros((nbLoc, nbTime)) # slope
    B0 = zeros((nbLoc, nbTime)) # intercept
    S0 = zeros((nbLoc, nbTime)) # noise variance sigma
    C0 = zeros((nbLoc, nbTime)) # correlation
    E0 = zeros((nbLoc, nbTime)) # residual error
    Er0 = zeros((nbLoc, nbTime)) # relative residual error

    # Same quantities, but with thermal delay compensation
    K = zeros((nbLoc, nbTime)) # slope
    B = zeros((nbLoc, nbTime)) # intercept
    S = zeros((nbLoc, nbTime)) # noise variance sigma
    C = zeros((nbLoc, nbTime)) # correlation
    E = zeros((nbLoc, nbTime)) # residual error
    Er = zeros((nbLoc, nbTime)) # relative residual error

    D = zeros((nbLoc, nbTime)) # thermal delay

    xt = zeros((nbLoc, wsize))
    yt = zeros((nbLoc, wsize))

    for tidx in range(nbTime):
        xt.fill(0); yt.fill(0) # reused for each tidx

        # estimation of the optimal delay and linear regression
        for n in range(nbLoc):
            # Estimation without compensation of thermal delay
            x0 = squeeze(Tools.safe_slice(X[n,:], tidx, wsize, mode='hard'))
            y0 = squeeze(Tools.safe_slice(Y[n,:], tidx, wsize, mode='hard'))
            C0[n, tidx] = corr(x0, y0) # correlation
            res, err, errel, sigma2 = linear_regression(np.vstack([x0, np.ones(len(x0))]).T, y0) # linear regression
            K0[n,tidx], B0[n,tidx] = res[0], res[1] # estimator of slope and intercept
            S0[n,tidx,:] = np.diag(sigma2 * linalg.pinv(A.T @ A)) # variance of the estimators (K, B)

            # Estimation with compensation of thermal delay
            xt[n,:] = squeeze(Tools.safe_slice(X[n,:], tidx, wsize, mode='soft'))
            yt[n,:] = squeeze(Tools.safe_slice(Y[n,:], tidx, wsize, mode='soft'))

            if regweight is None or tidx<wsize:
                D[n,tidx], (K[n,tidx], B[n,tidx], E[n,tidx]), C[n,tidx], xt[n,:]\
                = linear_regression_with_delay(X[n,:], yt[n,:], tidx, dlrange=dlrange)
            else:
                # apriori value for the thermal delay
                # regval = np.median(squeeze(Tools.safe_slice(D[n,:], tidx, wsize, mode='hard', causal=True)))
                regval = np.mean(squeeze(Tools.safe_slice(D[n,:], tidx, wsize, mode='hard', causal=True)))

                D[n,tidx], (K[n,tidx], B[n,tidx], E[n,tidx]), C[n,tidx], xt[n,:]\
                = linear_regression_with_delay(X[n,:], yt[n,:], tidx,
                                               dlrange=dlrange, reg={'weight':regweight, 'val':regval})

            Er[n,tidx] = E[n,tidx]/norm(yt[n,:])

        # multi linear regression
        A.append(op_linear_regression(xt, yt))

        # yt.fill(0)
        # for n in range(nbLoc):
        #     yt[n,:] = squeeze(Tools.safe_slice(Y[n,:], tidx+D[n,tidx], wsize, mode='soft'))
        # S.append(sensor_similarity(yt, nflag=False))

    return A, (K, B, E, Er), C, D
    # return A, (squeeze(K)[()], squeeze(B)[()], squeeze(E)[()], squeeze(Er)[()]),\
    #   squeeze(C)[()], squeeze(D)[()] #, squeeze(S)[()]



def linear_regression_with_delay(X, Y, tidx=None, dlrange=[-12,12], reg=None):
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

    # res[0]:slope, res[1]:intercept res[2,3]:residual error and relative error,
    # res[4]: noise variance
    res = zeros((5, dlrange[1] - dlrange[0])) 
    regerr = zeros(dlrange[1] - dlrange[0]) # regularization error

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
        res[:, n-dlrange[0]] = linear_regression(A, Y)

        if reg is not None: # compute the regularization error
            regerr[n-dlrange[0]] = sqrt(res[2,n-dlrange[0]]**2 + np.abs(n-reg['val'])**2 * reg['weight'])

    # the optimal delay is taken as the one minimizing the residual of LS
    if reg is None:
        nidx = np.argmin(res[2,:]) # argmin/argmax always return 0 if data contain nan
    else:
        nidx = np.argmin(regerr)

    dt = dlrange[0]+nidx

    if tidx is None:
        Xn = np.roll(X[:wsize], dt)
    else:
        Xn = squeeze(Tools.safe_slice(X, tidx-dt, wsize, mode='soft'))

    return dt, res[:, nidx], corr(Xn, Y), Xn





    # B = np.hstack([A, ones([A.shape[0],1])])
    # toto = inv(B.T @ B) @ (B.T @ Y)
    # return toto[:-1], toto[-1], norm(B @ toto - Y), norm(B @ toto - Y)/norm(Y)

        # tidx0 = max(0, tidx-int(wsize/2))
        # tidx1 = min(nbTime, tidx0+wsize)
        # x,y = X[tidx0:tidx1], Y[tidx0:tidx1] # data on the moving window
        # tidx2 = min(nbTime, tidx0+wsize+abs(t0))
        # yt = np.roll(Y[tidx0:tidx2], t0)[:(tidx1-tidx0)]
        # K[tidx], B[tidx] = estimate_linear_relation_reg(xt, yt)
        # W[tidx] = t0




def linear_regression_with_delay(X, Y, tidx=None, dlrange=[-12,12], reg=None):
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


def mw_estimate_DCOLR(X0, Y0, D0, wsize=24*10, **kwargs):
    """
    Moving window estimation of Delay, Correlation, and (Multivariate)
    Linear Regression (DCOLR).

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
    (k, b, e, er): k the slope and b the intercept in the relation yt = k*xt + b,
        e is the residual, and er is the relative residual e/norm(yt)
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

    regweight = kwargs['regweight'] if 'regweight' in kwargs else 1e-3

    nbLoc,nbTime = X.shape

    # quantities after compensation of thermal delay
    K = zeros((nbLoc, nbTime)) # slope
    B = zeros((nbLoc, nbTime)) # intercept
    S = zeros((nbLoc, nbTime)) # noise variance sigma
    C = zeros((nbLoc, nbTime)) # correlation
    E = zeros((nbLoc, nbTime)) # residual error
    Er = zeros((nbLoc, nbTime)) # relative residual error
    D = zeros((nbLoc, nbTime)) # thermal delay

    xt = zeros((nbLoc, wsize))
    yt = zeros((nbLoc, wsize))

    for tidx in range(nbTime):
        xt.fill(0); yt.fill(0) # reused for each tidx

        # estimation of the optimal delay and linear regression
        for n in range(nbLoc):
            # xt[n,:] = squeeze(Tools.safe_slice(X[n,:], tidx, wsize, mode='soft'))
            yt[n,:] = squeeze(Tools.safe_slice(Y[n,:], tidx, wsize, mode='soft'))

            if regweight is None or tidx<wsize:
                D[n,tidx], (K[n,tidx], B[n,tidx], E[n,tidx]), C[n,tidx], xt[n,:]\
                = linear_regression_with_delay(X[n,:], yt[n,:], tidx, dlrange=dlrange)
            else:
                # apriori value for the thermal delay
                # regval = np.median(squeeze(Tools.safe_slice(D[n,:], tidx, wsize, mode='hard', causal=True)))
                regval = np.mean(squeeze(Tools.safe_slice(D[n,:], tidx, wsize, mode='hard', causal=True)))

                D[n,tidx], (K[n,tidx], B[n,tidx], E[n,tidx]), C[n,tidx], xt[n,:]\
                = linear_regression_with_delay(X[n,:], yt[n,:], tidx,
                                               dlrange=dlrange, reg={'weight':regweight, 'val':regval})

            Er[n,tidx] = E[n,tidx]/norm(yt[n,:])

        # multi linear regression
        A.append(op_linear_regression(xt, yt))

        # yt.fill(0)
        # for n in range(nbLoc):
        #     yt[n,:] = squeeze(Tools.safe_slice(Y[n,:], tidx+D[n,tidx], wsize, mode='soft'))
        # S.append(sensor_similarity(yt, nflag=False))

    return A, (K, B, E, Er), C, D
    # return A, (squeeze(K)[()], squeeze(B)[()], squeeze(E)[()], squeeze(Er)[()]),\
    #   squeeze(C)[()], squeeze(D)[()] #, squeeze(S)[()]


def mw_estimate_delay(X0, Y0, wsize=24*10, dlrange=[-12,12], **kwargs):
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
    K = zeros((nbLoc, nbTime)) # slope
    B = zeros((nbLoc, nbTime)) # intercept
    E = zeros((nbLoc, nbTime)) # residual error
    Er = zeros((nbLoc, nbTime)) # relative residual error
    S = zeros((3, nbLoc, nbTime)) # noise variance sigma
    C = zeros((nbLoc, nbTime)) # correlation
    D = zeros((nbLoc, nbTime)) # thermal delay

    yt = zeros((nbLoc, wsize))

    for tidx in range(nbTime):
        yt.fill(0) # reused for each tidx

        # estimation of the optimal delay and linear regression
        for n in range(nbLoc):
            yt[n,:] = squeeze(Tools.safe_slice(Y[n,:], tidx, wsize, mode='soft'))

            D[n,tidx], toto, C[n,tidx], _\
              = linear_regression_with_delay(X[n,:], yt[n,:], tidx, dlrange=dlrange)

            K[n,tidx], B[n,tidx], E[n,tidx], Er[n,tidx], S[0,n,tidx], S[1,n,tidx], S[2,n,tidx] = toto

    return (K, B, E, Er, S), C, D
    


def safe_slice_old(X0, tidx, wsize, mode='hard', causal=False):

    if X0.ndim==1:
        X = X0[np.newaxis,:]
    elif X0.ndim==2:
        X = X0.copy()
    else:
        raise NotImplementedError()

    nbTime = X.shape[1]
    # assert(0<=tidx<nbTime)

    if mode=='soft':
        x = zeros((X.shape[0], wsize))

        if causal:
            x[:] = X[:, tidx0:tidx1]
        else:
            # virtual begining and ending positions
            tidxa = tidx-int(wsize/2)
            tidxb = tidxa+wsize

            if tidxa<nbTime and tidxb>0:
                # physical begining and ending positions
                tidx0 = max(0, tidxa)
                tidx1 = min(nbTime, tidxb)

                if tidx1==tidxb:
                    x[:, (tidx0-tidxa):] = X[:, tidx0:tidx1]
                else:
                    x[:, (tidx0-tidxa):(tidx1-tidxb)] = X[:, tidx0:tidx1]
    else:
        if causal: #causal version:
            tidx0 = max(0, tidx-wsize+1)
            tidx1 = min(nbTime, tidx+1)
        else:         # non-causal version:
            tidx0 = max(0, tidx-int(wsize/2))
            tidx1 = min(nbTime, tidx0+wsize)

        x = np.squeeze(X[:,tidx0:tidx1]) if X0.ndim==1 else X[:,tidx0:tidx1]
        # print(tidx0, tidx1, x)

    return x



def ARX_fit_circ(Yt, ny, Xt, nx, bflag=False):
    """
    Least-square fit of the ARX model:
        Y_t = \sum_{j=1}^p h_j * Y_{t-j} + \sum_{i=0}^{q-1} g_i * X_{t-i} + u_t

    This function compute the coefficients alpha_j and beta_i as well as the noise variance
    of z_t.

    Parameters
    ----------
    """
    assert(Xt.ndim==Yt.ndim==1)
    assert(Xt.size==Yt.size)

    # Construct the linear system corresponding to the convolutions operators
    Ay = scipy.linalg.circulant(Yt)[:,1:(ny+1)]  # this is \sum_{j=1}^{ny} alpha_j * Y_{t-j}
    # print(Ay)
    Ax = scipy.linalg.circulant(Xt)[:,:nx]  # this is \sum_{i=0}^{nx-1} beta_i * X_{t-i}
    
    A0 = np.hstack([Ay, Ax])
    A = A0
    
    # Drop the parts of boundary influence
    if bflag:
        nr, nc = A0.shape    
        nd = max(nx, ny) # or nx-1?
        assert(nr-nd>nd)
        A = A0[nd:, :]
        Yt = Yt[nd:]
        # A = A0[nd:nr-nd, :]
        # Yt = Yt[nd:nr-nd]
    
    toto = np.linalg.inv(A.T @ A) @ (A.T @ Yt)
    H = toto[:ny]
    G = toto[ny:]

    err = squeeze(A @ toto - Yt)
    return H, G, err, A



"""
State-space Decomposition Model with given transition matrices (SSDtm)
"""
class SSDtm(sm.tsa.statespace.MLEModel):
    """
    
    """
    def __init__(self, endog):
        # Model order
        self.polynomial_order = 2 # kord
        self.periodicity = 24 # pord
        self.power_order = 1 # lord
        
        k_states = k_posdef = 2

        # Initialize the statespace
        super(SSDtm, self).__init__(
            endog, k_states=k_states, k_posdef=k_posdef,
            initialization='approximate_diffuse',
            loglikelihood_burn=k_states
        )

        # Initialize the matrices

        kord = self.polynomial_order
        lord = self.power_order
        pord = self.periodicity
        
        self.Tcoeff = ImBk_coeff(self.polynomial_order)
        self.Scoeff = NSBl_coeff(self.power_order, self.periodicity)

        Self.Fm = vstack([self.Tcoeff, hstack([eye(self.polynomial_order-1), zeros((self.polynomial_order-1, 1))])])
        Self.Fd = vstack([self.Scoeff, hstack([eye(self.power_order*(self.periodicity-1)-1), zeros((self.power_order*(self.periodicity-1)-1, 1))])])

        Xdim = self.Fm.shape[0] + self.Fd.shape[0]
        F = zeros((Xdim, Xdim))
        F[:Fo.shape[0],:Fo.shape[0]] = Fo
        F[Fo.shape[0]:,Fo.shape[0]:] = Fd
        G = zeros((1,Xdim)); G[0,0]=1; G[0,kord]=1

        Q = zeros((Xdim, Xdim)); 
        Q[0,0] = sigmam2; Q[kord, kord] = sigmad2; R = 1e0
        # Q[0,0] = 1e-5; Q[kord, kord] = 1e-1; R = 1e-1

        # Observation matrix
        self.ssm['design'] = np.array([1, 0])
        # System state matrix
        self.ssm['transition'] = np.array([[1, 1],
                                       [0, 1]])
        self.ssm['selection'] = np.eye(k_states)

        # Cache some indices
        self._state_cov_idx = ('state_cov',) + np.diag_indices(k_posdef)

    @property
    def param_names(self):
        return ['sigma2.measurement', 'sigma2.level', 'sigma2.trend']

    @property
    def start_params(self):
        return [np.std(self.endog)]*3

    def transform_params(self, unconstrained):
        return unconstrained**2

    def untransform_params(self, constrained):
        return constrained**0.5

    def update(self, params, *args, **kwargs):
        params = super(LocalLinearTrend, self).update(params, *args, **kwargs)
        
        # Observation covariance
        self.ssm['obs_cov',0,0] = params[0]

        # State covariance
        self.ssm[self._state_cov_idx] = params[1:]

        

# Construct the system state matrix
# Non stationaray seasonal component

# kord = 1
# pord = 24
# lord = 1

# Tcoeff = T_coeff(kord)
# Scoeff = S_coeff(lord, pord)

# Fo = vstack([Tcoeff, hstack([eye(kord-1), zeros((kord-1, 1))])])
# Fd = vstack([Scoeff, hstack([eye(lord*(pord-1)-1), zeros((lord*(pord-1)-1, 1))])])


# Fo = vstack([Tcoeff, hstack([eye(kord-1), zeros((kord-1, 1))])])
# Fd = vstack([Scoeff, hstack([eye(lord*pord-1), zeros((lord*pord-1, 1))])])




# def ST_decomposition_Kalman(kord=2, pord=24, lord=1):
    
# Xdim = Fo.shape[0]+Fd.shape[0]
# F = zeros((Xdim, Xdim))
# F[:Fo.shape[0],:Fo.shape[0]] = Fo
# F[Fo.shape[0]:,Fo.shape[0]:] = Fd
# G = zeros((1,Xdim)); G[0,0]=1; G[0,kord]=1

# Q = zeros((Xdim, Xdim)); 
# Q[0,0] = sigmam2; Q[kord, kord] = sigmad2; R = 1e-1
# # Q[0,0] = 1e-5; Q[kord, kord] = 1e0; R = 1e0

# # Run Kalman
# from Seim import Kalman
# reload(Kalman)

# X0 = zeros(Xdim); S0 = eye(Xdim)
# # X0 = hstack([Ym[0]*ones(kord), Yt[0]*ones(lord*(pord-1))]); S0 = eye(Xdim)# * R
# # X0 = hstack([Ym[0]*ones(kord), Yt[0]*ones(lord*pord-1)]); S0 = eye(Xdim)# * R
# res = Kalman.Kalman_Filter(Yt, F, G, Q, R, X0, S0)

# LXtt, LPtt, LXtm, LPtm = res[0], res[1], res[2], res[3]

# # Kalman smoothing
# reload(Kalman)
# res_sm = Kalman.Kalman_Smoother(Yt, F, G, Q, R, X0, S0)

# LXtn, LPtn, LJt = res_sm[0], res_sm[1], res_sm[2]



# Kalman.py
# # For the following, their values are well defined for t>0, we add the initial state as the value at t=0.
# LXtm = [X0]  # X_{0|-1}
# LPtm = [P0]  # P_{0|-1}
# LSt = [St]  # S_{0}
# LKt = [Kt]  #K_{0}
# LEt = [Et]  # E_{0}

