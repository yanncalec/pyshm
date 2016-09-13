"""
.. module:: Kalman
   :platform: Unix, Windows
   :synopsis: NA

.. moduleauthor:: Han Wang <han@sivienn.com>


"""

import numpy as np
from numpy import newaxis, zeros, zeros_like, squeeze, asarray,\
     abs, linspace, fft, random, eye, ones, vstack
from numpy.linalg import norm, det, inv
import itertools
import numbers
from . import SPG, Tools


def Kalman_Filter(Y, A, B, Q, R, X0, P0, G=None):
    """
    Kalman Filter.

    KF on the dynamical system for t>=1:

    X_t = A_t X_{t-1} + G_t + U_t
    Y_t = B_t X_t + V_t

    Given the initial guess at t=0, KF computes the conditional mean X_{t|t}=Exp[X_t | Y_{1:t}].

    Parameters
    ----------
    Y : array
        data stream vectors, must be 2d array and the first dimension
        corresponds to time.
    A : array
        system state matrix, 2d or 3d. In case of 2d array the system transition
        matrix A is time-independent, in case of 3d array A is time-dependent and
        the 1st dimension corresponds to time.
    B : array
        observation matrix, 2d or 3d. In case of 3d array the 1st dimension
        corresponds to time.
    Q : array
        system noise covariance matix, 2d or 3d. In case of 3d array the 1st
        dimension corresponds to time.
    R : array
        observation noise covariance matrix, 2d or 3d. In case of 3d array the
        1st dimension corresponds to time.
    X0 : array
        guess for the initial state
    P0 : array
        guess for the covariance matrix of the initial state
    G : array
        input control vector, 1d or 2d. In case of 2d array the 1st dimension
        corresponds to time.

    Returns
    -------
    LXtt, LPtt, LXtm, LPtm, LEt, LSt, LKt, LLLHt,
    Lists of X_{t|t} for t=1...{Nt}, Nt is the length of Y
             P_{t|t}
             X_{t|t-1}
             P_{t|t-1}
             E_{t}
             S_{t}
             K_{t}
             Log-Likelihood_{t}
    """

    # Check dimension
    assert(Y.ndim == 2)
    assert(A.ndim == 3 or A.ndim == 2)
    assert(B.ndim == 3 or B.ndim == 2)
    assert(Q.ndim == 3 or Q.ndim == 2)
    assert(R.ndim == 3 or R.ndim == 2)

    dimX = B[0].shape[1] if B.ndim == 3 else B.shape[1]  # dimension of X
    dimY = B[0].shape[0] if B.ndim == 3 else B.shape[0]  # dimension of Y

    if G is not None:
        assert(G.ndim == 2 or G.ndim == 1)
        if G.ndim==2:
            assert(G[0].size == dimX)
        else:
            assert(G.size == dimX)

    assert(Y[0].size == dimY)

    Y = np.asarray(Y, dtype=np.float64) # Missing values: convert None to nan

    # Creat iterators
    AL = A if A.ndim == 3 else itertools.repeat(A, len(Y))
    BL = B if B.ndim == 3 else itertools.repeat(B, len(Y))
    QL = Q if Q.ndim == 3 else itertools.repeat(Q, len(Y))
    RL = R if R.ndim == 3 else itertools.repeat(R, len(Y))
    GL = G if G is not None and G.ndim == 2 else itertools.repeat(G, len(Y))

    # Initialization
    assert(norm(P0-P0.T) < 1e-8 and det(P0) >= 0)  # verify that P0 is symmetric and positive

    # Lists for keeping the results
    LXtt = [X0]  # X_{0|0}
    LPtt = [P0]  # P_{0|0}
    LmXt = [np.zeros_like(X0)]  # mean(X)_{0}

    LXtm = []  # X_{0|-1}
    LPtm = []  # P_{0|-1}
    LSt = []  # S_0
    LKt = []  # K_0
    LEt = []  # E_0
    LLLHt = []  # Log-Likelihood_0

    for t, At, Bt, Qt, Rt, Gt in zip(range(len(Y)), AL, BL, QL, RL, GL):
        # Prediction at t from Y_{1:t-1}
        Xtm = At @ LXtt[-1] if Gt is None else At @ LXtt[-1] + Gt  # X_{t,t-1}
        Ptm = At @ LPtt[-1] @ At.conjugate().T + Qt                   # P_{t,t-1}
        St = Bt @ Ptm @ Bt.conjugate().T + Rt                         # Cov(Epsilon_t)
        iSt = inv(St)
        Kt = Ptm @ Bt.conjugate().T @ iSt         # Gain

        # mX_{t}, this is optional and not used in KF
        mXt = At @ LmXt[-1] if Gt is None else At @ LmXt[-1] + Gt

        # Update
        if not np.isnan(Y[t]).all(): # if Y[t] is available
            Et = Y[t] - Bt @ Xtm    # Epsilon_{t}
            Xtt = Xtm + Kt @ Et     # X_{t,t}
            Ptt = Ptm - Kt @ Bt @ Ptm    # P_{t,t}
            # Log(Proba_(Y_t|Y_{1..t-1}))
            LLHt = -1/2 * (np.log(det(St)) + Et.conjugate().T @ iSt @ Et + dimY * np.log(2*np.pi))
        else:
            Et = np.nan     # Epsilon_{t}
            Xtt = Xtm.copy()    # X_{t,t}
            Ptt = Ptm.copy()    # P_{t,t}
            LLHt = 0

        # Save
        LXtt.append(Xtt)
        LPtt.append(Ptt)
        LmXt.append(mXt)
        LXtm.append(Xtm)
        LPtm.append(Ptm)
        LEt.append(Et)
        LSt.append(St)
        LKt.append(Kt)
        if t > 0:
            LLLHt.append(LLLHt[-1] + LLHt) # append the increment
        else:
            LLLHt.append(LLHt)

    # Pop the first element in these lists, such that all the outputs are for t=1..T
    LXtt.pop(0)
    LPtt.pop(0)
    LmXt.pop(0)

    return LXtt, LPtt, LXtm, LPtm, LEt, LSt, LKt, LmXt, LLLHt


def Kalman_Smoother(Y, A, B, Q, R, X0, P0, G=None):
    """
    Kalman Smoother.

    Unlike the KF, the Kalman Smoother compute the conditional expectation X_t
    given all observations from 1 to n:
        X_{t|n} = Exp[X_t | Y_{1:n}],  for 1 <= t <= n

    Parameters
    ----------
    same as in Kalman_Filter.

    Return
    ------
    LXtn, LPtn, LJt, res:
        List of X_{t|n}, P_{t|n}, J_t and the results of the Kalman filter.
    """

    # Run first the Kalman filter
    res = Kalman_Filter(Y, A, B, Q, R, X0, P0, G)

    LXtt, LPtt, LXtm, LPtm, LEt, LSt, LKt, LmXt, *_ = res

    # Lists for keeping the results
    LJt = []  # not defined for t == Nt-1 since Xtm == X_{Nt|Nt-1} is not available
    # Initialization
    LXtn = [LXtt[-1]]       # X_{t|n} with t=n is simply X{n|n}
    LPtn = [LPtt[-1]]

    for t in range(len(Y)-2, -1, -1):
        # short-hands
        AT = A[t].conjugate().T if A.ndim == 3 else A.conjugate().T
        Ptm, Xtm = LPtm[t+1], LXtm[t+1]

        Jt = LPtt[t] @ AT @ inv(Ptm)
        Xtn = LXtt[t] + Jt @ (LXtn[-1] - Xtm)

        if G is None:
            Ptn = LPtt[t] + Jt @ (LPtn[-1] - Ptm) @ Jt.conjugate().T
            # Ptn = LPtt[t] + Jt @ (LPtn[-1] - LPtm[t]) @ Jt.T
        else:
            if G.ndim == 2:
                toto = G[t][:, newaxis] @ LmXt[t][newaxis, :] @ AT
            else:
                toto = G[:, newaxis] @ LmXt[t][newaxis, :] @ AT
            toto += toto.conjugate().T

            Ptn = LPtt[t] + Jt @ (LPtn[-1] - Ptm + toto) @ Jt.conjugate().T

        LXtn.append(Xtn)
        LPtn.append(Ptn)
        LJt.append(Jt)

    return LXtn[::-1], LPtn[::-1], LJt[::-1], res


def Kalman_ARX(Ydata, Nh, Xdata, Ng, sigmaq2, sigmar2, X0=None, P0=None, cflag=True):
    """
    State-space solution for the ARX model:
        Y[t] = \sum_{i=1}^Nh h[t,i] Y[t-i] + \sum_{j=0}^{Ng-1} g[t,j] X[t-j] + c[t]
    with h, g and c following the Brownian motion:
        h[t,i] = h[t-1,i] + u[t],  for i=1..Nh
        g[t,j] = g[t-1,j] + u[t],  for j=0..Ng-1
        c[t] = c[t-1] + u[t]
    """
    assert(Nh>=0 and Ng>=0 and Nh+Ng>0)
    assert(len(Xdata) == len(Ydata))  # assume X and Y have the same length

    # dimension of the state vector
    Nq = Nh+Ng+1 if cflag else Nh+Ng

    A = eye(Nq)  # System state matrix

    # observation matrix
    By = Tools.construct_convolution_matrix(Ydata, Nh, tflag=False)  # tflag=False means Y[t] is removed from the convolution sum
    Bx = Tools.construct_convolution_matrix(Xdata, Ng, tflag=True)  # tflag=True means X[t] is used in the convolution sum
    B0 = np.hstack([By, Bx, ones((Bx.shape[0],1))]) if cflag else np.hstack([By, Bx])
    B = B0[:, newaxis, :] # dimension extension

    # Covariance matrix of the innovation noise, time-independent
    Q = eye(Nq) * sigmaq2 if isinstance(sigmaq2, numbers.Number) else np.diag(sigmaq2)
    assert(Q.shape[0] == Nq)
    # Covariance matrix of the observation noise
    R = np.atleast_2d(sigmar2) if isinstance(sigmar2, numbers.Number) else sigmar2[:, newaxis, newaxis]

    # Initial state
    if X0 is None:
        X0 = zeros(Nq)
    if P0 is None:
        P0 = eye(Nq)

    Y = np.atleast_2d(Ydata).T  # put Y into column vector form
    LXtn, LPtn, LJt, res = Kalman_Smoother(Y, A, B, Q, R, X0, P0)

    # Post-processing of results

    # Smoothing
    Xtn = np.asarray(LXtn)
    Ptn = np.asarray(LPtn)
    Ytn = np.sum(B0 * Xtn, axis=1)
    Etn = Ydata - Ytn # error of smoothing

    # Filtration
    Xtt = np.asarray(res[0])
    Ptt = np.asarray(res[1])
    Ytt = np.sum(B0 * Xtt, axis=1)
    Ett = Ydata - Ytt # error of filtration

    # Prediction
    Xtm = np.asarray(res[2])
    Ptm = np.asarray(res[3])
    Ytm = np.sum(B0 * Xtm, axis=1)
    Etm = Ydata - Ytm # error of prediction
    # Etm = squeeze(np.asarray(Res[2][4]))  # equivalent

    return (Xtn, Ptn, Ytn, Etn), (Xtt, Ptt, Ytt, Ett), (Xtm, Ptm, Ytm, Etm)


#### Utility functions for parameter estimation
def Kalman_ARX_wrapper(Y, Nh, X, Ng, sigmaq2, sigmar2, X0, P0, cflag=True):
    """
    Wrapper for the function Kalman_ARX.
    Use this function with FP_Parms_Estimation().
    """
    Res_tn, Res_tt, Res_tm = Kalman_ARX(Y, Nh, X, Ng, sigmaq2, sigmar2, X0=X0, P0=P0, cflag=cflag)

#     Xtn, Ptn, Etn = Res_tn
    Xtt, Ptt, Ytt, Ett = Res_tt
    Xtm, Ptm, Ytm, Etm = Res_tm

    # Use the tail to estimate sigmaq2 and sigmar2
    tail = int(Xtm.shape[0]/4) # use only a last part of the total length
    sidx = max(0, Xtm.shape[0]-tail)

    if isinstance(sigmaq2, numbers.Number):
        # if the input value of sigmaq2 is a number, set the output also to a number by taking the average
        sigmaq2 = np.mean(np.var(Xtm[sidx:,:], axis=0))
    else:
        sigmaq2 = np.var(Xtm[sidx:,:], axis=0)

    sigmar2 = np.var(Etm[sidx:])
    X0 = np.mean(Xtt[sidx:, :], axis=0)
    P0 = np.mean(Ptt[sidx:,], axis=0)

    return sigmaq2, sigmar2, X0, P0

# Example of inline lambda function
# ssfunc = lambda sigmaq2, sigmar2, X0, P0: \
#     Kalman_ARX_wrapper(Y, Nh, X, Ng, sigmaq2, sigmar2, X0, P0, cflag=True)

def FP_Parms_Estimation(func, sigmaq2=None, sigmar2=None, niter=100, verbose=False):
    """
    Fixed point parameter estimations.

    Parameters
    ----------
    func : function handle
        func takes as input sigmaq2, sigmar2, X0, P0 and returns the same group of values.
        X0 and P0 are respectively the initial state vector and the covariance matrix
    sigmaq2, sigmar2 : arrays or float
        initial values
    niter : integer
        number of FP iterations
    verbose : boolean
        print messages

    Returns
    -------
    sigmaq2, sigmar2, X0, P0
    """

    X0, P0 = None, None
    if sigmaq2 is None:
        sigmaq2 = 0
    else:
        sigmaq2 = np.zeros_like(sigmaq2)

    for k in range(niter):
        sigmaq2, sigmar2, X0, P0 = func(np.zeros_like(sigmaq2), sigmar2, X0, P0)
        # sigmaq2, sigmar2, X0, P0 = func(sigmaq2/10**6, sigmar2, X0, P0)

        if verbose:
            print('Iteration {}\nnorm(sigmaq2)={}\tsigmar2={}\tnorm(X0)={}'.format(k, norm(sigmaq2), sigmar2, norm(X0)))

    return sigmaq2, sigmar2, X0, P0


################



def Kalman_deconvolution_BM(Xdata, Ydata, Nq, sigmaq2, sigmar2, X0=None, P0=None):
    # System state and observation matrix
    A = eye(Nq)
    B0 = Tools.construct_convolution_matrix(Xdata, Nq, tflag=True)
    B = B0[:, newaxis, :] # dimension extension

    # Initial state
    if X0 is None:
        X0 = zeros(Nq)
    if P0 is None:
        P0 = eye(Nq)

    # Manualy setting the parameters
    if isinstance(sigmaq2, numbers.Number):
        Q = eye(Nq) * sigmaq2
    else:
        Q = np.diag(sigmaq2)

    if isinstance(sigmar2, numbers.Number):
        R = np.atleast_2d(sigmar2)
    else: # suppose R is 2d
        R = sigmar2[:, newaxis, newaxis]

    LXtn, LPtn, LJt, res = Kalman_Smoother(Ydata, A, B, Q, R, X0, P0)

    return LXtn, LPtn, res, B0


def Kalman_deconvolution(Tdata, Ydata, Cvec, sigmaq2=1e-4, sigmar2=1e-5, X0=None, P0=None):
    """
    X_{t,j} = C_j * X_{t-1,j} + U_{t,j}
    Y_t = \sum_j X_{t,j} T_{t-j} + V_t
    0 = \sum_j X_{t,j} + W_t
    """

    # System state and observation matrix
    A = np.diag(Cvec)
    Nq = len(Cvec)  # length of the convolution kernel

    B0 = Tools.construct_convolution_matrix(Tdata, Nq, tflag=True)

    B = zeros((B0.shape[0], 2, Nq))
    for t in range(B.shape[0]):
        B[t, 0, :] = B0[t, :]
        B[t, 1, :] = ones(Nq)

    Y = vstack([Ydata, zeros_like(Ydata)]).T
    # print(Y.shape)

    # Initial state
    if X0 is None:
        X0 = zeros(Nq)
    if P0 is None:
        P0 = eye(Nq)

    # Manualy setting the parameters
    if isinstance(sigmaq2, numbers.Number):
        Q = eye(Nq) * sigmaq2
    else:
        Q = np.diag(sigmaq2)

    # if isinstance(sigmar2, numbers.Number):
    #     R = np.atleast_2d(sigmar2)
    # else: # suppose R is 2d
    #     R = sigmar2[:, newaxis, newaxis]

    if isinstance(sigmar2, numbers.Number):
        R = sigmar2 * eye(2)
    else: # suppose R is 2d
        R = sigmar2[:, newaxis, newaxis] * eye(2)

    LXtn, LPtn, LJt, res = Kalman_Smoother(Y, A, B, Q, R, X0, P0)

    return LXtn, LPtn, res, B0



#### TODO

def Kalman_MLE_cnl(Y, A, B, rflag=True, maxIter=1000,
                   sigmaq2=None, sigmar2=None,
                   pinit=None, verbose=1):
    """
    Maximum likelihood estimation of constant noise level in Kalman filter.

    The constant noise level means that the variance of the system state noise sigmaq^2
    and the observation noise sigmar^2 remain constant through time.

    Parameters
    ----------
    same as in Kalman_Filter.
    maxIter : integer
        maximum number of iterations.

    verbose : integer
        1: silent, >2: print messages.

    Return
    ------
    Estimated parameters
    """

    assert(A.ndim == 3 or A.ndim == 2)
    assert(B.ndim == 3 or B.ndim == 2)

    dimX = B[0].shape[1] if B.ndim == 3 else B.shape[1]  # dimension of X
    dimY = B[0].shape[0] if B.ndim == 3 else B.shape[0]  # dimension of Y
    dimD = 2 if rflag else dimX + dimY  # dimension of the derivative (number of parameters)

    # Initial state for KF
    X0 = np.zeros(dimX)
    P0 = np.eye(dimX)

    # Objective function using KF
    def funObj(x):
        Q = np.eye(dimX)*x[0] if rflag else np.diag(x[:dimX])
        R = np.eye(dimY)*x[1] if rflag else np.diag(x[dimX:])
        res = Kalman_Filter_LH(Y, A, B, Q, R, X0, P0, rflag=rflag)
        LLHt = np.asarray(res[8])[-1] # -1 * log-Likelihood
        dLLHt = np.asarray(res[9])[-1] # derivative of LLHt

        return LLHt, dLLHt

    # Constraint of positivity
    funProj = lambda x : SPG.projectBound(x, 1e-8)

    # Options for SPG
    spg_options = SPG.default_options
    spg_options.maxIter = maxIter
    spg_options.curvilinear = True
    spg_options.interp = 2
    spg_options.optTol = 1e-11
    spg_options.progTol = 1e-11
    spg_options.verbose = verbose # be silent

    # Initialization
    if pinit == None:
        pinit = np.random.rand(dimD)
    elif isinstance(pinit, numbers.Number):
        pinit = np.ones(dimD) * pinit
    else:
        pinit = np.asarray(pinit)

    # Minimization by SPG method
    Parms, vF = SPG.SPG(funObj, funProj, pinit, spg_options)

    return Parms


def Kalman_Filter_LH(Y, A, B, Q, R, X0, P0, G=None, rflag=True):
    """
    Kalman Filter with computation of the likelihood function.

    KF on the dynamical system:

    X_t = A_t X_{t-1} + G_t + U_t, for t>=1
    Y_t = B_t X_t + V_t, for t>=0

    Given the initial guess at t=0, KF computes the conditional mean X_{t|t}=Exp[X_t | Y_{0:t}] for t>=0.

    Parameters
    ----------
    Y : array
        data stream vectors, in case of 2d array the first dimension corresponds to time
    A : array
        system state matrix, 2d or 3d. In case of 2d array the system transition matrix A is time-invariant, in case of 3d array A is time-variant and the 1st dimension corresponds to time
    B : array
        observation matrix, 2d or 3d. In case of 3d array the 1st dimension corresponds to time
    Q : array
        system noise covariance matix, assumed to be diagonal for computation of derivative.
    R : array
        observation noise covariance matrix, assumed to be diagonal for computation of derivative.
    X0 : array
        guess for the initial state
    P0 : array
        guess for the covariance matrix of the initial state
    G : array
        input control vector starting from t=1
    rflag : boolean
        if True use the reduced set of parameters in the computation of derivatives,
        which means Q = sigma_q^2 * Id and R = sigma_r^2 * Id. Otherwise use
        diagonal matrices: Q = diag([sigma_q1^2,... sigma_qn^2]), R = ...

    Returns
    -------
    LXtt, LPtt, LXtm, LPtm, LEt, LSt, LKt, LLLHt, LdLLHt
    Lists of X_{t|t} for t=0...{Nt-1}, Nt is the length of Y
             P_{t|t} for t=0...{Nt-1}
             X_{t|t-1} for t=0...{Nt-1}, with X_{0|-1} == X0
             P_{t|t-1} for t=0...{Nt-1}, with P_{0|-1} == P0
             E_{t} for t=0...{Nt-1}
             S_{t} for t=0...{Nt-1}
             K_{t} for t=0...{Nt-1}
             Log-Likelihood_{t} for t=0...{Nt-1}
             derivative of Log-Likelihood_{t} for t=0...{Nt-1}
    """

    # Conventions:
    # Total length of Kalman filtering is len(Y)=Nt
    # Time index of Y: t=0:Nt-1
    # Prediction at {0|0} is computed from the initialization and Y0

    assert(A.ndim == 3 or A.ndim == 2)
    assert(B.ndim == 3 or B.ndim == 2)

    dimX = B[0].shape[1] if B.ndim == 3 else B.shape[1] # dimension of X
    dimY = B[0].shape[0] if B.ndim == 3 else B.shape[0] # dimension of Y
    dimD = 2 if rflag else dimX + dimY  # dimension of the derivative (number of parameters)

    assert(dimX == np.atleast_2d(Q).shape[0]) # check dimension of the matrix Q
    assert(dimY == np.atleast_2d(R).shape[0]) # check dimension of the matrix R

    if G != None:
        assert(len(G) == len(Y))

    Y = np.array(Y, dtype=np.float64) # Missing values: convert None to nan

    def dQ(i):
        """
        i-th partial derivative of Q=cov(U), i=0..,dimD-1
        """
        if rflag: # if the set of parameters is reduced
            if i == 0:
                return np.eye(dimX)
            else:
                return np.zeros((dimX,dimX))
        else:
            dM = np.zeros((dimX,dimX))
            if 0 <= i < dimX:
                dM[i, i] = 1
            return dM

    def dR(i):
        """
        i-th partial derivative of R=cov(V)
        """
        if rflag:
            if i == 1:
                return np.eye(dimY)
            else:
                return np.zeros((dimY,dimY))
        else:
            dM = np.zeros((dimY,dimY))
            if dimX <= i < dimX+dimY:
                dM[i-dimX, i-dimX] = 1
            return dM

    # Initialization
    assert(norm(P0-P0.T) < 1e-8 and det(P0)>=0) # verify that P0 is symmetric and positive

    B0 = B[0] if B.ndim == 3 else B # matrix B at t=0
    St = B0 @ P0 @ B0.T + R
    iSt = inv(St)
    dSt = np.asarray([dR(i) for i in range(dimD)])
    # dSt = np.atleast_1d(np.squeeze(np.asarray([dR(i) for i in range(dimD)])))
    Kt = P0 @ B0.T @ iSt
    Et = Y[0] - B0 @ X0

    Xinit = X0 + Kt @ Et
    Pinit = P0 - Kt @ B0 @ P0

    # Lists for keeping the results
    # For the following, their values are well defined for t>0 and we extend the value at t=0
    LXtm = [X0]  # X_{0|-1}
    LPtm = [P0]  # P_{0|-1}
    LSt = [St]  # S_0
    LKt = [Kt]  # K_0
    LEt = [Et]  # E_0
    # LXtm = []; LPtm = []; LSt = []; LKt = []; LEt = []
    #
    # For the following, their values are well defined for t>=0
    LXtt = [Xinit] # X_{0|0}
    LPtt = [Pinit] # P_{0|0}
    LmXt = [X0]  # mean(X)_{t} for t=0...{Nt-1}

    # Initialization of derivatives
    dXtt = [- P0 @ B0.T @ iSt @ dR(i) @ iSt @ Et for i in range(dimD)] # of dimension dimD * dimX
    dPtt = [P0 @ B0.T @ iSt @ dR(i) @ iSt @ B0 @ P0 for i in range(dimD)] # of dimension dimD * dimX * dimX

    # Initialization of likelihood
    LLLHt = [1/2 * (np.log(np.linalg.det(St)) + Et @ iSt @ Et + dimY * np.log(2*np.pi))] # Log-Likelihood
    LdLLHt = [np.asarray([1/2 * (np.trace(iSt @ dSt[i]) - (iSt @ Et) @ dSt[i] @ (iSt @ Et))
                                     for i in range(dimD)])] # derivative of Log-Likelihood

    # Creat iterators
    AL = A if A.ndim == 3 else itertools.repeat(A, len(Y))
    BL = B[1:,:,:] if B.ndim == 3 else itertools.repeat(B, len(Y))

    for t, At, Bt in zip(range(1, len(Y)), AL, BL):
        ## Part I: For Kalman filter
        # Prediction at t from Y_{1:t-1}
        Xtm = At @ LXtt[-1] if G is None else At @ LXtt[-1] + G[t]  # X_{t,t-1}
        Ptm = At @ LPtt[-1] @ At.T + Q                   # P_{t,t-1}
        St = Bt @ Ptm @ Bt.T + R                         # Cov(Epsilon_t)
        iSt = inv(St)
        Kt = Ptm @ Bt.T @ iSt         # Gain

        # mX_{t}, this is optional and not used in KF
        mXt = At @ LmXt[-1] if G is None else At @ LmXt[-1] + G[t]

        # Update
        if not np.isnan(Y[t]).all():
            # assert(not np.isnan(np.sum(Bt)))
            # assert(not np.isnan(np.sum(St)))
            # assert(not np.isnan(np.sum(Kt)))

            Et = Y[t] - Bt @ Xtm    # Epsilon_{t}
            Xtt = Xtm + Kt @ Et     # X_{t,t}
            Ptt = Ptm - Kt @ Bt @ Ptm    # P_{t,t}
            LLHt = 1/2 * (np.log(det(St)) + Et @ iSt @ Et + dimY * np.log(2*np.pi)) # Log-likelihood Proba_(Y_t|Y_{0..t-1})
        else:
            Et = np.nan     # Epsilon_{t}
            Xtt = Xtm.copy()    # X_{t,t}
            Ptt = Ptm.copy()    # P_{t,t}
            LLHt = 0

        # Save
        LXtt.append(Xtt)
        LPtt.append(Ptt)
        LmXt.append(mXt)
        LXtm.append(Xtm)
        LPtm.append(Ptm)
        LEt.append(Et)
        LSt.append(St)
        LKt.append(Kt)
        LLLHt.append(LLLHt[-1] + LLHt) # append the increment

        # print(At.shape, Ptm.shape, Kt.shape, Ptt.shape, LXtt[-1].shape)

        ## Part II: For the derivatives
        # Prediction
        dXtm = [At @ dXtt[i] for i in range(dimD)]
        dPtm = [At @ dPtt[i] @ At.T + dQ(i) for i in range(dimD)]
        dSt = [Bt @ dPtm[i] @ Bt.T + dR(i) for i in range(dimD)]
        dKt = [dPtm[i] @ Bt.T @ iSt - Ptm @ Bt.T @ iSt @ dSt[i] @ iSt for i in range(dimD)]

        # Update
        if not np.isnan(Y[t]).all():
            dEt = [-Bt @ dXtm[i] for i in range(dimD)]
            dXtt = [dXtm[i] + dKt[i] @ Et + Kt @ dEt[i] for i in range(dimD)]
            dPtt = [-dKt[i] @ Bt @ Ptm + dPtm[i] - Kt @ Bt @ dPtm[i] for i in range(dimD)]
            dLLHt = np.asarray([1/2 * (np.trace(iSt @ dSt[i]) - (iSt @ Et) @ dSt[i] @ (iSt @ Et)
                                           + 2 * dEt[i] @ iSt @ Et) for i in range(dimD)]) # increment
            # dLLHt = dXtm
        else:
            dXtt = dXtm.copy()
            dPtt = dPtm.copy()
            dLLHt = np.zeros(dimD)

        # Derivative of Log-Likelihood
        LdLLHt.append(LdLLHt[-1] + dLLHt)
        # LdLLHt.append(dLLHt)

    return LXtt, LPtt, LXtm, LPtm, LEt, LSt, LKt, LmXt, LLLHt, LdLLHt


def Kalman_deconvolution_old(Xdata, Ydata, Cvec, Mvec, sigmaq2=1e-4, sigmar2=1e-5, X0=None, P0=None):

    # System state and observation matrix
    A = np.diag(Cvec)
    Nq = len(Cvec)  # length of the convolution kernel
    Gvec = (1-Cvec) * Mvec

    B0 = Tools.construct_convolution_matrix(Xdata, Nq, tflag=True)
    B = B0[:, newaxis, :] # dimension extension

    # Initial state
    if X0 is None:
        X0 = zeros(Nq)
    if P0 is None:
        P0 = eye(Nq)

    # Manualy setting the parameters
    if isinstance(sigmaq2, numbers.Number):
        Q = eye(Nq) * sigmaq2
    else:
        Q = np.diag(sigmaq2)

    if isinstance(sigmar2, numbers.Number):
        R = np.atleast_2d(sigmar2)
    else: # suppose R is 2d
        R = sigmar2[:, newaxis, newaxis]

    LXtn, LPtn, LJt, res = Kalman_Smoother(Ydata, A, B, Q, R, X0, P0, G = Gvec)

    return LXtn, LPtn, res, B0
