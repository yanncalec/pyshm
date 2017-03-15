"""
.. module:: Kalman
   :platform: Unix, Windows
   :synopsis: NA

.. moduleauthor:: Han Wang <han@sivienn.com>
"""

import numpy as np
import numpy.linalg as la
import itertools
import numbers
from . import Tools
# from . import SPG, Tools


class Kalman:
    transition_matrices = None
    observation_matrices = None
    transition_covariance = None
    observation_covariance = None
    control_vector = None
    init_state = None
    init_covariance = None

    observation = None
    dimsys = None
    dimobs = None

    def __init__(self, Y, A, B, G=None, Q=None, R=None, X0=None, P0=None):
        assert(Y.ndim == 2)
        assert(A.ndim == 3 or A.ndim == 2)
        assert(B.ndim == 3 or B.ndim == 2)

        self.dimsys = B[0].shape[1] if B.ndim == 3 else B.shape[1]
        self.dimobs = B[0].shape[0] if B.ndim == 3 else B.shape[0]

        assert Y.shape[0] == self.dimobs

        self.observation = np.asarray(Y, dtype=np.float64)  # Missing values: convert None to nan
        self.transition_matrices = A
        self.observation_matrices = B

        if G is not None:
            assert(G.ndim == 2 or G.ndim == 1)
            if G.ndim==2:
                assert(G[0].size == self.dimsys)
            else:
                assert(G.size == self.dimsys)
        self.control_vector = G

        # Covariance matrix of the innovation noise, time-independent
        if isinstance(Q, numbers.Number) and Q>0:
            self.transition_covariance = np.eye(self.dimsys) * Q
        elif isinstance(Q, np.ndarray):
            # check is simplified here
            if Q.ndim==1:
                assert(len(Q)) == self.dimsys
                self.transition_covariance = np.diag(Q)
            elif Q.ndim==2:
                assert(Tools.issymmetric(Q) and Q.shape[0]==self.dimsys)
                assert(Tools.ispositivedefinite(Q))
                self.transition_covariance = Q
            elif Q.ndim==3:
                assert(Q.shape[1]==Q.shape[2]==self.dimsys)
                self.transition_covariance = Q
            else:
                raise TypeError("Q must be a 1d or 2d array")
        elif Q is None:
            self.transition_covariance = None
        else:
            pass
            # raise TypeError('Q must be a number or an array')

        # Covariance matrix of the observation noise
        if isinstance(R, numbers.Number) and R>0:
            self.observation_covariance = np.eye(self.dimobs) * R
        elif isinstance(R, np.ndarray):
            # check is simplified here
            if R.ndim==1:
                assert(len(R)) == self.dimobs
                self.observation_covariance = np.diag(R)
            elif R.ndim==2:
                assert(Tools.issymmetric(R) and R.shape[0]==self.dimobs)
                assert(Tools.ispositivedefinite(R))
                self.observation_covariance = R
            elif R.ndim==3:
                assert(R.shape[1]==R.shape[2]==self.dimobs)
                self.observation_covariance = R
            else:
                raise TypeError('R must be a 1d or 2d array')
        elif R is None:
            self.observation_covariance = None
        else:
            pass
            # raise TypeError('R must be a number or an array')

        if isinstance(X0, numbers.Number):
            self.init_state = np.ones((self.dimsys, 1)) * X0
        elif isinstance(X0, np.ndarray):
            assert(X0.ndim == 2 and X0.shape[0] == self.dimsys)
            self.init_state = X0
        elif X0 is None:
            self.init_state = None
        else:
            pass
            # raise TypeError('X0 must be a number or None')

        if isinstance(P0, numbers.Number) and P0>0:
            self.init_covariance = np.eye(self.dimsys) * P0
        elif isinstance(P0, np.ndarray):
            assert(Tools.issymmetric(P0) and P0.shape[0]==self.dimsys)
            # assert(Tools.ispositivedefinite(P0))
            self.init_covariance = P0
        elif P0 is None:
            self.init_covariance = None
        else:
            pass
            # raise TypeError('P0 must be a positive number or a symmetric matrix')

    def parms_estimation(self):
        if self.init_state is None or self.init_covariance is None or self.transition_covariance is None or self.observation_covariance is None:
            raise NotImplementedError('Parameter estimation')

    def filter(self):
        return Kalman_filter(self.observation,
                             self.transition_matrices,
                             self.observation_matrices,
                             self.control_vector,
                             self.transition_covariance,
                             self.observation_covariance,
                             self.init_state,
                             self.init_covariance)

    def smoother(self):
        return Kalman_smoother(self.observation,
                               self.transition_matrices,
                               self.observation_matrices,
                               self.control_vector,
                               self.transition_covariance,
                               self.observation_covariance,
                               self.init_state,
                               self.init_covariance)


def Kalman_filter(Y, A, B, G, Q, R, X0, P0):
    """Kalman filter.

    Consider the dynamical system (for t>=1):
        X_t = A_t X_{t-1} + G_t + U_t
        Y_t = B_t X_t + V_t
    Given the initial guess at t=0, Kalman filter computes the conditional mean
        X_{t|t}=Exp[X_t | Y_{1:t}].

    Args:
        Y (2d array): observation vectors, must be 2d array and each row corresponds to an observation (ie, the 1st dimension corresponds to time).
        A (array): system state matrix, 2d or 3d. In case of 2d array the system transition matrix is time-independent, in case of 3d array it is time-dependent and the 1st dimension corresponds to time.
        B (array): observation matrix, 2d or 3d. In case of 3d array the 1st dimension corresponds to time.
        G (array): input control vector, 1d or 2d. In case of 2d array the 1st dimension corresponds to time. Set G to None if there is no control vector.
        Q (array): system noise covariance matix, 2d or 3d. In case of 3d array the 1st dimension corresponds to time.
        R (array): observation noise covariance matrix, 2d or 3d. In case of 3d array the 1st dimension corresponds to time.
        X0 (array): guess for the initial state
        P0 (array): guess for the covariance matrix of the initial state

    Returns:
        LXtt: X_{t|t} for t=1...{Nt}, Nt is the length of Y
        LPtt: P_{t|t}
        LXtm: X_{t|t-1}
        LPtm: P_{t|t-1}
        LEt: E{t}
        LSt: S{t}
        LKt: K{t}
        LLLHt: Log-Likelihood_{t}
    """

    # dimX = B[0].shape[1] if B.ndim == 3 else B.shape[1]  # dimension of X
    dimY = B[0].shape[0] if B.ndim == 3 else B.shape[0]  # dimension of Y
    Nt = Y.shape[1]  # length of observation

    # Creat iterators
    AL = A if A.ndim == 3 else itertools.repeat(A, Nt)
    BL = B if B.ndim == 3 else itertools.repeat(B, Nt)
    QL = Q if Q.ndim == 3 else itertools.repeat(Q, Nt)
    RL = R if R.ndim == 3 else itertools.repeat(R, Nt)
    GL = G if G is not None and G.ndim == 2 else itertools.repeat(G, Nt)

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

    for t, At, Bt, Qt, Rt, Gt in zip(range(Nt), AL, BL, QL, RL, GL):
        # Prediction at t from Y_{1:t-1}
        Xtm = At @ LXtt[-1] if Gt is None else At @ LXtt[-1] + Gt  # X_{t,t-1}
        Ptm = At @ LPtt[-1] @ At.conjugate().T + Qt                # P_{t,t-1}
        St = Bt @ Ptm @ Bt.conjugate().T + Rt                      # Cov(Epsilon_t)
        iSt = la.inv(St)
        Kt = Ptm @ Bt.conjugate().T @ iSt         # Gain

        # mX_{t}, this is optional and not used in KF
        mXt = At @ LmXt[-1] if Gt is None else At @ LmXt[-1] + Gt

        # Update
        if not np.isnan(Y[:,t]).any(): # if Y[t] is available
            Et = Y[:,[t]] - Bt @ Xtm    # Epsilon_{t}
            # assert(Et.shape[1]==1)
            Xtt = Xtm + Kt @ Et     # X_{t,t}
            Ptt = Ptm - Kt @ Bt @ Ptm    # P_{t,t}
            # Log(Proba_(Y_t|Y_{1..t-1}))
            LLHt = -1/2 * (np.log(la.det(St)) + Et.conjugate().T @ iSt @ Et + dimY * np.log(2*np.pi))
        else:
            Et = np.nan * np.zeros((dimY,1))     # Epsilon_{t}
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


def Kalman_smoother(Y, A, B, G, Q, R, X0, P0):
    """Kalman Smoother.

    Unlike Kalman filter, Kalman smoother compute the conditional expectation X_t
    given all observations from 1 to n:
        X_{t|n} = Exp[X_t | Y_{1:n}],  for 1 <= t <= n

    Args:
        same as filter.
    Returns:
        LXtn, LPtn, LJt, res: List of X_{t|n}, P_{t|n}, J_t and the results of the Kalman filter.
    """

    # Run first the Kalman filter
    res = Kalman_filter(Y, A, B, G, Q, R, X0, P0)

    LXtt, LPtt, LXtm, LPtm, LEt, LSt, LKt, LmXt, *_ = res

    # Lists for keeping the results
    LJt = []  # not defined for t == Nt-1 since Xtm == X_{Nt|Nt-1} is not available
    # Initialization
    LXtn = [LXtt[-1]]       # X_{t|n} with t=n is simply X{n|n}
    LPtn = [LPtt[-1]]
    Nt = Y.shape[1]

    for t in range(Nt-2, -1, -1):
        # short-hands
        AT = A[t].conjugate().T if A.ndim == 3 else A.conjugate().T
        Ptm, Xtm = LPtm[t+1], LXtm[t+1]

        Jt = LPtt[t] @ AT @ la.inv(Ptm)
        Xtn = LXtt[t] + Jt @ (LXtn[-1] - Xtm)

        if G is None:
            Ptn = LPtt[t] + Jt @ (LPtn[-1] - Ptm) @ Jt.conjugate().T
            # Ptn = LPtt[t] + Jt @ (LPtn[-1] - LPtm[t]) @ Jt.T
        else:
            if G.ndim == 2:
                toto = G[t][:, np.newaxis] @ LmXt[t][np.newaxis, :] @ AT
            else:
                toto = G[:, np.newaxis] @ LmXt[t][np.newaxis, :] @ AT
            toto += toto.conjugate().T

            Ptn = LPtt[t] + Jt @ (LPtn[-1] - Ptm + toto) @ Jt.conjugate().T

        LXtn.append(Xtn)
        LPtn.append(Ptn)
        LJt.append(Jt)

    return LXtn[::-1], LPtn[::-1], LJt[::-1], res


#### obsolete ####

def FP_Parms_Estimation(func, sigmaq2=None, sigmar2=None, niter=100, verbose=False):
    """Fixed point parameter estimations.

    Args:
        func (function handle): func takes as input sigmaq2, sigmar2, X0, P0 and returns the same group of values. X0 and P0 are respectively the initial state vector and the covariance matrix
        sigmaq2, sigmar2 (arrays or float): initial values
        niter (int): number of FP iterations
        verbose (bool): print messages
    Returns:
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


def Kalman_Deconv(Ydata, lagx, Xdata, sigmaq2, sigmar2, X0=None, P0=None, constflag=True):
    """
    State-space solution for the convolution model:
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
