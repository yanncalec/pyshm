import numpy as np
import scipy
from scipy import optimize
from numpy import newaxis
from numpy.linalg import inv, norm, det
import warnings

class Kriging:
    """
    Kriging for the model:

    y(t) = f(x(t)) + g(t) + e(t)

    t is the time variable, x(t) is the system state (vector), y(t) is the system
    output (scalar), f, g and e are Gaussian process.

    Parameters:
    ----------
    Tobs: list
        Timestramp of observations.
    Xobs: numpy.ndarray
        Observations of system state covariates at the timestramps of Tobs. Each
        observation is a n-dimensional vector, each column corresponds to one
        observation.
    Yobs: numpy.ndarray
        Observations of system output at the timestramps of Tobs. Each observation
        is a scalar.

    Returns:
    -------


    """
    nbThetaa = 2 # by default use first order polynomial for mu
    nbThetas = 6 # number of parameters in the set theta^s
    powcst = 2 # power constant in the exponential kernel of f and g

    # Naming convention for the set of parameters:
    # x[0]: sigmaf^2, x[1]: lcf^2, x[2]: betaf^2, x[3]: sigmag^2, x[4]: lcg^2, x[5]: sigmae^2,
    # x[6]..x[-1]: alpha[0]..alpha[P]

    # Covariance matrices of the vectors f,g,and y at observation points
    def Kfoofunc(self, x):
        return x[0] * np.exp(-self.XmXp/x[1]) + x[2]*self.XpX

    def Kgoofunc(self, x):
        return x[3] * np.exp(-self.TmTp/x[4])

    def Kyoofunc(self, x):
        return self.Kfoofunc(x) + self.Kgoofunc(x) + x[5] * np.eye(self.nbObs)

    def mufunc(self, x):
        # toto = np.asarray([x[self.nbThetas+p] * self.Tobs**p for p in range(self.nbThetaa)]).sum(axis=0)
        # print(toto.shape)
        return np.asarray([x[self.nbThetas+p] * self.Tobs**p for p in range(self.nbThetaa)]).sum(axis=0)

    def dKyoofunc(self, x):
        return [np.exp(-self.XmXp/x[1]),
                x[0] * np.exp(-self.XmXp/x[1]) * self.XmXp/(x[1]**2),
                self.XpX,
                np.exp(-self.TmTp/x[4]),
                x[3] * np.exp(-self.TmTp/x[4]) * self.TmTp/(x[4]**2),
                np.eye(self.nbObs)]

    def d2Kyoofunc(self, x):
        d2Kyoo = [[np.zeros_like(self.XmXp) for j in range(self.nbThetas)] for i in range(self.nbThetas)]

        d2Kyoo[0][1] = np.exp(-self.XmXp/x[1]) * self.XmXp / (x[1]**2)
        d2Kyoo[1][0] = np.copy(d2Kyoo[0][1])
        d2Kyoo[1][1] = x[0] * np.exp(-self.XmXp/x[1]) * (self.XmXp**2/(x[1]**4) - 2*self.XmXp/(x[1]**3))
        d2Kyoo[3][4] = np.exp(-self.TmTp/x[4]) * self.TmTp / (x[4]**2)
        d2Kyoo[4][3] = np.copy(d2Kyoo[3][4])
        d2Kyoo[4][4] = x[3] * np.exp(-self.TmTp/x[4]) * (self.TmTp**2/(x[4]**4) - 2*self.TmTp/(x[4]**3))

        return d2Kyoo

    def func(self, x):
        """
        -1*log-likelihood function (with irrelevant constant removed)
        """
        Kyoo = self.Kyoofunc(x)
        mYobs = self.mufunc(x)
        # return -1/2 * (np.log(np.linalg.det(Kyoo)) + (self.Yobs - mYobs) @ (np.linalg.solve(Kyoo, self.Yobs - mYobs))) # log-like
        return 1/2 * (np.log(np.linalg.det(Kyoo)) + (self.Yobs - mYobs) @ (np.linalg.solve(Kyoo, self.Yobs - mYobs))) # -1*log-like

    def jac(self, x):
        """
        Jacobian of the log-likelihood function
        """
        Kyoo = self.Kyoofunc(x)
        # print(np.linalg.cond(Kyoo))
        Kyooinv = np.linalg.inv(Kyoo)
        dKyoo = self.dKyoofunc(x)
        mYobs = self.mufunc(x)

        # first part: d J/d thetas
        xi = Kyooinv @ (self.Yobs - mYobs)
        xixit = xi[:,newaxis] * xi[newaxis,:]
        ds = -1/2 * np.asarray([np.trace((xixit - Kyooinv) @ dKyoo[n])
                           for n in range(self.nbThetas)])

        # second part: d J/d thetaa
        da = -(self.Yobs - mYobs) @ (Kyooinv @ self.dmuda)

        return np.concatenate((ds,da))

    def hess(self, x):
        """
        Hessian of the log-likelihood function
        """
        Kyoo = self.Kyoofunc(x)
        Kyooinv = np.linalg.inv(Kyoo)
        dKyoo = self.dKyoofunc(x)
        d2Kyoo = self.d2Kyoofunc(x)
        mYobs = self.mufunc(x)

        A = [[dKyoo[i] @ Kyooinv @ dKyoo[j] for j in range(self.nbThetas)] for i in range(self.nbThetas)]

        xi = Kyooinv @ (self.Yobs - mYobs)
        xixit = xi[:,newaxis] * xi[newaxis,:]

        H_thetass = -1/2 * np.asarray([[np.trace((xixit - Kyooinv) @ (d2Kyoo[i][j] - A[i][j]) - xixit @ A[j][i])
              for j in range(self.nbThetas)] for i in range(self.nbThetas)])

        H_thetaaa = self.dmuda.T @ (Kyooinv @ self.dmuda)
        H_thetaas = np.asarray([[(self.Yobs - mYobs) @ (Kyooinv @ dKyoo[j]) @ (Kyooinv @ self.dmuda[:,i]) for j in range(self.nbThetas)] for i in range(self.nbThetaa)])
        # print(H_thetass.shape, H_thetaas.shape, H_thetaaa.shape)
        H = np.r_[np.c_[H_thetass, H_thetaas.T],np.c_[H_thetaas, H_thetaaa]]
        return (H + H.T)/2 # force symmetry


    def get_parms(self):
        parmsdict = {'sigmaf2': self.sigmaf2,
               'lcf2': self.lcf2,
               'betaf2': self.betaf2,
               'sigmag2': self.sigmag2,
               'lcg2': self.lcg2,
               'sigmae2': self.sigmae2,
               'alpha': self.alpha}

        return np.r_[self.sigmaf2, self.lcf2, self.betaf2, self.sigmag2, self.lcg2, self.sigmae2, self.alpha], parmsdict


    def set_parms(self, x):
        self.sigmaf2 = x[0]
        self.lcf2 = x[1]
        self.betaf2 = x[2]
        self.sigmag2 = x[3]
        self.lcg2 = x[4]
        self.sigmae2 = x[5]
        self.alpha = x[6:]

    def set_parms_dic(self, pdic):
        self.sigmaf2 = pdic['sigmaf2']
        self.lcf2 = pdic['lcf2']
        self.betaf2 = pdic['betaf2']
        self.sigmag2 = pdic['sigmag2']
        self.lcg2 = pdic['lcg2']
        self.sigmae2 = pdic['sigmae2']
        self.alpha = pdic['alpha']


    def parameters_estimation(self, x, maxiter=50000, xtol=1e-3, gtol=1e-3, step=1e-3, c1=1e-4, c2=1-1e-1, verbose=10):
        """
        Paramters:
        ---------
        maxiter: integer
            number of iterations
        xtol: float, default = 1e-5
            tolerance threshold for the relative changes
        gtol: float, default = 1e-2
            tolerance threshold for the norm of Jacobian
        """

        n = 0 # iteration counter
        converged = False

        while n < maxiter and not converged:
            J0 = self.jac(x); J = J0[self.mask]
            H0 = self.hess(x); H = H0[self.mask,:][:,self.mask]

            # Hinv = np.linalg.inv(H)
            # print(J @ (Hinv @ J))

            dx_jac = -J; dx_jac /= np.linalg.norm(dx_jac) # direction decreasing the objective function value
            dx_hess = -np.linalg.solve(H, J); dx_hess /= np.linalg.norm(dx_hess)  # direction decreasing the modulus of the gradient

            if (dx_hess @ dx_jac >= 0):
                dx = dx_hess
            else:
                # tau = np.random.rand()
                # dx = (1-tau)*dx_jac + tau*dx_hess
                dx = (dx_jac + dx_hess)/2
                dx /= np.linalg.norm(dx)

            if dx @ dx_jac  < 0 or dx @ dx_hess < 0: # check if the update direction is good
                ee,vv = np.linalg.eig(H)
                raise ValueError('Update direction error: rho_max(H)={}, rho_min(H)={}, dx@dx_jac={}, dx@dx_hess={}'.format(np.max(ee), np.min(ee), dx @ dx_jac, dx @ dx_hess))

            # Line search
            F0 = self.func(x)

            def Wolf1(a):
                x1 = np.copy(x); x1[self.mask] += a*dx
                # print(self.func(x1)-c1 * a * dx @ J, F0)
                return self.func(x1) <= (F0 + c1 * a * dx @ J)

            def Wolf2(a):
                x1 = np.copy(x); x1[self.mask] += a*dx
                J1 = self.jac(x1)
                # print(np.abs(dx @ J1[self.mask]), c2 * np.abs(dx @ J))
                # return np.abs(dx @ J1[self.mask]) <= c2 * np.abs(dx @ J) # strong
                # print(dx @ J1[self.mask], c2 * dx @ J)
                # return dx @ J1[self.mask] >= (c2 * dx @ J) # weak
                return dx @ J1[self.mask] >= c2 * dx @ J # weak

            k=0; stepk = step
            # while k<20 and not (Wolf1(stepk) and Wolf2(stepk)): # wolf2 does not work
            while k<20 and not Wolf1(stepk):
                stepk /= 2
                k += 1

            # Condition of convergence
            # if (np.linalg.norm(stepk*dx) / np.linalg.norm(x[self.mask])) < xtol or np.linalg.norm(J0) < gtol:
            if np.linalg.norm(J) < gtol:
                converged = True
            else:
                x[self.mask] += stepk * dx

            # Forced positivity
            idx = np.where(x[:self.nbThetas] < 0); x[idx] = np.random.rand(len(idx))

            if verbose and n % (maxiter//verbose) == 0:
                parmstr = 'sigmaf2={:.2e}, lcf2={:.2e}, betaf2={:.2e}, sigmag2={:.2e}, lcg2={:.2e}, sigmae2={:.2e}, '.format(x[0], x[1], x[2], x[3], x[4], x[5])
                toto = ['alpha[{}]={:.2e}, '.format(m, x[self.nbThetas+m]) for m in range(self.nbThetaa)]
                parmstr += ''.join(toto) + ' step={:.4e}, k={}'.format(stepk, k)
                ee,vv = np.linalg.eig(H)
                print("Iteration: {}\nL({})={:.2e}, |dL|={:.2e}, rho_max(H)={:.2e}, rho_min(H)={:.2e}\n".format(n, parmstr, self.func(x), np.linalg.norm(J), np.max(ee), np.min(ee)))

            n += 1

        return x


    def __init__(self, Tobs, Xobs, Yobs, initval=False, **kwargs):
        """
        initval: bool, default=False
            if True the given keyword arguments are used as initial value to the
            numerical optimization algorithm, otherwise they are taken as fixed
            value and the optimization algorithm are applied on the parameters that
            the value is not set in the keyword arguments.
        """

        if 'alpha' in kwargs.keys():
            self.nbThetaa = len(kwargs['alpha']) # if the coefficients alpha are given

        self.Tobs = np.copy(Tobs)
        self.nbObs = len(Tobs)

        if Xobs.ndim==1:
            Xobs = np.reshape(Xobs, (1,-1))
        assert(Xobs.shape[1] == self.nbObs) # check that the provided Xobs contains same number of observations as Tobs.
        self.Xobs = np.copy(Xobs)

        assert(len(Yobs) == self.nbObs)
        self.Yobs = np.copy(Yobs)

        self.XmXp = np.asarray([[(np.linalg.norm(self.Xobs[:,i]-self.Xobs[:,j]))**self.powcst for j in range(self.nbObs)] for i in range(self.nbObs)])
        self.XpX = self.Xobs.T @ self.Xobs # np.asarray([[Xobs[:,i] @ Xobs[:,j] for j in range(nbObs)] for i in range(nbObs)])
        self.TmTp = np.abs(self.Tobs[:,newaxis]-self.Tobs[newaxis,:])**self.powcst
        self.dmuda = np.asarray([self.Tobs**p for p in range(self.nbThetaa)]).T # d mu/d thetas, i-th column corresponds to (d mu / d alpha_i)

        # Optimization
        Mask = np.zeros(self.nbThetas+self.nbThetaa, dtype=np.bool)
        Mask[0] = 'sigmaf2' in kwargs.keys() and kwargs['sigmaf2']>=0
        Mask[1] = 'lcf2'    in kwargs.keys() and kwargs['lcf2']>=0
        Mask[2] = 'betaf2'  in kwargs.keys() and kwargs['betaf2']>=0
        Mask[3] = 'sigmag2' in kwargs.keys() and kwargs['sigmag2']>=0
        Mask[4] = 'lcg2'    in kwargs.keys() and kwargs['lcg2']>=0
        Mask[5] = 'sigmae2' in kwargs.keys() and kwargs['sigmae2']>=0
        Mask[6:] = 'alpha'  in kwargs.keys()

        rgx = np.max(self.XmXp)
        max_xpx = np.max(self.XpX)
        rgt = np.max(self.TmTp)

        # The following initial guess are obtained by manual test
        # An example:
        # {'betaf2': 0.041706114761408147, 'sigmag2': 0.0086947224893822068, 'lcg2': 13.865969927679043, 'alpha': array([ 0.43511868, -0.175873  ]), 'sigmae2': 0.00026165794144624815, 'sigmaf2': 0.0035336216411931288, 'lcf2': 19.079947271111443} 
        
        self.sigmaf2 = kwargs['sigmaf2'] if Mask[0] else 1e-1
        self.lcf2    = kwargs['lcf2']    if Mask[1] else 1e-1 * rgx
        self.betaf2  = kwargs['betaf2']  if Mask[2] else 1e-1 / max_xpx
        self.sigmag2 = kwargs['sigmag2'] if Mask[3] else 1e-3
        self.lcg2    = kwargs['lcg2']    if Mask[4] else 1e1 * rgt
        self.sigmae2 = kwargs['sigmae2'] if Mask[5] else 1e-3
        self.alpha   = kwargs['alpha']   if np.sum(Mask[6:]) else np.zeros(self.nbThetaa)

        if not(np.sum(Mask) == len(Mask)) or initval:
            self.mask = np.logical_or(initval, np.logical_not(Mask))

            # initial guesses
            x0, _ = self.get_parms()

            print('Maximum likelihood estimation of optimal parameters')
            print('Initial guess={}\nmask={}\n'.format(x0, self.mask))

            x = self.parameters_estimation(x0)

            H0 = self.hess(x); H = H0[self.mask,:][:,self.mask]
            ee,vv = np.linalg.eig(H)

            if np.min(ee) < 0:
                print('Optimization failed: rho_min(H)={:.2e}'.format(np.min(ee)))

            self.set_parms(x)
            # self.sigmaf2, self.lcf2, self.betaf2, self.sigmag2, self.lcg2, self.sigmae2 = x[:self.nbThetas]
            # self.alpha = x[self.nbThetas:]


    def Prediction(self, Xprd=None, Tprd=None):
        """
        Make predictions of f(x), g(t), y(t) at given time instants t and system states x(t).

        Parameters:
        ----------
        Xprd: ndarray
            system states for prediction, each column is an observation
        Tprd: 1darray
            time instants for prediction
        """
        
        parms,val = self.get_parms()
        mYobs = self.mufunc(parms)
        Kyoo = self.Kyoofunc(parms)
        Kyooinv = np.linalg.inv(Kyoo)
        xi = Kyooinv @ (self.Yobs - mYobs)
        
        if not Xprd is None: # Xprd given
            if Xprd.ndim==1:
                Xprd = np.reshape(Xprd, (1,-1))

            nbXprd = Xprd.shape[1]
            XpmXp = np.asarray([[norm(Xprd[:,i]-Xprd[:,j])**self.powcst for j in range(nbXprd)] for i in range(nbXprd)])
            XpmXo = np.asarray([[norm(Xprd[:,i]-self.Xobs[:,j])**self.powcst for j in range(self.nbObs)] for i in range(nbXprd)])

            Kfpp = self.sigmaf2 * np.exp(-XpmXp/self.lcf2) + self.betaf2 * Xprd.T @ Xprd
            Kfpo = self.sigmaf2 * np.exp(-XpmXo/self.lcf2) + self.betaf2 * Xprd.T @ self.Xobs
            
            Fprd = Kfpo @ xi # equivalent to Fprd = Kfpo @ np.linalg.solve(Kyoo, Yobs-mYobs)
            VFprd = np.diag(Kfpp - Kfpo @ (Kyooinv @ Kfpo.T))
        else:
            Fprd, VFprd = None, None

        if not Tprd is None: # Tprd given
            nbTprd = len(Tprd)
            TpmTp = (Tprd[:,newaxis]-Tprd[newaxis,:])**self.powcst
            TpmTo = (Tprd[:,newaxis]-self.Tobs[newaxis,:])**self.powcst

            Kgpp = self.sigmag2 * np.exp(-TpmTp/self.lcg2)
            Kgpo = self.sigmag2 * np.exp(-TpmTo/self.lcg2)
            mGprd = np.asarray([self.alpha[p]*Tprd**p for p in range(self.nbThetaa)]).sum(axis=0)
            
            Gprd = mGprd + Kgpo @ xi
            VGprd = np.diag(Kgpp - Kgpo @ (Kyooinv @ Kgpo.T))
        else:
            Gprd, VGprd = None, None

        if not Tprd is None and not Xprd is None and nbTprd == nbXprd: # both Tprd and Xprd are given and have the same length
            Kypp = Kfpp + Kgpp + self.sigmae2 * np.eye(nbTprd)
            # Kypo = Kfpo + Kgpo + self.sigmae2 * (Xprd[:,newaxis] == self.Xobs[newaxis,:])
            Kypo = Kfpo + Kgpo
            
            Yprd = mGprd + Kypo @ xi
            VYprd = np.diag(Kypp - Kypo @ (Kyooinv @ Kypo.T))
        else:
            Yprd, VYprd = None, None
            
        return [Fprd, VFprd], [Gprd, VGprd], [Yprd, VYprd]

        # def Kfoo(self):
    #     return self.sigmaf**2 * np.exp(-self.XmXp/self.lcf**2) + self.betaf**2 * self.XpX

    # def Kgoo(self):
    #     return self.sigmag**2 * np.exp(-self.TmTp/self.lcg**2)

    # def muo(self):
    #     return np.asarray([self.alpha[p] * Tobs**p for p in range(self.nbAlpha)]).sum(axis=0)

    # def dKyoo(self):
    #     """
    #     Derivatives of Kyoo wrt the parameter set (except alpha)
    #     """
    #     val = []
    #     val.append(np.exp(-self.XmXp/self.lcf**2))
    #     val.append(self.sigmaf**2 * np.exp(-self.XmXp/self.lcf**2) * self.XmXp/(self.lcf**4))
    #     val.append(self.XpX)
    #     val.append(np.exp(-self.TmTp/self.lcg**2))
    #     val.append(self.sigmag**2 * np.exp(-self.TmTp/self.lcg**2) * self.TmTp/(self.lcg**4))
    #     val.append(np.eye(nbObs))
    #     return val

    # def d2Kyoo(self):
    #     d2Kyoo = [[np.zeros_like(XmXp) for j in range(6)] for i in range(6)]

    #     d2Kyoo[0][1] = np.exp(-XmXp/x[1]) * XmXp / (x[1]**2)
    #     d2Kyoo[1][0] = d2Kyoo[0][1]
    #     d2Kyoo[1][1] = x[0] * np.exp(-XmXp/x[1]) * (XmXp**2/(x[1]**4) - 2*XmXp/(x[1]**3))
    #     d2Kyoo[3][4] = np.exp(-TmTp/x[4]) * TmTp / (x[4]**2)
    #     d2Kyoo[4][3] = d2Kyoo[3][4]
    #     d2Kyoo[4][4] = x[3] * np.exp(-TmTp/x[4]) * (TmTp**2/(x[4]**4) - 2*TmTp/(x[4]**3))

    #     return d2Kyoo
