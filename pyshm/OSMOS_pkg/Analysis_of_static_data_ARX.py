'''Analysis of static data using the ARX model.'''

import os
import pickle
# import statsmodels.api as sm
import numpy as np

from .. import Tools
from . import OSMOS

class Options:
    tidx0=None
    tidx1=None
    component='AllDiff-AllDiff'
    penalh=5e-1
    penalg=5e-1
    Nh=10
    Ng=24
    Ntrn=3*30*24
    sidx=0
    const=False
    verbose=False

def Analysis_of_static_data_ARX(infile, loc, outdir0, options=None):
    if options is None:
        options = Options()

    # Load raw data
    with open(infile, 'rb') as fp:
        All = pickle.load(fp)['Static_Data']

    # All locations of the current project
    Locations = list(All.keys()); Locations.sort()

    if loc not in Locations:
        raise Exception('Invalid location keyid')

    if options.verbose:
        print('\nLocation: {}'.format(loc))

    # dictionary for saving results
    Res = {}

    # Data truncation
    if options.tidx0 is not None and options.tidx1 is not None:
        Data0 = All[loc][options.tidx0:options.tidx1]
    elif options.tidx0 is not None:
        Data0 = All[loc][options.tidx0:]
    elif options.tidx1 is not None:
        Data0 = All[loc][:options.tidx1]
    else:
        Data0 = All[loc]

    Xall = Data0['Temperature'].copy()
    Yall = Data0['Elongation'].copy()
    Tidx = Xall.index  # time index
    Nidx = np.isnan(np.asarray(Xall)) # nan index

    # Components for training
    Xraw, Yraw = OSMOS.choose_component(Data0, options.component)  # pandas format
    Xdata, Ydata = np.asarray(Xraw), np.asarray(Yraw)  # numpy format

    # Training data
    Xtrn = Xdata[options.sidx:options.sidx+options.Ntrn].copy()
    Ytrn = Ydata[options.sidx:options.sidx+options.Ntrn].copy()
    Xtrn[np.isnan(Xtrn)] = 0
    Ytrn[np.isnan(Ytrn)] = 0

    # Optimal length of kernels
    if options.Nh < 0 or options.Nh > 24:
        AIC, BIC = Tools.optimal_kernel_length_AR(Xtrn, Ytrn)
        options.Nh = np.argmin(BIC)+1
        if options.verbose:
            print('Optimal length of AR kernel: {}'.format(options.Nh))

    if options.Ng < 0 or options.Ng > 24:
        AIC, BIC = Tools.optimal_kernel_length_conv(Xtrn, Ytrn)
        options.Ng = np.argmin(BIC)+1
        if options.verbose:
            print('Optimal length of convolution kernel: {}'.format(options.Ng))

    if (options.Nh==0 and options.Ng==0):
        raise ValueError('Nh and Ng must not be both == 0')

    Nq = options.Nh+options.Ng+int(options.const)  # total dimension of the state vector

    # penalization on the decay of kernel
    wh = Tools.exponential_weight(options.Nh, w0=options.penalh) if options.Nh>0 else np.zeros(options.Nh)
    wg = Tools.exponential_weight(options.Ng, w0=options.penalg) if options.Ng>0 else np.zeros(options.Ng)
    pcoef = np.hstack([wh, wg, 0]) if options.const else np.hstack([wh, wg])

    h0, g0, c0, err0, A0 = Tools.ARX_fit(Ytrn, options.Nh, Xtrn, options.Ng, bflag=True, pcoef=None, cflag=options.const)  # without penalization
    h1, g1, c1, err1, _ = Tools.ARX_fit(Ytrn, options.Nh, Xtrn, options.Ng, bflag=True, pcoef=pcoef, cflag=options.const)  # with penalization

    # save result in a dictionary
    Res['AR.Kernel_np'] = h0  # AR kernel, without penalization
    Res['Conv.Kernel_np'] = g0 # Convolution kernel, without penalization
    Res['Constant_np'] = c0 # constant trend, without penalization
    Res['AR.Kernel'] = h1  # AR kernel
    Res['AR.Penal'] = options.penalh  # penalization for AR kernel
    Res['Conv.Kernel'] = g1  # Conv. kernel
    Res['Conv.Penal'] = options.penalg  # penalization for conv.kernel
    Res['Constant'] = c1  # constant trend
    Res['Cond.Number'] = np.linalg.cond(A0)  # condition number of the linear regression matrix
    Res['Xall'] = Xall  # Temperature, raw
    Res['Yall'] = Yall  # Elongation, raw
    Res['sidx'] = options.sidx  # starting index of the training data
    Res['Ntrn'] = options.Ntrn  # length of the training data
    Res['component'] = options.component  # type of component for whole data set
    # Res['cmptrn'] = options.cmptrn  # type of component for training data set
    Res['Nh'] = options.Nh
    Res['Ng'] = options.Ng

    # Components for prediction
    Xraw, Yraw = OSMOS.choose_component(Data0, options.component)
    Xdata, Ydata = np.asarray(Xraw), np.asarray(Yraw)

    # Prediction with the estimated or precomputed kernel
    Yprd = Tools.ARX_prediction(Ydata, h1, X=Xdata, g=g1, c=c1)
    Yprd[Nidx] = np.nan
    Res['Xdata'] = Xdata  # prediction data set: Temperature
    Res['Ydata'] = Ydata  # prediction data set: Elongation
    Res['Yprd'] = Yprd  # results of prediction
    Res['Rel.Error'] = Tools.safe_norm(Yprd-Ydata)/Tools.safe_norm(Ydata)

    if options.verbose:
        print('AR.Kernel: ', Res['AR.Kernel'])
        print('Conv.Kernel: ', Res['Conv.Kernel'])
        print('Constant: ', Res['Constant'])
        print('Condition number: {}'.format(Res['Cond.Number']))
        print('Rel.Error: ', Res['Rel.Error'])

    outdir = os.path.join(outdir0,'ARX_[{}]_[Nh={}_Ng={}_const={}]'.format(options.component, options.Nh, options.Ng, options.const), '{}'.format(loc))
    try:
        os.makedirs(outdir)
    except OSError:
        pass

    fname = os.path.join(outdir, 'Results.pkl')
    try:
        with open(fname, 'wb') as fp:
            pickle.dump(Res, fp)
        if options.verbose:
            print('\nResults saved in {}'.format(fname))
    except Exception as msg:
        print(msg)
        # print(Fore.RED + 'Warning: ', msg)
        # print(Style.RESET_ALL)
