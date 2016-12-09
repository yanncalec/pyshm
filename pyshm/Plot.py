"""Some handy functions for plot.
"""

import numpy as np

import matplotlib
matplotlib.use("qt5agg")
import matplotlib.pyplot as plt


def tep_plot(xobs0, yobs0, yprd0, Tidx, loc, Midx=[],
                xobs_label='Temperature', yobs_label='Elongation', yprd_label='Prediction'):
    assert(len(xobs0)==len(yobs0)==len(yprd0)==len(Tidx))
    xobs = xobs0.copy()
    xobs[Midx] = np.nan

    yobs = yobs0.copy()
    yobs[Midx] = np.nan

    yprd = yprd0.copy()
    yprd[Midx] = np.nan

    v0 = yobs[~np.isnan(yobs)][0]
    v1 = yprd0[~np.isnan(yprd0)][0]
    yprd += v0-v1

    yerr = yobs - yprd

    fig, axes = plt.subplots(2,1,figsize=(20,10),sharex=True)
    axa = axes[0]
    axa.plot(Tidx, yobs, 'b', label=yobs_label)
    axa.plot(Tidx, yprd, 'c', label=yprd_label)
    axa.legend(loc='upper left')
    axb = axa.twinx()
    axb.patch.set_alpha(0)
#     axb.plot(Tidx, yerr, 'g', linewidth=2, label='{}-{}'.format(yobs_label, yprd_label))
    axb.plot(Tidx, yerr, 'g', linewidth=2, label='Residual')
    axb.legend(loc='upper right')
    axa.set_title('Location {}'.format(loc))

    axa = axes[1]
    axa.plot(Tidx, yobs, 'b', label=yobs_label)
    axa.legend(loc='upper left')
    axb = axa.twinx()
    axb.patch.set_alpha(0)
    axb.plot(Tidx, xobs, 'r', label=xobs_label)
    axb.legend(loc='upper right')
#     axa.set_title('Raw data')
    plt.tight_layout()

    return fig, axes


def twin_plot(y1, y2, index=None):
    fig, axes = plt.subplots(1,1,figsize=(20,5),sharex=True)
    axa = axes
    axb = axa.twinx()
    axb.patch.set_alpha(0.0)
    axa.plot(y1, 'r')
    axb.plot(y2, 'b')
    return fig, (axa, axb)

def two_plot(y1, y2):
    fig, axes = plt.subplots(1,1,figsize=(20,5),sharex=True)
    axa = axes
    _two_plot(axa, y1, y2, twin=False)
    return fig, axes


def _two_plot(axa, y1, y2, *args, tidx=None, twin=True):
    if tidx is None:
        tidx1 = range(len(y1))
        tidx2 = range(len(y2))
    else:
        assert(len(y1)==len(y2))
        tidx1 = tidx[:len(y1)]
        tidx2 = tidx[:len(y1)]

    if len(args)>0:
        axa.plot(tidx1, y1, 'r', label=args[0])
        axa.legend(loc='upper left')
    else:
        axa.plot(tidx1, y1, 'r')

    axb = axa.twinx() if twin else axa
    # axb.patch.set_alpha(0.0)

    txtb = args[1] if len(args)>1 else ''
    if len(args)>1:
        axb.plot(tidx2, y2, 'b', label=args[1])
        axb.legend(loc='upper right')
    else:
        axb.plot(tidx2, y2, 'b')

#     twinx_adjust_yrng(axa, axb)


def twinx_adjust_yrng(axa, axb):
    ya1, ya2 = axa.get_ylim()
    ya = max(abs(ya1), abs(ya2))
    _ = axa.set_ylim(-ya, ya)

    yb1, yb2 = axb.get_ylim()
    yb = max(abs(yb1), abs(yb2))
    _ = axb.set_ylim(-yb, yb)
