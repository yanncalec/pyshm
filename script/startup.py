import importlib
from importlib import reload

import sys, os, glob
from optparse import OptionParser       # command line arguments parser
import requests
import json
import pickle, datetime,dateutil
from collections import namedtuple
import warnings
import itertools
import copy
import numbers

import statsmodels.api as sm

from sklearn.decomposition import PCA

import pandas as pd
pd.options.mode.chained_assignment = None  # Turn off pandas copy warning

import scipy
from scipy import interpolate, stats, signal, sparse

import numpy as np
# from numpy import *

from Pyshm import Tools, Stat, Kalman, OSMOS

from colorama import Fore, Back, Style # for color output

# homedir = os.path.expanduser('~')+"/Sivienn/Projects/Osmos/"
homedir = os.path.expanduser('~')+"/OSMOS/"
datadir = homedir+"/Data/"
outdir = homedir+"/Outputs/"
# figdir = outdir+"/figures/"

# try:
#     os.mkdir(datadir+'/Static/')
#     os.mkdir(outdir+'/Static/')
# except OSError:
#     pass

# import matplotlib.colors as colors
# color_list = list(colors.cnames.keys())
color_list = ['red', 'green', 'blue', 'magenta', 'cyan', 'pink', 'lightgrey', 'yelow',
              'purple', 'mediumorchid', 'chocolate', 'blue', 'blueviolet', 'brown']

# from bokeh.client import push_session
# from bokeh.plotting import figure, curdoc
# from bokeh.charts import TimeSeries
# from bokeh.resources import CDN, INLINE
# from bokeh.embed import file_html, components
# from bokeh.util.string import encode_utf8
# from bokeh.io import hplot, vplot, gridplot
# # output_notebook() # for bokeh plotting in notebook

# bokehtools = 'pan, wheel_zoom, box_zoom, crosshair, hover, reset, save'

# disable deprecation warning
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import mpld3
# mpld3.enable_notebook()
