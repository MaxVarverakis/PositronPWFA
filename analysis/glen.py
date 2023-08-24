import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from openpmd_viewer import OpenPMDTimeSeries
from openpmd_viewer.addons import LpaDiagnostics
from scipy import constants

from matplotlib import patches
from matplotlib import ticker

import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import ConnectionPatch

import sys
sys.path.append('/Users/max/HiPACE/hipace/tools/')
import read_insitu_diagnostics as diag

plt.rc('text', usetex = True)
plt.rcParams['figure.facecolor'] = 'w'
plt.rcParams['figure.figsize'] = [10.0, 6.0]
plt.rcParams['font.size'] = 16
plt.style.use('classic')

ts = OpenPMDTimeSeries('path', check_all_files = True)

zd, uzd = ts.get_particle(var_list = ['z', 'uz'], species = 'drive', iteration = 100)
zw, uzw = ts.get_particle(var_list = ['z', 'uz'], species = 'witness', iteration = 100)

plt.figure(figsize = (10, 6))
plt.scatter(zd, uzd, s = 0.001, edgecolor = 'darkred')
plt.scatter(zw, uzw, s = 0.001, edgecolor = 'blue')
plt.xlabel('$k_p\zeta$')
plt.ylabel('$u_z/(m_e c)$')
plt.ylim(0, 1.05e4)
plt.show()