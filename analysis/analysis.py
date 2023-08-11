# </path/to>/build/bin/hipace <input>
import sys
import numpy as np
import matplotlib.pyplot as plt
from openpmd_viewer import OpenPMDTimeSeries
from openpmd_viewer.addons import LpaDiagnostics
from scipy import constants

from matplotlib import patches
from matplotlib import ticker

import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import ConnectionPatch

# sys.path.append('/Users/max/HiPACE/src/hipace/tools/')
# import read_insitu_diagnostics as diag
# all_data = diag.read_file("/Users/max/HiPACE/recovery/diags/insitu/positron/reduced_witness.0000.txt")
# print("Available diagnostics:", all_data.dtype.names)

i = 0

plt.rc('text', usetex = True)
plt.rc('font', family = 'serif')

# path to output directory
ts = LpaDiagnostics('/Users/max/HiPACE/recovery/diags/hdf5/positron', check_all_files = False)

def efficiency(path: str, iteration = 0, plot = True, check_files = False, insitu_path: str = None) -> float:
    
    """
    Calculates energy transfer efficiency from drive to witness bunch.

    Parameters
    ----------
    path : str
        path to output directory

    iteration : int, optional
        iteration number. Defaults to 0.

    plot : bool, optional
        plots plasma wakefield, transverse electric field (Ez), and particle beams in normalized coordinates. Defaults to True.

    check_files :bool, optional
        check all files in directory are of the same form. Defaults to False.

    insitu_path : str, optional
        path to insitu diagnostics file. Defaults to None.

    Returns
    -------
    eta : float (in percent)
        energy transfer efficiency from drive to witness bunch
    """

    ts = LpaDiagnostics(path, check_all_files = check_files)

    i = iteration

    Qd = ts.get_charge(species = 'drive', iteration = 0)
    Qw = ts.get_charge(species = 'witness', iteration = 0) + ts.get_charge(species = 'witness2', iteration = 0)

    ExmBy, _ = ts.get_field(field = 'ExmBy', iteration = i)
    Ez_raw, info = ts.get_field(field = 'Ez', iteration = i)
    xd, zd = ts.get_particle(species = 'drive', iteration = i, var_list = ['x', 'z'])
    xw, zw = ts.get_particle(species = 'witness', iteration = i, var_list = ['x', 'z'])
    xw2, zw2 = ts.get_particle(species = 'witness2', iteration = i, var_list = ['x', 'z'])

    Ez = np.transpose(Ez_raw)[len(info.x) // 2, :]

    driveMin, driveMax = min(zd), max(zd)
    witnessMin, witnessMax = min(zw), max(zw) # note that 'witness2' is spatially contained in 'witness'

    # driveWidth = driveMax - driveMin
    # witnessWidth = witnessMax - witnessMin

    maskD = np.logical_and(driveMin <= info.z, info.z <= driveMax)
    maskW = np.logical_and(witnessMin <= info.z, info.z <= witnessMax)

    EzDAvg = np.mean(Ez[maskD])
    EzWAvg = np.mean(Ez[maskW])

    eta = - (Qw * EzWAvg) / (Qd * EzDAvg) * 1e2 # percent

    if plot:

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        im = ax.pcolormesh(info.z, info.x, np.transpose(ExmBy), cmap = 'RdBu', vmin = -2, vmax = 2)

        plt.scatter(zd, xd, s = .02, color = 'darkred')
        plt.scatter(zw, xw, s = .02, color = 'darkblue')
        plt.scatter(zw2, xw2, s = .02, color = 'darkblue')
        plt.legend(['$e^-$ Drive Bunch', '$e^+$ Witness Bunch'], loc = 'upper left', markerscale = 10)
        
        if insitu_path:
            import sys
            sys.path.append('/Users/max/HiPACE/src/hipace/tools/')
            import read_insitu_diagnostics as diag
            all_data = diag.read_file(insitu_path)
            # print("Available diagnostics:", all_data.dtype.names)
            # print(all_data['normalized_density_factor'])
            # print(all_data['is_normalized_units'])
            # print(diag.total_charge(all_data))
            # border = [all_data[key] for key in ['z_lo', 'z_hi', 'x_lo', 'x_hi']]

            ax.set_xlim(all_data['z_lo'], all_data['z_hi'])
        else:
            ax.set_xlim(-34, 6)
        
        ax.set_ylim(-6, 6)
        ax.set_ylabel('$k_px$')
        ax.set_xlabel('$k_p\zeta$')

        # plt.style.use()

        # matplotlib.rc('axes', edgecolor='k')

        # ax.tick_params(colors='k')
        # ax.tick_params(axis='y', colors='k')
        # ax.yaxis.label.set_color('k')
        # ax.xaxis.label.set_color('k')

        ax2 = ax.twinx()
        ax2.plot(info.z, Ez, color = 'black')
        ax2.set_ylim(-.8, .8)
        ax2.set_ylabel(r'$E_z/E_0$',  labelpad = 1) 

        divider2 = make_axes_locatable(ax)
        cax2 = divider2.append_axes("right", size = "4%", pad = .8)
        divider3 = make_axes_locatable(ax2)
        cax3 = divider3.append_axes("right", size = "4%", pad = .8)
        cax3.remove()

        cb2 = plt.colorbar(im, cax=cax2)
        cb2.set_label(r'$(E_x - B_y)/E_0 $')
        plt.show()

    return eta

ExmBy, info_x = ts.get_field(field='ExmBy', iteration = i)
Ez, info_x = ts.get_field(field='Ez', iteration = i)
x, z = ts.get_particle(species = 'drive', iteration = i, var_list = ['x', 'z'])
xp, zp = ts.get_particle(species = 'witness', iteration = i, var_list = ['x', 'z'])
xw, zw = ts.get_particle(species = 'witness2', iteration = i, var_list = ['x', 'z'])
# ts.get_field?

fig, ax =plt.subplots(1, 1, figsize=(10,6))


im = ax.pcolormesh(info_x.z, info_x.x, np.transpose(ExmBy), cmap = 'RdBu', vmin = -2, vmax = 2)

plt.scatter(z, x, s = .02, color = 'darkred')
plt.scatter(zp, xp, s = .02, color = 'darkblue')
plt.scatter(zw, xw, s = .02, color = 'darkblue')
plt.legend(['$e^-$ Drive Bunch', '$e^+$ Witness Bunch'], loc = 'upper left', markerscale = 10)

ax.set_xlim(-34, 6)
ax.set_ylim(-6, 6)
ax.set_ylabel('$k_px$')
ax.set_xlabel('$k_p\zeta$')

# plt.style.use()

# matplotlib.rc('axes', edgecolor='k')

# ax.tick_params(colors='k')
# ax.tick_params(axis='y', colors='k')
# ax.yaxis.label.set_color('k')
# ax.xaxis.label.set_color('k')

ax2 = ax.twinx()
ax2.plot(info_x.z, np.transpose(Ez)[len(info_x.x)//2,:], color = 'black')
ax2.set_ylim(-.8, .8)
ax2.set_ylabel(r'$E_z/E_0$',  labelpad = 1) 

divider2 = make_axes_locatable(ax)
cax2 = divider2.append_axes("right", size = "4%", pad = .8)
divider3 = make_axes_locatable(ax2)
cax3 = divider3.append_axes("right", size = "4%", pad = .8)
cax3.remove()

cb2 = plt.colorbar(im, cax=cax2)
cb2.set_label(r'$(E_x - B_y)/E_0 $')
plt.show()

# if __name__ == '__main__':
#     p = '/Users/max/HiPACE/recovery/diags/hdf5/positron'
#     ip = '/Users/max/HiPACE/recovery/diags/insitu/positron/reduced_drive.0000.txt'
#     print(efficiency(path = p, insitu_path = ip, plot = False))