import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from openpmd_viewer import OpenPMDTimeSeries
from openpmd_viewer.addons import LpaDiagnostics
from scipy import constants

from mpl_toolkits.axes_grid1 import make_axes_locatable

import sys
sys.path.append('/Users/max/HiPACE/hipace/tools/')
import read_insitu_diagnostics as diag

class Functions():
    def __init__(self, path: str, insitu_path: str, n0: float, iteration: int, normalized: bool, recovery: bool, check = True):
        self.n0 = n0 # cm^-3
        self.iteration = iteration
        self.normalized = normalized
        self.recovery = recovery
        self.ts = OpenPMDTimeSeries(path, check_all_files = check)
        self.kp_inv = self.skinDepth(self.n0) # m
        self.kp = self.kp_inv**-1 # m^-1
        self.E0 = self.E0(self.n0) # V/m
        self.ExmBy, self.info, self.Ez, self.xd, self.zd, self.wd, self.xw, self.zw, self.ww, self.xr, self.zr, self.wr = self.getPlotData(self.iteration, self.recovery)
        self.driveInsitu, self.witnessInsitu, self.recoveryInsitu = self.insitu(insitu_path, self.recovery)
        self.maskD, self.maskW, self.maskR = self.bunchMask(self.iteration, self.recovery)
        self.rho = self.ts.get_field(field = 'rho', iteration = self.iteration, coord = 'z')[0]
        self.jz_beam = self.ts.get_field(field = 'jz_beam', iteration = self.iteration, coord = 'z')[0]
        self.profile = abs(self.getZ(self.jz_beam, self.info))

    def customCMAP(self, names = ['RdBu', 'PuOr', 'PRGn', 'bwr', 'bwr_r', 'PuOr_r'], ncolors: int = 256):
        for cmap in names:
            color_array = plt.get_cmap(cmap)(range(ncolors))
            color_array[:,-1] = abs(np.linspace(-1.0, 1.0, ncolors))
            map_object = mcolors.LinearSegmentedColormap.from_list(name = cmap + 'T', colors = color_array)
            plt.register_cmap(cmap = map_object)

    def skinDepth(self, ne):
        """
        Calculates plasma skin depth based off plasma electron density

        Parameters
        ----------
        ne : float
            plasma electron density (in cm^-3)
        
        Returns
        -------
        kp_inv : float
            plasma skin depth in m
        """

        wp = 1e-3 * np.sqrt((ne * constants.e**2) / (constants.epsilon_0 * constants.m_e)) # SI
        
        kp_inv = 1e-6 * constants.c / wp # m
        
        return kp_inv

    def E0(self, ne: float) -> float:
        """
        Calculates the cold-nonrelativistic wave breaking field E0

        Parameters
        ----------
        ne : float
            plasma electron density (in cm^-3)
        
        Returns
        -------
        E0 : float
            cold-nonrelativistic wave breaking field (in V/m)
        """
        kp = self.skinDepth(ne) ** -1 # m^-1
        return constants.m_e * constants.c**2 * kp / constants.e # V/m

    def GeV2P(self, pG: float) -> float:
        """
        Calculates momentum value for HiPACE++ input file from GeV/c value
        
        Parameters
        ----------
        pG : float
            momentum in GeV/c
        
        Returns
        -------
        p : float
            momentum in HiPACE++ input file units
        """
        pG *= 1e9 * constants.e / constants.c # 1 [GeV/c] = eV / (m/s) = kg m / s

        return pG / (constants.m_e * constants.c)

    def getZ(self, F, info):
        """
        returns on-axis longitudinal slice of field
        """
        return F[:, len(info.x)//2].T

    def getPlotData(self, iteration: int, recovery: bool) -> tuple:
        i = iteration

        ExmBy, info = self.ts.get_field(field = 'ExmBy', iteration = i)
        Ez = self.getZ(self.ts.get_field(field = 'Ez', iteration = i)[0], info)
        xd, zd, wd = self.ts.get_particle(species = 'drive', iteration = i, var_list = ['x', 'z', 'w'])
        xw, zw, ww = self.ts.get_particle(species = 'witness', iteration = i, var_list = ['x', 'z', 'w'])
        if recovery:
            xr, zr, wr = self.ts.get_particle(species = 'recovery', iteration = i, var_list = ['x', 'z', 'w'])
        else:
            xr, zr, wr = np.zeros_like(xd), np.zeros_like(zd), np.zeros_like(xd)

        return ExmBy, info, Ez, xd, zd, wd, xw, zw, ww, xr, zr, wr

    def EDensitySim(self, iteration: int):
        """
        From HiPACE++ simulation (normalized input file)
        """
        
        i = iteration

        Ez, _ = self.ts.get_field(field = 'Ez', iteration = i)
        ExmBy, _ = self.ts.get_field(field = 'ExmBy', iteration = i)
        EypBx, _ = self.ts.get_field(field = 'EypBx', iteration = i)
        By, _ = self.ts.get_field(field = 'By', iteration = i)
        Bx, _ = self.ts.get_field(field = 'Bx', iteration = i)
        Bz, _ = self.ts.get_field(field = 'Bz', iteration = i)

        if self.normalized:
            Ex = ExmBy + By
            Ey = EypBx - Bx
            u = self.E0**2 / 2 * constants.epsilon_0 * ((Ex**2 + Ey**2 + Ez**2) + (Bx**2 + By**2 + Bz**2)) # from normalized sim
        else:
            Ex = ExmBy + constants.c * By
            Ey = EypBx - constants.c * Bx
            u = 1 / 2 * (constants.epsilon_0 * (Ex**2 + Ey**2 + Ez**2) + 1 / constants.mu_0 * (Bx**2 + By**2 + Bz**2))
        
        return u

    def charge(self, q: float, ne = None) -> float:
        """
        Calculates charge in Coulombs based off of normalized charge

        Parameters
        ----------
        q : float
            normalized charge
        ne : float
            plasma electron density (in m^-3).  Defaults to 1e23 m^-3

        Returns
        -------
        q0 * q : float
            charge in Coulombs
        """
        # n_b = Q/((2*pi)^(3/2)* std_x * std_y * std_z)
        
        if not ne:
            ne = self.n0

        q0 = constants.c**3 * constants.epsilon_0**(3/2) * constants.m_e**(3/2) / (np.sqrt(ne) * constants.e**2)
        
        return q0 * q

    def bunchMask(self, iteration: int, recovery: bool) -> tuple:
        i = iteration
        _, info = self.ts.get_field(field = 'Ez', iteration = i, coord = 'z')
        zd = self.ts.get_particle(species = 'drive', iteration = i, var_list = ['z'])[0]
        zw = self.ts.get_particle(species = 'witness', iteration = i, var_list = ['z'])[0]

        driveMin, driveMax = min(zd), max(zd)
        witnessMin, witnessMax = min(zw), max(zw) # note that 'witness2' is spatially contained in 'witness'

        maskD = np.logical_and(driveMin <= info.z, info.z <= driveMax)
        maskW = np.logical_and(witnessMin <= info.z, info.z <= witnessMax)

        if recovery:
            zr = self.ts.get_particle(species = 'recovery', iteration = i, var_list = ['z'])[0]
            recoveryMin, recoveryMax = min(zr), max(zr)
            maskR = np.logical_and(recoveryMin <= info.z, info.z <= recoveryMax)
        else:
            maskR = np.zeros_like(maskD)

        return maskD, maskW, maskR

    def insitu(self, insitu_path, recovery: bool):
        driveInsitu = diag.read_file(insitu_path + 'reduced_drive.0000.txt')
        witnessInsitu = diag.read_file(insitu_path + 'reduced_witness.0000.txt')

        if recovery:
            recoveryInsitu = diag.read_file(insitu_path + 'reduced_recovery.0000.txt')
        else:
            recoveryInsitu = np.zeros_like(driveInsitu)

        return driveInsitu, witnessInsitu, recoveryInsitu

    def quickEfficiency(self, iteration: int) -> float:
        """
        Calculates energy transfer efficiency from drive to witness bunch (only uses on-axis slices).

        Parameters
        ----------
        path : str
            path to output directory
        
        insitu_path : str
            path to insitu diagnostics file

        iteration : int, optional
            iteration number. Defaults to 0.
        
        check_files :bool, optional
            check all files in directory are of the same form. Defaults to False.

        Returns
        -------
        eta : float (in percent)
            energy transfer efficiency from drive to witness bunch
        """
        # ts = LpaDiagnostics(self, path, check_all_files = check_files)

        i = iteration

        # driveInsitu, witnessInsitu, recoveryInsitu = self.insitu(insitu_path, recovery)

        Qd_slices = diag.per_slice_charge(self.driveInsitu)[i]
        Qw_slices = diag.per_slice_charge(self.witnessInsitu)[i]
        Qr_slices = diag.per_slice_charge(self.recoveryInsitu)[i]

        Ez_raw, info = self.ts.get_field(field = 'Ez', iteration = i, coord = 'z')
        
        Ez = self.getZ(Ez_raw, info) # on-axis slice of Ez

        d = Ez @ Qd_slices
        w = Ez @ Qw_slices
        r = Ez @ Qr_slices

        eta = - ((w + r) / d) * 1e2 # percent

        return eta

    def efficiency3D(self, iteration: int) -> float:
        """
        Calculates energy transfer efficiency from drive to witness bunch (includes entire 3D domain in calculation).

        NOTE: `diagnostic.names = driver_diag witness_diag` must be set with xyz Ez field diagnostics enabled in order for this to work

        Parameters
        ----------
        path : str
            path to output directory
        
        insitu_path : str
            path to insitu diagnostics file

        iteration : int, optional
            iteration number. Defaults to 0.
        
        check_files :bool, optional
            check all files in directory are of the same form. Defaults to False.

        Returns
        -------
        eta : float (in percent)
            energy transfer efficiency from drive to witness bunch
        """
        
        # ts = LpaDiagnostics(path, check_all_files = check_files)
        
        i = iteration

        # driveInsitu, witnessInsitu, recoveryInsitu = insitu(insitu_path, recovery)

        z = diag.z_axis(self.driveInsitu)

        EzDrive, infoD = self.ts.get_field(field = 'Ez_driver_diag', iteration = i)
        EzWitness, infoW = self.ts.get_field(field = 'Ez_witness_diag', iteration = i)

        if self.recovery:
            EzRecovery, infoR = self.ts.get_field(field = 'Ez_recovery_diag', iteration = i)
            maskR = np.logical_and(z >= infoR.zmin, z <= infoR.zmax)
            EzR = np.array([np.mean(EzRecovery[i,:,:]) for i in range(EzRecovery.shape[0])])
            Qr_slices = diag.per_slice_charge(self.recoveryInsitu)[i][maskR]
            r = EzR @ Qr_slices
        else:
            r = 0

        # average Ez for each z-slice over each bunch
        EzD = np.array([np.mean(EzDrive[i,:,:]) for i in range(EzDrive.shape[0])])
        EzW = np.array([np.mean(EzWitness[i,:,:]) for i in range(EzWitness.shape[0])])

        maskD = np.logical_and(z >= infoD.zmin, z <= infoD.zmax)
        maskW = np.logical_and(z >= infoW.zmin, z <= infoW.zmax)

        Qd_slices = diag.per_slice_charge(self.driveInsitu)[i][maskD]
        Qw_slices = diag.per_slice_charge(self.witnessInsitu)[i][maskW]

        # print(Qd_slices.shape, Qw_slices.shape, Qr_slices.shape)
        # print(EzD.shape, EzW.shape)

        d = EzD @ Qd_slices
        w = EzW @ Qw_slices

        eta = - ((w + r) / d) * 1e2 # percent

        return eta

    def getProfile(self, iteration: int, plot = False):
        i = iteration
        # maskD, maskW, maskR = self.bunchMask(i, self.recovery)

        # Ez_raw, info = self.ts.get_field(field = 'Ez', iteration = i, coord = 'z')
        Ez = self.getZ(self.ts.get_field(field = 'Ez', iteration = i, coord = 'z')[0], self.info) # on-axis slice of Ez
        # zd, wd = self.ts.get_particle(species = 'drive', iteration = i, var_list = ['z', 'w'])
        # zw, ww = self.ts.get_particle(species = 'witness', iteration = i, var_list = ['z', 'w'])
        

        plt.close()
        plt.figure(figsize = (10, 6))
        plt.rcParams['agg.path.chunksize'] = 10000
        nD = plt.hist(self.zd, bins = len(Ez[self.maskD]), facecolor = 'r', linewidth = 0.2, weights = self.wd)[0]
        nW = plt.hist(self.zw, bins = (len(Ez[self.maskW])), facecolor = 'b', linewidth = 0.2, weights = self.ww)[0]
        if self.recovery:
            # zr, wr = self.ts.get_particle(species = 'recovery', iteration = i, var_list = ['z', 'w'])
            nR = plt.hist(self.zr, bins = (len(Ez[self.maskR])), facecolor = 'g', linewidth = 0.2, weights = self.wr)[0]
        else:
            nR = 0
        plt.close()

        if plot:
            plt.plot(self.info.z[self.maskD], nD, 'r', alpha = .5)
            plt.plot(self.info.z[self.maskW], nW, 'r', alpha = .5)
            plt.plot(self.info.z[self.maskR], nR, 'r', alpha = .5)


        return nD, nW, nR

    def emittance(self, sigma_x, sigma_ux, normalized: bool):
        """
        Calculates normalized emittance

        Parameters
        ----------
        normalized : bool
            If True, assumes sigma_x and sigma_ux are normalized by the skin depth and m_e*c, respectively.
        n0 : float
            Plasma density (in cm^-3)
        """

        if normalized:
            return sigma_x * self.kp_inv * sigma_ux # m rad
        else:
            return sigma_x * sigma_ux # m rad

    def ux(self, eps_x, sigma_x, normalized: bool):
        """
        Calculates normalized transverse momentum

        Parameters
        ----------
        eps_x : float
            normalized transverse emittance (in m rad)
        normalized : bool
            If True, assumes sigma_x is normalized by the skin depth
        n0 : float
            Plasma density (in cm^-3)
        
        Returns
        -------
        ux : float
            transverse momentum normalized to m_e*c
        """
        if normalized:
            return eps_x / (sigma_x * self.kp_inv)
        else:
            return eps_x / sigma_x



