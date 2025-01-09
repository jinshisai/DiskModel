# import modules
import numpy as np
import matplotlib.pyplot as plt
import sys
import copy
from scipy.interpolate import griddata, interpn
from scipy.interpolate import RegularGridInterpolator
from astropy import constants, units

from .funcs import beam_convolution, gaussian2d
from .grid import Nested2DGrid
from diskmodel2d import spectra


### constants
Ggrav  = constants.G.cgs.value        # Gravitational constant
Msun   = constants.M_sun.cgs.value    # Solar mass (g)
Lsun   = constants.L_sun.cgs.value    # Solar luminosity (erg s^-1)
Rsun   = constants.R_sun.cgs.value    # Solar radius (cm)
clight = constants.c.cgs.value        # light speed (cm s^-1)
kb     = constants.k_B.cgs.value      # Boltzman coefficient
sigsb  = constants.sigma_sb.cgs.value # Stefan-Boltzmann constant (erg s^-1 cm^-2 K^-4)
mH     = constants.m_p.cgs.value      # Proton mass (g)
hp     = constants.h.cgs.value # Planck constant [erg s]

# unit
auTOcm = units.au.to('cm') # 1 au (cm)

# Ignore divide-by-zero warning
np.seterr(divide='ignore')


class Builder(object):
    '''
    A disk model with Two Thick Layers (TTL) with a thin dust layer.

    '''

    def __init__(self, model,
        axes_model: list, axes_sky: list, 
        xlim: list | None = None, ylim: list | None = None,
        nsub: list | None = None, reslim: float = 10,
        beam: list | None = None,
        coordinate_type = 'polar'):
        '''
        Set up model grid and initialize model.

        Parameters
        ----------
        axes (list of axes): Three dimensional coordinates aligned plane of sky (au).
        '''
        super(Builder, self).__init__()

        # model
        self.model = model()
        self._model = model

        # model grid
        if coordinate_type == 'polar':
            self.build_polar_grid(axes_model)

        # sky grid
        self.reslim = reslim
        self.nsub = nsub
        x, y, v  = axes_sky
        self.v = v
        self.nv = len(v)
        self.nx = len(x)
        self.ny = len(y)
        self.build_sky_grid(x, y, xlim, ylim, nsub, reslim)

        # beam
        if beam is not None:
            self.define_beam(beam)
        else:
            self.beam = beam


    def define_beam(self, beam):
        '''
        Parameters
        ----------
         beam (list): Observational beam. Must be given in a format of 
                      [major (au), minor (au), pa (deg)].
        '''
        # save beam info
        self.beam = beam
        # define Gaussian beam
        nx, ny = self.skygrid.nx, self.skygrid.ny
        xx = self.skygrid.xx.copy()
        yy = self.skygrid.yy.copy()
        gaussbeam = gaussian2d(xx, yy, 1., 
            xx[ny//2 - 1 + ny%2, nx//2 - 1 + nx%2],
            yy[ny//2 - 1 + ny%2, nx//2 - 1 + nx%2],
            beam[1] / 2.35, beam[0] / 2.35, beam[2], peak=True)
        gaussbeam /= np.sum(gaussbeam)
        self.gaussbeam = gaussbeam


    def build_sky_grid(self, x, y, xlim = None, ylim = None, nsub = None, reslim = 10):
        self.skygrid = Nested2DGrid(x, y, xlim, ylim, nsub, reslim)


    def build_polar_grid(self, axes):
        r, phi = axes
        self.r = r
        self.phi = phi
        self.nr = len(r)
        self.nphi = len(phi)

        rr, phph = np.meshgrid(r, phi, indexing = 'ij')
        self.rs = rr.ravel()
        self.phis = phph.ravel()


    def set_model(self, params):
        self.model.set_params(**params)
        # geometric parameters
        #self.dx0 = params['dx0']
        #self.dy0 = params['dy0']
        self.inc = params['inc']
        self.__inc_rad = np.radians(self.inc)
        self.pa = params['pa']
        self.__pa_rad = np.radians(self.pa)


    def skygrid_info(self):
        self.skygrid.gridinfo()


    def build_model(self):
        I_int, vlos, dv = self.model.build(self.rs, self.phis)
        return I_int, vlos, dv


    def project_grid(self):
        # Convert spherical to Cartesian coordinates
        x = self.rs * np.cos(self.phis)
        y = self.rs * np.sin(self.phis)

        # Apply inclination and position angle
        y_rot = y * np.cos(self.__inc_rad)
        rot_ang = self.__pa_rad + 0.5 * np.pi
        x_rot = x * np.cos(rot_ang) + y_rot * np.sin(rot_ang)
        y_rot = -x * np.sin(rot_ang) + y_rot * np.cos(rot_ang)

        #self._x = x
        #self._y = y
        self.xrot = x_rot
        self.yrot = y_rot


    def project_quantity(self, q):
        # interpolator
        #interpolator = RegularGridInterpolator(
        #    (self.xrot, self.yrot), q, bounds_error=False, fill_value=0)
        # projection
        #q_proj = interp_int((self.skygrid.xnest, self.skygrid.ynest))
        q_proj = griddata(
            (self.xrot, self.yrot), q, (self.skygrid.xnest, self.skygrid.ynest),
            method = 'linear', fill_value = 0.)
        #q_proj = interpn((self.xrot, self.yrot), 
        #    q, np.array([self.skygrid.xnest, self.skygrid.ynest]),
        #    bounds_error = False, fill_value = 0.)
        return q_proj



    def build_cube(self):
        I_int, vlos, dv = self.build_model()
        self.project_grid()

        ''' new version; line profile first
        lnprofs = spectra.glnprof_series(self.v, vlos, dv) # x,y,v
        lnprofs[np.isnan(lnprofs)] = 0.
        _Iv = np.tile(I_int, (self.nv, 1,)) * lnprofs

        Iv = np.array([
            self.project_quantity(_Iv[i,:]) for i in range(self.nv)
            ])
        '''

        #'''old version; projection first
        I_proj = self.project_quantity(I_int)
        v_proj = self.project_quantity(vlos)
        dv_proj = self.project_quantity(dv)

        # line profile function
        lnprofs = spectra.glnprof_series(self.v, v_proj, dv_proj) # x,y,v
        Iv = np.tile(I_proj, (self.nv, 1,)) * lnprofs
        #'''

        # collapse
        Iv = self.skygrid.high_dimensional_collapse(Iv, 
            fill = 'zero',
            collapse_mode = 'mean')


        # Convolve beam if given
        if self.beam is not None:
            Iv = beam_convolution(
                self.skygrid.xx.copy(), 
                self.skygrid.yy.copy(), Iv, 
                self.beam, self.gaussbeam)

        return Iv


def rot2d(x, y, ang):
    return x * np.cos(ang) - y * np.sin(ang), x * np.sin(ang) + y * np.cos(ang)

def xrot(x, y, z, ang):
    return x, y * np.cos(ang) - z * np.sin(ang), y * np.sin(ang) + z * np.cos(ang)

def yp2y(yp, z, inc):
    """
    Deproject y' of the plane of the sky (PoS) coordindates to y of the disk coordinates.
    The height on the disk coordindate, z, must be given.
     yp = y cos(i) + z sin(i)
    therefore,
     y = (yp - z sin(i)) / cos(i)

    Parameters
    ----------
     yp (float or array): y' of PoS coordinates (any unit of distance.
     z (float or array): z of the disk local coordinates (any unit of distance).
     inc (float): Inclination angle of the disk (rad).
    """
    return (yp - z * np.sin(inc)) / np.cos(inc)


# Planck function
def Bv(T,v):
    '''
    Calculate Plack function of frequency in cgs.
    Unit is [erg s-1 cm-2 Hz-1 str-1].

    T: temprature [K]
    v: frequency [GHz]
    '''
    v = v * 1.e9 # GHz --> Hz
    #print((hp*v)/(kb*T))
    exp=np.exp((hp*v)/(kb*T)) - 1.0
    fterm=(2.0*hp*v*v*v)/(clight*clight)
    Bv=fterm/exp
    #print(exp, T, v)
    return Bv



# Planck function
def Bvppx(T, v, px, py, dist = 140., au = True):
    '''
    Calculate Plack function of frequency in cgs.
    Unit is [erg s-1 cm-2 Hz-1 str-1].

    T: temprature [K]
    v: frequency [GHz]
    '''
    # unit
    v = v * 1.e9 # GHz --> Hz

    # Bv in cgs
    exp = np.exp((hp*v)/(kb*T)) - 1.0
    fterm = (2.0*hp*v*v*v)/(clight*clight)
    Bv = fterm / exp

    # From cgs to Jy/str
    Bv = Bv*1.e-7*1.e4 # cgs --> MKS
    Bv = Bv*1.0e26     # MKS --> Jy (Jy = 10^-26 Wm-2Hz-1)

    # Jy/str -> Jy/pixel
    if au:
        px = np.radians(px / dist / 3600.) # au --> radian
        py = np.radians(py / dist / 3600.) # au --> radian
    else:
        px = np.radians(px) # deg --> rad
        py = np.radians(py) # deg --> rad
    # one_pixel_area = pixel*pixel (rad^2)
    # Exactly, one_pixel_area = 4.*np.arcsin(np.sin(psize*0.5)*np.sin(psize*0.5))
    #  but the result is almost the same pixel cuz pixel area is much small.
    # (When psize = 20 au and dist = 140 pc, S_apprx/S_acc = 1.00000000000004)
    # I [Jy/pixel]   = I [Jy/sr] * one_pixel_area
    one_pixel_area = np.abs(px*py)
    Bv *= one_pixel_area # Iv (Jy per pixel)
    return Bv



# Jy/beam
def Bv_Jybeam(T,v,bmaj,bmin):
    '''
    Calculate Plack function of frequency in cgs.
    Unit is [erg s-1 cm-2 Hz-1 str-1].

    T: temprature [K]
    v: frequency [GHz]
    bmaj, bmin: beamsize [arcsec]
    '''

    # units
    bmaj = np.radians(bmaj / 3600.) # arcsec --> radian
    bmin = np.radians(bmin / 3600.) # arcsec --> radian
    v = v * 1.e9 # GHz --> Hz

    # coefficient for unit convertion
    # Omg_beam (sr) = (pi/4ln(2))*beam (rad^2)
    # I [Jy/beam] / Omg_beam = I [Jy/sr]
    C2 = np.pi/(4.*np.log(2.))  # beam(rad) -> beam (sr)
    bTOstr = bmaj * bmin * C2  # beam --> str


    #print(np.nanmax((hp*v)/(kb*T)), np.nanmin(T))
    exp = np.exp((hp*v)/(kb*T)) - 1.0
    #print(np.nanmax(exp))
    fterm=(2.0*hp*v*v*v)/(clight*clight)
    Bv = fterm / exp

    # cgs --> Jy/beam
    Bv = Bv*1.e-7*1.e4 # cgs --> MKS
    Bv = Bv*1.0e26     # MKS --> Jy (Jy = 10^-26 Wm-2Hz-1)
    Bv = Bv*bTOstr     # Jy/str --> Jy/beam
    return Bv