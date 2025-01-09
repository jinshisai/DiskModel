import numpy as np
import matplotlib.pyplot as plt
import time

from diskmodel2d import Builder, SSDisk
from diskmodel2d.grid import SubGrid2D
from imfits import Imfits
from imfits.drawmaps import AstroCanvas


def main():
    # --------- input ---------
    # model params
    Ic, rc, beta, gamma = [1., 600., 1.5, 1.]
    inc = 70.
    pa = 69.
    ms = 1.6
    vsys = 7.4
    # power-law line width profile (km/s)
    dv, pdv = 0.2, 0.25 # 0.07 at 100 au, T \propto r^-0.5
    # constant line width
    dv, pdv = 0.07, 0.

    params = {'Ic': Ic, 'rc': rc, 'beta': beta, 'gamma': gamma,
    'inc': inc, 'pa': pa, 'ms': ms, 'vsys': vsys, 'dv': dv, 'pdv': pdv}

    # object
    f = 'l1489.c18o.contsub.gain01.rbp05.mlt100.cf15.pbcor.croped.fits'
    dist = 140.

    # model grid
    rmin = 0.1
    re = np.logspace(np.log10(rmin), 3, 512+1)
    phie = np.linspace(0., 2. * np.pi, 256+1)
    rc = 0.5 * (re[1:] + re[:-1])
    phic = 0.5 * (phie[1:] + phie[:-1])
    # -------------------------


    # --------- main ----------
    # read fits file
    cube = Imfits(f)
    cube.trim_data([-5., 5.,], [-5.,5.],)# [4.4, 10.4])
    _x = cube.xaxis * 3600. * dist # in au
    _y = cube.yaxis * 3600. * dist # in au
    xx = cube.xx * 3600. * dist # in au
    yy = cube.yy * 3600. * dist # in au
    v = cube.vaxis # km/s
    delv = cube.delv
    beam_au = [cube.beam[0] * dist, cube.beam[1] * dist, cube.beam[2]]
    #beam_au = None

    # model on finner grid
    subgrid = SubGrid2D(_x, _y, 1) # use number > 1 if refine grid
    x = subgrid.x_sub
    y = subgrid.y_sub


    # model
    model = Builder(SSDisk, [rc, phic], [x, y, v],
        nsub = [2,2,2,2,2,2,2,2], reslim = 40, beam = beam_au) #2,2,2,2,2,2
    model.set_model(params)
    model.skygrid_info()

    # visualize grid
    I_int, vlos, dv = model.build_model()
    model.project_grid()
    I_proj = model.project_quantity(I_int)
    I_proj[I_proj <= 0.] = np.nan # for plot
    v_proj = model.project_quantity(vlos)
    dv_proj = model.project_quantity(dv)
    for q, l in zip([np.log10(I_proj), v_proj, dv_proj], ['intensity', 'vlos', 'dv']):
        ax = model.skygrid.visualize_grid(q, showfig = False, 
            outname = 'grid_visualize_%s_demo.png'%l, savefig = True,)
        ax.text(0.1, 0.9, l, transform = ax.transAxes, ha = 'left', va = 'top')
        plt.show()
        plt.close()

    # build cube
    start = time.time()
    modelcube = model.build_cube()
    end = time.time()
    print('Takes %.2fs'%(end-start))


    # rebin
    modelcube = subgrid.binning_onsubgrid_layered(modelcube)

    # plot
    canvas = AstroCanvas((7,10),(0,0), imagegrid=True)
    canvas.channelmaps(cube, contour=True, color=False,
        clevels = np.array([-3,3.,6.,9.,12.,15])*5e-3)
    vmin, vmax = np.nanmin(modelcube)*0.5, np.nanmax(modelcube)*0.5
    for i, im in enumerate(modelcube):
        if i < len(canvas.axes):
            ax = canvas.axes[i]
            ax.pcolormesh(xx / dist, yy / dist, im, shading='auto', rasterized=True,
                vmin = vmin, vmax = vmax, cmap='PuBuGn')
        else:
            break
    canvas.savefig('channel_model_demo_new')
    plt.show()


    cube_out = cube.copy()
    nv, ny, nx = modelcube.shape
    cube_out.data = modelcube.reshape(1, nv, ny, nx)
    cube_out.writeout('model_demo_new.fits', overwrite= True)
    # -------------------------

if __name__ == '__main__':
    main()