# modules
import numpy as np
import math
import matplotlib.pyplot as plt



class Nested2DGrid(object):
    """docstring for NestedGrid"""
    def __init__(self, x, y,
        xlim: list | None = None, ylim: list | None = None, 
        nsub: list | None = None, reslim = 10,):
        super(Nested2DGrid, self).__init__()
        # save axes of the mother grid
        self.x = x
        self.y = y
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        self.dx = dx
        self.dy = dy
        xe = np.hstack([x - self.dx * 0.5, x[-1] + self.dx * 0.5])
        ye = np.hstack([y - self.dy * 0.5, y[-1] + self.dy * 0.5])
        self.xe, self.ye = xe, ye
        ny, nx = len(y), len(x)
        self.ny, self.nx = ny, nx
        self.xx, self.yy = np.meshgrid(x, y)
        self.Lx, self.Ly = xe[-1] - xe[0], ye[-1] - ye[0]


        # nested grid
        self.nsub = nsub
        nlevels = 1 if nsub is None else len(nsub) + 1
        self.nlevels = nlevels
        # original 1D axes
        self.xaxes = [None] * nlevels
        self.yaxes = [None] * nlevels
        self.xaxes[0], self.yaxes[0] = x, y
        # grid sizes
        self.ngrids = np.zeros((nlevels, 2)).astype(int)
        self.ngrids[0,:] = np.array([ny, nx])
        # nested grid
        self.xnest = self.xx.ravel()
        self.ynest = self.yy.ravel()
        self.partition = self.xnest.size
        # starting and ending indices
        self.xinest = [-1,-1]
        self.yinest = [-1,-1]
        # nest
        if self.nlevels > 1:
            if (np.array([xlim, ylim]) == None).any():
                _xlim, _ylim = self.get_nestinglim(reslim = reslim)
                if xlim is None: xlim = _xlim
                if ylim is None: ylim = _ylim
            self.xlim, self.ylim = xlim.copy(), ylim.copy()
            self.xlim.insert(0, [xe[0], xe[-1]])
            self.ylim.insert(0, [ye[0], ye[-1]])
            self.nest()
        else:
            self.xlim = [xe[0], xe[-1]]
            self.ylim = [ye[0], ye[-1]]


    def get_nestinglim(self, reslim = 5):
        xlim = []
        ylim = []
        _dx, _dy = np.abs(self.dx), np.abs(self.dy)
        for l in range(self.nlevels - 1):
            xlim.append([-_dx * reslim, _dx * reslim])
            ylim.append([-_dy * reslim, _dy * reslim])
            _dx, _dy = np.abs(np.array([_dx, _dy])) / self.nsub[l]

        return xlim, ylim


    def check_symmetry(self, decimals = 5):
        nx, ny = self.nx, self.ny
        xc = np.round(self.xc, decimals)
        yc = np.round(self.yc, decimals)
        _xcent = (xc == 0.) if nx%2 == 1 else (xc == - np.round(self.xx[ny//2 - 1, nx//2 - 1], decimals))
        _ycent = (yc == 0.) if ny%2 == 1 else (yc == - np.round(self.yy[ny//2 - 1, nx//2 - 1], decimals))
        delxs = (self.xx[1:,1:] - self.xx[:-1,:-1]) / self.dx
        delys = (self.yy[1:,1:] - self.yy[:-1,:-1]) / self.dy
        _xdel = (np.round(delxs, decimals) == 1. ).all()
        _ydel = (np.round(delys, decimals)  == 1. ).all()
        cond = [_xdel, _ydel] # _xcent, _ycent,
        return all(cond), cond


    def get_grid(self, l):
        '''
        Get grid on the l layer.
        '''
        _ny, _nx = self.ngrids[l]
        partition = self.partition[l:l+2]
        # if it is not collapsed
        if self.xnest[partition[0]:].size == _nx * _ny:
            #partition = self.partition[l:l+2]
            xx = self.xnest[partition[0]:].reshape(_nx, _ny)
            yy = self.ynest[partition[0]:].reshape(_nx, _ny)
        else:
            # else
            x, y = self.xaxes[l], self.yaxes[l]
            xx, yy = np.meshgrid(x, y)
        return xx, yy


    def nest(self):
        '''
        l - 1 is the mother grid layer. l is the child grid layer.
        '''
        # initialize
        partition = [0]
        xnest = np.array([])
        ynest = np.array([])
        dxnest = []
        dynest = []
        for l in range(1, self.nlevels):
            # axes of the parental grid
            x, y = self.xaxes[l-1], self.yaxes[l-1]
            dxnest.append(x[1] - x[0])
            dynest.append(y[1] - y[0])

            # make childe grid
            ximin, ximax, yimin, yimax, x_sub, y_sub = \
            nestgrid_2D(x, y, self.xlim[l], self.ylim[l], self.nsub[l-1])
            self.xinest += [ximin, ximax] # starting and ending indices on the upper-layer grid
            self.yinest += [yimin, yimax]
            self.xaxes[l], self.yaxes[l] = x_sub, y_sub
            self.ngrids[l,:] = np.array([len(x_sub), len(y_sub)])

            # parental grid
            _nx, _ny = self.ngrids[l-1,:]
            #else:
            xx, yy = np.meshgrid(x, y)

            # devide the upper grid into six sub-regions
            # Region 1:  x from 0 to ximin, all y
            R1x = xx[:, :ximin].ravel()
            R1y = yy[:, :ximin].ravel()
            # Region 2: x from ximax+1 to nx, all y and z
            R2x = xx[:, ximax+1:].ravel()
            R2y = yy[:, ximax+1:].ravel()
            # Region 3: x from xi0 to ximax, y from 0 to yimin
            R3x = xx[:yimin, ximin:ximax+1].ravel()
            R3y = yy[:yimin, ximin:ximax+1].ravel()
            # Region 4: x from xi0 to ximax, y from yimax+1 to ny, and all z
            R4x = xx[yimax+1:, ximin:ximax+1].ravel()
            R4y = yy[yimax+1:, ximin:ximax+1].ravel()

            # update
            Rx = np.concatenate([R1x, R2x, R3x, R4x])
            partition.append(partition[l-1] + Rx.size)
            xnest = np.concatenate([xnest, Rx]) # update
            ynest = np.concatenate([ynest, R1y, R2y, R3y, R4y]) # update

        # child grid
        xx_sub, yy_sub = np.meshgrid(x_sub, y_sub)
        xnest = np.concatenate([xnest, xx_sub.ravel()]) # update
        ynest = np.concatenate([ynest, yy_sub.ravel()]) # update
        dxnest.append(x_sub[1] - x_sub[0])
        dynest.append(y_sub[1] - y_sub[0])
        nd = xnest.size
        partition.append(nd)
        self.xnest = xnest
        self.ynest = ynest
        self.dxnest = dxnest
        self.dynest = dynest
        self.partition = partition
        self.nd = nd


    def to_perpix(self, d):
        ndim = len(d.shape)
        if ndim == 1:
            nd = d.size
            d = d.reshape((1, nd))
            shape = (nd)
        elif ndim >= 2:
            shape = d.shape
            _n = math.prod(list(shape)[:-1])
            d = d.reshape((_n, shape[-1]))

        #print('to per pixel')
        for l in range(self.nlevels):
            partition = self.partition[l:l+2]
            dx = self.dxnest[l]
            dy = self.dynest[l]
            #print('(l,dx,dy) = (%i, %.2e, %.2e)'%(l, dx, dy))
            d[:,partition[0]:partition[1]] *= np.abs(dx*dy)

        d = d.reshape(shape)
        return d


    def collapse(self, d, upto = None):
        '''
        Collapse given data to the mother grid.

        Parameters
        ----------
        d (list): List of data on the nested grid
        '''
        lmax = 0 if upto is None else upto
        #print(self.partition[-2]:self.partition[-1])
        d_col = d[self.partition[-2]:self.partition[-1]] # starting from the inner most grid
        d_col = d_col.reshape(tuple(self.ngrids[-1,:]))
        for l in range(self.nlevels-1,lmax,-1):
            nsub = self.nsub[l-1]
            ximin, ximax = self.xinest[l*2:(l+1)*2]
            yimin, yimax = self.yinest[l*2:(l+1)*2]
            # collapse data on the inner grid
            _d = self.binning_onsubgrid_layered(d_col, nsub)
            #print(ximin, ximax, yimin, yimax, zimin, zimax)

            # go upper layer
            nx, ny = self.ngrids[l-1,:] # size of the upper layer
            d_col = np.full((nx, ny), np.nan)

            # insert collapsed data
            d_col[yimin:yimax+1, ximin:ximax+1] = _d

            # fill upper layer data
            d_up = d[self.partition[l-1]:self.partition[l]]

            # Region 1: x from zero to ximin, all y and z
            d_col[:, :ximin] = \
            d_up[:ximin * ny].reshape((ny, ximin))
            # Region 2: x from ximax to nx, all y and z
            i0 = ximin * ny
            i1 = i0 + (nx - ximax - 1) * ny
            d_col[:, ximax+1:] = \
            d_up[i0:i1].reshape(
                (ny, nx - ximax - 1))
            # Region 3
            i0 = i1
            i1 = i0 + (ximax + 1 - ximin) * yimin
            d_col[:yimin, ximin:ximax+1] = \
            d_up[i0:i1].reshape(
                (yimin, ximax + 1 - ximin))
            # Region 4
            i0 = i1
            i1 = i0 + (ximax + 1 - ximin) * (ny - yimax - 1)
            d_col[yimax+1:, ximin:ximax+1] = \
            d_up[i0:i1].reshape(
                (ny - yimax - 1, ximax + 1 - ximin))

        return d_col


    def high_dimensional_collapse(self, 
        d, upto = None, fill = 'zero', collapse_mode = 'mean'):
        ndim = len(d.shape)
        if ndim == 1:
            dcol = self.collapse(d, upto = upto)
        elif ndim == 2:
            dcol = self.collapse_extra_1d(d, 
                upto = upto, fill = fill, 
                collapse_mode = collapse_mode)
        else:
            print('ERROR\thigh_dimensional_collapse: currently only upto ndim=2 is supported.')
            return 0
        return dcol


    def collapse_extra_1d(self, d, 
        upto = None, fill = 'zero', collapse_mode = 'mean'):
        '''
        Collapse given data to the mother grid.

        Parameters
        ----------
        d (list): List of data on the nested grid
        '''
        lmax = 0 if upto is None else upto
        nv, _ = d.shape
        d_col = d[:,self.partition[-2]:self.partition[-1]] # starting from the inner most grid
        d_col = d_col.reshape((nv, self.ngrids[-1,:][0], self.ngrids[-1,:][1]))
        for l in range(self.nlevels-1,lmax,-1):
            nsub = self.nsub[l-1]
            ximin, ximax = self.xinest[l*2:(l+1)*2]
            yimin, yimax = self.yinest[l*2:(l+1)*2]
            # collapse data on the inner grid
            if (collapse_mode == 'mean') | (collapse_mode == 'sum'):
                d_col[np.isnan(d_col)] = 0.
                _d = self.binning_onsubgrid_layered(d_col, nsub, 
                    binning_mode = collapse_mode)
            elif collapse_mode == 'integrate':
                #d_col *= np.abs(self.dxnest[l] * self.dynest[l]) # per pix
                _d = self.binning_onsubgrid_layered(
                    d_col, nsub, 
                    binning_mode = 'sum')
                _d *= np.abs(self.dxnest[l] * self.dynest[l]) / np.abs(self.dxnest[l-1] * self.dynest[l-1]) # per a^-2 with a of the pixel unit)
                print('(l, Rarea_l-1,l) = (%i, %.2f)'%(l, np.abs(self.dxnest[l] * self.dynest[l]) / np.abs(self.dxnest[l-1] * self.dynest[l-1])))
            #print(ximin, ximax, yimin, yimax)

            # go upper layer
            ny, nx = self.ngrids[l-1,:] # size of the upper layer
            if fill == 'nan':
                d_col = np.full((nv, ny, nx), np.nan)
            elif fill == 'zero':
                d_col = np.zeros((nv, ny, nx))
            else:
                d_col = np.full((nv, ny, nx), fill)

            # insert collapsed data
            d_col[:, yimin:yimax+1, ximin:ximax+1] = _d

            # fill upper layer data
            d_up = d[:, self.partition[l-1]:self.partition[l]]

            # Region 1: x from zero to ximin, all y
            d_col[:, :, :ximin] = \
            d_up[:, :ximin * ny].reshape((nv, ny, ximin))
            # Region 2: x from ximax to nx, all y and z
            i0 = ximin * ny
            i1 = i0 + (nx - ximax - 1) * ny
            d_col[:, :, ximax+1:] = \
            d_up[:, i0:i1].reshape(
                (nv, ny, nx - ximax - 1))
            # Region 3
            i0 = i1
            i1 = i0 + (ximax + 1 - ximin) * yimin
            d_col[:, :yimin, ximin:ximax+1] = \
            d_up[:, i0:i1].reshape(
                (nv, yimin, ximax + 1 - ximin))
            # Region 4
            i0 = i1
            i1 = i0 + (ximax + 1 - ximin) * (ny - yimax - 1)
            d_col[:, yimax+1:, ximin:ximax+1] = \
            d_up[:, i0:i1].reshape(
                (nv, ny - yimax - 1, ximax + 1 - ximin))

        return d_col


    def nest_sub(self, xlim, ylim, nsub):
        # error check
        if (len(xlim) != 2) | (len(ylim) != 2):
            print('ERROR\tnest: Input xlim and ylim must be list as [min, max].')
            return 0
        # decimals
        xlim = [np.round(xlim[0], self.decimals), np.round(xlim[1], self.decimals)]
        ylim = [np.round(ylim[0], self.decimals), np.round(ylim[1], self.decimals)]

        self.nsub = nsub
        self.xlim_sub, self.ylim_sub, self.zlim_sub = xlim, ylim, zlim
        ximin, ximax = index_between(self.x, xlim, mode='edge')[0]
        yimin, yimax = index_between(self.y, ylim, mode='edge')[0]
        zimin, zimax = index_between(self.z, zlim, mode='edge')[0]
        _nx = ximax - ximin + 1
        _ny = yimax - yimin + 1
        _nz - zimax - zimin + 1
        xemin, xemax = self.xe[ximin], self.xe[ximax + 1]
        yemin, yemax = self.ye[yimin], self.ye[yimax + 1]
        zemin, zemax = self.ze[zimin], self.ze[zimax + 1]
        self.xi0, self.xi1 = ximin, ximax # Starting and ending indices of nested grid
        self.yi0, self.yi1 = yimin, yimax # Starting and ending indices of nested grid
        self.zi0, self.zi1 = zimin, zimax # Starting and ending indices of nested grid

        # nested grid
        xe_sub = np.linspace(xemin, xemax, _nx * nsub + 1)
        ye_sub = np.linspace(yemin, yemax, _ny * nsub + 1)
        ze_sub = np.linspace(zemin, zemax, _nz * nsub + 1)
        x_sub = 0.5 * (xe_sub[:-1] + xe_sub[1:])
        y_sub = 0.5 * (ye_sub[:-1] + ye_sub[1:])
        z_sub = 0.5 * (ze_sub[:-1] + ze_sub[1:])
        xx_sub, yy_sub, zz_sub = np.meshgrid(x_sub, y_sub, z_sub, indexing = 'ij')
        self.xe_sub, self.ye_sub, self.ze_sub = xe_sub, ye_sub, ze_sub
        self.x_sub, self.y_sub, z_sub = x_sub, y_sub, z_sub
        self.xx_sub, self.yy_sub, self.zz_sub = xx_sub, yy_sub, zz_sub
        self.dx_sub, self.dy_sub, self.dz_sub = self.dx / nsub, self.dy / nsub, self.dz / nsub
        self.nx_sub, self.ny_sub, self.nz_sub = len(x_sub), len(y_sub), len(z_sub)
        return xx_sub, yy_sub, zz_sub


    def binning_onsubgrid(self, data):
        nbin = self.nsub
        d_avg = np.array([
            data[j::nbin, i::nbin]
            for j in range(nbin) for i in range(nbin)
            ])
        return np.nanmean(d_avg, axis = 0)


    def binning_onsubgrid_layered(self, 
        data, nbin, binning_mode = 'mean'):
        dshape = len(data.shape)
        if dshape == 2:
            d_avg = np.array([
                data[j::nbin, i::nbin]
                for j in range(nbin) for i in range(nbin)
                ])
        elif dshape == 3:
            d_avg = np.array([
                data[:, j::nbin, i::nbin]
                for j in range(nbin) for i in range(nbin)
                ])
        elif dshape ==4:
            d_avg = np.array([
                data[:, :, j::nbin, i::nbin]
                for j in range(nbin) for i in range(nbin)
                ])
        else:
            print('ERROR\tbinning_onsubgrid_layered: only Nd of data of 3-5 is now supported.')
            return 0

        if binning_mode == 'mean':
            return np.nanmean(d_avg, axis = 0)
        elif binning_mode == 'sum':
            return np.nansum(d_avg, axis = 0)
        else:
            print('ERROR\binning_onsubgrid_layered: binning_mode must be mean or sum.')
            return 0


    def gridinfo(self, units = ['au', 'au']):
        ux, uy = units
        print('Nesting level: %i'%self.nlevels)
        print('Resolutions:')
        for l in range(self.nlevels):
            dx = self.xaxes[l][1] - self.xaxes[l][0]
            dy = self.yaxes[l][1] - self.yaxes[l][0]
            print('   l=%i: (dx, dy) = (%.2e %s, %.2e %s)'%(l, dx, ux, dy, uy))
            print('      : (xlim, ylim) = (%.2e to %.2e %s, %.2e to %.2e %s)'%(
                self.xlim[l][0], self.xlim[l][1], ux,
                self.ylim[l][0], self.ylim[l][1], uy))



    def edgecut_indices(self, xlength, ylength):
        # odd or even
        x_oddeven = self.nx%2
        y_oddeven = self.ny%2
        # edge indices for subgrid
        xi = int(xlength / self.dx_sub) if self.dx_sub > 0 else int(- xlength / self.dx_sub)
        yi = int(ylength / self.dy_sub)
        _nx_resub = int(self.nx_sub - 2 * xi) // self.nsub # nx of subgrid after cutting edge
        _ny_resub = int(self.ny_sub - 2 * yi) // self.nsub # ny of subgrid after cutting edge
        # fit odd/even
        if _nx_resub%2 != x_oddeven: _nx_resub += 1
        if _ny_resub%2 != y_oddeven: _ny_resub += 1
        # nx, ny of the new subgrid and new xi and yi
        nx_resub = _nx_resub * self.nsub
        ny_resub = _ny_resub * self.nsub
        xi = (self.nx_sub - nx_resub) // 2
        yi = (self.ny_sub - ny_resub) // 2
        # for original grid
        xi0 = int((self.nx - nx_resub / self.nsub) * 0.5)
        yi0 = int((self.ny - ny_resub / self.nsub) * 0.5)
        #print(nx_resub / self.nsub, self.nx - xi0 - xi0)
        return xi, yi, xi0, yi0


    def visualize_grid(self, d, 
        ax = None, vmin = None, vmax = None,
        showfig = False, cmap = 'viridis', outname = None,
        savefig = False):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        if vmax is None: vmax = np.nanmax(d)
        if vmin is None: vmin = np.nanmin(d)
        # from upper to lower
        for l in range(self.nlevels):
            nx, ny = self.ngrids[l,:]
            xmin, xmax = self.xlim[l]
            ymin, ymax = self.ylim[l]

            d_plt = self.collapse(
                d, upto = l)

            # hide parental layer
            if l <= self.nlevels-2:
                ximin, ximax = self.xinest[(l+1)*2:(l+2)*2]
                yimin, yimax = self.xinest[(l+1)*2:(l+2)*2]
                d_plt[yimin:yimax+1, ximin:ximax+1] = np.nan

            #ax.imshow(d_plt, extent = (zmin, zmax, xmin, xmax),
            #    alpha = 1., vmax = vmax, vmin = vmin, origin = 'upper', cmap = cmap)

            _xx, _yy = self.get_grid(l)
            im = ax.pcolormesh(_xx, _yy, d_plt, 
                alpha = 1., vmax = vmax, vmin = vmin, cmap = cmap)
            rect = plt.Rectangle((xmin, ymin), 
                xmax - xmin, ymax - ymin, edgecolor = 'white', facecolor = "none",
                linewidth = 0.5, ls = '--')
            ax.add_patch(rect)

        ax.set_xlim(self.xlim[0])
        ax.set_ylim(self.ylim[0])
        plt.colorbar(im, ax = ax)

        if showfig: plt.show()
        if savefig: plt.savefig(outname, transparent = True, dpi = 300)
        return ax





class Nested3DGrid(object):
    """docstring for NestedGrid"""
    def __init__(self, x, y, z, 
        xlim = None, ylim = None, zlim = None, 
        nsub = None, reslim = 5,):
        super(Nested3DGrid, self).__init__()
        # save axes of the mother grid
        self.x = x
        self.y = y
        self.z = z
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        dz = z[1] - z[0]
        self.dx = dx
        self.dy = dy
        self.dz = dz
        xe = np.hstack([x - self.dx * 0.5, x[-1] + self.dx * 0.5])
        ye = np.hstack([y - self.dy * 0.5, y[-1] + self.dy * 0.5])
        ze = np.hstack([z - self.dz * 0.5, z[-1] + self.dz * 0.5])
        self.xe, self.ye, self.ze = xe, ye, ze
        nz, ny, nx = len(z), len(y), len(x)
        self.nz, self.ny, self.nx = nz, ny, nx
        self.xx, self.yy, self.zz = np.meshgrid(x, y, z, indexing='ij')
        self.Lx, self.Ly, self.Lz = xe[-1] - xe[0], ye[-1] - ye[0], ze[-1] - ze[0]


        # nested grid
        self.nsub = nsub
        nlevels = 1 if nsub is None else len(nsub) + 1
        self.nlevels = nlevels
        # original 1D axes
        self.xaxes = [None] * nlevels
        self.yaxes = [None] * nlevels
        self.zaxes = [None] * nlevels
        self.xaxes[0], self.yaxes[0], self.zaxes[0] = x, y, z
        # grid sizes
        self.ngrids = np.zeros((nlevels, 3)).astype(int)
        self.ngrids[0,:] = np.array([nx, ny, nz])
        # nested grid
        self.xnest = self.xx.ravel() # save all grid info in 1D array
        self.ynest = self.yy.ravel()
        self.znest = self.zz.ravel()
        self.partition = [0, self.xnest.size] # partition indices
        # starting and ending indices
        self.xinest = [-1,-1]
        self.yinest = [-1,-1]
        self.zinest = [-1,-1]
        # nest
        if self.nlevels > 1:
            if (np.array([xlim, ylim, zlim]) == None).any():
                _xlim, _ylim, _zlim = self.get_nestinglim(reslim = reslim)
                if xlim is None: xlim = _xlim
                if ylim is None: ylim = _ylim
                if zlim is None: zlim = _zlim
            self.xlim, self.ylim, self.zlim = xlim.copy(), ylim.copy(), zlim.copy()
            self.xlim.insert(0, [xe[0], xe[-1]])
            self.ylim.insert(0, [ye[0], ye[-1]])
            self.zlim.insert(0, [ze[0], ze[-1]])

            self.nest()
        else:
            self.xlim = [xe[0], xe[-1]]
            self.ylim = [ye[0], ye[-1]]
            self.zlim = [ze[0], ze[-1]]


    def get_nestinglim(self, reslim = 5):
        xlim = []
        ylim = []
        zlim = []
        _dx, _dy, _dz = np.abs(self.dx), np.abs(self.dy), np.abs(self.dz)
        for l in range(self.nlevels - 1):
            xlim.append([-_dx * reslim, _dx * reslim])
            ylim.append([-_dy * reslim, _dy * reslim])
            zlim.append([-_dz * reslim, _dz * reslim])
            _dx, _dy, _dz = np.abs(np.array([_dx, _dy, _dz])) / self.nsub[l]

        return xlim, ylim, zlim


    def get_grid(self, l):
        '''
        Get grid on the l layer.
        '''
        _nx, _ny, _nz = self.ngrids[l,:]
        partition = self.partition[l:l+2]
        # if it is not collapsed
        if self.xnest[partition[0]:partition[1]].size == _nx * _ny * _nz:
            xx = self.xnest[partition[0]:partition[1]].reshape(_nx, _ny, _nz)
            yy = self.ynest[partition[0]:partition[1]].reshape(_nx, _ny, _nz)
            zz = self.znest[partition[0]:partition[1]].reshape(_nx, _ny, _nz)
        else:
            # else
            x, y, z = self.xaxes[l], self.yaxes[l], self.zaxes[l]
            xx, yy, zz = np.meshgrid(x, y, z, indexing = 'ij')
        return xx, yy, zz


    def nest(self,):
        '''
        l - 1 is the mother grid layer. l is the child grid layer.
        '''
        # initialize
        partition = [0]
        xnest = np.array([])
        ynest = np.array([])
        znest = np.array([])
        for l in range(1, self.nlevels):
            # axes of the parental grid
            x, y, z = self.xaxes[l-1], self.yaxes[l-1], self.zaxes[l-1]

            # make childe grid
            ximin, ximax, yimin, yimax, zimin, zimax, x_sub, y_sub, z_sub = \
            nestgrid_3D(x, y, z, self.xlim[l], self.ylim[l], self.zlim[l], self.nsub[l-1])
            self.xinest += [ximin, ximax] # starting and ending indices on the upper-layer grid
            self.yinest += [yimin, yimax]
            self.zinest += [zimin, zimax]
            self.xaxes[l], self.yaxes[l], self.zaxes[l] = x_sub, y_sub, z_sub
            self.ngrids[l,:] = np.array([len(x_sub), len(y_sub), len(z_sub)])

            # parental grid
            _nx, _ny, _nz = self.ngrids[l-1,:]
            #if self.xnest[l-1].size == _nx * _ny * _nz:
            #    xx = self.xnest[l-1].reshape(_nx, _ny, _nz)
            #    yy = self.ynest[l-1].reshape(_nx, _ny, _nz)
            #    zz = self.znest[l-1].reshape(_nx, _ny, _nz)
            #else:
            xx, yy, zz = np.meshgrid(x, y, z, indexing = 'ij')

            # devide the upper grid into six sub-regions
            # Region 1:  x from 0 to ximin, all y and z
            R1x = xx[:ximin, :, :].ravel()
            R1y = yy[:ximin, :, :].ravel()
            R1z = zz[:ximin, :, :].ravel()
            # Region 2: x from ximax+1 to nx, all y and z
            R2x = xx[ximax+1:, :, :].ravel()
            R2y = yy[ximax+1:, :, :].ravel()
            R2z = zz[ximax+1:, :, :].ravel()
            # Region 3: x from xi0 to ximax, y from 0 to yimin, and all z
            R3x = xx[ximin:ximax+1, :yimin, :].ravel()
            R3y = yy[ximin:ximax+1, :yimin, :].ravel()
            R3z = zz[ximin:ximax+1, :yimin, :].ravel()
            # Region 4: x from xi0 to ximax, y from yimax+1 to ny, and all z
            R4x = xx[ximin:ximax+1, yimax+1:, :].ravel()
            R4y = yy[ximin:ximax+1, yimax+1:, :].ravel()
            R4z = zz[ximin:ximax+1, yimax+1:, :].ravel()
            # Region 5: x from xi0 to ximax, y from yimin to yimax and z from 0 to zimin
            R5x = xx[ximin:ximax+1, yimin:yimax+1, :zimin].ravel()
            R5y = yy[ximin:ximax+1, yimin:yimax+1, :zimin].ravel()
            R5z = zz[ximin:ximax+1, yimin:yimax+1, :zimin].ravel()
            # Region 6: x from xi0 to ximax, y from yimin to yimax and z from zimax+1 to nz
            R6x = xx[ximin:ximax+1, yimin:yimax+1, zimax+1:].ravel()
            R6y = yy[ximin:ximax+1, yimin:yimax+1, zimax+1:].ravel()
            R6z = zz[ximin:ximax+1, yimin:yimax+1, zimax+1:].ravel()

            Rx = np.concatenate([R1x, R2x, R3x, R4x, R5x, R6x])
            nl = Rx.size
            partition.append(partition[l-1] + nl)
            xnest = np.concatenate([xnest, Rx]) # update
            ynest = np.concatenate([ynest, R1y, R2y, R3y, R4y, R5y, R6y]) # update
            znest = np.concatenate([znest, R1z, R2z, R3z, R4z, R5z, R6z]) # update

        # the deepest child grid
        xx_sub, yy_sub, zz_sub = np.meshgrid(x_sub, y_sub, z_sub, indexing = 'ij')
        xnest = np.concatenate([xnest, xx_sub.ravel()]) # update
        ynest = np.concatenate([ynest, yy_sub.ravel()]) # update
        znest = np.concatenate([znest, zz_sub.ravel()]) # update
        self.xnest = xnest
        self.ynest = ynest
        self.znest = znest
        self.partition = partition


    def collapse(self, d, upto = None):
        '''
        Collapse given data to the mother grid.

        Parameters
        ----------
        d (list): List of data on the nested grid
        '''
        lmax = 0 if upto is None else upto
        d_col = d[self.partition[-1]:] # starting from the inner most grid
        d_col = d_col.reshape(tuple(self.ngrids[-1,:]))
        for l in range(self.nlevels-1,lmax,-1):
            nsub = self.nsub[l-1]
            ximin, ximax = self.xinest[l*2:(l+1)*2]
            yimin, yimax = self.yinest[l*2:(l+1)*2]
            zimin, zimax = self.zinest[l*2:(l+1)*2]
            # collapse data on the inner grid
            _d = self.binning_onsubgrid_layered(d_col, nsub)
            #print(ximin, ximax, yimin, yimax, zimin, zimax)

            # go upper layer
            nx, ny, nz = self.ngrids[l-1,:] # size of the upper layer
            d_col = np.full((nx, ny, nz), np.nan)

            # insert collapsed data
            d_col[ximin:ximax+1, yimin:yimax+1, zimin:zimax+1] = _d

            # fill upper layer data
            d_up = d[self.partition[l-1]:self.partition[l]]
            # Region 1: x from zero to ximin, all y and z
            d_col[:ximin, :, :] = \
            d_up[:ximin * ny * nz].reshape((ximin, ny, nz))
            # Region 2: x from ximax to nx, all y and z
            i0 = ximin * ny * nz
            i1 = i0 + (nx - ximax - 1) * ny * nz
            d_col[ximax+1:, :, :] = \
            d_up[i0:i1].reshape(
                (nx - ximax - 1, ny, nz))
            # Region 3
            i0 = i1
            i1 = i0 + (ximax + 1 - ximin) * yimin * nz
            d_col[ximin:ximax+1, :yimin, :] = \
            d_up[i0:i1].reshape(
                (ximax + 1 - ximin, yimin, nz))
            # Region 4
            i0 = i1
            i1 = i0 + (ximax + 1 - ximin) * (ny - yimax - 1) * nz
            d_col[ximin:ximax+1, yimax+1:, :] = \
            d_up[i0:i1].reshape(
                (ximax + 1 - ximin, ny - yimax - 1, nz))
            # Region 5
            i0 = i1
            i1 = i0 + (ximax + 1 - ximin) * (yimax + 1 - yimin) * zimin
            d_col[ximin:ximax+1, yimin:yimax+1, :zimin] = \
            d_up[i0:i1].reshape(
                (ximax + 1 - ximin, yimax + 1 - yimin, zimin))
            # Region 6
            i0 = i1
            i1 = i0 + (ximax + 1 - ximin) * (yimax + 1 - yimin) * (nz - zimax -1)
            d_col[ximin:ximax+1, yimin:yimax+1, zimax+1:] = \
            d_up[i0:].reshape(
                (ximax + 1 - ximin, yimax + 1 - yimin, nz - zimax -1))

        return d_col


    def high_dimensional_collapse(self, d, upto = None, fill = 'nan'):
        if len(d.shape) == 2:
            d_col = self.collapse_extra_1d(d, upto = upto, fill = fill)
            return d_col
        else:
            print('ERROR\thigh_dimensional_collapse: currently only 2d data are supported.')
            return 0


    def collapse_extra_1d(self, d, upto = None, fill = 'nan'):
        '''
        Collapse given data to the mother grid.

        Parameters
        ----------
        d (list): List of data on the nested grid
        '''
        lmax = 0 if upto is None else upto
        nv, nd = d.shape
        d_col = d[:, self.partition[-1]:] # starting from the inner most grid
        d_col = d_col.reshape((nv, self.ngrids[-1,:][0], self.ngrids[-1,:][1], self.ngrids[-1,:][2]))
        for l in range(self.nlevels-1,lmax,-1):
            nsub = self.nsub[l-1]
            ximin, ximax = self.xinest[l*2:(l+1)*2]
            yimin, yimax = self.yinest[l*2:(l+1)*2]
            zimin, zimax = self.zinest[l*2:(l+1)*2]
            # collapse data on the inner grid
            _d = self.binning_onsubgrid_layered(d_col, nsub)
            #print(ximin, ximax, yimin, yimax, zimin, zimax)

            # go upper layer
            nx, ny, nz = self.ngrids[l-1,:] # size of the upper layer
            if fill == 'nan':
                d_col = np.full((nv, nx, ny, nz), np.nan)
            elif fill == 'zero':
                d_col = np.zeros((nv, nx, ny, nz))
            else:
                d_col = np.full((nv, nx, ny, nz), fill)

            # insert collapsed data
            d_col[:, ximin:ximax+1, yimin:yimax+1, zimin:zimax+1] = _d

            # fill upper layer data
            d_up = d[:, self.partition[l-1]:self.partition[l]]
            # Region 1: x from zero to ximin, all y and z
            d_col[:, :ximin, :, :] = \
            d_up[:, :ximin * ny * nz].reshape((nv, ximin, ny, nz))
            # Region 2: x from ximax to nx, all y and z
            i0 = ximin * ny * nz
            i1 = i0 + (nx - ximax - 1) * ny * nz
            d_col[:, ximax+1:, :, :] = \
            d_up[:, i0:i1].reshape(
                (nv, nx - ximax - 1, ny, nz))
            # Region 3
            i0 = i1
            i1 = i0 + (ximax + 1 - ximin) * yimin * nz
            d_col[:, ximin:ximax+1, :yimin, :] = \
            d_up[:, i0:i1].reshape(
                (nv, ximax + 1 - ximin, yimin, nz))
            # Region 4
            i0 = i1
            i1 = i0 + (ximax + 1 - ximin) * (ny - yimax - 1) * nz
            d_col[:, ximin:ximax+1, yimax+1:, :] = \
            d_up[:, i0:i1].reshape(
                (nv, ximax + 1 - ximin, ny - yimax - 1, nz))
            # Region 5
            i0 = i1
            i1 = i0 + (ximax + 1 - ximin) * (yimax + 1 - yimin) * zimin
            d_col[:, ximin:ximax+1, yimin:yimax+1, :zimin] = \
            d_up[:, i0:i1].reshape(
                (nv, ximax + 1 - ximin, yimax + 1 - yimin, zimin))
            # Region 6
            i0 = i1
            i1 = i0 + (ximax + 1 - ximin) * (yimax + 1 - yimin) * (nz - zimax -1)
            d_col[:, ximin:ximax+1, yimin:yimax+1, zimax+1:] = \
            d_up[:, i0:].reshape(
                (nv, ximax + 1 - ximin, yimax + 1 - yimin, nz - zimax -1))

        return d_col


    def nest_sub(self, xlim,  ylim, zlim, nsub):
        # error check
        if (len(xlim) != 2) | (len(ylim) != 2) | (len(zlim) != 2):
            print('ERROR\tnest: Input xlim/ylim/zlim must be list as [min, max].')
            return 0
        # decimals
        xlim = [np.round(xlim[0], self.decimals), np.round(xlim[1], self.decimals)]
        ylim = [np.round(ylim[0], self.decimals), np.round(ylim[1], self.decimals)]

        self.nsub = nsub
        self.xlim_sub, self.ylim_sub, self.zlim_sub = xlim, ylim, zlim
        ximin, ximax = index_between(self.x, xlim, mode='edge')[0]
        yimin, yimax = index_between(self.y, ylim, mode='edge')[0]
        zimin, zimax = index_between(self.z, zlim, mode='edge')[0]
        _nx = ximax - ximin + 1
        _ny = yimax - yimin + 1
        _nz - zimax - zimin + 1
        xemin, xemax = self.xe[ximin], self.xe[ximax + 1]
        yemin, yemax = self.ye[yimin], self.ye[yimax + 1]
        zemin, zemax = self.ze[zimin], self.ze[zimax + 1]
        self.xi0, self.xi1 = ximin, ximax # Starting and ending indices of nested grid
        self.yi0, self.yi1 = yimin, yimax # Starting and ending indices of nested grid
        self.zi0, self.zi1 = zimin, zimax # Starting and ending indices of nested grid

        # nested grid
        xe_sub = np.linspace(xemin, xemax, _nx * nsub + 1)
        ye_sub = np.linspace(yemin, yemax, _ny * nsub + 1)
        ze_sub = np.linspace(zemin, zemax, _nz * nsub + 1)
        x_sub = 0.5 * (xe_sub[:-1] + xe_sub[1:])
        y_sub = 0.5 * (ye_sub[:-1] + ye_sub[1:])
        z_sub = 0.5 * (ze_sub[:-1] + ze_sub[1:])
        xx_sub, yy_sub, zz_sub = np.meshgrid(x_sub, y_sub, z_sub, indexing = 'ij')
        self.xe_sub, self.ye_sub, self.ze_sub = xe_sub, ye_sub, ze_sub
        self.x_sub, self.y_sub, z_sub = x_sub, y_sub, z_sub
        self.xx_sub, self.yy_sub, self.zz_sub = xx_sub, yy_sub, zz_sub
        self.dx_sub, self.dy_sub, self.dz_sub = self.dx / nsub, self.dy / nsub, self.dz / nsub
        self.nx_sub, self.ny_sub, self.nz_sub = len(x_sub), len(y_sub), len(z_sub)
        return xx_sub, yy_sub, zz_sub


    def where_subgrid(self):
        return np.where(
            (self.xx >= self.xlim_sub[0]) * (self.xx <= self.xlim_sub[1]) \
            * (self.yy >= self.ylim_sub[0]) * (self.yy <= self.ylim_sub[1]))


    def binning_onsubgrid(self, data):
        nbin = self.nsub
        d_avg = np.array([
            data[i::nbin, i::nbin, i::nbin]
            for i in range(nbin)
            ])
        return np.nanmean(d_avg, axis = 0)


    def binning_onsubgrid_layered(self, data, nbin):
        dshape = len(data.shape)
        if dshape == 3:
            d_avg = np.array([
                data[i::nbin, i::nbin, i::nbin]
                for i in range(nbin)
                ])
        elif dshape == 4:
            d_avg = np.array([
                data[:, i::nbin, i::nbin, i::nbin]
                for i in range(nbin)
                ])
        elif dshape ==5:
            d_avg = np.array([
                data[:, :, i::nbin, i::nbin, i::nbin]
                for i in range(nbin)
                ])
        else:
            print('ERROR\tbinning_onsubgrid_layered: only Nd of data of 3-5 is now supported.')
            return 0
        return np.nanmean(d_avg, axis = 0)


    def gridinfo(self, units = ['au', 'au', 'au']):
        ux, uy, uz = units
        print('Nesting level: %i'%self.nlevels)
        print('Resolutions:')
        for l in range(self.nlevels):
            dx = self.xaxes[l][1] - self.xaxes[l][0]
            dy = self.yaxes[l][1] - self.yaxes[l][0]
            dz = self.zaxes[l][1] - self.zaxes[l][0]
            print('   l=%i: (dx, dy, dz) = (%.2e %s, %.2e %s, %.2e %s)'%(l, dx, ux, dy, uy, dz, uz))
            print('      : (xlim, ylim, zlim) = (%.2e to %.2e %s, %.2e to %.2e %s, %.2e to %.2e %s, )'%(
                self.xlim[l][0], self.xlim[l][1], ux,
                self.ylim[l][0], self.ylim[l][1], uy,
                self.zlim[l][0], self.zlim[l][1], uz))


    def edgecut_indices(self, xlength, ylength):
        # odd or even
        x_oddeven = self.nx%2
        y_oddeven = self.ny%2
        # edge indices for subgrid
        xi = int(xlength / self.dx_sub) if self.dx_sub > 0 else int(- xlength / self.dx_sub)
        yi = int(ylength / self.dy_sub)
        _nx_resub = int(self.nx_sub - 2 * xi) // self.nsub # nx of subgrid after cutting edge
        _ny_resub = int(self.ny_sub - 2 * yi) // self.nsub # ny of subgrid after cutting edge
        # fit odd/even
        if _nx_resub%2 != x_oddeven: _nx_resub += 1
        if _ny_resub%2 != y_oddeven: _ny_resub += 1
        # nx, ny of the new subgrid and new xi and yi
        nx_resub = _nx_resub * self.nsub
        ny_resub = _ny_resub * self.nsub
        xi = (self.nx_sub - nx_resub) // 2
        yi = (self.ny_sub - ny_resub) // 2
        # for original grid
        xi0 = int((self.nx - nx_resub / self.nsub) * 0.5)
        yi0 = int((self.ny - ny_resub / self.nsub) * 0.5)
        #print(nx_resub / self.nsub, self.nx - xi0 - xi0)
        return xi, yi, xi0, yi0




class Nested3DObsGrid(object):
    """
    3D Cartesian grid with nesting along x- and y-axis and with adoptive z-axis.

    """
    def __init__(self, x, y, z, 
        xlim = None, ylim = None,
        nsub = None, zstrech = None, reslim = 20,
        preserve_z = False):
        super(Nested3DObsGrid, self).__init__()
        # save axes of the mother grid
        self.x = x
        self.y = y
        self.z = z
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        dz = z[1] - z[0]
        self.dx = dx
        self.dy = dy
        self.dz = dz
        xe = np.hstack([x - self.dx * 0.5, x[-1] + self.dx * 0.5])
        ye = np.hstack([y - self.dy * 0.5, y[-1] + self.dy * 0.5])
        ze = np.hstack([z - self.dz * 0.5, z[-1] + self.dz * 0.5])
        self.xe, self.ye, self.ze = xe, ye, ze
        nz, ny, nx = len(z), len(y), len(x)
        self.nz, self.ny, self.nx = nz, ny, nx
        self.xx, self.yy, self.zz = np.meshgrid(x, y, z, indexing='ij')
        self.Lx, self.Ly, self.Lz = xe[-1] - xe[0], ye[-1] - ye[0], ze[-1] - ze[0]
        #self.dzs = np.full((nx, ny, nz), dz)


        # nested grid
        self.nsub = nsub
        self.zstrech = zstrech
        if len(nsub) != len(zstrech):
            print('ERROR\tNested3DObsGrid: nsub and zstrech must have the same dimension.')
            return 0
        nlevels = 1 if nsub is None else len(nsub) + 1
        self.nlevels = nlevels
        # original 1D axes
        self.xaxes = [None] * nlevels
        self.yaxes = [None] * nlevels
        self.zaxes = [None] * nlevels
        self.xaxes[0], self.yaxes[0], self.zaxes[0] = x, y, z
        # grid sizes
        self.ngrids = np.zeros((nlevels, 3)).astype(int)
        self.ngrids[0,:] = np.array([nx, ny, nz])
        # nested grid
        self.xnest = self.xx.ravel() # save all grid info in 1D array
        self.ynest = self.yy.ravel()
        self.znest = self.zz.ravel()
        self.dznest = dz
        self.partition = [0, self.xnest.size] # partition indices
        self.xypartition = [0, nx * ny] # partition indices
        # starting and ending indices
        self.xinest = [-1,-1]
        self.yinest = [-1,-1]
        # nest
        self.preserve_z = preserve_z
        if self.nlevels > 1:
            self.get_zlim()
            self.zlim.insert(0, [ze[0], ze[-1]])
            if (np.array([xlim, ylim]) == None).any():
                _xlim, _ylim = self.get_nestinglim(reslim = reslim)
                if xlim is None: xlim = _xlim
                if ylim is None: ylim = _ylim
            self.xlim, self.ylim = xlim.copy(), ylim.copy()
            self.xlim.insert(0, [xe[0], xe[-1]])
            self.ylim.insert(0, [ye[0], ye[-1]])
            self.nest(preserve_z = preserve_z)
        else:
            self.xlim = [xe[0], xe[-1]]
            self.ylim = [ye[0], ye[-1]]
            self.zlim = [ze[0], ze[-1]]


    def get_nestinglim(self, reslim = 5):
        xlim = []
        ylim = []
        _dx, _dy = np.abs(self.dx), np.abs(self.dy)
        for l in range(self.nlevels - 1):
            xlim.append([-_dx * reslim, _dx * reslim])
            ylim.append([-_dy * reslim, _dy * reslim])
            _dx, _dy = np.abs(np.array([_dx, _dy])) / self.nsub[l]

        return xlim, ylim


    def get_zlim(self):
        z = self.zaxes[0]
        zlim = []
        for l in range(1, self.nlevels):
            dz = z[1] - z[0]
            _zmax = (z[-1] + 0.5 * dz) / self.zstrech[l-1]
            _zmin = (z[0] - 0.5 * dz) / self.zstrech[l-1]
            zlim.append([_zmin, _zmax])

            # renew z axis
            ze_sub = np.linspace(_zmin, _zmax, self.nz + 1)
            z = 0.5 * (ze_sub[1:] + ze_sub[:-1])
        self.zlim = zlim


    def get_grid(self, l):
        '''
        Get grid on the l layer.
        '''
        _nx, _ny, _nz = self.ngrids[l,:]
        partition = self.partition[l:l+2]
        if self.xnest[partition[0]:].size == _nx * _ny * _nz:
            if self.preserve_z:
                partition = self.xypartition[l:l+2] #(l+1)*2:(l+2)*2
            else:
                partition = self.partition[l:l+2]
            xx = self.xnest[partition[0]:partition[1]].reshape(_nx, _ny, _nz)
            yy = self.ynest[partition[0]:partition[1]].reshape(_nx, _ny, _nz)
            zz = self.znest[partition[0]:partition[1]].reshape(_nx, _ny, _nz)
        else:
            # else
            x, y, z = self.xaxes[l], self.yaxes[l], self.zaxes[l]
            xx, yy, zz = np.meshgrid(x, y, z, indexing = 'ij')
        return xx, yy, zz


    def nest(self, preserve_z = False):
        if preserve_z:
            self.nest_z_preserved()
        else:
            self.nest_flatten()


    def nest_flatten(self):
        '''
        l - 1 is the mother grid layer. l is the child grid layer.
        '''
        # initialize
        partition = [0]
        xypartition = [0]
        xnest = np.array([])
        ynest = np.array([])
        znest = np.array([])
        dznest = np.array([])
        for l in range(1, self.nlevels):
            # axes of the parental grid
            x, y, z = self.xaxes[l-1], self.yaxes[l-1], self.zaxes[l-1]

            # make childe grid
            ximin, ximax, yimin, yimax, x_sub, y_sub = \
            nestgrid_2D(x, y, self.xlim[l], self.ylim[l], self.nsub[l-1])
            self.xinest += [ximin, ximax] # starting and ending indices on the upper-layer grid
            self.yinest += [yimin, yimax]

            # new z axis
            dz = z[1] - z[0]
            _zmax = (z[-1] + 0.5 * dz) / self.zstrech[l-1]
            _zmin = (z[0] - 0.5 * dz) / self.zstrech[l-1]
            ze_sub = np.linspace(_zmin, _zmax, self.nz + 1)
            z_sub = 0.5 * (ze_sub[1:] + ze_sub[:-1])
            dz_sub = ze_sub[1] - ze_sub[0]

            self.xaxes[l], self.yaxes[l], self.zaxes[l] = x_sub, y_sub, z_sub
            self.ngrids[l,:] = np.array([len(x_sub), len(y_sub), len(z_sub)])

            # parental grid
            _nx, _ny, _nz = self.ngrids[l-1,:]
            xx, yy, zz = np.meshgrid(x, y, z, indexing = 'ij')

            # devide the upper grid into six sub-regions
            # Region 1:  x from 0 to ximin, all y and z
            #_nxy = ximin * _ny
            R1x = xx[:ximin, :, :].ravel()
            R1y = yy[:ximin, :, :].ravel()
            R1z = zz[:ximin, :, :].ravel()
            # Region 2: x from ximax+1 to nx, all y and z
            #_nxy = (_nx - ximax - 1) * _ny
            R2x = xx[ximax+1:, :, :].ravel()
            R2y = yy[ximax+1:, :, :].ravel()
            R2z = zz[ximax+1:, :, :].ravel()
            # Region 3: x from ximin to ximax, y from 0 to yimin, and all z
            #_nxy = (ximax + 1 - ximin) * yimin
            R3x = xx[ximin:ximax+1, :yimin, :].ravel()
            R3y = yy[ximin:ximax+1, :yimin, :].ravel()
            R3z = zz[ximin:ximax+1, :yimin, :].ravel()
            # Region 4: x from ximin to ximax, y from yimax+1 to ny, and all z
            #_nxy = (ximax + 1 - ximin) * (_ny - yimax - 1)
            R4x = xx[ximin:ximax+1, yimax+1:, :].ravel()
            R4y = yy[ximin:ximax+1, yimax+1:, :].ravel()
            R4z = zz[ximin:ximax+1, yimax+1:, :].ravel()

            # save in the shape of (nxy, nz)
            Rx = np.concatenate([R1x, R2x, R3x, R4x])
            nl = Rx.size
            partition.append(partition[l-1] + nl)
            #xypartition.append(xypartition[l-1] + nxy)

            xnest = np.concatenate([xnest, Rx]) # update
            ynest = np.concatenate([ynest, R1y, R2y, R3y, R4y]) # update
            znest = np.concatenate([znest, R1z, R2z, R3z, R4z]) # update
            dznest = np.concatenate([dznest, np.full(nl, dz)])


        # the deepest child grid
        xx_sub, yy_sub, zz_sub = np.meshgrid(x_sub, y_sub, z_sub, indexing = 'ij')
        xnest = np.concatenate([xnest, xx_sub.ravel()]) # update
        ynest = np.concatenate([ynest, yy_sub.ravel()]) # update
        znest = np.concatenate([znest, zz_sub.ravel()]) # update
        dznest = np.concatenate([dznest, np.full(xx_sub.size, dz_sub)])

        self.xnest = xnest
        self.ynest = ynest
        self.znest = znest
        self.dznest = dznest
        self.partition = partition


    def nest_z_preserved(self):
        '''
        l - 1 is the mother grid layer. l is the child grid layer.
        '''
        # initialize
        partition = [0]
        xypartition = [0]
        xnest = np.array([])
        ynest = np.array([])
        znest = np.array([])
        dznest = np.array([])
        for l in range(1, self.nlevels):
            # axes of the parental grid
            x, y, z = self.xaxes[l-1], self.yaxes[l-1], self.zaxes[l-1]

            # make childe grid
            ximin, ximax, yimin, yimax, x_sub, y_sub = \
            nestgrid_2D(x, y, self.xlim[l], self.ylim[l], self.nsub[l-1])
            self.xinest += [ximin, ximax] # starting and ending indices on the upper-layer grid
            self.yinest += [yimin, yimax]

            # new z axis
            dz = z[1] - z[0]
            _zmax = (z[-1] + 0.5 * dz) / self.zstrech[l-1]
            _zmin = (z[0] - 0.5 * dz) / self.zstrech[l-1]
            ze_sub = np.linspace(_zmin, _zmax, self.nz + 1)
            z_sub = 0.5 * (ze_sub[1:] + ze_sub[:-1])
            dz_sub = ze_sub[1] - ze_sub[0]

            self.xaxes[l], self.yaxes[l], self.zaxes[l] = x_sub, y_sub, z_sub
            self.ngrids[l,:] = np.array([len(x_sub), len(y_sub), len(z_sub)])

            # parental grid
            _nx, _ny, _nz = self.ngrids[l-1,:]
            xx, yy, zz = np.meshgrid(x, y, z, indexing = 'ij')

            # devide the upper grid into six sub-regions
            # Region 1:  x from 0 to ximin, all y and z
            _nxy = ximin * _ny
            R1x = xx[:ximin, :, :].reshape((_nxy, _nz)) #.ravel()
            R1y = yy[:ximin, :, :].reshape((_nxy, _nz)) #.ravel()
            R1z = zz[:ximin, :, :].reshape((_nxy, _nz)) #.ravel()
            # Region 2: x from ximax+1 to nx, all y and z
            _nxy = (_nx - ximax - 1) * _ny
            R2x = xx[ximax+1:, :, :].reshape((_nxy, _nz)) #.ravel()
            R2y = yy[ximax+1:, :, :].reshape((_nxy, _nz)) #.ravel()
            R2z = zz[ximax+1:, :, :].reshape((_nxy, _nz)) #.ravel()
            # Region 3: x from ximin to ximax, y from 0 to yimin, and all z
            _nxy = (ximax + 1 - ximin) * yimin
            R3x = xx[ximin:ximax+1, :yimin, :].reshape((_nxy, _nz)) #.ravel()
            R3y = yy[ximin:ximax+1, :yimin, :].reshape((_nxy, _nz)) #.ravel()
            R3z = zz[ximin:ximax+1, :yimin, :].reshape((_nxy, _nz)) #.ravel()
            # Region 4: x from ximin to ximax, y from yimax+1 to ny, and all z
            _nxy = (ximax + 1 - ximin) * (_ny - yimax - 1)
            R4x = xx[ximin:ximax+1, yimax+1:, :].reshape((_nxy, _nz)) #.ravel()
            R4y = yy[ximin:ximax+1, yimax+1:, :].reshape((_nxy, _nz)) #.ravel()
            R4z = zz[ximin:ximax+1, yimax+1:, :].reshape((_nxy, _nz)) #.ravel()

            # save in the shape of (nxy, nz)
            Rx = np.vstack([R1x, R2x, R3x, R4x])
            nl = Rx.size
            nxy, nz = Rx.shape
            partition.append(partition[l-1] + nl)
            xypartition.append(xypartition[l-1] + nxy)

            if l == 1:
                xnest = Rx
                ynest = np.vstack([R1y, R2y, R3y, R4y])
                znest = np.vstack([R1z, R2z, R3z, R4z])
                dznest = np.full((nxy, nz), dz)
            else:
                xnest = np.vstack([xnest, Rx]) # update
                ynest = np.vstack([ynest, R1y, R2y, R3y, R4y]) # update
                znest = np.vstack([znest, R1z, R2z, R3z, R4z]) # update
                dznest = np.vstack([dznest, np.full((nxy, nz), dz)])


        # the deepest child grid
        xx_sub, yy_sub, zz_sub = np.meshgrid(x_sub, y_sub, z_sub, indexing = 'ij')
        nx, ny, nz = xx_sub.shape
        #xnest = np.concatenate([xnest, xx_sub.ravel()]) # update
        #ynest = np.concatenate([ynest, yy_sub.ravel()]) # update
        #znest = np.concatenate([znest, zz_sub.ravel()]) # update
        #dznest = np.concatenate([dznest, np.full(xx_sub.size, dz_sub)])
        xnest = np.vstack([xnest, xx_sub.reshape((nx*ny, nz))])
        ynest = np.vstack([ynest, yy_sub.reshape((nx*ny, nz))])
        znest = np.vstack([znest, zz_sub.reshape((nx*ny, nz))])
        dznest = np.vstack([dznest, np.full((nx*ny, nz), dz_sub)])

        self.xnest = xnest
        self.ynest = ynest
        self.znest = znest
        self.dznest = dznest
        self.partition = partition
        self.xypartition = xypartition

        nxy, nz = xnest.shape
        self.nxy = nxy


    def collapse(self, d, upto = None):
        if self.preserve_z:
            d_col = self.collapse_z_preserved(d, upto = upto)
        else:
            d_col = self.collapse_flatten(d, upto = upto)
        return d_col


    def collapse_z_preserved(self, d, upto = None):
        '''
        Collapse given data to the mother grid.

        Parameters
        ----------
        d (2D array): List of data on the nested grid
        '''
        lmax = 0 if upto is None else upto
        d_col = d[self.xypartition[-1]:,:] # starting from the inner most grid
        d_col = d_col.reshape(tuple(self.ngrids[-1,:]))
        for l in range(self.nlevels-1,lmax,-1):
            nsub = self.nsub[l-1]
            ximin, ximax = self.xinest[l*2:(l+1)*2]
            yimin, yimax = self.yinest[l*2:(l+1)*2]
            # collapse data on the inner grid
            _d = self.binning_onsubgrid_layered(d_col, nsub)
            #print(ximin, ximax, yimin, yimax, zimin, zimax)

            # go upper layer
            nx, ny, nz = self.ngrids[l-1,:] # size of the upper layer
            d_col = np.full((nx, ny, nz), np.nan)

            # insert collapsed data
            d_col[ximin:ximax+1, yimin:yimax+1, :] = _d

            # fill upper layer data
            d_up = d[self.xypartition[l-1]:self.xypartition[l],:]
            # Region 1: x from zero to ximin, all y and z
            d_col[:ximin, :, :] = \
            d_up[:ximin * ny,: ].reshape((ximin, ny, nz))
            # Region 2: x from ximax to nx, all y and z
            i0 = ximin * ny
            i1 = i0 + (nx - ximax - 1) * ny
            d_col[ximax+1:, :, :] = \
            d_up[i0:i1,:].reshape(
                (nx - ximax - 1, ny, nz))
            # Region 3
            i0 = i1
            i1 = i0 + (ximax + 1 - ximin) * yimin
            d_col[ximin:ximax+1, :yimin, :] = \
            d_up[i0:i1,:].reshape(
                (ximax + 1 - ximin, yimin, nz))
            # Region 4
            i0 = i1
            i1 = i0 + (ximax + 1 - ximin) * (ny - yimax - 1)
            d_col[ximin:ximax+1, yimax+1:, :] = \
            d_up[i0:i1,:].reshape(
                (ximax + 1 - ximin, ny - yimax - 1, nz))

        return d_col


    def collapse_flatten(self, d, upto = None):
        '''
        Collapse given data to the mother grid.

        Parameters
        ----------
        d (1D array): List of data on the nested grid
        '''
        lmax = 0 if upto is None else upto
        d_col = d[self.partition[-1]:] # starting from the inner most grid
        d_col = d_col.reshape(tuple(self.ngrids[-1,:]))
        for l in range(self.nlevels-1,lmax,-1):
            nsub = self.nsub[l-1]
            ximin, ximax = self.xinest[l*2:(l+1)*2]
            yimin, yimax = self.yinest[l*2:(l+1)*2]
            # collapse data on the inner grid
            _d = self.binning_onsubgrid_layered(d_col, nsub)
            #print(ximin, ximax, yimin, yimax, zimin, zimax)

            # go upper layer
            nx, ny, nz = self.ngrids[l-1,:] # size of the upper layer
            d_col = np.full((nx, ny, nz), np.nan)

            # insert collapsed data
            d_col[ximin:ximax+1, yimin:yimax+1, :] = _d

            # fill upper layer data
            d_up = d[self.partition[l-1]:self.partition[l]]
            # Region 1: x from zero to ximin, all y and z
            d_col[:ximin, :, :] = \
            d_up[:ximin * ny * nz].reshape((ximin, ny, nz))
            # Region 2: x from ximax to nx, all y and z
            i0 = ximin * ny * nz
            i1 = i0 + (nx - ximax - 1) * ny * nz
            d_col[ximax+1:, :, :] = \
            d_up[i0:i1].reshape(
                (nx - ximax - 1, ny, nz))
            # Region 3
            i0 = i1
            i1 = i0 + (ximax + 1 - ximin) * yimin * nz
            d_col[ximin:ximax+1, :yimin, :] = \
            d_up[i0:i1].reshape(
                (ximax + 1 - ximin, yimin, nz))
            # Region 4
            i0 = i1
            i1 = i0 + (ximax + 1 - ximin) * (ny - yimax - 1) * nz
            d_col[ximin:ximax+1, yimax+1:, :] = \
            d_up[i0:i1].reshape(
                (ximax + 1 - ximin, ny - yimax - 1, nz))

        return d_col


    def collapse2D(self, d, upto = None, fill = 'nan'):
        ndim = len(d.shape)
        #print(ndim)
        if ndim == 1:
            d_col = self.collapse2D_no_extradim(d, upto = upto, fill = fill)
        elif ndim == 2:
            d_col = self.collapse2D_extra_1d(d, upto = upto, fill = fill)
        else:
            print('ERROR\tcollapse2D: currently only ndim=2 is supported.')
            return 0
        return d_col


    def collapse2D_no_extradim(self, d, upto = None, fill = 'nan'):
        '''
        Collapse given data to the mother grid.

        Parameters
        ----------
        d (2D array): Data on nested grid
        '''
        lmax = 0 if upto is None else upto
        d_col = d[self.xypartition[-1]:] # starting from the inner most grid
        d_col = d_col.reshape(tuple(self.ngrids[-1,:-1])) # to (nx, ny)
        for l in range(self.nlevels-1,lmax,-1):
            nsub = self.nsub[l-1]
            ximin, ximax = self.xinest[l*2:(l+1)*2]
            yimin, yimax = self.yinest[l*2:(l+1)*2]
            # collapse data on the inner grid
            _d = self.binning_onsubgrid_xy(d_col, nsub)
            #print(l, ximin, ximax, yimin, yimax)
            #print(np.nanmax(_d))

            # go upper layer
            nx, ny, _ = self.ngrids[l-1,:] # size of the upper layer
            if fill == 'nan':
                d_col = np.full((nx, ny), np.nan)
            elif fill == 'zero':
                d_col = np.zeros((nx, ny))
            else:
                d_col = np.full((nx, ny), fill)

            # insert collapsed data
            d_col[ximin:ximax+1, yimin:yimax+1] = _d

            # fill upper layer data
            d_up = d[self.xypartition[l-1]:self.xypartition[l]]
            # Region 1: x from zero to ximin, all y and z
            d_col[:ximin, :] = \
            d_up[:ximin * ny].reshape((ximin, ny))
            # Region 2: x from ximax to nx, all y and z
            i0 = ximin * ny
            i1 = i0 + (nx - ximax - 1) * ny
            d_col[ximax+1:, :] = \
            d_up[i0:i1].reshape(
                (nx - ximax - 1, ny))
            # Region 3
            i0 = i1
            i1 = i0 + (ximax + 1 - ximin) * yimin
            d_col[ximin:ximax+1, :yimin] = \
            d_up[i0:i1].reshape(
                (ximax + 1 - ximin, yimin))
            # Region 4
            i0 = i1
            i1 = i0 + (ximax + 1 - ximin) * (ny - yimax - 1)
            d_col[ximin:ximax+1, yimax+1:] = \
            d_up[i0:i1].reshape(
                (ximax + 1 - ximin, ny - yimax - 1))

        return d_col


    def collapse2D_extra_1d(self, d, upto = None, fill = 'nan'):
        '''
        Collapse given data to the mother grid.

        Parameters
        ----------
        d (2D array): Data on nested grid
        '''
        lmax = 0 if upto is None else upto
        nv, nd = d.shape
        d_col = d[:, self.xypartition[-1]:] # starting from the inner most grid
        d_col = d_col.reshape(
            (nv, self.ngrids[-1,:][0], self.ngrids[-1,:][1])) # to (nv, nx, ny)
        for l in range(self.nlevels-1,lmax,-1):
            nsub = self.nsub[l-1]
            ximin, ximax = self.xinest[l*2:(l+1)*2]
            yimin, yimax = self.yinest[l*2:(l+1)*2]
            # collapse data on the inner grid
            _d = self.binning_onsubgrid_xy(d_col, nsub)
            #print(ximin, ximax, yimin, yimax, zimin, zimax)

            # go upper layer
            nx, ny, _ = self.ngrids[l-1,:] # size of the upper layer
            if fill == 'nan':
                d_col = np.full((nv, nx, ny), np.nan)
            elif fill == 'zero':
                d_col = np.zeros((nv, nx, ny))
            else:
                d_col = np.full((nv, nx, ny), fill)

            # insert collapsed data
            d_col[:,ximin:ximax+1, yimin:yimax+1] = _d

            # fill upper layer data
            d_up = d[:,self.xypartition[l-1]:self.xypartition[l]]
            # Region 1: x from zero to ximin, all y and z
            d_col[:, :ximin, :] = \
            d_up[:, :ximin * ny].reshape((nv, ximin, ny,))
            # Region 2: x from ximax to nx, all y and z
            i0 = ximin * ny
            i1 = i0 + (nx - ximax - 1) * ny
            d_col[:, ximax+1:, :] = \
            d_up[:, i0:i1].reshape(
                (nv, nx - ximax - 1, ny))
            # Region 3
            i0 = i1
            i1 = i0 + (ximax + 1 - ximin) * yimin
            d_col[:, ximin:ximax+1, :yimin] = \
            d_up[:, i0:i1].reshape(
                (nv, ximax + 1 - ximin, yimin))
            # Region 4
            i0 = i1
            i1 = i0 + (ximax + 1 - ximin) * (ny - yimax - 1)
            d_col[:, ximin:ximax+1, yimax+1:] = \
            d_up[:, i0:i1].reshape(
                (nv, ximax + 1 - ximin, ny - yimax - 1))

        return d_col


    def high_dimensional_collapse(self, d, upto = None, fill = 'nan'):
        if len(d.shape) == 2:
            d_col = self.collapse_extra_1d(d, upto = upto, fill = fill)
            return d_col
        else:
            print('ERROR\thigh_dimensional_collapse: currently only 2d data are supported.')
            return 0


    def collapse_extra_1d(self, d, upto = None, fill = 'nan'):
        '''
        Collapse given data to the mother grid.

        Parameters
        ----------
        d (list): List of data on the nested grid
        '''
        lmax = 0 if upto is None else upto
        nv, nd = d.shape
        d_col = d[:, self.partition[-1]:] # starting from the inner most grid
        d_col = d_col.reshape(
            (nv, self.ngrids[-1,:][0], self.ngrids[-1,:][1], self.ngrids[-1,:][2]))
        for l in range(self.nlevels-1,lmax,-1):
            nsub = self.nsub[l-1]
            ximin, ximax = self.xinest[l*2:(l+1)*2]
            yimin, yimax = self.yinest[l*2:(l+1)*2]
            # collapse data on the inner grid
            _d = self.binning_onsubgrid_layered(d_col, nsub)
            #print(ximin, ximax, yimin, yimax, zimin, zimax)

            # go upper layer
            nx, ny, nz = self.ngrids[l-1,:] # size of the upper layer
            if fill == 'nan':
                d_col = np.full((nv, nx, ny, nz), np.nan)
            elif fill == 'zero':
                d_col = np.zeros((nv, nx, ny, nz))
            else:
                d_col = np.full((nv, nx, ny, nz), fill)

            # insert collapsed data
            d_col[:, ximin:ximax+1, yimin:yimax+1, :] = _d

            # fill upper layer data
            d_up = d[:, self.partition[l-1]:self.partition[l]]
            # Region 1: x from zero to ximin, all y and z
            d_col[:, :ximin, :, :] = \
            d_up[:, :ximin * ny * nz].reshape((nv, ximin, ny, nz))
            # Region 2: x from ximax to nx, all y and z
            i0 = ximin * ny * nz
            i1 = i0 + (nx - ximax - 1) * ny * nz
            d_col[:, ximax+1:, :, :] = \
            d_up[:, i0:i1].reshape(
                (nv, nx - ximax - 1, ny, nz))
            # Region 3
            i0 = i1
            i1 = i0 + (ximax + 1 - ximin) * yimin * nz
            d_col[:, ximin:ximax+1, :yimin, :] = \
            d_up[:, i0:i1].reshape(
                (nv, ximax + 1 - ximin, yimin, nz))
            # Region 4
            i0 = i1
            i1 = i0 + (ximax + 1 - ximin) * (ny - yimax - 1) * nz
            d_col[:, ximin:ximax+1, yimax+1:, :] = \
            d_up[:, i0:i1].reshape(
                (nv, ximax + 1 - ximin, ny - yimax - 1, nz))

        return d_col


    def nest_sub(self, xlim,  ylim, zlim, nsub):
        # error check
        if (len(xlim) != 2) | (len(ylim) != 2) | (len(zlim) != 2):
            print('ERROR\tnest: Input xlim/ylim/zlim must be list as [min, max].')
            return 0
        # decimals
        xlim = [np.round(xlim[0], self.decimals), np.round(xlim[1], self.decimals)]
        ylim = [np.round(ylim[0], self.decimals), np.round(ylim[1], self.decimals)]

        self.nsub = nsub
        self.xlim_sub, self.ylim_sub, self.zlim_sub = xlim, ylim, zlim
        ximin, ximax = index_between(self.x, xlim, mode='edge')[0]
        yimin, yimax = index_between(self.y, ylim, mode='edge')[0]
        zimin, zimax = index_between(self.z, zlim, mode='edge')[0]
        _nx = ximax - ximin + 1
        _ny = yimax - yimin + 1
        _nz - zimax - zimin + 1
        xemin, xemax = self.xe[ximin], self.xe[ximax + 1]
        yemin, yemax = self.ye[yimin], self.ye[yimax + 1]
        zemin, zemax = self.ze[zimin], self.ze[zimax + 1]
        self.xi0, self.xi1 = ximin, ximax # Starting and ending indices of nested grid
        self.yi0, self.yi1 = yimin, yimax # Starting and ending indices of nested grid
        self.zi0, self.zi1 = zimin, zimax # Starting and ending indices of nested grid

        # nested grid
        xe_sub = np.linspace(xemin, xemax, _nx * nsub + 1)
        ye_sub = np.linspace(yemin, yemax, _ny * nsub + 1)
        ze_sub = np.linspace(zemin, zemax, _nz * nsub + 1)
        x_sub = 0.5 * (xe_sub[:-1] + xe_sub[1:])
        y_sub = 0.5 * (ye_sub[:-1] + ye_sub[1:])
        z_sub = 0.5 * (ze_sub[:-1] + ze_sub[1:])
        xx_sub, yy_sub, zz_sub = np.meshgrid(x_sub, y_sub, z_sub, indexing = 'ij')
        self.xe_sub, self.ye_sub, self.ze_sub = xe_sub, ye_sub, ze_sub
        self.x_sub, self.y_sub, z_sub = x_sub, y_sub, z_sub
        self.xx_sub, self.yy_sub, self.zz_sub = xx_sub, yy_sub, zz_sub
        self.dx_sub, self.dy_sub, self.dz_sub = self.dx / nsub, self.dy / nsub, self.dz / nsub
        self.nx_sub, self.ny_sub, self.nz_sub = len(x_sub), len(y_sub), len(z_sub)
        return xx_sub, yy_sub, zz_sub


    def where_subgrid(self):
        return np.where(
            (self.xx >= self.xlim_sub[0]) * (self.xx <= self.xlim_sub[1]) \
            * (self.yy >= self.ylim_sub[0]) * (self.yy <= self.ylim_sub[1]))


    def binning_onsubgrid(self, data):
        nbin = self.nsub
        d_avg = np.array([
            data[i::nbin, i::nbin, :]
            for i in range(nbin)
            ])
        return np.nanmean(d_avg, axis = 0)


    def binning_onsubgrid_layered(self, data, nbin):
        dshape = len(data.shape)
        if dshape == 3:
            d_avg = np.array([
                data[i::nbin, i::nbin, :]
                for i in range(nbin)
                ])
        elif dshape == 4:
            d_avg = np.array([
                data[:, i::nbin, i::nbin, :]
                for i in range(nbin)
                ])
        elif dshape ==5:
            d_avg = np.array([
                data[:, :, i::nbin, i::nbin, :]
                for i in range(nbin)
                ])
        else:
            print('ERROR\tbinning_onsubgrid_layered: only Nd of data of 3-5 is now supported.')
            return 0
        return np.nanmean(d_avg, axis = 0)


    def binning_onsubgrid_xy(self, data, nbin):
        dshape = len(data.shape)
        if dshape == 2:
            d_avg = np.array([
                data[i::nbin, i::nbin]
                for i in range(nbin)
                ])
        elif dshape == 3:
            d_avg = np.array([
                data[:, i::nbin, i::nbin]
                for i in range(nbin)
                ])
        elif dshape == 4:
            d_avg = np.array([
                data[:, :, i::nbin, i::nbin]
                for i in range(nbin)
                ])
        else:
            print('ERROR\tbinning_onsubgrid_xy: only Nd of data of 2-4 is now supported.')
            return 0
        return np.nanmean(d_avg, axis = 0)


    def gridinfo(self, units = ['au', 'au', 'au']):
        ux, uy, uz = units
        print('Nesting level: %i'%self.nlevels)
        print('Resolutions:')
        for l in range(self.nlevels):
            dx = self.xaxes[l][1] - self.xaxes[l][0]
            dy = self.yaxes[l][1] - self.yaxes[l][0]
            dz = self.zaxes[l][1] - self.zaxes[l][0]
            print('   l=%i: (dx, dy, dz) = (%.2e %s, %.2e %s, %.2e %s)'%(l, dx, ux, dy, uy, dz, uz))
            print('      : (xlim, ylim, zlim) = (%.2e to %.2e %s, %.2e to %.2e %s, %.2e to %.2e %s, )'%(
                self.xlim[l][0], self.xlim[l][1], ux,
                self.ylim[l][0], self.ylim[l][1], uy,
                self.zlim[l][0], self.zlim[l][1], uz))


    def visualize_xz(self, d, 
        ax = None, vmin = None, vmax = None,
        showfig = False, cmap = 'viridis'):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        # from upper to lower
        for l in range(self.nlevels):
            nx, ny, nz = self.ngrids[l,:]
            xmin, xmax = self.xlim[l]
            zmin, zmax = self.zlim[l]

            if self.preserve_z:
                d_plt = self.collapse(
                    d, upto = l)[:, ny//2, :]
            else:
                d_plt = self.collapse(
                    d, upto = l)[:, ny//2, :]

            # hide parental layer
            if l <= self.nlevels-2:
                ximin, ximax = self.xinest[(l+1)*2:(l+2)*2]
                d_plt[ximin:ximax+1,:] = np.nan

            #ax.imshow(d_plt, extent = (zmin, zmax, xmin, xmax),
            #    alpha = 1., vmax = vmax, vmin = vmin, origin = 'upper', cmap = cmap)

            _xx, _yy, _zz = self.get_grid(l)
            _xx = _xx[:, ny//2, :]
            _zz = _zz[:, ny//2, :]
            ax.pcolormesh(_zz, _xx, d_plt, 
                alpha = 1., vmax = vmax, vmin = vmin, cmap = cmap)
            rect = plt.Rectangle((zmin, xmin), 
                zmax - zmin, xmax - xmin, edgecolor = 'white', facecolor = "none",
                linewidth = 0.5, ls = '--')
            ax.add_patch(rect)

        ax.set_xlim(self.zlim[0])
        ax.set_ylim(self.xlim[0])

        if showfig: plt.show()
        return ax

def nestgrid_2D(x, y, xlim, ylim, nsub, decimals = 4.):
    # error check
    if (len(xlim) != 2) | (len(ylim) != 2):
        print('ERROR\tnest: Input xlim and ylim must be list as [min, max].')
        return 0
    # decimals
    #xlim = [np.round(xlim[0], self.decimals), np.round(xlim[1], self.decimals)]
    #ylim = [np.round(ylim[0], self.decimals), np.round(ylim[1], self.decimals)]

    dx = x[1] - x[0]
    dy = y[1] - y[0]
    ximin, ximax = index_between(x, xlim, mode='edge')[0] # starting and ending index of the subgrid
    yimin, yimax = index_between(y, ylim, mode='edge')[0] # starting and ending index of the subgrid
    _nx = ximax - ximin + 1
    _ny = yimax - yimin + 1
    xemin, xemax = x[ximin] - 0.5 * dx, x[ximax] + 0.5 * dx
    yemin, yemax = y[yimin] - 0.5 * dy, y[yimax] + 0.5 * dy

    # nested grid
    xe_sub = np.linspace(xemin, xemax, _nx * nsub + 1)
    ye_sub = np.linspace(yemin, yemax, _ny * nsub + 1)
    x_sub = 0.5 * (xe_sub[:-1] + xe_sub[1:])
    y_sub = 0.5 * (ye_sub[:-1] + ye_sub[1:])
    #xx_sub, yy_sub = np.meshgrid(x_sub, y_sub)
    return ximin, ximax, yimin, yimax, x_sub, y_sub



def nestgrid_3D(x, y, z, xlim, ylim, zlim, nsub, decimals = 4.):
    # error check
    if (len(xlim) != 2) | (len(ylim) != 2) | (len(zlim) != 2):
        print('ERROR\tnest: Input xlim/ylim/zlim must be list as [min, max].')
        return 0
    # decimals
    #xlim = [np.round(xlim[0], self.decimals), np.round(xlim[1], self.decimals)]
    #ylim = [np.round(ylim[0], self.decimals), np.round(ylim[1], self.decimals)]

    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = z[1] - z[0]
    ximin, ximax = index_between(x, xlim, mode='edge')[0] # starting and ending index of the subgrid
    yimin, yimax = index_between(y, ylim, mode='edge')[0] # starting and ending index of the subgrid
    zimin, zimax = index_between(z, zlim, mode='edge')[0] # starting and ending index of the subgrid
    _nx = ximax - ximin + 1
    _ny = yimax - yimin + 1
    _nz = zimax - zimin + 1
    xemin, xemax = x[ximin] - 0.5 * dx, x[ximax] + 0.5 * dx
    yemin, yemax = y[yimin] - 0.5 * dy, y[yimax] + 0.5 * dy
    zemin, zemax = z[zimin] - 0.5 * dz, z[zimax] + 0.5 * dz

    # nested grid
    xe_sub = np.linspace(xemin, xemax, _nx * nsub + 1)
    ye_sub = np.linspace(yemin, yemax, _ny * nsub + 1)
    ze_sub = np.linspace(zemin, zemax, _nz * nsub + 1)
    x_sub = 0.5 * (xe_sub[:-1] + xe_sub[1:])
    y_sub = 0.5 * (ye_sub[:-1] + ye_sub[1:])
    z_sub = 0.5 * (ze_sub[:-1] + ze_sub[1:])
    #xx_sub, yy_sub, zz_sub = np.meshgrid(x_sub, y_sub, z_sub, indexing = 'ij')
    return ximin, ximax, yimin, yimax, zimin, zimax, x_sub, y_sub, z_sub



class Nested2DGrid_old(object):
    """docstring for NestedGrid"""
    def __init__(self, xx, yy, precision = 4):
        super(Nested2DGrid, self).__init__()
        self.xx = xx
        self.yy = yy
        ny, nx = xx.shape
        self.ny, self.nx = ny, nx
        self.dx = xx[0,1] - xx[0,0]
        self.dy = yy[1,0] - yy[0,0]
        self.xc = xx[ny//2, nx//2]
        self.yc = yy[ny//2, nx//2]
        self.yci, self.xci = ny//2, nx//2

        if (_check := self.check_symmetry(precision))[0]:
            pass
        else:
            print('ERROR\tNested2DGrid: Input grid must be symmetric but not.')
            print('ERROR\tNested2DGrid: Condition.')
            print('ERROR\tNested2DGrid: [xcent, ycent, dx, dy]')
            print('ERROR\tNested2DGrid: ', _check[1])
            return None

        # retrive x and y
        x = self.xx[0,:]
        y = self.yy[:,0]
        xe = np.hstack([x - self.dx * 0.5, x[-1] + self.dx * 0.5])
        ye = np.hstack([y - self.dy * 0.5, y[-1] + self.dy * 0.5])
        self.x, self.y = x, y
        self.xe, self.ye = xe, ye
        self.decimals = precision


    def check_symmetry(self, decimals = 5):
        nx, ny = self.nx, self.ny
        xc = np.round(self.xc, decimals)
        yc = np.round(self.yc, decimals)
        _xcent = (xc == 0.) if nx%2 == 1 else (xc == - np.round(self.xx[ny//2 - 1, nx//2 - 1], decimals))
        _ycent = (yc == 0.) if ny%2 == 1 else (yc == - np.round(self.yy[ny//2 - 1, nx//2 - 1], decimals))
        delxs = (self.xx[1:,1:] - self.xx[:-1,:-1]) / self.dx
        delys = (self.yy[1:,1:] - self.yy[:-1,:-1]) / self.dy
        _xdel = (np.round(delxs, decimals) == 1. ).all()
        _ydel = (np.round(delys, decimals)  == 1. ).all()
        cond = [_xdel, _ydel] # _xcent, _ycent,
        return all(cond), cond
    

    def nest(self, xlim,  ylim, nsub = 2):
        # error check
        if (len(xlim) != 2) | (len(ylim) != 2):
            print('ERROR\tnest: Input xlim and/or ylim is not valid.')
            return 0
        # decimals
        xlim = [np.round(xlim[0], self.decimals), np.round(xlim[1], self.decimals)]
        ylim = [np.round(ylim[0], self.decimals), np.round(ylim[1], self.decimals)]

        self.nsub = nsub
        self.xlim_sub, self.ylim_sub = xlim, ylim
        ximin, ximax = index_between(self.x, xlim, mode='edge')[0]
        yimin, yimax = index_between(self.y, ylim, mode='edge')[0]
        _nx = ximax - ximin + 1
        _ny = yimax - yimin + 1
        xemin, xemax = self.xe[ximin], self.xe[ximax + 1]
        yemin, yemax = self.ye[yimin], self.ye[yimax + 1]
        self.xi0, self.xi1 = ximin, ximax # Starting and ending indices of nested grid
        self.yi0, self.yi1 = yimin, yimax # Starting and ending indices of nested grid

        # nested grid
        xe_sub = np.linspace(xemin, xemax, _nx * nsub + 1)
        ye_sub = np.linspace(yemin, yemax, _ny * nsub + 1)
        x_sub = 0.5 * (xe_sub[:-1] + xe_sub[1:])
        y_sub = 0.5 * (ye_sub[:-1] + ye_sub[1:])
        xx_sub, yy_sub = np.meshgrid(x_sub, y_sub)
        self.xe_sub, self.ye_sub = xe_sub, ye_sub
        self.x_sub, self.y_sub = x_sub, y_sub
        self.xx_sub, self.yy_sub = xx_sub, yy_sub
        self.dx_sub, self.dy_sub = self.dx / nsub, self.dy / nsub
        self.nx_sub, self.ny_sub = len(x_sub), len(y_sub)
        return xx_sub, yy_sub


    def where_subgrid(self):
        return np.where(
            (self.xx >= self.xlim_sub[0]) * (self.xx <= self.xlim_sub[1]) \
            * (self.yy >= self.ylim_sub[0]) * (self.yy <= self.ylim_sub[1]))


    def binning_onsubgrid(self, data):
        nbin = self.nsub
        d_avg = np.array([
            data[i::nbin, i::nbin]
            for i in range(nbin)
            ])
        return np.nanmean(d_avg, axis = 0)


    def edgecut_indices(self, xlength, ylength):
        # odd or even
        x_oddeven = self.nx%2
        y_oddeven = self.ny%2
        # edge indices for subgrid
        xi = int(xlength / self.dx_sub) if self.dx_sub > 0 else int(- xlength / self.dx_sub)
        yi = int(ylength / self.dy_sub)
        _nx_resub = int(self.nx_sub - 2 * xi) // self.nsub # nx of subgrid after cutting edge
        _ny_resub = int(self.ny_sub - 2 * yi) // self.nsub # ny of subgrid after cutting edge
        # fit odd/even
        if _nx_resub%2 != x_oddeven: _nx_resub += 1
        if _ny_resub%2 != y_oddeven: _ny_resub += 1
        # nx, ny of the new subgrid and new xi and yi
        nx_resub = _nx_resub * self.nsub
        ny_resub = _ny_resub * self.nsub
        xi = (self.nx_sub - nx_resub) // 2
        yi = (self.ny_sub - ny_resub) // 2
        # for original grid
        xi0 = int((self.nx - nx_resub / self.nsub) * 0.5)
        yi0 = int((self.ny - ny_resub / self.nsub) * 0.5)
        #print(nx_resub / self.nsub, self.nx - xi0 - xi0)
        return xi, yi, xi0, yi0


    def binning(self, nbin):
        if nbin%2 == 0:
            xx, yy = self.shift()
        else:
            xx, yy = self.xx.copy(), self.yy.copy()

        xcut = self.nx%nbin
        ycut = self.ny%nbin
        _xx = xx[ycut//2:-ycut//2, xcut//2:-xcut//2]
        _yy = yy[ycut//2:-ycut//2, xcut//2:-xcut//2]
        xx_avg = np.array([
            _xx[i::nbin, i::nbin]
            for i in range(nbin)
            ])
        yy_avg = np.array([
            _yy[i::nbin, i::nbin]
            for i in range(nbin)
            ])

        return np.average(xx_avg, axis= 0), np.average(yy_avg, axis= 0)


    def shift(self):
        rex = np.arange(-self.nx//2, self.nx//2+1, 1) + 0.5
        rex *= self.dx
        rey = np.arange(-self.ny//2, self.ny//2+1, 1) + 0.5
        rey *= self.dy
        return np.meshgrid(rex, rey)



class SubGrid2D(object):
    """docstring for NestedGrid"""
    def __init__(self, x, y, nsub = 2):
        super(SubGrid2D, self).__init__()
        # save grid info
        self.x = x
        self.y = y
        ny, nx = len(y), len(x)
        self.ny, self.nx = ny, nx
        self.dx = x[1] - x[0]
        self.dy = y[1] - y[0]
        self.xc = x[nx//2]
        self.yc = y[ny//2]
        self.yci, self.xci = ny//2, nx//2

        # cell edges
        self.xe = np.hstack([x - self.dx * 0.5, x[-1] + self.dx * 0.5])
        self.ye = np.hstack([y - self.dy * 0.5, y[-1] + self.dy * 0.5])

        # grid
        self.xx, self.yy = np.meshgrid(x, y)

        # subgrid
        self.subgrid(nsub = nsub)


    def subgrid(self, nsub = 2):
        self.nsub = nsub
        nx_sub, ny_sub = self.nx * nsub, self.ny * nsub
        self.nx_sub, self.ny_sub = nx_sub, ny_sub

        # sub grid
        xemin, xemax = self.xe[0], self.xe[-1] # edge of the original grid
        yemin, yemax = self.ye[0], self.ye[-1] # edge of the original grid
        xe_sub = np.linspace(xemin, xemax, nx_sub + 1)
        ye_sub = np.linspace(yemin, yemax, ny_sub + 1)
        x_sub = 0.5 * (xe_sub[:-1] + xe_sub[1:])
        y_sub = 0.5 * (ye_sub[:-1] + ye_sub[1:])
        xx_sub, yy_sub = np.meshgrid(x_sub, y_sub)
        self.xe_sub, self.ye_sub = xe_sub, ye_sub
        self.x_sub, self.y_sub = x_sub, y_sub
        self.xx_sub, self.yy_sub = xx_sub, yy_sub
        self.dx_sub, self.dy_sub = self.dx / nsub, self.dy / nsub
        #self.nx_sub, self.ny_sub = len(x_sub), len(y_sub)
        return xx_sub, yy_sub


    def binning_onsubgrid(self, data):
        nbin = self.nsub
        d_avg = np.array([
            data[i::nbin, i::nbin]
            for i in range(nbin)
            ])
        return np.nanmean(d_avg, axis = 0)


    def binning_onsubgrid_layered(self, data):
        nbin = self.nsub
        dshape = len(data.shape)
        if dshape == 2:
            d_avg = np.array([
                data[i::nbin, i::nbin]
                for i in range(nbin)
                ])
        elif dshape == 3:
            d_avg = np.array([
                data[:, i::nbin, i::nbin]
                for i in range(nbin)
                ])
        elif dshape ==4:
            d_avg = np.array([
                data[:, :, i::nbin, i::nbin]
                for i in range(nbin)
                ])
        else:
            print('ERROR\tbinning_onsubgrid_layered: only Nd of data of 2-4 is now supported.')
            return 0
        return np.nanmean(d_avg, axis = 0)


    def binning(self, nbin):
        if nbin%2 == 0:
            xx, yy = self.shift()
        else:
            xx, yy = self.xx.copy(), self.yy.copy()

        xcut = self.nx%nbin
        ycut = self.ny%nbin
        _xx = xx[ycut//2:-ycut//2, xcut//2:-xcut//2]
        _yy = yy[ycut//2:-ycut//2, xcut//2:-xcut//2]
        xx_avg = np.array([
            _xx[i::nbin, i::nbin]
            for i in range(nbin)
            ])
        yy_avg = np.array([
            _yy[i::nbin, i::nbin]
            for i in range(nbin)
            ])

        return np.average(xx_avg, axis= 0), np.average(yy_avg, axis= 0)


    def shift(self):
        rex = np.arange(-self.nx//2, self.nx//2+1, 1) + 0.5
        rex *= self.dx
        rey = np.arange(-self.ny//2, self.ny//2+1, 1) + 0.5
        rey *= self.dy
        return np.meshgrid(rex, rey)



class SubGrid2D_old(object):
    """docstring for NestedGrid"""
    def __init__(self, xx, yy, nsub = 2):
        super(SubGrid2D, self).__init__()
        # save grid info
        self.xx = xx
        self.yy = yy
        ny, nx = xx.shape
        self.ny, self.nx = ny, nx
        self.dx = xx[0,1] - xx[0,0]
        self.dy = yy[1,0] - yy[0,0]
        self.xc = xx[ny//2, nx//2]
        self.yc = yy[ny//2, nx//2]
        self.yci, self.xci = ny//2, nx//2

        # retrive x and y
        x = self.xx[0,:]
        y = self.yy[:,0]
        xe = np.hstack([x - self.dx * 0.5, x[-1] + self.dx * 0.5])
        ye = np.hstack([y - self.dy * 0.5, y[-1] + self.dy * 0.5])
        self.x, self.y = x, y
        self.xe, self.ye = xe, ye

        # subgrid
        self.subgrid(nsub = nsub)


    def subgrid(self, nsub = 2):
        self.nsub = nsub
        nx_sub, ny_sub = self.nx * nsub, self.ny * nsub
        self.nx_sub, self.ny_sub = nx_sub, ny_sub

        # sub grid
        xemin, xemax = self.xe[0], self.xe[-1] # edge of the original grid
        yemin, yemax = self.ye[0], self.ye[-1] # edge of the original grid
        xe_sub = np.linspace(xemin, xemax, nx_sub + 1)
        ye_sub = np.linspace(yemin, yemax, ny_sub + 1)
        x_sub = 0.5 * (xe_sub[:-1] + xe_sub[1:])
        y_sub = 0.5 * (ye_sub[:-1] + ye_sub[1:])
        xx_sub, yy_sub = np.meshgrid(x_sub, y_sub)
        self.xe_sub, self.ye_sub = xe_sub, ye_sub
        self.x_sub, self.y_sub = x_sub, y_sub
        self.xx_sub, self.yy_sub = xx_sub, yy_sub
        self.dx_sub, self.dy_sub = self.dx / nsub, self.dy / nsub
        #self.nx_sub, self.ny_sub = len(x_sub), len(y_sub)
        return xx_sub, yy_sub


    def binning_onsubgrid(self, data):
        nbin = self.nsub
        d_avg = np.array([
            data[i::nbin, i::nbin]
            for i in range(nbin)
            ])
        return np.nanmean(d_avg, axis = 0)


    def binning_onsubgrid_layered(self, data):
        nbin = self.nsub
        dshape = len(data.shape)
        if dshape == 2:
            d_avg = np.array([
                data[i::nbin, i::nbin]
                for i in range(nbin)
                ])
        elif dshape == 3:
            d_avg = np.array([
                data[:, i::nbin, i::nbin]
                for i in range(nbin)
                ])
        elif dshape ==4:
            d_avg = np.array([
                data[:, :, i::nbin, i::nbin]
                for i in range(nbin)
                ])
        else:
            print('ERROR\tbinning_onsubgrid_layered: only Nd of data of 2-4 is now supported.')
            return 0
        return np.nanmean(d_avg, axis = 0)


    def binning(self, nbin):
        if nbin%2 == 0:
            xx, yy = self.shift()
        else:
            xx, yy = self.xx.copy(), self.yy.copy()

        xcut = self.nx%nbin
        ycut = self.ny%nbin
        _xx = xx[ycut//2:-ycut//2, xcut//2:-xcut//2]
        _yy = yy[ycut//2:-ycut//2, xcut//2:-xcut//2]
        xx_avg = np.array([
            _xx[i::nbin, i::nbin]
            for i in range(nbin)
            ])
        yy_avg = np.array([
            _yy[i::nbin, i::nbin]
            for i in range(nbin)
            ])

        return np.average(xx_avg, axis= 0), np.average(yy_avg, axis= 0)


    def shift(self):
        rex = np.arange(-self.nx//2, self.nx//2+1, 1) + 0.5
        rex *= self.dx
        rey = np.arange(-self.ny//2, self.ny//2+1, 1) + 0.5
        rey *= self.dy
        return np.meshgrid(rex, rey)


def index_between(t, tlim, mode='all'):
    if not (len(tlim) == 2):
        if mode=='all':
            return np.full(np.shape(t), True)
        elif mode == 'edge':
            if len(t.shape) == 1:
                return tuple([[0, len(t)-1]])
            else:
                return tuple([[0, t.shape[i]] for i in range(len(t.shape))])
        else:
            print('index_between: mode parameter is not right.')
            return np.full(np.shape(t), True)
    else:
        if mode=='all':
            return (tlim[0] <= t) * (t <= tlim[1])
        elif mode == 'edge':
            nonzero = np.nonzero((tlim[0] <= t) * (t <= tlim[1]))
            return tuple([[np.min(i), np.max(i)] for i in nonzero])
        else:
            print('index_between: mode parameter is not right.')
            return (tlim[0] <= t) * (t <= tlim[1])


class Nested1DGrid(object):
    """docstring for NestedGrid"""
    def __init__(self, x, precision = 4):
        super(Nested1DGrid, self).__init__()
        self.x = x
        nx = len(x)
        self.nx = nx
        self.dx = x[1] - x[0]
        self.xc = x[nx//2]
        self.xci = nx//2

        if (_check := self.check_symmetry(precision))[0]:
            pass
        else:
            print('ERROR\tNested2DGrid: Input grid must be symmetric but not.')
            print('ERROR\tNested2DGrid: Condition.')
            print('ERROR\tNested2DGrid: [dx]')
            print('ERROR\tNested2DGrid: ', _check[1])
            return None

        # retrive x and y
        xe = np.hstack([x - self.dx * 0.5, x[-1] + self.dx * 0.5])
        self.xe = xe
        self.decimals = precision


    def check_symmetry(self, decimals = 5):
        nx = self.nx
        xc = np.round(self.xc, decimals)
        #_xcent = (xc == 0.) if nx%2 == 1 else (xc == - np.round(self.x[nx//2 - 1], decimals))
        delxs = (self.x[1:] - self.x[:-1]) / self.dx
        _xdel = (np.round(delxs, decimals) == 1. ).all()
        cond = [_xdel] # _xcent
        return all(cond), cond
    

    def nest(self, nsub = 3, xlim = None):
        # subgrid
        self.nsub = nsub
        self.xlim_sub = xlim
        if xlim is not None:
            # error check
            if len(xlim) != 2:
                print('ERROR\tnest: Input xlim is not valid.')
                return 0
            # decimals
            xlim = [np.round(xlim[0], self.decimals), np.round(xlim[1], self.decimals)]
            ximin, ximax = index_between(self.x, xlim, mode='edge')[0]
            _nx = ximax - ximin + 1
            xemin, xemax = self.xe[ximin], self.xe[ximax + 1]
            self.xi0, self.xi1 = ximin, ximax # Starting and ending indices of nested grid

            # nested grid
            xe_sub = np.linspace(xemin, xemax, _nx * nsub + 1)
            x_sub = 0.5 * (xe_sub[:-1] + xe_sub[1:])
            self.xe_sub = xe_sub
            self.x_sub = x_sub
            self.dx_sub = self.dx / nsub
            self.nx_sub = len(x_sub)
        else:
            xe_sub = np.arange(0., self.nx * nsub + 1, 1.) # delta in pixel
            dx_sub = self.dx / nsub # delta for subgrid
            xe_sub *= dx_sub # increase
            xe_sub += self.xe[0] # cell edge
            x_sub = 0.5 * (xe_sub[:-1] + xe_sub[1:]) # cell center
            # save
            self.xe_sub = xe_sub
            self.x_sub = x_sub
            self.dx_sub = dx_sub
            self.nx_sub = len(x_sub)
        return x_sub


    def binning_onsubgrid(self, data):
        nbin = self.nsub
        if len(data.shape) == 1:
            d_avg = np.array([
                data[i::nbin]
                for i in range(nbin)
                ])
        elif len(data.shape) == 2:
            d_avg = np.array([
                data[i::nbin, :]
                for i in range(nbin)
                ])
        elif len(data.shape) == 3:
            d_avg = np.array([
                data[i::nbin, :, :]
                for i in range(nbin)
                ])
        else:
            print('Ndim is expected to be <= 3.')
            return 0
        return np.nanmean(d_avg, axis = 0)


    def shift(self):
        rex = np.arange(-self.nx//2, self.nx//2+1, 1) + 0.5
        rex *= self.dx
        return rex



def main():
    # ---------- input -----------
    nx, ny = 32, 33
    xe = np.linspace(-10, 10, nx+1)
    #xe = np.logspace(-1, 1, nx+1)
    ye = np.linspace(-10, 10, ny+1)
    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    xx, yy = np.meshgrid(xc, yc)
    # ----------------------------


    # ---------- debug ------------
    '''
    # 2D
    # model on an input grid
    dd = np.exp( - (xx**2. / 18.) - (yy**2. / 18.))

    # nested grid
    gridder = Nested2DGrid(xx,yy)
    xx_sub, yy_sub = gridder.nest([-3., 3.], [-3., 3.], 2)
    # model on the nested grid
    dd_sub = np.exp( - (xx_sub**2. / 18.) - (yy_sub**2. / 18.))
    # binned
    dd_binned = gridder.binning_onsubgrid(dd_sub)
    dd_re = dd.copy()
    #print(gridder.where_subgrid())
    dd_re[gridder.where_subgrid()] = dd_binned.ravel()



    # plot
    fig, axes = plt.subplots(1,3)
    ax1, ax2, ax3 = axes

    xx_plt, yy_plt = np.meshgrid(xe, ye)
    ax1.pcolor(xx_plt, yy_plt, dd, vmin = 0., vmax = 1.)
    #ax1.pcolor(xx_sub_plt, yy_sub_plt, dd_sub)

    xx_sub_plt, yy_sub_plt = np.meshgrid(gridder.xe_sub, gridder.ye_sub)
    ax2.pcolor(xx_sub_plt, yy_sub_plt, dd_sub, vmin = 0., vmax = 1)
    c = ax3.pcolor(xx_plt, yy_plt, dd - dd_re, vmin = 0., vmax = 1)


    for axi in [ax1, ax2, ax3]:
        axi.set_xlim(-10,10)
        axi.set_ylim(-10,10)

    for axi in [ax2, ax3]:
        axi.set_xticklabels('')
        axi.set_yticklabels('')

    cax = ax3.inset_axes([1.03, 0., 0.03, 1.])
    plt.colorbar(c, cax=cax)
    plt.show()
    '''


    # 1D
    # model on an input grid
    model = lambda x: np.exp( - (x**2. / 18.))
    d = model(xc)
    # nested grid
    nstg1D = Nested1DGrid(xc)
    x_sub = nstg1D.nest(3)
    # model on the nested grid
    d_sub = model(x_sub)
    # binned
    d_binned = nstg1D.binning_onsubgrid(d_sub)

    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for xi, di, label, ls in zip(
        [xc, x_sub, xc],
        [d, d_sub, d_binned],
        ['Original', 'Subgrid', 'Binned'],
        ['-', '-', '--']
        ):
        ax.step(xi, di, where = 'mid', lw = 2., alpha = 0.5, ls = ls)

    plt.show()



if __name__ == '__main__':
    main()