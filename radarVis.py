import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as pcolor

import netCDF4 as nc
import wradlib as wrl


def ppi_vis(pol_var, save_path, var_name, range=None, azimuth=None, title=None, colorbar_label=None, cmap=None, norm=None, noData=None):
    if noData is None:
        if var_name == "DifferentialPhase":
            noData = -2.0
        elif var_name == "KDP":
            noData = -5.0
        elif var_name == "Reflectivity":
            noData = -33.0
        else:
            raise NotImplementedError("Unknown Variable")

    pol_var = np.where(pol_var == noData, np.nan, pol_var)

    if cmap is None and norm is None:
        if var_name == "DifferentialPhase":
            colors = np.array([[133 / 255, 216 / 255, 253 / 255], [41 / 255, 237 / 255, 238 / 255], [29 / 255, 175 / 255, 243 / 255],
                               [10 / 255, 35 / 255, 244 / 255], [41 / 255, 253 / 255, 47 / 255], [30 / 255, 199 / 255, 34 / 255],
                               [19 / 255, 144 / 255, 22 / 255], [254 / 255, 253 / 255, 56 / 255], [230 / 255, 191 / 255, 43 / 255],
                               [251 / 255, 144 / 255, 37 / 255], [249 / 255, 14 / 255, 28 / 255], [209 / 255, 11 / 255, 21 / 255],
                               [189 / 255, 8 / 255, 19 / 255], [219 / 255, 102 / 255, 252 / 255], [186 / 255, 36 / 255, 235 / 255],
                               [92 / 255, 14 / 255, 116 / 255]])
            nvals = np.linspace(0, 180, 17)
        elif var_name == "KDP":
            colors = np.array([[133 / 255, 216 / 255, 253 / 255], [41 / 255, 237 / 255, 238 / 255], [29 / 255, 175 / 255, 243 / 255],
                               [10/255, 35/255, 244/ 255], [41/255, 253/255, 47/255], [30/ 255, 199/255, 34/255],
                               [19 / 255, 144 / 255, 22 / 255], [254 / 255, 253 / 255, 56 / 255], [230 / 255, 191 / 255, 43 / 255],
                               [251 / 255, 144 / 255, 37 / 255], [249 / 255, 14 / 255, 28 / 255], [209 / 255, 11 / 255, 21 / 255],
                               [189 / 255, 8 / 255, 19 / 255], [219 / 255, 102 / 255, 252 / 255], [186 / 255, 36 / 255, 235 / 255],
                               [92 / 255, 14 / 255, 116 / 255]])
            nvals = np.linspace(-1.0, 3.0, 17)
        elif var_name == "Reflectivity":
            colors = np.array([[133 / 255, 216 / 255, 253 / 255], [41 / 255, 237 / 255, 238 / 255], [29 / 255, 175 / 255, 243 / 255],
                               [10 / 255, 35 / 255, 244 / 255], [41 / 255, 253 / 255, 47 / 255], [30 / 255, 199 / 255, 34 / 255],
                               [19 / 255, 144 / 255, 22 / 255], [254 / 255, 253 / 255, 56 / 255], [230 / 255, 191 / 255, 43 / 255],
                               [251 / 255, 144 / 255, 37 / 255], [249 / 255, 14 / 255, 28 / 255], [209 / 255, 11 / 255, 21 / 255],
                               [189 / 255, 8 / 255, 19 / 255], [219 / 255, 102 / 255, 252 / 255], [186 / 255, 36 / 255, 235 / 255]])
            nvals = np.linspace(0.0, 75.0, 16)
        else:
            raise NotImplementedError("Unknown Variable")

        cmap, norm = pcolor.from_levels_and_colors(nvals, colors)

    fig = plt.figure(figsize=(10, 8))
    ax, cf = wrl.vis.plot_ppi(data=pol_var, r=range, az=azimuth, fig=fig, cmap=cmap, norm=norm)

    if title is not None:
        ax.set_title(title)

    cbar = fig.colorbar(cf)
    if colorbar_label is not None:
        cbar.set_label(colorbar_label)

    plt.grid(color="grey")
    # plt.show()
    plt.savefig(save_path, dpi=400)


def plot_ZH(zH_array, save_path, title=None):
    colors = np.array([[133/255, 216/255, 253/255], [41/255, 237/255, 238/255], [29/255, 175/255, 243/255],
                       [10/255, 35/255, 244/255], [41/255, 253/255, 47/255], [30/255, 199/255, 34/255],
                       [19/255, 144/255, 22/255], [254/255, 253/255, 56/255], [230/255, 191/255, 43/255],
                       [251/255, 144/255, 37/255], [249/255, 14/255, 28/255], [209/255, 11/255, 21/255],
                       [189/255, 8/255, 19/255], [219/255, 102/255, 252/255], [186/255, 36/255, 235/255]])
    nvals = np.linspace(0.0, 45.0, 16)
    cmap, norm = pcolor.from_levels_and_colors(nvals, colors)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax_c = ax.imshow(zH_array, cmap=cmap, norm=norm)

    if title is not None:
        ax.set_title(title)

    cbar = fig.colorbar(ax_c)
    plt.show()
    # plt.savefig(save_path, dpi=400)
