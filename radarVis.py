import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as pcolor

import netCDF4 as nc
import wradlib as wrl


def ppi_vis(pol_var, range=None, title=None, colorbar_label=None, cmap=None, norm=None, noData=None):
    if noData is not None:
        pol_var = np.where(pol_var==noData, np.nan, pol_var)

    if cmap is None and norm is None:
        colors = np.array([[127/255, 127/255, 127/255], [45/255, 255/255, 255/255], [26/255, 155/255, 255/255],
                            [0/255, 15/255, 150/255], [40/255, 255/255, 47/255], [26/255, 177/255, 30/255],
                            [15/255, 125/255, 18/255], [250/255, 253/255, 55/255], [255/255, 213/255, 48/255],
                            [237/255, 143/255, 37/255], [255/255, 12/255, 24/255], [189/255, 7/255, 17/255],
                            [124/255, 3/255, 8/255], [187/255, 44/255, 232/255], [108/255, 31/255, 120/255]])
        nvals = np.linspace(0, 180, 16)
        cmap, norm = pcolor.from_levels_and_colors(nvals, colors)

    fig = plt.figure(figsize=(10, 8))
    ax, cf = wrl.vis.plot_ppi(data=pol_var, r=range, fig=fig, cmap=cmap, norm=norm)

    if title is not None:
        ax.set_title(title)

    cbar = fig.colorbar(cf)
    if colorbar_label is not None:
        cbar.set_label(colorbar_label)

    plt.grid(color="grey")
    # plt.show()
    plt.savefig("LP_phiDP.png", dpi=400)


raw_dir = "Input"
raw_fname = "BJXFS_2.5_20190909_180000.netcdf"
nc_ds = nc.Dataset(os.path.join(raw_dir, raw_fname), "r")

# convert mm to km
GateWidth = np.array(nc_ds.variables["GateWidth"]) / 1000.0 / 1000.0
zDr = np.array(nc_ds.variables["DifferentialReflectivity"])
Phi_dp = np.array(nc_ds.variables["DifferentialPhase"])
rho_hv = np.array(nc_ds.variables["CrossPolCorrelation"])
KDP = np.array(nc_ds.variables["KDP"])
reflectivity = np.array(nc_ds.variables["Reflectivity"])

num_radial, num_gate = Phi_dp.shape
GateWidth_cum = np.full(num_gate, GateWidth[0])
GateWidth_cum = np.cumsum(GateWidth_cum)

ppi_vis(Phi_dp, range=GateWidth_cum, title="original $\Phi_{dp}$ at 2019/09/09 18:00:00", colorbar_label="$\Phi_{dp}$ [Degrees]", noData=-2.0)