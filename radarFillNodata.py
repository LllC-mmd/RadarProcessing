import timeit
import os
import re
import sys
import numpy as np
from scipy import interpolate
from scipy import ndimage
import matplotlib.pyplot as plt
import netCDF4 as nc
import cv2


def nc_interploate(dataset_path, new_dir, suffix, method, rho_thres, smooth_factor):
    nc_name = re.findall(r"\/(.+)\.netcdf", dataset_path)[0]
    os.system("cp %s %s" % (dataset_path, os.path.join(new_dir, nc_name + suffix + ".netcdf")))
    nc_ds = nc.Dataset(os.path.join(new_dir, nc_name + suffix + ".netcdf"), "a")

    #start = timeit.default_timer()
    Dp = np.array(nc_ds.variables["DifferentialPhase"])
    rho = np.array(nc_ds.variables["CrossPolCorrelation"])

    # set the mask area to determine which data used for interpolation
    # mask_tmp = ~np.isnan(rho)
    # mask_tmp[mask_tmp] &= rho[mask_tmp] < 0.4
    # rho_mask = np.ma.masked_where(mask_tmp, rho)
    rho_mask = np.ma.masked_where(rho < rho_thres, rho)
    grid_X, grid_Y = np.mgrid[0:rho.shape[0], 0:rho.shape[1]]
    X_use = grid_X[~rho_mask.mask]
    Y_use = grid_Y[~rho_mask.mask]
    pt_use = np.concatenate((np.reshape(X_use, (-1, 1)), np.reshape(Y_use, (-1, 1))), axis=1)
    pt_inter = np.concatenate((np.reshape(grid_X.flatten(), (-1, 1)), np.reshape(grid_Y.flatten(), (-1, 1))), axis=1)

    # interpolate rho
    rho[rho <= -0.025] = np.NaN
    rho_use = rho[~rho_mask.mask]
    rho_new = interpolate.griddata(pt_use, rho_use, pt_inter, method=method)
    rho_new = np.reshape(rho_new, (rho.shape[0], rho.shape[1]))

    # interpolate Dp
    Dp[Dp == -2] = np.NaN
    Dp_use = Dp[~rho_mask.mask]
    Dp_new = interpolate.griddata(pt_use, Dp_use, pt_inter, method=method)
    Dp_new = np.reshape(Dp_new, (Dp.shape[0], Dp.shape[1]))

    # adjust the reserved area
    # larger erode kernel & iter and open kernel, more area would be reserved
    # larger close kernel, less area would be reserved
    kernel_erode = np.ones((2, 2), np.int8)
    kernel_open = np.ones((4, 4), np.int8)
    kernel_close = np.ones((10, 10), np.int8)
    mask_res = rho_mask.mask.astype(np.uint8)
    rho_inside_mask = cv2.erode(mask_res, kernel_erode, iterations=4)
    rho_inside_mask = cv2.morphologyEx(rho_inside_mask, cv2.MORPH_OPEN, kernel_open)
    rho_inside_mask = cv2.morphologyEx(rho_inside_mask, cv2.MORPH_CLOSE, kernel_close)

    # restrict rho's range
    rho_new[rho_inside_mask.astype(np.bool)] = np.NaN
    mask_tmp = ~np.isnan(rho_new)
    mask_tmp[mask_tmp] &= rho_new[mask_tmp] > 1.0
    rho_new[mask_tmp] = 1.0
    mask_tmp = ~np.isnan(rho_new)
    mask_tmp[mask_tmp] &= rho_new[mask_tmp] < -1.0
    rho_new[mask_tmp] = -1.0
    nc_ds.variables["CrossPolCorrelation"][:, :] = rho_new

    # smooth Dp by Gaussian filter and restrict Dp's range
    Dp_new[rho_inside_mask.astype(np.bool)] = np.NaN
    Dp_new = np.array([ndimage.gaussian_filter1d(ax, sigma=smooth_factor) for ax in Dp_new])
    mask_tmp = ~np.isnan(Dp_new)
    mask_tmp[mask_tmp] &= Dp_new[mask_tmp] > 360
    Dp_new[mask_tmp] = 360.0
    mask_tmp = ~np.isnan(Dp_new)
    mask_tmp[mask_tmp] &= Dp_new[mask_tmp] < -360
    Dp_new[mask_tmp] = -360.0
    nc_ds.variables["DifferentialPhase"][:, :] = Dp_new

    # see the reserved area (px value==0)
    '''
    plt.imshow(rho_inside_mask, cmap='gray')
    plt.colorbar()
    plt.show()
    '''
    # see the interpolated result
    '''
    fig, ax = plt.subplots(1, figsize=(16, 8))
    ax1 = ax.imshow(X=Dp_new, cmap="rainbow", vmax=np.nanmax(Dp), vmin=np.nanmin(Dp))
    fig.colorbar(ax1)
    plt.savefig("Dp_GS_filter.png", dpi=400)
    '''
    #stop = timeit.default_timer()
    #print("Run-time of Interpolation: ", stop - start)
    nc_ds.close()


data_list = os.listdir(sys.argv[1])
data_newDir = sys.argv[2]
for d in data_list:
    if d.find("netcdf") != -1:
        d_path = os.path.join(sys.argv[1], d)
        nc_interploate(d_path, data_newDir, suffix=sys.argv[3], method=sys.argv[4], rho_thres=float(sys.argv[5]), smooth_factor=float(sys.argv[6]))
