import numpy as np
import scipy.optimize as opt
import scipy.integrate as integrate
import cv2

import netCDF4 as nc
import os

import pulp as pl

from radarVis import *


fuzzy_dict = {"clutter":
                {"Z": {"m": -99900.0, "a": -99900.0, "b": -99900.0, "w":-99900.0},
                 "zDr": {"m": -99900.0, "a": -99900.0, "b": -99900.0, "w": -99900.0},
                 "zDr_std": {"m": 2.0, "al": 0.8, "bl": 1.8, "ar": 1000, "br": 2, "w": 0.5},
                 "PhiDP": {"m": -99900.0, "a": -99900.0, "b": -99900.0, "w": -99900.0},
                 "PhiDP_std": {"m": 40, "al": 16.0, "bl": 1.8, "ar": 1000, "br": 2, "w": 1.0},
                 "RhoHV": {"m": 0.5, "al": 0.4, "bl": 1.0, "ar": 0.3, "br": 2.0, "w": 0.2},
                 "KDP": {"m": -99900.0, "a": -99900.0, "b": -99900.0, "w": -99900.0},
                 "T": {"m": -99900.0, "a": -99900.0, "b": -99900.0, "w": -99900.0}
                }
            }


# ************************* General Math functions *************************
def rolling_window_1D(dta, window_length):
    shape = dta.shape[:-1] + (dta.shape[-1] - window_length + 1, window_length)
    strides = dta.strides + (dta.strides[-1], )
    dta_window = np.lib.stride_tricks.as_strided(dta, shape=shape, strides=strides)
    return dta_window


def rolling_window_2D(dta, kernel_size):
    shape = dta.shape[:-2] + (dta.shape[-2] - kernel_size[-2] + 1, dta.shape[-1] - kernel_size[-1] + 1, kernel_size[-2], kernel_size[-1])
    strides = dta.strides + (dta.strides[-2], dta.strides[-1], )
    dta_window = np.lib.stride_tricks.as_strided(dta, shape=shape, strides=strides)
    return dta_window


def memFunc(x, m, al, bl, ar=None, br=None):
    if ar is None:
        ar = al
    if br is None:
        br = bl
    mf = np.where(x<=m, 1.0/(1.0+np.power(np.abs((x-m)/al), 2.0*bl)), 1.0/(1.0+np.power(np.abs((x-m)/ar), 2.0*br)))

    return mf


def get_dispersion(dta, axis=None, deg=True, nan_threshold=0.6):
    # ------If Phi_dp is represented in Degrees, convert it into Radian representation first
    if deg:
        dta = dta * np.pi / 180.0
    dta_c = np.cos(dta)
    dta_s = np.sin(dta)

    dispersion_raw = np.sqrt(np.nanmean(dta_c, axis=axis)**2 + np.nanmean(dta_s, axis=axis)**2)

    nan_count = np.sum(~np.isnan(dta), axis=-1) / dta.shape[-1]
    dispersion_raw[nan_count < nan_threshold] = np.nan

    return dispersion_raw


def get_good_point(dispersion, d_max, axis=None):
    test_sample = (dispersion > d_max)
    # test_sample = np.all(dispersion > d_max, axis=axis)
    return np.where(test_sample)[0]


def get_bad_point(dispersion, Rho_HV, d_max, rho_max, axis=None):
    test_sample = np.logical_and(np.logical_or(dispersion < d_max, np.isnan(dispersion)), Rho_HV < rho_max)
    # test_sample = np.logical_and(np.all(dispersion < d_max, axis=axis), Rho_HV < rho_max)
    return np.where(test_sample)[0]


def get_LinearCoef(y, x, axis=None):
    y_mean = np.mean(y, axis=axis)
    x_mean = np.mean(x, axis=axis)

    if axis is not None:
        y_mean = y_mean.reshape((-1, 1))
        x_mean = x_mean.reshape((-1, 1))

    coef = np.mean((y - y_mean)*(x - x_mean), axis=axis) / np.mean((x - x_mean)*(x - x_mean), axis=axis)

    return coef


def get_invW(dta, axis=None):
    invW = np.sqrt(np.var(dta/180.0*np.pi, axis=axis))

    return np.diag(invW)


def complex2deg(complex_array):
    deg_array = np.angle(complex_array, deg=True)
    return np.where(deg_array < 0.0, deg_array + 2*180.0, deg_array)


def ConnectedComponent_masking(dta, noData, population_min=30, nc=8):
    img_gray = np.full_like(dta, 255)
    img_gray[dta==noData] = 0
    num_labels, labels, stats, centers = cv2.connectedComponentsWithStats(img_gray.astype('uint8'), connectivity=nc, ltype=cv2.CV_32S)

    cid_minority = np.where(stats[1:, 4] < population_min)[0] + 1
    mask = np.zeros_like(dta)
    mask[dta == noData] = 1.0
    for cid in cid_minority:
        mask[labels == cid] = 1.0

    return mask.astype(np.bool)


def SpatialTexture_masking(dta, num_range, threshold, noData=None):
    dta_block = rolling_window_2D(dta, [2*num_range+1, 2*num_range+1])
    dta_block_SD = np.nanstd(dta_block, axis=(-1, -2))
    mask_SD = (dta_block_SD > threshold)

    if noData is not None:
        mask_nodata = (dta == noData)
        mask = np.logical_or(mask_SD, mask_nodata[num_range:-num_range, num_range:-num_range])
    else:
        mask = mask_SD

    mask = np.pad(mask, pad_width=(num_range, num_range), constant_values=(True, True))

    return mask.astype(np.bool)


def Spike_filter_2D(dta, num_range, threshold, noData=None):
    if noData is not None:
        dta = np.where(dta==noData, np.nan, dta)

    dta_block = rolling_window_2D(dta, [2*num_range+1, 2*num_range+1])

    dta_block_mean = np.nanmean(dta_block, axis=(-1, -2))
    dta_block_central = dta_block[:, :, num_range, num_range]
    dta_block_deviation = np.abs(dta_block_mean - dta_block_central)

    dta_smooth = np.where(dta_block_deviation > threshold, dta_block_mean, dta_block_central)
    dta[num_range:-num_range, num_range:-num_range] = dta_smooth

    return dta


# ************************* Visualization *************************
def plot_label(GateWidth_r, Phi_dp_array, labels):
    fig, ax = plt.subplots(3, 2, figsize=(10, 12))
    loc_dic = {0: [0, 0], 60: [0, 1], 120: [1, 0], 180: [1, 1], 240: [2, 0], 300: [2, 1]}

    for r in loc_dic.keys():
        ir, jr = loc_dic[r]
        n_color = len(np.bincount(labels)) - 1
        cmap = plt.cm.get_cmap("jet", n_color)

        clutter_loc = np.nonzero(labels == 0)[0]
        ax[ir, jr].scatter(GateWidth_r[clutter_loc], Phi_dp_array[r, clutter_loc], c="darkgrey", label=0, s=0.5)

        for l in range(0, n_color):
            weather_loc = np.nonzero(labels == l+1)[0]
            ax[ir, jr].scatter(GateWidth_r[weather_loc], Phi_dp_array[r, weather_loc], c=[cmap(l)], label=l+1, s=1)

        ax[ir, jr].set_title("No. Radial = %d" % r)
        ax[ir, jr].legend(fontsize="small", ncol=4)

    # plt.show()
    plt.savefig("DROPs_masking_pop1_combine.png", dpi=400)


# ************************* Quality Control Algorithms *************************
# ************* [0] Clutter Identification *************
# ---Phi_dp unfolding (necessary for non-angular method)

# ************* [1] Phase reconstruction *************
# ---Phi_dp unfolding (necessary for non-angular method)
def PhaseUnfolding(Phi_dp_array, rho_hv_array, GateWidth_array, max_phaseDiff=-80, dphase=180):
    std_start = 5.0
    rho_hv_start = 0.9
    std_ref = 15.0
    slope_min_ref = -5.0
    slope_max_ref = 20.0

    ref = 0.0
    pt_start = 0
    num_radial, num_gate = Phi_dp_array.shape
    for r in range(0, num_radial):
        w_r = GateWidth_array[r]
        GateWidth_r = np.ones(num_gate) * w_r
        GateWidth_r = np.cumsum(GateWidth_r)
        # ------determine the start point of meaningful Phi_dp
        for i in range(5, num_gate - 9):
            if (np.std(Phi_dp_array[r, i:i+10]) < std_start) and (np.all(rho_hv_array[r, i-5:i] > rho_hv_start)):
                ref = np.mean(Phi_dp_array[r, i-5:i])
                pt_start = i
                break

        for i in range(pt_start, num_gate - 9):
            sigma = np.std(Phi_dp_array[r, i:i+10])
            # ---------calculate the slope of Linear Regression over the last 5 gates
            # ---------slope = Cov(Phi_dp, r) / Var(r)
            cov_mat = np.cov(GateWidth_r[i-5:i], Phi_dp_array[r, i-5:i])
            slope = cov_mat[0, 1] / cov_mat[0, 0]
            if (sigma < std_ref) and (slope > slope_min_ref) and (slope < slope_max_ref):
                ref += slope * w_r
            if Phi_dp_array[r, i] - ref < max_phaseDiff:
                Phi_dp_array[r, i] += dphase

    return Phi_dp_array


# ---reconstruct Phi_dp using Linear Programming
def LP_solver(Phi_dp_array, weights=None):
    if weights is None:
        weights = np.ones_like(Phi_dp_array)

    num_gate = len(Phi_dp_array)

    # ------define the Problem object
    lp_prob = pl.LpProblem("Reconstruction", pl.LpMinimize)
    # ------define the decision variable
    z = ["z" + str(i) for i in range(0, num_gate)]
    x = ["x" + str(i) for i in range(0, num_gate)]
    var = z + x
    lp_var = pl.LpVariable.dicts("var", var, cat='Continuous')

    # ------define the objective function
    gate_used = np.where(weights != 0.0)[0]
    lp_prob += pl.lpSum([lp_var[var[i]] for i in gate_used])

    for i in range(0, num_gate):
        if weights[i] != 0:
            # ------add the fidelity constraints
            lp_prob += weights[i] * (lp_var[var[i]] - lp_var[var[i + num_gate]]) >= -Phi_dp_array[i]
            lp_prob += weights[i] * (lp_var[var[i]] + lp_var[var[i + num_gate]]) >= Phi_dp_array[i]

    for i in range(num_gate + 2, 2*num_gate - 2):
        w_i = weights[i-num_gate]
        if w_i != 0:
            # ------add the montonicity constraints via five-point Savitzky–Golay derivative filter
            lp_prob += w_i * (-0.2*lp_var[var[i-2]] + (-0.1)*lp_var[var[i-1]] + 0.1*lp_var[var[i+1]] + 0.2*lp_var[var[i+2]]) >= 0.0

    # ------solve the LP problem
    # ---------use GNU Linear Programming Kit solver
    # lp_prob.solve(pl.GLPK())
    # ---------use default CBC solver
    lp_prob.solve()
    # ------check the status of solution
    if lp_prob.status == 1:
        phi_rec = np.array([lp_var[var[i]].varValue for i in range(num_gate, 2*num_gate)])
        phi_rec[np.equal(phi_rec, None)] = np.nan
        # ------smoothing filter to prevent subﬁlter-length oscillations
        Phi_dp_array[2:-2] = 0.1*phi_rec[0:-4] + 0.25*phi_rec[1:-3] + 0.3*phi_rec[2:-2] + 0.25*phi_rec[3:-1] + 0.1*phi_rec[4:]
        Phi_dp_array[-2:] = Phi_dp_array[-3]
    else:
        print("Linear Programming failed !!!")

    return Phi_dp_array


def PhaseRec_LP(Phi_dp_array, KDP_array, rho_hv_array, GateWidth_array, num_good=10, num_bad=5, d_max=0.98, rho_max=0.9, population_min=5, masked=True):
    num_radial, num_gate = Phi_dp_array.shape

    for r in range(0, num_radial):
        # ------rain cell segmentation
        w_r = GateWidth_array[r]
        GateWidth_r = np.full(num_gate, w_r)
        GateWidth_r = np.cumsum(GateWidth_r)
        labels = dataMasking_DROPs(Phi_dp_array[r], GateWidth_r, rho_hv_array[r], num_good, num_bad, d_max, rho_max, population_min)
        labels = labels.astype(np.int64)

        # mask non-rain cell
        if masked:
            mask_r = (labels == 0)
            Phi_dp_array[r, mask_r] = -2.0
            KDP_array[r, mask_r] = -5.0

        # ------use Linear Programming in every rain cell
        n_cell = len(np.bincount(labels)) - 1
        for cid in range(1, n_cell + 1):
            cell_loc = np.nonzero(labels == cid)[0]
            Phi_dp_array[r, cell_loc] = LP_solver(Phi_dp_array[r, cell_loc])
            array_split = rolling_window_1D(Phi_dp_array[r, cell_loc], 5)
            KDP_array_r = (-0.2*array_split[:, 0] - 0.1*array_split[:, 1] + 0.1*array_split[:, 3] + 0.2*array_split[:, 4]) / w_r
            KDP_array[r, cell_loc] = np.concatenate(([KDP_array_r[0]], [KDP_array_r[0]], KDP_array_r, [KDP_array_r[-1]], [KDP_array_r[-1]]), axis=0)

    return Phi_dp_array, KDP_array


def PhaseRec_LP_mask(Phi_dp_array, KDP_array, rho_hv_array, GateWidth_array, num_good=10, num_bad=5, d_max=0.98, rho_max=0.9, population_min=5, masked=True):
    num_radial, num_gate = Phi_dp_array.shape

    for r in range(0, num_radial):
        # ------rain cell segmentation
        w_r = GateWidth_array[r]
        GateWidth_r = np.full(num_gate, w_r)
        GateWidth_r = np.cumsum(GateWidth_r)
        labels = dataMasking_DROPs(Phi_dp_array[r], GateWidth_r, rho_hv_array[r], num_good, num_bad, d_max, rho_max, population_min)
        labels = labels.astype(np.int64)

        weights = np.ones_like(Phi_dp_array[r])

        # mask non-rain cell, i.e., Non-Precipitation Echoes (NPE)
        if masked:
            mask_r = (labels == 0)
            weights[mask_r] = 0.0
            Phi_dp_array[r, mask_r] = -2.0
            KDP_array[r, mask_r] = -5.0

        # ------use Linear Programming for one radial observation excluding NPE
        Phi_dp_array[r] = LP_solver(Phi_dp_array[r], weights=weights)
        array_split = rolling_window_1D(Phi_dp_array[r], 5)
        KDP_array_r = (-0.2*array_split[:, 0] - 0.1*array_split[:, 1] + 0.1*array_split[:, 3] + 0.2*array_split[:, 4]) / w_r
        KDP_array[r] = np.concatenate(([KDP_array_r[0]], [KDP_array_r[0]], KDP_array_r, [KDP_array_r[-1]], [KDP_array_r[-1]]), axis=0)

    return Phi_dp_array, KDP_array


# ---reconstruct Phi_dp using Fuzzy Logic
def PhaseRec_fuzzy(reflectivity_array, zDr_array, Phi_dp_array, RhoHV_array, KDP_array, T_array):
    num_radial, num_gate = Phi_dp_array.shape
    num_sample = 11
    num_padding = int((num_sample-1)/2)

    for r in range(0, num_radial):
        zDr_std = np.std(rolling_window_1D(zDr_array[r], num_sample), 1)
        zDr_std = np.concatenate(([zDr_std[0] for i in range(0, num_padding)], zDr_std, [zDr_std[-1] for i in range(0, num_padding)]), axis=0)
        mf_zDr = memFunc(zDr_std, fuzzy_dict["clutter"]["zDr_std"]["m"], fuzzy_dict["clutter"]["zDr_std"]["al"], fuzzy_dict["clutter"]["zDr_std"]["bl"],
                         fuzzy_dict["clutter"]["zDr_std"]["ar"], fuzzy_dict["clutter"]["zDr_std"]["br"])
        phi_dp_std = np.std(rolling_window_1D(Phi_dp_array[r], num_sample), 1)
        phi_dp_std = np.concatenate(([phi_dp_std[0] for i in range(0, num_padding)], phi_dp_std, [phi_dp_std[-1] for i in range(0, num_padding)]), axis=0)
        mf_phi_dp = memFunc(phi_dp_std, fuzzy_dict["clutter"]["PhiDP_std"]["m"], fuzzy_dict["clutter"]["PhiDP_std"]["al"], fuzzy_dict["clutter"]["PhiDP_std"]["bl"],
                         fuzzy_dict["clutter"]["PhiDP_std"]["ar"], fuzzy_dict["clutter"]["PhiDP_std"]["br"])
        mf_RhoHV = memFunc(RhoHV_array[r], fuzzy_dict["clutter"]["RhoHV"]["m"], fuzzy_dict["clutter"]["RhoHV"]["al"], fuzzy_dict["clutter"]["RhoHV"]["bl"],
                            fuzzy_dict["clutter"]["RhoHV"]["ar"], fuzzy_dict["clutter"]["RhoHV"]["br"])
        s = (fuzzy_dict["clutter"]["zDr_std"]["w"] * mf_zDr + fuzzy_dict["clutter"]["PhiDP_std"]["w"] * mf_phi_dp
             + fuzzy_dict["clutter"]["RhoHV"]["w"] * mf_RhoHV) / (fuzzy_dict["clutter"]["zDr_std"]["w"] + fuzzy_dict["clutter"]["PhiDP_std"]["w"] + fuzzy_dict["clutter"]["RhoHV"]["w"])

        clutter_loc = np.nonzero(s >= 0.5)
        Phi_dp_array[r, clutter_loc] = -2.0
        KDP_array[r, clutter_loc] = -5.0

    return Phi_dp_array, KDP_array


# ---reconstruct Phi_dp followed the method proposed by DROPs
def dataMasking_DROPs(Phi_dp_array, GateWidth_array, rho_hv_array, num_good, num_bad, d_max, rho_max, population_min):
    Phi_dp_array[Phi_dp_array == -2.0] = np.nan
    # ------calculate the dispersion of PhiDP
    d_PhiDP_good = get_dispersion(rolling_window_1D(Phi_dp_array, num_good), axis=1)
    num_pad_good = int((num_good - 1) / 2)

    d_PhiDP_bad = get_dispersion(rolling_window_1D(Phi_dp_array, num_bad), axis=1)
    num_pad_bad = int((num_bad - 1) / 2)

    # ------As for selection rules, please ref to
    # Chen, Haonan, et al. “Urban Hydrological Applications of Dual-Polarization X-Band Radar: Case Study in Korea.”
    # Journal of Hydrologic Engineering, vol. 22, no. 5, 2017.
    id_start = get_good_point(d_PhiDP_good, d_max) + num_pad_good
    id_end = get_bad_point(d_PhiDP_bad, rho_hv_array[num_pad_bad:-num_pad_bad], d_max, rho_max) + num_pad_bad

    Phi_dp_array[np.isnan(Phi_dp_array)] = -2.0

    # ------rain cell segmentation
    labels = np.zeros_like(GateWidth_array)
    cell_id = 1

    if len(id_end) == 0:
        labels[id_start[0]:] = cell_id
        return labels
    else:
        counter = 0
        num_start = len(id_start)
        num_end = len(id_end)
        while counter < num_start:
            i_start = id_start[counter]
            # --------search the index of the end position of this rain cell
            counter = np.searchsorted(id_end, i_start, side="right")
            # -----------stop if end index exceeds
            if counter >= num_end:
                labels[i_start:] = cell_id
                break
            else:
                i_end = id_end[counter]
                labels[i_start:i_end + 1] = cell_id
                cell_id += 1
                counter = np.searchsorted(id_start, i_end, side="right")

        id_test = np.where(labels == 0)[0]
        check = id_test[1:] - id_test[:-1]
        discontinuity_set = np.where(check > 1)[0]

        if len(discontinuity_set) == 0:
            if labels.shape[0] - id_test[-1] <= population_min:
                labels[id_test[-1]:] = 0
            else:
                labels[id_test[-1]:] = 1
            return labels
        # --------if there are multiple discontinuous rain cells, let us refine them by joining and discarding
        else:
            # ----------combine two neighbouring rain cells
            id_start = id_test[discontinuity_set] + 1
            id_end = id_test[discontinuity_set + 1]

            if id_test[-1] < labels.shape[0] and labels.shape[0] - id_test[-1] > population_min:
                id_start = np.append(id_start, id_test[-1] + 1)
                id_end = np.append(id_end, labels.shape[0])

            cid = 1
            labels_refined = np.zeros_like(labels)
            for i in range(0, len(id_end)):
                # ----------only keep rain cells which contain enough data points
                if id_end[i] - id_start[i] + 1 > population_min:
                    labels_refined[id_start[i]:id_end[i]] = cid
                    cid += 1
            return labels_refined


def dataMasking_direct(Phi_dp_array, population_min):
    pass


def PhaseRec_DROPs(Phi_dp_array, GateWidth_array, rho_hv_array, KDP_array, num_good=15, num_bad=10, d_max=0.98, rho_max=0.9, population_min=5, masked=True):
    num_radial, num_gate = Phi_dp_array.shape
    num_sample = 15

    for r in range(0, num_radial):
        w_r = GateWidth_array[r]
        GateWidth_r = np.full(num_gate, w_r)
        GateWidth_r = np.cumsum(GateWidth_r)
        labels = dataMasking_DROPs(Phi_dp_array[r], GateWidth_r, rho_hv_array[r], num_good, num_bad, d_max, rho_max, population_min)
        labels = labels.astype(np.int64)

        # mask non-rain cell
        if masked:
            mask_r = (labels == 0)
            Phi_dp_array[r, mask_r] = -2.0
            KDP_array[r, mask_r] = -5.0

        # ------Do cubic spline fitting in every rain cell
        n_cell = len(np.bincount(labels)) - 1
        for cid in range(1, n_cell+1):
            cell_loc = np.nonzero(labels == cid)[0]
            h_vec = np.full(cell_loc.shape[0], w_r)  # len(h_vec) = M
            Phi_dp_i = np.cos(Phi_dp_array[r, cell_loc]/180.0*np.pi) + 1.0j * np.sin(Phi_dp_array[r, cell_loc]/180.0*np.pi)
            # ------First, we need to do non-adaptive fitting to obtain the overall trend of the Kdp proﬁle
            val_lambda = 0.1 * w_r
            # ------------prepare coefficient matrix M, Q
            p_vec = np.full(cell_loc.shape[0] - 2, 4 * w_r)
            M_mat = np.diag(p_vec) + np.diag(h_vec[1:-2], k=1) + np.diag(h_vec[1:-2], k=-1)
            l_vec = 3.0 / h_vec[:-1]     # len(l_vec) = M-1
            u_vec = -l_vec[1:] - l_vec[:-1]    # len(u_vec) = M-2
            Q_mat = np.diag(l_vec[:-1]) + np.diag(u_vec[:-1], k=1) + np.diag(l_vec[1:-2], k=2)  # size(Q_mat) = (M-2)*M
            Q_last_col = np.zeros((Q_mat.shape[0], 1))
            Q_mat = np.concatenate((Q_mat, Q_last_col, Q_last_col), axis=1)
            Q_mat[-1, -1] = l_vec[-1]
            Q_mat[-2, -2] = l_vec[-2]
            Q_mat[-1, -2] = u_vec[-1]
            b_ini = np.linalg.inv(M_mat + 2.0/3.0/val_lambda*np.dot(Q_mat, np.transpose(Q_mat)))
            b_ini = np.dot(np.dot(b_ini, Q_mat), Phi_dp_i)
            d_ini = Phi_dp_i - 2.0/3.0/val_lambda * np.dot(np.transpose(Q_mat), b_ini)
            b_ini = np.concatenate(([0], b_ini, [0]), axis=0)
            c_ini = (d_ini[1:] - d_ini[:-1]) / h_vec[:-1] - (b_ini[1:] + 2.0*b_ini[:-1]) * h_vec[:-1] / 3.0
            KDP_ini = 0.5 * np.imag(c_ini / d_ini[:-1])
            KDP_ini = np.concatenate((KDP_ini, [KDP_ini[-1]]), axis=0)
            # ------Then, we do adaptive cubic spline fitting in every rain cell
            val_lambda = 1.1 * w_r
            # ---------calculate the weighting matrix W for the precision of fitting
            num_pad_s = int((num_sample-1)/2)
            num_pad_e = num_sample - num_pad_s - 1
            cell_sample = np.linspace(cell_loc[0]-num_pad_s, cell_loc[-1]+num_pad_e, cell_loc.shape[0]+num_sample-1).astype(np.int64)
            cell_sample = np.where(cell_sample < 0, 0, cell_sample)
            cell_sample = np.where(cell_sample >= Phi_dp_array[r].shape[0], Phi_dp_array[r].shape[0]-1, cell_sample)
            W_inv_mat = get_invW(rolling_window_1D(Phi_dp_array[r, cell_sample], num_sample), 1)
            # ---------calculate the weighting matrix Mq for the smoothness of fitting
            KDP_ini[KDP_ini < 0.1] = 0.1
            w_q = 1.0 / (2.0 * KDP_ini)
            qh_vec = w_q[:-1] * h_vec[:-1]
            p_vec = 2 * (qh_vec[:-1] + qh_vec[1:])
            Mq_mat = np.diag(p_vec) + np.diag(qh_vec[1:-1], k=1) + np.diag(qh_vec[1:-1], k=-1)
            # ---------calculate the solution of cubic spline fitting
            # ------------coefficient matrix remains unchanged
            # ------------solve b = [b2, b3, ..., b_{M-1}]^T and d = [d1, d2, ..., d_M]^T
            cum_mat = np.dot(W_inv_mat, np.transpose(Q_mat))
            cum_mat = np.dot(cum_mat, np.linalg.inv(M_mat))
            cum_mat = np.dot(cum_mat, Mq_mat)
            b_vec = np.linalg.inv(M_mat + 2.0/3.0/val_lambda * np.dot(Q_mat, cum_mat))
            b_vec = np.dot(np.dot(b_vec, Q_mat), Phi_dp_i)   # len(b_vec) = M-2
            # ---------------Note that d corresponds to the smoothed angular Phi_dp
            d_vec = Phi_dp_i - 2.0/3.0/val_lambda * np.dot(cum_mat, b_vec)   # len(d_vec) = M
            Phi_dp_array[r, cell_loc] = complex2deg(d_vec)
            # ---------------b1, b_M are set to 0 because of the imposed natural condition
            b_vec = np.concatenate(([0], b_vec, [0]), axis=0)
            # ------------solve a = [a1, a2, ..., a_{M-1}]^T and c = [c1, c2, ..., c_{M-1}]^T
            c_vec = (d_vec[1:] - d_vec[:-1]) / h_vec[:-1] - (b_vec[1:] + 2.0*b_vec[:-1]) * h_vec[:-1] / 3.0
            KDP_array[r, cell_loc[:-1]] = 0.5 * np.imag(c_vec/d_vec[:-1])

    return Phi_dp_array, KDP_array


# ---reconstruct Phi_dp followed the hybrid method proposed by Hao Huang (2016)
def get_KDP_lower(KDP_DROPs, KDP_lower_est):
    KDP_lower = np.where(KDP_DROPs < 0.0, 0.5 * KDP_lower_est, KDP_DROPs)
    KDP_lower = np.where(KDP_DROPs >= KDP_lower_est, KDP_lower_est, KDP_lower)
    return KDP_lower


def get_KDP_higher(KDP_higher_est, ZH_array):
    KDP_higher = np.copy(KDP_higher_est)
    KDP_higher[np.logical_and(KDP_higher_est > 8.0, ZH_array < 35.0)] = 8.0
    KDP_higher[np.logical_and(KDP_higher_est > 10.0, ZH_array < 45.0)] = 10.0

    return KDP_higher


def PhaseRec_hybrid(Phi_dp_array, GateWidth_array, reflectivity_array, zDr_array, rho_hv_array, KDP_array, kdp_zh_zdr_para, num_good=10, num_bad=5, d_max=0.98, rho_max=0.9, population_min=5, masked=True):
    num_radial, num_gate = Phi_dp_array.shape
    # ------KDP = c * Zh^alpha * Zdr^beta
    c, alpha, beta = kdp_zh_zdr_para

    for r in range(0, num_radial):
        # ------rain cell segmentation
        w_r = GateWidth_array[r]
        GateWidth_r = np.full(num_gate, w_r)
        GateWidth_r = np.cumsum(GateWidth_r)
        labels = dataMasking_DROPs(Phi_dp_array[r], GateWidth_r, rho_hv_array[r], num_good, num_bad, d_max, rho_max, population_min)
        labels = labels.astype(np.int64)

        # Phi_dp_array[r] = np.where(np.isnan(Phi_dp_array[r]), -2.0, Phi_dp_array[r])

        # mask non-rain cell
        if masked:
            mask_r = (labels == 0)
            Phi_dp_array[r, mask_r] = -2.0
            KDP_array[r, mask_r] = -5.0

        # ------use Linear Programming with KDP constraints in every rain cell
        n_cell = len(np.bincount(labels)) - 1
        for cid in range(1, n_cell + 1):
            cell_loc = np.nonzero(labels == cid)[0]
            # ---------run non-adaptive fitting proposed by DROPs first to obtain the overall trend of the Kdp proﬁle
            h_vec = np.full(cell_loc.shape[0], w_r)
            Phi_dp_i = np.cos(Phi_dp_array[r, cell_loc]/180.0*np.pi) + 1.0j * np.sin(Phi_dp_array[r, cell_loc]/180.0*np.pi)
            val_lambda = 0.1 * w_r
            p_vec = np.full(cell_loc.shape[0] - 2, 4 * w_r)
            M_mat = np.diag(p_vec) + np.diag(h_vec[1:-2], k=1) + np.diag(h_vec[1:-2], k=-1)
            l_vec = 3.0 / h_vec[:-1]
            u_vec = -l_vec[1:] - l_vec[:-1]
            Q_mat = np.diag(l_vec[:-1]) + np.diag(u_vec[:-1], k=1) + np.diag(l_vec[1:-2], k=2)
            Q_last_col = np.zeros((Q_mat.shape[0], 1))
            Q_mat = np.concatenate((Q_mat, Q_last_col, Q_last_col), axis=1)
            Q_mat[-1, -1] = l_vec[-1]
            Q_mat[-2, -2] = l_vec[-2]
            Q_mat[-1, -2] = u_vec[-1]
            b_ini = np.linalg.inv(M_mat + 2.0 / 3.0 / val_lambda * np.dot(Q_mat, np.transpose(Q_mat)))
            b_ini = np.dot(np.dot(b_ini, Q_mat), Phi_dp_i)
            d_ini = Phi_dp_i - 2.0 / 3.0 / val_lambda * np.dot(np.transpose(Q_mat), b_ini)
            b_ini = np.concatenate(([0], b_ini, [0]), axis=0)
            c_ini = (d_ini[1:] - d_ini[:-1]) / h_vec[:-1] - (b_ini[1:] + 2.0 * b_ini[:-1]) * h_vec[:-1] / 3.0
            KDP_ini = 0.5 * np.imag(c_ini / d_ini[:-1])
            KDP_ini = np.concatenate((KDP_ini, [KDP_ini[-1]]), axis=0)
            # ---------determine the lower and higher bound of KDP using KDP_ini, KDP estimated from ZH and ZDR
            Zh_sample = np.power(10, reflectivity_array[r, cell_loc]/10.0)
            Zdr_sample = np.power(10, zDr_array[r, cell_loc]/10.0)
            KDP_est = c * np.power(Zh_sample, alpha) * np.power(Zdr_sample, beta)
            KDP_lower_est = (1 - 0.25) * KDP_est
            KDP_higher_est = (1 + 0.25) * KDP_est
            KDP_lower = get_KDP_lower(KDP_ini, KDP_lower_est)
            KDP_higher = get_KDP_higher(KDP_higher_est, reflectivity_array[r, cell_loc])
            Phi_dp_array[r, cell_loc] = LP_solver_constraints(Phi_dp_array[r, cell_loc], w_r, KDP_lower, KDP_higher)
            array_split = rolling_window_1D(Phi_dp_array[r, cell_loc], 5)
            KDP_array_r = (-0.2 * array_split[:, 0] - 0.1 * array_split[:, 1] + 0.1 * array_split[:, 3] + 0.2 * array_split[:, 4]) / w_r
            KDP_array[r, cell_loc] = np.concatenate(([KDP_array_r[0]], [KDP_array_r[0]], KDP_array_r, [KDP_array_r[-1]], [KDP_array_r[-1]]), axis=0)

    return Phi_dp_array, KDP_array


def PhaseRec_hybrid_mask(Phi_dp_array, GateWidth_array, reflectivity_array, zDr_array, rho_hv_array, KDP_array, kdp_zh_zdr_para, num_good=10, num_bad=5, d_max=0.98, rho_max=0.9, population_min=5, masked=True):
    num_radial, num_gate = Phi_dp_array.shape
    # ------KDP = c * Zh^alpha * Zdr^beta
    c, alpha, beta = kdp_zh_zdr_para

    for r in range(0, num_radial):
        # ------rain cell segmentation
        w_r = GateWidth_array[r]
        GateWidth_r = np.full(num_gate, w_r)
        GateWidth_r = np.cumsum(GateWidth_r)
        labels = dataMasking_DROPs(Phi_dp_array[r], GateWidth_r, rho_hv_array[r], num_good, num_bad, d_max, rho_max, population_min)
        labels = labels.astype(np.int64)

        weights = np.ones_like(Phi_dp_array[r])
        KDP_higher = np.full_like(Phi_dp_array[r], -5.0)
        KDP_lower = np.full_like(Phi_dp_array[r], -5.0)

        # mask non-rain cell
        if masked:
            mask_r = (labels == 0)
            weights[mask_r] = 0.0
            Phi_dp_array[r, mask_r] = -2.0
            KDP_array[r, mask_r] = -5.0

        # ------use Linear Programming with KDP constraints in every rain cell
        n_cell = len(np.bincount(labels)) - 1
        for cid in range(1, n_cell + 1):
            cell_loc = np.nonzero(labels == cid)[0]
            # ---------run non-adaptive fitting proposed by DROPs first to obtain the overall trend of the Kdp proﬁle
            h_vec = np.full(cell_loc.shape[0], w_r)
            Phi_dp_i = np.cos(Phi_dp_array[r, cell_loc]/180.0*np.pi) + 1.0j * np.sin(Phi_dp_array[r, cell_loc]/180.0*np.pi)
            val_lambda = 0.1 * w_r
            p_vec = np.full(cell_loc.shape[0] - 2, 4 * w_r)
            M_mat = np.diag(p_vec) + np.diag(h_vec[1:-2], k=1) + np.diag(h_vec[1:-2], k=-1)
            l_vec = 3.0 / h_vec[:-1]
            u_vec = -l_vec[1:] - l_vec[:-1]
            Q_mat = np.diag(l_vec[:-1]) + np.diag(u_vec[:-1], k=1) + np.diag(l_vec[1:-2], k=2)
            Q_last_col = np.zeros((Q_mat.shape[0], 1))
            Q_mat = np.concatenate((Q_mat, Q_last_col, Q_last_col), axis=1)
            Q_mat[-1, -1] = l_vec[-1]
            Q_mat[-2, -2] = l_vec[-2]
            Q_mat[-1, -2] = u_vec[-1]
            b_ini = np.linalg.inv(M_mat + 2.0 / 3.0 / val_lambda * np.dot(Q_mat, np.transpose(Q_mat)))
            b_ini = np.dot(np.dot(b_ini, Q_mat), Phi_dp_i)
            d_ini = Phi_dp_i - 2.0 / 3.0 / val_lambda * np.dot(np.transpose(Q_mat), b_ini)
            b_ini = np.concatenate(([0], b_ini, [0]), axis=0)
            c_ini = (d_ini[1:] - d_ini[:-1]) / h_vec[:-1] - (b_ini[1:] + 2.0 * b_ini[:-1]) * h_vec[:-1] / 3.0
            KDP_ini = 0.5 * np.imag(c_ini / d_ini[:-1])
            KDP_ini = np.concatenate((KDP_ini, [KDP_ini[-1]]), axis=0)
            # ---------determine the lower and higher bound of KDP using KDP_ini, KDP estimated from ZH and ZDR
            Zh_sample = np.power(10, reflectivity_array[r, cell_loc]/10.0)
            Zdr_sample = np.power(10, zDr_array[r, cell_loc]/10.0)
            KDP_est = c * np.power(Zh_sample, alpha) * np.power(Zdr_sample, beta)
            KDP_lower_est = (1 - 0.25) * KDP_est
            KDP_higher_est = (1 + 0.25) * KDP_est
            KDP_lower[cell_loc] = get_KDP_lower(KDP_ini, KDP_lower_est)
            KDP_higher[cell_loc] = get_KDP_higher(KDP_higher_est, reflectivity_array[r, cell_loc])

        Phi_dp_array[r] = LP_solver_constraints(Phi_dp_array[r], w_r, KDP_lower, KDP_higher, weights)
        array_split = rolling_window_1D(Phi_dp_array[r], 5)
        KDP_array_r = (-0.2*array_split[:, 0] - 0.1*array_split[:, 1] + 0.1*array_split[:, 3] + 0.2*array_split[:, 4]) / w_r
        KDP_array[r] = np.concatenate(([KDP_array_r[0]], [KDP_array_r[0]], KDP_array_r, [KDP_array_r[-1]], [KDP_array_r[-1]]), axis=0)

    return Phi_dp_array, KDP_array


def LP_solver_constraints(Phi_dp_array, width, KDP_lower_bound, KDP_higher_bound, weights=None):
    if weights is None:
        weights = np.ones_like(Phi_dp_array)

    num_gate = len(Phi_dp_array)

    # ------define the Problem object
    lp_prob = pl.LpProblem("Reconstruction_SC", pl.LpMinimize)
    # ------define the decision variable
    z = ["z" + str(i) for i in range(0, num_gate)]
    x = ["x" + str(i) for i in range(0, num_gate)]
    var = z + x
    lp_var = pl.LpVariable.dicts("var", var, cat='Continuous')

    # ------define the objective function
    gate_used = np.where(weights != 0.0)[0]
    lp_prob += pl.lpSum([lp_var[var[i]] for i in gate_used])

    for i in range(0, num_gate):
        if weights[i] != 0:
            # ------add the fidelity constraints
            lp_prob += weights[i] * (lp_var[var[i]] - lp_var[var[i + num_gate]]) >= -Phi_dp_array[i]
            lp_prob += weights[i] * (lp_var[var[i]] + lp_var[var[i + num_gate]]) >= Phi_dp_array[i]

    for i in range(num_gate + 2, 2*num_gate - 2):
        w_i = weights[i - num_gate]
        if w_i != 0:
            # ------add the montonicity constraints via five-point Savitzky–Golay derivative filter
            lp_prob += w_i * (-0.2*lp_var[var[i-2]] + (-0.1)*lp_var[var[i-1]] + 0.1*lp_var[var[i+1]] + 0.2*lp_var[var[i+2]]) >= 0.0
            # ------add the self-consistency constraints via five-point Savitzky–Golay derivative filter
            lp_prob += w_i * (-0.2*lp_var[var[i-2]] + (-0.1)*lp_var[var[i-1]] + 0.1*lp_var[var[i+1]] + 0.2*lp_var[var[i+2]]) / width >= KDP_lower_bound[i-num_gate]
            lp_prob += w_i * (-0.2*lp_var[var[i-2]] + (-0.1)*lp_var[var[i-1]] + 0.1*lp_var[var[i+1]] + 0.2*lp_var[var[i+2]]) / width <= KDP_higher_bound[i-num_gate]

    # ------solve the LP problem
    # ---------use GNU Linear Programming Kit solver
    # lp_prob.solve(pl.GLPK())
    # ---------use default CBC solver
    lp_prob.solve()
    # ------check the status of solution
    if lp_prob.status == 1:
        phi_rec = np.array([lp_var[var[i]].varValue for i in range(num_gate, 2*num_gate)])
        phi_rec[np.equal(phi_rec, None)] = np.nan
        # ------smoothing filter to prevent subﬁlter-length oscillations
        Phi_dp_array[2:-2] = 0.1*phi_rec[0:-4] + 0.25*phi_rec[1:-3] + 0.3*phi_rec[2:-2] + 0.25*phi_rec[3:-1] + 0.1*phi_rec[4:]
        Phi_dp_array[-2:] = Phi_dp_array[-3]
    else:
        print("Linear Programming failed !!!")

    return Phi_dp_array


# ************* [2] Attenuation correction *************
# ---attenuation correction by Z-PHI method
def get_para_a(ZH_integral, PIA_start, PIA_end, b=0.7339):
    # ------Ref to
    # Testud, Jacques, et al. “The Rain Profiling Algorithm Applied to Polarimetric Weather Radar.”
    # Journal of Atmospheric and Oceanic Technology, vol. 17, no. 3, 2000, pp. 332–356.
    return (np.exp(-0.23*b*PIA_start) - np.exp(-0.23*b*PIA_end)) / 0.46 / b / ZH_integral


def phase_loss(c_coef, Zh, Phi_dp, GateWidth, num_sample, b=0.7339):
    PIA = c_coef * (Phi_dp - Phi_dp[0])
    ZH_integral = np.array([integrate.simps(y=np.power(Zh[0:i+1], b), x=GateWidth[0:i+1]) for i in range(0, num_sample)])
    a = get_para_a(ZH_integral[-1], 0.0, PIA[-1], b=b)
    attenuation = a * np.power(Zh, b) / (1.0 - 0.46 * a * b * ZH_integral)
    attenuation_int = attenuation / c_coef
    Phi_dp_rec = np.array([integrate.simps(y=attenuation_int[0:i+1], x=GateWidth[0:i+1]) for i in range(0, num_sample)])
    return np.mean(np.abs(Phi_dp[0] + Phi_dp_rec - Phi_dp))


def correct_ZPHI(ZH_array, zDr_array, Phi_dp_array, GateWidth_array, r_start, c_min=0.01, c_max=0.25, b_coef=0.7339):
    num_radial, num_gate = zDr_array.shape

    step = 0.01
    c_search = np.linspace(c_min, c_max, int((c_max - c_min) / step)+1)

    # ------set the value of ZH in the noData region to 0 for further integral computation
    Phi_dp_noData = -2.0
    ZH_array = np.where(Phi_dp_array == Phi_dp_noData, 0.0, ZH_array)

    for r in range(0, num_radial):
        # ------determine the end point of attenuation correction
        # ---------which is the end point of the last rain cell
        noData_loc = np.where(Phi_dp_array[r] < 0.0)[0]
        check = noData_loc[1:] - noData_loc[:-1]
        discontinuity_set = np.where(check > 1)[0]
        if len(discontinuity_set) == 0:
            if len(check) == num_gate - 1:
                continue
            else:
                id_end = noData_loc[len(check)]
        else:
            id_end = noData_loc[discontinuity_set[-1]+1]
        Zh_i = np.power(10.0, ZH_array[r, r_start:id_end]/10.0)
        Phi_dp_i = Phi_dp_array[r, r_start:id_end]
        num_correct = len(Zh_i)
        w_r = GateWidth_array[r]
        GateWidth_r = np.full(num_correct, w_r)
        GateWidth_r = np.cumsum(GateWidth_r)
        # ------find the optimal c by minimizing cost function
        phase_loss_list = [phase_loss(c_i, Zh_i, Phi_dp_i, GateWidth_r, num_correct, b_coef) for c_i in c_search]
        c_coef = c_search[np.argmin(phase_loss_list)]
        '''
        # ------ use some optimization algorithms from SciPy (not recommended)
        opt_res = opt.minimize(fun=phase_loss, x0=c_max, args=(Zh_i, Phi_dp_i, GateWidth_r, num_correct, b_coef),
                               method="SLSQP", tol=1e-4, bounds=[(c_min, c_max)])
        c_coef = opt_res["x"]
        '''
        # ------calculate Path-Integrated Attenuation (PIA)
        PIA = c_coef * (Phi_dp_i - Phi_dp_i[0])
        # ------calculate coefficient a
        ZH_integral = np.array([integrate.simps(y=np.power(Zh_i[0:i+1]/10.0, b_coef), x=GateWidth_r[0:i+1]) for i in range(0, num_correct)])
        a_coef = get_para_a(ZH_integral[-1], 0.0, PIA[-1], b=b_coef)
        Zh_i_corrected = Zh_i / np.power(1.0 - 0.46 * a_coef * b_coef * ZH_integral, 1.0/b_coef)
        ZH_array[r, r_start:id_end] = 10.0 * np.log10(Zh_i_corrected)
        print(r, c_coef)

    return ZH_array


if __name__ == "__main__":
    raw_dir = "Input"
    raw_fname = "BJXFS_20190909_180000_5_1000.nc"
    nc_ds = nc.Dataset(os.path.join(raw_dir, raw_fname), "r")

    # convert mm to km
    GateWidth = np.array(nc_ds.variables["GateWidth"]) / 1000.0 / 1000.0
    zDr = np.array(nc_ds.variables["DifferentialReflectivity"])
    Phi_dp = np.array(nc_ds.variables["DifferentialPhase"])
    rho_hv = np.array(nc_ds.variables["CrossPolCorrelation"])
    KDP = np.array(nc_ds.variables["KDP"])
    reflectivity = np.array(nc_ds.variables["Reflectivity"])
    # reflectivity = np.where(reflectivity==0.0, -33.0, reflectivity)

    temperature = np.zeros_like(reflectivity)

    mask_Phi_dp = SpatialTexture_masking(Phi_dp, num_range=2, threshold=5.0, noData=-2.0)
    mask_zDr = SpatialTexture_masking(zDr, num_range=2, threshold=1.0, noData=-8.125)
    mask = np.logical_and(np.logical_and(mask_Phi_dp, mask_zDr), rho_hv < 0.9)
    Phi_dp[mask] = -2.0

    mask = ConnectedComponent_masking(Phi_dp, noData=-2.0, population_min=30)
    Phi_dp[mask] = -2.0

    reflectivity[Phi_dp == -2.0] = -33.0
    KDP[Phi_dp == -2.0] = -5.0
    rho_hv[Phi_dp == -2.0] = -0.025
    zDr[Phi_dp == -2.0] = -0.625

    # Phi_dp_unfold = PhaseUnfolding(Phi_dp, rho_hv, GateWidth, max_phaseDiff=-180, dphase=360)
    # Phi_dp_rec, Kdp_rec = PhaseRec_LP(Phi_dp, KDP, rho_hv, GateWidth)
    # Phi_dp_rec, Kdp_rec = PhaseRec_LP_mask(Phi_dp, KDP, rho_hv, GateWidth)
    # Phi_dp_rec, Kdp_rec = PhaseRec_DROPs(Phi_dp, GateWidth, rho_hv, KDP, d_max=0.97, rho_max=0.7, population_min=10)
    # Phi_dp_rec, Kdp_rec = PhaseRec_hybrid(Phi_dp, GateWidth, reflectivity, zDr, rho_hv, KDP, [7.1827e-5, 1.1339, -3.3701], d_max=0.97, rho_max=0.7, population_min=5)
    Phi_dp_rec, Kdp_rec = PhaseRec_hybrid_mask(Phi_dp, GateWidth, reflectivity, zDr, rho_hv, KDP, [7.1827e-5, 1.1339, -3.3701], d_max=0.97, rho_max=0.7, population_min=5)
    # Phi_dp_rec, Kdp_rec = PhaseRec_fuzzy(reflectivity, zDr, Phi_dp_rec, rho_hv, Kdp_rec, temperature)

    #id_test = np.logical_and(Phi_dp_rec <= 0.0, Phi_dp_rec != -2.0)
    #Phi_dp_rec[id_test] = 0.0

    #np.save("Output/Phidp_hybrid.npy", Phi_dp_rec)
    #np.save("Output/Kdp_hybrid.npy", Kdp_rec)

    '''
    Phi_dp_rec = np.load("Output/Phidp_hybrid.npy")
    Kdp_rec = np.load("Output/Kdp_hybrid.npy")

    ZH_correct = correct_ZPHI(reflectivity, zDr, Phi_dp_rec, GateWidth, 12, c_min=0.01, c_max=0.25, b_coef=0.7339)
    ZH_correct = np.where(ZH_correct==0.0, -33.0, ZH_correct)
    '''

    num_radial, num_gate = Phi_dp.shape
    GateWidth_cum = np.full(num_gate, GateWidth[0])
    GateWidth_cum = np.cumsum(GateWidth_cum)
    ppi_vis(Phi_dp_rec, "Output_img/Phi_dp_SD+CC_LP_2d5+hybrid.png", "DifferentialPhase", range=GateWidth_cum,
            title="SD+CC+Hybrid-processed $\Phi_{dp}$ at 2019/09/09 18:00:00", colorbar_label="$\Phi_{dp}$ [Degrees]")
