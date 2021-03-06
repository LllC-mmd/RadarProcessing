import os
import datetime
import bz2
import numpy as np

import netCDF4 as nc
import wradlib.ipol as ipol

from functools import reduce
import imageio
import matplotlib.pyplot as plt
from radarVis import *


def MetSTARDataReader(dta_path):
    variable_encode = {1: "dbt", 2: "dbz", 7: "zdr", 9: "cc", 10: "phidp", 11: "kdp"}

    # [1] Read raw data produced by Beijing METSTAR radar Co.,LTD.
    fid = open(dta_path, "rb")
    # Shift value for file pointers of different block
    # ---generic header block pointer (len=32)
    pt_common_block = 0
    # ---site configuration information block pointer (len=128)
    pt_site_block = 0 + 32
    # ---task configuration information block pointer (len=256)
    pt_task_block = 0 + 32 + 128
    # ---1st angle of elevation block pointer
    # ------The shift value of N-th angle of elevation is: 416 + (N-1) * 256
    pt_1st_ele = 0 + 32 + 128 + 256

    siteinfo = {}
    taskinfo = {}
    eleinfo = {}
    radinfo = {}

    f = {'radinfo': {}, 'dbt': {}, 'dbz': {}, 'zdr': {}, 'cc': {}, 'phidp': {}, 'kdp': {}}

    # ---read site configuration information
    fid.seek(pt_site_block, 0)
    # ------site code
    siteinfo['code'] = ''.join([chr(item) for item in np.fromfile(file=fid, dtype=np.int8, count=8)])
    # ------site name
    siteinfo['name'] = ''.join([chr(item) for item in np.fromfile(fid, np.int8, 32)])
    # ------latitude
    siteinfo['lat'] = np.fromfile(fid, np.float32, 1)[0]
    # ------longitude
    siteinfo['lon'] = np.fromfile(fid, np.float32, 1)[0]
    # ------antenna height
    siteinfo['atennaasl'] = np.fromfile(fid, np.int32, 1)[0]
    # ------ground height
    siteinfo['baseasl'] = np.fromfile(fid, np.int32, 1)[0]
    # ------frequency
    siteinfo['freq'] = np.fromfile(fid, np.float32, 1)[0]
    # ------beam width (Horizontal)
    siteinfo['beamhwidth'] = np.fromfile(fid, np.float32, 1)[0]
    # ------beam width (Vertical)
    siteinfo['beamvwidth'] = np.fromfile(fid, np.float32, 1)[0]

    # ---read task configuration information
    fid.seek(pt_task_block, 0)
    # ------task name
    taskinfo['name'] = ''.join([chr(item) for item in np.fromfile(fid, np.int8, 32)])
    # ------task description
    taskinfo['description'] = ''.join([chr(item) for item in np.fromfile(fid, np.int8, 128)])
    # ------polarization type
    taskinfo['polmode'] = np.fromfile(fid, np.int32, 1)[0]
    # ------scan type: 1 for Plan Position Indicator (PPI), 2 for Range Height Indicator (RHI)
    taskinfo['scantype'] = np.fromfile(fid, np.int32, 1)[0]
    # ------pulse width
    taskinfo['pulsewidth'] = np.fromfile(fid, np.int32, 1)[0]
    # ------!!! scan start time (UTC, start from 1970/01/01 00:00)
    taskinfo['startime'] = np.fromfile(fid, np.int32, 1)[0]
    # ------!!! cut number
    taskinfo['cutnum'] = np.fromfile(fid, np.int32, 1)[0]
    # ------horizontal noise
    taskinfo['hnoise'] = np.fromfile(fid, np.float32, 1)[0]
    # ------vertical noise
    taskinfo['vnoise'] = np.fromfile(fid, np.float32, 1)[0]
    # ------horizontal calibration
    taskinfo['hsyscal'] = np.fromfile(fid, np.float32, 1)[0]
    # ------vertical calibration
    taskinfo['vsyscal'] = np.fromfile(fid, np.float32, 1)[0]
    # ------horizontal noise temperature
    taskinfo['hte'] = np.fromfile(fid, np.float32, 1)[0]
    # ------vertical noise temperature
    taskinfo['vte'] = np.fromfile(fid, np.float32, 1)[0]
    # ------ZDR calibration
    taskinfo['zdrbias'] = np.fromfile(fid, np.float32, 1)[0]
    # ------PhiDP calibration
    taskinfo['phasebias'] = np.fromfile(fid, np.float32, 1)[0]
    # ------LDR calibration
    taskinfo['ldrbias'] = np.fromfile(fid, np.float32, 1)[0]

    # ---read angle of elevation block
    for ct in range(1, int(taskinfo['cutnum']) + 1):
        fid.seek(pt_1st_ele + (ct - 1) * 256, 0)
        info_dict = {}
        # ------process mode
        info_dict['mode'] = np.fromfile(fid, np.int32, 1)[0]
        # ------wave form
        info_dict['waveform'] = np.fromfile(fid, np.int32, 1)[0]
        # ------pulse repetition frequency 1 (PRF1)
        info_dict['prf1'] = np.fromfile(fid, np.float32, 1)[0]
        # ------pulse repetition frequency 2 (PRF2)
        info_dict['prf2'] = np.fromfile(fid, np.float32, 1)[0]
        # ------de-aliasing mode
        info_dict['unfoldmode'] = np.fromfile(fid, np.int32, 1)[0]
        # ------azimuth for RHI mode
        info_dict['azi'] = np.fromfile(fid, np.float32, 1)[0]
        # ------elevation for PPI mode
        info_dict['ele'] = np.fromfile(fid, np.float32, 1)[0]
        # ------start angle, i.e., start azimuth for PPI mode or highest elevation for RHI mode
        info_dict['startangle'] = np.fromfile(fid, np.float32, 1)[0]
        # ------end angle, i.e., end azimuth for PPI mode or lowest elevation for RHI mode
        info_dict['endangle'] = np.fromfile(fid, np.float32, 1)[0]
        # ------angular resolution (only for PPI mode)
        info_dict['angleres'] = np.fromfile(fid, np.float32, 1)[0]
        # ------scan speed
        info_dict['scanspeed'] = np.fromfile(fid, np.float32, 1)[0]
        # ------log resolution
        info_dict['logres'] = np.fromfile(fid, np.int32, 1)[0]
        # ------Doppler Resolution
        info_dict['dopres'] = np.fromfile(fid, np.int32, 1)[0]
        # ------Maximum Range corresponding to PRF1
        info_dict['maxrange1'] = np.fromfile(fid, np.int32, 1)[0]
        # ------Maximum Range corresponding to PRF2
        info_dict['maxrange2'] = np.fromfile(fid, np.int32, 1)[0]
        # ------start range
        info_dict['startrange'] = np.fromfile(fid, np.int32, 1)[0]
        # ------number of samples corresponding to PRF1
        info_dict['samplenum1'] = np.fromfile(fid, np.int32, 1)[0]
        # ------number of samples corresponding to PRF2
        info_dict['samplenum2'] = np.fromfile(fid, np.int32, 1)[0]
        # ------phase mode
        info_dict['phasemode'] = np.fromfile(fid, np.int32, 1)[0]
        # ------atmosphere loss
        info_dict['atmosloss'] = np.fromfile(fid, np.float32, 1)[0]
        # ------Nyquist speed
        info_dict['vmax'] = np.fromfile(fid, np.float32, 1)[0]
        # ------moments mask
        info_dict['mask'] = np.fromfile(fid, np.int32, 1)[0]
        # ------moments size mask
        info_dict['masksize'] = np.fromfile(fid, np.float32, 1)[0]
        info_dict['datasizemask'] = [mas for mas in np.fromfile(fid, np.float64, 64)]
        # ------misc filter mask
        info_dict['filtermask'] = np.fromfile(fid, np.int32, 1)[0]
        # ------SQI threshold
        info_dict['sqi'] = np.fromfile(fid, np.float32, 1)[0]
        # ------SIG threshold
        info_dict['sig'] = np.fromfile(fid, np.float32, 1)[0]
        # ------CSR threshold
        info_dict['csr'] = np.fromfile(fid, np.float32, 1)[0]
        # ------LOG threshold
        info_dict['log'] = np.fromfile(fid, np.float32, 1)[0]
        # ------CPA threshold
        info_dict['cpa'] = np.fromfile(fid, np.float32, 1)[0]
        # ------PMI threshold
        info_dict['pmi'] = np.fromfile(fid, np.float32, 1)[0]
        # ------reserved threshold
        info_dict['threshold'] = np.fromfile(fid, np.int8, 8)[0]
        # ------dBT threshold
        info_dict['dbtmask'] = np.fromfile(fid, np.int32, 1)[0]
        # ------dBZ mask
        info_dict['dbzmask'] = np.fromfile(fid, np.int32, 1)[0]
        # ------velocity mask
        info_dict['vmask'] = np.fromfile(fid, np.int32, 1)[0]
        # ------spectrum width mask
        info_dict['wmask'] = np.fromfile(fid, np.int32, 1)[0]
        # ------DP mask
        info_dict['zdrmask'] = np.fromfile(fid, np.int32, 1)[0]
        # ------mask reserved
        info_dict['maskreserved'] = np.fromfile(fid, np.int8, 12)
        # ------flag for scan synchronization
        info_dict['scansync'] = np.fromfile(fid, np.int32, 1)[0]
        # ------scan direction
        info_dict['scandirection'] = np.fromfile(fid, np.int32, 1)[0]
        # ------ground clutter classifier type
        info_dict['cmap'] = np.fromfile(fid, np.int16, 1)[0]
        # ------ground clutter filter type
        info_dict['cfiltertype'] = np.fromfile(fid, np.int16, 1)[0]
        # ------ground clutter filter notch width
        info_dict['cnotchwidth'] = np.fromfile(fid, np.int16, 1)[0]
        # ------ground clutter filter window
        info_dict['cfilterwin'] = np.fromfile(fid, np.int16, 1)[0]
        # ------reserved
        info_dict['twin'] = np.fromfile(fid, np.int8, 1)[0]
        eleinfo[str(ct)] = info_dict

    # ---read radial data block
    pt_radial_block = pt_1st_ele + int(taskinfo['cutnum']) * 256
    fid.seek(pt_radial_block, 0)
    a = 0
    # ------we have 12 angle of elevation
    for i in range(1, 12):
        radinfo[str(i)] = {}
        # ------radial state
        radinfo[str(i)]['state'] = []
        # ------spot blank
        radinfo[str(i)]['spotblank'] = []
        # ------sequence number
        radinfo[str(i)]['seqnum'] = []
        # ------radial number
        radinfo[str(i)]['curnum'] = []
        # ------elevation number
        radinfo[str(i)]['elenum'] = []
        # ------azimuth
        radinfo[str(i)]['azi'] = []
        # ------elevation
        radinfo[str(i)]['ele'] = []
        # ------seconds
        radinfo[str(i)]['sec'] = []
        # ------microseconds
        radinfo[str(i)]['micro'] = []
        # ------length of data
        radinfo[str(i)]['datalen'] = []
        # ------moment number
        radinfo[str(i)]['momnum'] = []
        radinfo[str(i)]['reserved'] = []

    for i in range(1, 12):
        f['dbt'][i] = {}
        f['dbz'][i] = {}
        f['zdr'][i] = {}
        f['cc'][i] = {}
        f['phidp'][i] = {}
        f['kdp'][i] = {}

    while True:
        # ------radial state
        state = np.fromfile(fid, np.int32, 1)[0]
        # ------spot blank
        spotblank = np.fromfile(fid, np.int32, 1)[0]
        # ------sequence number
        seqnum = np.fromfile(fid, np.int32, 1)[0]
        # ------radial number
        curnum = np.fromfile(fid, np.int32, 1)[0]
        # ------elevation number
        elenum = np.fromfile(fid, np.int32, 1)[0]

        radinfo[str(int(elenum))]['state'].append(state)
        radinfo[str(int(elenum))]['spotblank'].append(spotblank)
        radinfo[str(int(elenum))]['seqnum'].append(seqnum)
        radinfo[str(int(elenum))]['curnum'].append(curnum)
        radinfo[str(int(elenum))]['elenum'].append(elenum)

        # ------azimuth
        radinfo[str(int(elenum))]['azi'].append(np.fromfile(fid, np.float32, 1)[0])
        # ------elevation
        radinfo[str(int(elenum))]['ele'].append(np.fromfile(fid, np.float32, 1)[0])
        # ------seconds
        radinfo[str(int(elenum))]['sec'].append(np.fromfile(fid, np.int32, 1)[0])
        # ------microseconds
        radinfo[str(int(elenum))]['micro'].append(np.fromfile(fid, np.int32, 1)[0])
        # ------length of data
        datalen = np.fromfile(fid, np.int32, 1)[0]
        radinfo[str(int(elenum))]['datalen'].append(datalen)
        # ------moment number
        num_moment = np.fromfile(fid, np.int32, 1)[0]
        radinfo[str(int(elenum))]['momnum'].append(num_moment)
        radinfo[str(int(elenum))]['reserved'].append(np.fromfile(fid, np.int8, 20))

        radcnt = curnum
        b = 0
        for n in range(0, num_moment):
            # ------data type
            var_type = np.fromfile(fid, np.int32, 1)[0]
            # ------scale
            scale = np.fromfile(fid, np.int32, 1)[0]
            # ------offset
            offset = np.fromfile(fid, np.int32, 1)[0]
            # ------!!! bin length: bytes for one bin storage
            binbytenum = np.fromfile(fid, np.int16, 1)[0]
            # ------flags
            flag = np.fromfile(fid, np.int16, 1)[0]
            # ------length
            bin_length = np.fromfile(fid, np.int32, 1)[0]
            reserved = np.fromfile(fid, np.int8, 12)

            bin_num = bin_length / binbytenum

            if binbytenum == 1:
                data_raw = np.fromfile(fid, np.uint8, int(bin_num))
            else:
                data_raw = np.fromfile(fid, np.uint16, int(bin_num))

            b += 1

            if elenum != 2 and elenum != 4:
                if var_type in variable_encode.keys():
                    # Note: When operated between unsigned int8 and int32, numpy might give some mis-conversion
                    f[variable_encode[var_type]][int(elenum)][str(int(radcnt))] = (1.0 * data_raw - offset) / scale

        a += 1
        if state == 6 or state == 4:
            break

    return siteinfo, taskinfo, eleinfo, radinfo, f


def DROPsNetCDFGen(nc_name, siteinfo, taskinfo, eleinfo, radinfo, radar_data, num_gate=1000, elev_id=3):
    nc_ds = nc.Dataset(nc_name, mode="w")
    elev_id = str(elev_id)

    nc_ds.NetCDFRevision = "lyy_data"
    nc_ds.GenDate = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    nc_ds.RadarName = siteinfo['name']
    nc_ds.Latitude = siteinfo['lat']
    nc_ds.Longitude = siteinfo['lon']
    nc_ds.Height = siteinfo['atennaasl']
    nc_ds.NumGates = num_gate
    nc_ds.ScanId = 3
    nc_ds.ScanFlag = 1
    nc_ds.ScanType = taskinfo['scantype']
    nc_ds.AntennaGain = 44.7
    nc_ds.AntennaBeamwidth = 0.94

    num_rad = len(radinfo[elev_id]['azi'])
    RadDimId = nc_ds.createDimension("Radial", num_rad)
    GatDimId = nc_ds.createDimension("Gate", num_gate)

    az_id = nc_ds.createVariable("Azimuth", np.float64, ("Radial",))
    az_id.Units = "Degrees"
    az_id[:] = radinfo[elev_id]['azi']
    el_id = nc_ds.createVariable("Elevation", np.float64, ("Radial",))
    el_id.Units = "Degrees"
    el_id[:] = radinfo[elev_id]['ele']
    gw_id = nc_ds.createVariable("GateWidth", np.float64, ("Radial",))
    gw_id.Units = "Millimeters"
    gw_id[:] = np.full(num_rad, eleinfo[elev_id]['logres'] * 1000)
    str_id = nc_ds.createVariable("StartRange", np.float64, ("Radial",))
    str_id.Units = "Millimeters"
    str_id[:] = np.full(num_rad, eleinfo[elev_id]['startrange'] * 1000000)
    t_id = nc_ds.createVariable("Time", np.int64, ("Radial",))
    t_id.Units = "Seconds"
    t_id[:] = radinfo[elev_id]['sec']
    tf_id = nc_ds.createVariable("TxFrequency", np.float64, ("Radial",))
    tf_id.Units = "Hertz"
    tf_id[:] = np.full(num_rad, siteinfo['freq'] * 1000000)

    zh_id = nc_ds.createVariable("Reflectivity", np.float32, ("Radial", "Gate"))
    zh_id.Units = "dBz"
    zh_id[:, :] = np.array(
        [v[0:num_gate] for k, v in sorted(radar_data['dbz'][int(elev_id)].items(), key=lambda item: int(item[0]))])
    zd_id = nc_ds.createVariable("DifferentialReflectivity", np.float32, ("Radial", "Gate"))
    zd_id.Units = "dB"
    zd_id[:, :] = np.array(
        [v[0:num_gate] for k, v in sorted(radar_data['zdr'][int(elev_id)].items(), key=lambda item: int(item[0]))])
    phiDP_id = nc_ds.createVariable("DifferentialPhase", np.float32, ("Radial", "Gate"))
    phiDP_id.Units = "Degrees"
    phiDP_id[:, :] = np.array(
        [v[0:num_gate] for k, v in sorted(radar_data['phidp'][int(elev_id)].items(), key=lambda item: int(item[0]))])
    rhoHV_id = nc_ds.createVariable("CrossPolCorrelation", np.float32, ("Radial", "Gate"))
    rhoHV_id.Units = "Unitless"
    rhoHV_id[:, :] = np.array(
        [v[0:num_gate] for k, v in sorted(radar_data['cc'][int(elev_id)].items(), key=lambda item: int(item[0]))])
    snr_id = nc_ds.createVariable("KDP", np.float32, ("Radial", "Gate"))
    snr_id.Units = "Unitless"
    snr_id[:, :] = np.array(
        [v[0:num_gate] for k, v in sorted(radar_data['kdp'][int(elev_id)].items(), key=lambda item: int(item[0]))])

    nc_ds.close()


def getRadarDataByExtent(radar_nc_path, save_path, extent):
    x_min, x_max, y_min, y_max = extent

    scale = 111.194925
    nc_ds = nc.Dataset(radar_nc_path, "r")
    lon = nc_ds.Longitude
    lat = nc_ds.Latitude

    zDr = np.array(nc_ds.variables["DifferentialReflectivity"])
    Phi_dp = np.array(nc_ds.variables["DifferentialPhase"])
    KDP = np.array(nc_ds.variables["KDP"])
    reflectivity = np.array(nc_ds.variables["Reflectivity"])

    '''
	zDr_loc = np.where(zDr.flatten() != -8.125)[0]
	reflect_loc = np.where(reflectivity.flatten() != -33.0)[0]
	Kdp_loc = np.where(KDP.flatten() != -5.0)[0]
	sample_loc = reduce(np.union1d, (zDr_loc, reflect_loc, Kdp_loc))
	'''

    # ------convert mm to m
    GateWidth = np.array(nc_ds.variables["GateWidth"]) / 1000.0
    w_r = GateWidth[0] / scale / 1000.0
    dis_cum = np.cumsum(np.full(Phi_dp.shape[1], w_r))

    # ---Determine the center coordinates of observed radar data
    azimuth = np.array(nc_ds.variables["Azimuth"]) * np.pi / 180.0
    x_obs_loc = lon + np.outer(np.cos(azimuth), dis_cum)
    y_obs_loc = lat + np.outer(np.sin(azimuth), dis_cum)
    xy_obs_loc = np.concatenate((np.reshape(x_obs_loc.flatten(), (-1, 1)), np.reshape(y_obs_loc.flatten(), (-1, 1))),
                                axis=1)
    # xy_obs_loc = xy_obs_loc[sample_loc]

    # ---Determine the center coordinates of cells which are to be interpolated
    num_row = int(np.ceil((y_max - y_min) / w_r))
    num_col = int(np.ceil((x_max - x_min) / w_r))

    x_center_loc = np.linspace(x_min, x_min + (num_col - 1) * w_r, num_col) + 0.5 * w_r
    y_center_loc = np.linspace(y_max - (num_row - 1) * w_r, y_max, num_row) - 0.5 * w_r
    x_center_loc, y_center_loc = np.meshgrid(x_center_loc, y_center_loc)
    xy_center_loc = np.concatenate(
        (np.reshape(x_center_loc.flatten(), (-1, 1)), np.reshape(y_center_loc.flatten(), (-1, 1))), axis=1)

    # ---construct an Interpolation Object
    interp_obj = ipol.Nearest(xy_obs_loc, xy_center_loc)

    # zH_interploate = OrdinaryKriging_obj(reflectivity.flatten()[sample_loc])
    zH_interploate = interp_obj(reflectivity.flatten())
    zH_interploate = np.reshape(zH_interploate, (num_row, num_col))

    # zDr_interploate = OrdinaryKriging_obj(zDr.flatten()[sample_loc])
    zDr_interploate = interp_obj(zDr.flatten())
    zDr_interploate = np.reshape(zDr_interploate, (num_row, num_col))

    # Kdp_interploate = OrdinaryKriging_obj(KDP.flatten()[sample_loc])
    Kdp_interploate = interp_obj(KDP.flatten())
    Kdp_interploate = np.reshape(Kdp_interploate, (num_row, num_col))

    # ---save the output .nc file
    output_ds = nc.Dataset(save_path, mode="w")

    output_ds.NetCDFRevision = "lyy_thu_data"
    output_ds.GenDate = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_ds.num_col = num_col
    output_ds.num_row = num_row
    output_ds.Resolution = w_r
    output_ds.X_min = x_min
    output_ds.Y_max = y_max

    RowDimId = output_ds.createDimension("row", num_row)
    ColDimId = output_ds.createDimension("col", num_col)

    zh_id = output_ds.createVariable("Reflectivity", np.float64, ("row", "col"))
    zh_id.Units = "dBz"
    zh_id[:, :] = zH_interploate

    zdr_id = output_ds.createVariable("DifferentialReflectivity", np.float64, ("row", "col"))
    zdr_id.Units = "dB"
    zdr_id[:, :] = zDr_interploate

    kdp_id = output_ds.createVariable("KDP", np.float64, ("row", "col"))
    kdp_id.Units = "unitless"
    kdp_id[:, :] = Kdp_interploate

    output_ds.close()
    nc_ds.close()


def exportToPNG(nc_dir, export_dir, field="Reflectivity", scale=1.0):
    nc_list = sorted([f for f in os.listdir(nc_dir) if f.endswith("nc")])
    for f in nc_list:
        nc_ds = nc.Dataset(os.path.join(nc_dir, f), "r")
        zH = np.array(nc_ds.variables[field]) * scale

        d = f.split("_")[1]
        t = f.split("_")[2]
        obs_time = datetime.datetime.strptime(d + " " + t, "%Y%m%d %H%M%S")
        obs_title = "$Z_{H}$(dBZ) at " + obs_time.strftime("%Y/%m/%d %H:%M:%S")
        plot_ZH(zH, os.path.join(export_dir, "THU_" + d + "_" + t + ".png"), title=obs_title)
        nc_ds.close()


def PNGToGIF(png_dir, save_path):
    images = [f for f in os.listdir(png_dir) if f.endswith(".png")]
    images = sorted(images)

    with imageio.get_writer(save_path, mode="I", duration=0.3) as writer:
        for i in images:
            im = imageio.imread(os.path.join(png_dir, i))
            writer.append_data(im)


if __name__ == "__main__":
	'''
	# [1] Radar Raw Data -> NetCDF Dataset (batch processing)
	dta_dir = "*******Write Your Directory of Radar Raw Data which ends with .AR2 Here*******"
	for f in os.listdir(dta_dir):
		if f.endswith(".AR2"):
			num_gate = 1000
			elev_id = 3
			newNetCDFFile = "_".join(f.split(".")[:-1]) + "_" + str(elev_id) + "_" + str(num_gate) + ".nc"
			newNetCDFFile = os.path.join("Temp", newNetCDFFile)
			site, task, elev, rad, data_pol = MetSTARDataReader(os.path.join(dta_dir, f))
			DROPsNetCDFGen(newNetCDFFile, site, task, elev, rad, data_pol, num_gate=num_gate, elev_id=elev_id)
	'''
	
	'''
	# [2] NetCDF Dataset after Quality Control -> NetCDF Dataset cliped by a specified bounding box (batch processing)
	for f in os.listdir("Temp"):
		if f.startswith("QC"):
			station_name = f.split("_")[1]
			d = f.split("_")[2]
			t = f.split("_")[3]
			save_path = os.path.join("Temp_Output", "THU_"+d+"_"+t+"_"+station_name+".nc")
			getRadarDataByExtent(radar_nc_path=os.path.join("Temp", f), save_path=save_path, extent=[116.297059, 116.344256, 39.983916, 40.020212])
	'''
