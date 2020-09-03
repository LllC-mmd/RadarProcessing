import os
import datetime
import bz2
import pandas as pd
import numpy as np

import netCDF4 as nc

import matplotlib.pyplot as plt


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
	f = pd.DataFrame(columns=['radinfo', 'dbt', 'dbz', 'zdr', 'cc', 'phidp', 'kdp'])

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
	for ct in range(1, int(taskinfo['cutnum'])+1):
		fid.seek(pt_1st_ele + (ct-1) * 256, 0)
		info_dict = {}
		# ------process mode
		info_dict['mode'] = np.fromfile(fid, np.int32, 1)[0]
		# ------wave form
		info_dict['waveform'] = np.fromfile(fid, np.int32, 1)[0]
		# ------pulse repetition frequency 1 (PRF1)
		info_dict['prf1'] = np.fromfile(fid,np.float32, 1)[0]
		# ------pulse repetition frequency 2 (PRF2)
		info_dict['prf2'] = np.fromfile(fid, np.int32, 1)[0]
		# ------de-aliasing mode
		info_dict['unfoldmode'] = np.fromfile(fid, np.int32, 1)[0]
		# ------azimuth for RHI mode
		info_dict['azi'] = np.fromfile(fid, np.float32, 1)[0]
		# ------elevation for PPI mode
		info_dict['ele'] = np.fromfile(fid, np.float32, 1)[0]
		# ------start angle, i.e., start azimuth for PPI mode or highest elevation for RHI mode
		info_dict['startangle'] = np.fromfile(fid, np.int32, 1)[0]
		# ------end angle, i.e., end azimuth for PPI mode or lowest elevation for RHI mode
		info_dict['endangle'] = np.fromfile(fid, np.float32, 1)[0]
		# ------angular resolution (only for PPI mode)
		info_dict['angleres'] = np.fromfile(fid, np.float32, 1)[0]
		# ------scan speed
		info_dict['scanspeed'] = np.fromfile(fid, np.float32, 1)[0]
		# ------log resolution
		info_dict['logres'] = np.fromfile(fid, np.float32, 1)[0]
		# ------Doppler Resolution
		info_dict['dopres'] = np.fromfile(fid, np.float32, 1)[0]
		# ------Maximum Range corresponding to PRF1
		info_dict['maxrange1'] = np.fromfile(fid, np.float32, 1)[0]
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
		info_dict['atmosloss'] = np.fromfile(fid, np.int32, 1)[0]
		# ------Nyquist speed
		info_dict['vmax'] = np.fromfile(fid, np.int32, 1)[0]
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
		f['dbt'].loc[i] = {}
		f['dbz'].loc[i] = {}
		f['zdr'].loc[i] = {}
		f['cc'].loc[i] = {}
		f['phidp'].loc[i] = {}
		f['kdp'].loc[i] = {}

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
					# Note !!!
					# When unsigned int8 meets int32, numpy might give some mis-conversion
					f[variable_encode[var_type]].loc[int(elenum)][str(int(radcnt))] = (1.0 * data_raw - offset)/scale

		a += 1
		if state == 6 or state == 4:
			break

	return siteinfo, taskinfo, eleinfo, radinfo, f


def DROPsNetCDFGen(nc_name, siteinfo, taskinfo, eleinfo, radinfo, radar_data):
	nc_ds = nc.Dataset(nc_name, mode="w")

	nc_ds.NetCDFRevision = "lyy_data"
	nc_ds.GenDate = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
	nc_ds.RadarName = siteinfo['name']
	nc_ds.Latitude = siteinfo['lat']
	nc_ds.Longitude = siteinfo['lon']
	nc_ds.Height = siteinfo['atennaasl']
	nc_ds.NumGates = 1000
	nc_ds.ScanId = 3
	nc_ds.ScanFlag = 1
	nc_ds.ScanType = taskinfo['scantype']
	nc_ds.AntennaGain = 44.7
	nc_ds.AntennaBeamwidth = 0.94

	RadDimId = nc_ds.createDimension("Radial", )
	GatDimId = nc_ds.createDimension("Gate", )


radarDir = "rawData/Xband"
radarDataFile = "BJXSY.20170822.080000.AR2"

site, task, elev, rad, data_pol = MetSTARDataReader(os.path.join(radarDir, radarDataFile))

counter = 1
for i in data_pol["kdp"].loc[11]["365"]:
	print(counter, i)
	counter += 1
